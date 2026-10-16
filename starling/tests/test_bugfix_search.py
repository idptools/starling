"""
Regression tests for the similarity-search engine and the converter / pretokenize CLIs.

Everything here runs on CPU against tiny synthetic shards and indexes built in a
temporary directory. The STARLING encoder is never loaded: wherever the code under
test would encode sequences, ``sequence_encoder_backend`` is replaced by a fake
that maps each sequence to a fixed pseudo-random vector. The stored index vectors
are made with the same fake, so "exact" rerank scores are known exactly.
"""

import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

# faiss-cpu and torch each ship their own OpenMP runtime on macOS; loading both in one
# process aborts (or segfaults) unless duplicate runtimes are allowed and OpenMP is kept
# single-threaded. These must be set before faiss is first imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss  # noqa: E402

from starling.configs import DEFAULT_IONIC_STRENGTH  # noqa: E402
from starling.inference import generation  # noqa: E402
from starling.scripts import starling_converter, starling_pretokenize, starling_search  # noqa: E402
from starling.search import IndexBuilder, SearchEngine, SequenceStore  # noqa: E402
from starling.search.search_utils import Candidate, SequenceIdentityFilter  # noqa: E402
from starling.structure import ensemble  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic data parameters (all tiny, chosen only so FAISS can train quickly)
# ---------------------------------------------------------------------------
EMBED_DIM = 8
N_PER_SHARD = 300
DEFAULT_SHARD_DIRS = ("uniref50_idrs_only_000000", "uniref50_idrs_only_000001")
N_TOTAL = N_PER_SHARD * len(DEFAULT_SHARD_DIRS)
SEQ_LEN_RANGE = (10, 40)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DATA_SEED = 0

# IVF-PQ settings small enough for N_TOTAL training vectors
NLIST = 4
PQ_M = 2
PQ_NBITS = 4
BUILD_NPROBE = 2

# overfetch large enough that k * overfetch covers every vector in the index
TOP_K = 5
OVERFETCH_ALL = N_TOTAL

# The fake encoder shifts every embedding component by this much per mM away from
# DEFAULT_IONIC_STRENGTH, so vectors made at different ionic strengths differ
IONIC_SHIFT_PER_MM = 0.001
QUERY_IONIC_STRENGTH_MM = 20

# Tiny CA-only trajectory written for the converter tests: an extended chain with
# 0.38 nm between consecutive CA atoms (the CA-CA virtual bond length) plus small
# per-frame noise
TEST_SEQUENCE = "ASPAPSPASPASPASPAP"
N_TRAJ_FRAMES = 4
CA_CA_SPACING_NM = 0.38
COORD_NOISE_NM = 0.02

# Inter-residue spacings (Angstrom) for synthetic distance maps: 3.8 is a plausible
# extended chain, 10 is far beyond the per-bond bound used by the error check
GOOD_SPACING_A = 3.8
BAD_SPACING_A = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_embedding(sequence: str, ionic_strength: float = DEFAULT_IONIC_STRENGTH) -> torch.Tensor:
    """
    Return a deterministic stand-in for the STARLING embedding of a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    ionic_strength : float
        Ionic strength in mM; each component is shifted by ``IONIC_SHIFT_PER_MM``
        per mM away from ``DEFAULT_IONIC_STRENGTH``.

    Returns
    -------
    torch.Tensor
        Vector of shape (EMBED_DIM,).
    """
    seed = int(hashlib.md5(sequence.encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(EMBED_DIM, generator=generator)
    return base + (ionic_strength - DEFAULT_IONIC_STRENGTH) * IONIC_SHIFT_PER_MM


class FakeEncoder:
    """Drop-in replacement for ``sequence_encoder_backend`` that records every call."""

    def __init__(self) -> None:
        """Start with an empty call log of (sequence names, ionic strength) pairs."""
        self.calls: List[Tuple[List[str], float]] = []

    def __call__(
        self,
        sequence_dict: Dict[str, str],
        device: str,
        batch_size: int,
        ionic_strength: float,
        aggregate: bool = True,
        output_directory: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode sequences with ``_fake_embedding`` and log the call.

        Parameters
        ----------
        sequence_dict : dict
            Mapping name -> sequence.
        device, batch_size, aggregate, output_directory, kwargs
            Accepted for signature compatibility and ignored.
        ionic_strength : float
            Ionic strength in mM.

        Returns
        -------
        dict
            Mapping name -> embedding.
        """
        self.calls.append((list(sequence_dict), ionic_strength))
        return {name: _fake_embedding(seq, ionic_strength) for name, seq in sequence_dict.items()}


def _random_sequences(n_sequences: int, seed: int) -> List[str]:
    """
    Return reproducible random amino acid sequences.

    Parameters
    ----------
    n_sequences : int
        Number of sequences.
    seed : int
        Seed for the generator.

    Returns
    -------
    list of str
        Sequences with lengths drawn uniformly from ``SEQ_LEN_RANGE``.
    """
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(n_sequences):
        length = int(rng.integers(SEQ_LEN_RANGE[0], SEQ_LEN_RANGE[1] + 1))
        sequences.append("".join(rng.choice(list(AMINO_ACIDS), size=length)))
    return sequences


def _write_shards(
    root: str,
    tokens_dir: Optional[str],
    shard_dirs: Tuple[str, ...] = DEFAULT_SHARD_DIRS,
    seed: int = DATA_SEED,
) -> Dict[str, str]:
    """
    Write feature shards (and optionally tokens files) for a synthetic dataset.

    Features are stored keyed by header in the reverse order of the tokens records,
    so the builder has to pair them by header rather than by position.

    Parameters
    ----------
    root : str
        Directory that will contain ``<shard dir>/sequence_features.pt``.
    tokens_dir : str or None
        If given, ``<shard dir>.tokens.pt`` is written here for every shard.
    shard_dirs : tuple of str
        Shard directory names.
    seed : int
        Seed for the random sequences.

    Returns
    -------
    dict
        Mapping header -> sequence over all shards.
    """
    header_to_seq: Dict[str, str] = {}
    sequences = _random_sequences(N_PER_SHARD * len(shard_dirs), seed)
    for shard_index, shard_dir in enumerate(shard_dirs):
        shard_seqs = sequences[shard_index * N_PER_SHARD : (shard_index + 1) * N_PER_SHARD]
        records = [
            {"header": f"{shard_dir}_seq{i}", "sequence": seq, "tokenized": [0]}
            for i, seq in enumerate(shard_seqs)
        ]
        header_to_seq.update({r["header"]: r["sequence"] for r in records})

        features = {r["header"]: _fake_embedding(r["sequence"]) for r in reversed(records)}
        os.makedirs(os.path.join(root, shard_dir), exist_ok=True)
        torch.save({"features": features}, os.path.join(root, shard_dir, "sequence_features.pt"))

        if tokens_dir is not None:
            os.makedirs(tokens_dir, exist_ok=True)
            torch.save(records, os.path.join(tokens_dir, f"{shard_dir}.tokens.pt"))
    return header_to_seq


def _build_kwargs(**overrides) -> dict:
    """
    Return the tiny-index ``build_index`` keyword arguments used throughout.

    Parameters
    ----------
    overrides
        Keyword arguments that replace the defaults.

    Returns
    -------
    dict
        Keyword arguments for ``IndexBuilder.build_index``.
    """
    kwargs = dict(
        sample_size=N_TOTAL,
        nlist=NLIST,
        m=PQ_M,
        nbits=PQ_NBITS,
        use_gpu=False,
        nprobe=BUILD_NPROBE,
        use_opq=False,
    )
    kwargs.update(overrides)
    return kwargs


def _stored_vectors(engine: SearchEngine) -> torch.Tensor:
    """
    Return the exact (unquantized) vector of every gid, in gid order.

    Parameters
    ----------
    engine : SearchEngine
        Engine with an attached sequence store.

    Returns
    -------
    torch.Tensor
        Shape (engine.total, EMBED_DIM).
    """
    assert engine.seq_store is not None
    return torch.stack([_fake_embedding(engine.seq_store.get_seq(gid)) for gid in range(engine.total)])


def _building_files(directory: str) -> List[str]:
    """
    List leftover SequenceStore temp files in a directory.

    Parameters
    ----------
    directory : str
        Directory to inspect.

    Returns
    -------
    list of str
        File names containing ``.building.``.
    """
    return [name for name in os.listdir(directory) if ".building." in name]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built_indexes(tmp_path_factory) -> Dict[str, str]:
    """
    Build one L2 index (no OPQ) and one cosine index (with OPQ) over the same shards.

    Returns
    -------
    dict
        Paths keyed by 'root', 'tokens', 'l2' and 'cosine'.
    """
    base = tmp_path_factory.mktemp("search_indexes")
    root = str(base / "features")
    tokens = str(base / "tokens")
    _write_shards(root, tokens)

    paths = {"root": root, "tokens": tokens}
    for metric, use_opq in (("l2", False), ("cosine", True)):
        index_path = str(base / f"{metric}.faiss")
        builder = IndexBuilder(root=root, metric=metric, verbose=False)
        builder.build_index(index_path=index_path, tokens_dir=tokens, **_build_kwargs(use_opq=use_opq))
        paths[metric] = index_path
    return paths


@pytest.fixture
def fake_encoder(monkeypatch) -> FakeEncoder:
    """Replace the STARLING encoder backend with a recording ``FakeEncoder``."""
    encoder = FakeEncoder()
    monkeypatch.setattr(generation, "sequence_encoder_backend", encoder)
    return encoder


# ---------------------------------------------------------------------------
# SearchEngine: metric
# ---------------------------------------------------------------------------
def test_load_uses_index_metric_by_default(built_indexes):
    """With no metric given, load() takes the metric the index was built with."""
    assert SearchEngine.load(built_indexes["l2"], verbose=False).metric == "l2"
    assert SearchEngine.load(built_indexes["cosine"], verbose=False).metric == "cosine"


def test_load_rejects_explicit_metric_mismatch(built_indexes):
    """Querying an L2 index as cosine used to silently return the wrong neighbours."""
    with pytest.raises(ValueError, match="built for 'l2'"):
        SearchEngine.load(built_indexes["l2"], metric="cosine", verbose=False)


def test_load_rejects_manifest_that_disagrees_with_index(built_indexes, tmp_path):
    """A manifest recording a different metric than the index is refused."""
    index_copy = str(tmp_path / "copy.faiss")
    faiss.write_index(faiss.read_index(built_indexes["l2"]), index_copy)
    with open(built_indexes["l2"] + ".manifest.json") as fh:
        manifest = json.load(fh)
    manifest["metric"] = "cosine"
    with open(index_copy + ".manifest.json", "w") as fh:
        json.dump(manifest, fh)

    with pytest.raises(ValueError, match="do not belong together"):
        SearchEngine.load(index_copy, verbose=False)


def test_manifest_records_opq_only_when_used(built_indexes):
    """The manifest used to claim OPQ even for plain IVF-PQ builds."""
    with open(built_indexes["l2"] + ".manifest.json") as fh:
        assert json.load(fh)["opq"] is None
    with open(built_indexes["cosine"] + ".manifest.json") as fh:
        assert json.load(fh)["opq"] == {"m": PQ_M}


# ---------------------------------------------------------------------------
# SearchEngine: nprobe under length gating
# ---------------------------------------------------------------------------
def test_length_gated_search_keeps_index_nprobe(built_indexes):
    """With nprobe=None the selector path used FAISS's default nprobe=1, not the index's."""
    engine = SearchEngine.load(built_indexes["l2"], verbose=False)
    faiss.extract_index_ivf(engine.index).nprobe = NLIST  # probe every list
    queries = np.random.default_rng(DATA_SEED).standard_normal((3, EMBED_DIM)).astype("float32")
    all_gids = list(range(engine.total))

    _, idxs = engine._ann_search(queries, fetch_k=engine.total, nprobe=None, selector_ids=all_gids)

    # probing all lists reaches every vector; nprobe=1 would reach only one list's worth
    assert ((idxs >= 0).sum(axis=1) == engine.total).all()


# ---------------------------------------------------------------------------
# SearchEngine: thresholds after rerank
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_rerank_applies_thresholds_to_exact_scores(built_indexes, fake_encoder, metric):
    """
    After rerank, every hit must honour the score threshold on its exact score, and
    with the whole index in the pool the result must equal the brute-force answer.
    """
    engine = SearchEngine.load(built_indexes[metric], verbose=False)
    stored = _stored_vectors(engine)
    queries = torch.from_numpy(np.random.default_rng(1).standard_normal((4, EMBED_DIM)).astype("float32"))

    if metric == "l2":
        exact = torch.cdist(queries, stored)
        threshold = float(torch.quantile(exact, 0.3))
        kwargs = {"min_l2_distance": threshold}
        passes = exact >= threshold
        descending = False
    else:
        exact = torch.nn.functional.normalize(queries, dim=1) @ torch.nn.functional.normalize(stored, dim=1).T
        threshold = float(torch.quantile(exact, 0.7))
        kwargs = {"max_cosine_similarity": threshold, "return_similarity": True}
        passes = exact <= threshold
        descending = True

    results = engine.search(queries, k=TOP_K, nprobe=NLIST, rerank=True, overfetch=OVERFETCH_ALL, **kwargs)

    for qi, row in enumerate(results):
        scores = np.array([score for score, *_ in row])
        if metric == "l2":
            assert (scores >= threshold).all()
        else:
            assert (scores <= threshold).all()

        masked = torch.where(passes[qi], exact[qi], torch.tensor(float("nan")))
        candidate_gids = torch.nonzero(passes[qi]).flatten()
        order = torch.argsort(masked[candidate_gids], descending=descending)
        expected_gids = candidate_gids[order][:TOP_K].tolist()
        assert [gid for _, gid, _, _ in row] == expected_gids
        np.testing.assert_allclose(scores, exact[qi, expected_gids].numpy(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# SearchEngine: rerank ionic strength
# ---------------------------------------------------------------------------
def test_rerank_compares_query_and_candidates_at_same_ionic_strength(built_indexes, fake_encoder):
    """
    Query vectors made at 20 mM used to be compared with candidates re-encoded at
    150 mM. The query is now re-encoded at the rerank ionic strength as well.
    """
    engine = SearchEngine.load(built_indexes["l2"], verbose=False)
    assert engine.seq_store is not None
    target_gid = 7
    target_seq = engine.seq_store.get_seq(target_gid)
    query = _fake_embedding(target_seq, QUERY_IONIC_STRENGTH_MM).unsqueeze(0)

    results = engine.search(
        query, k=TOP_K, nprobe=NLIST, query_sequences=[target_seq], rerank=True, overfetch=OVERFETCH_ALL
    )

    top_score, top_gid, _, _ = results[0][0]
    assert top_gid == target_gid
    assert top_score == pytest.approx(0.0, abs=1e-5)

    # every rerank encoding (candidates and query) used the same ionic strength
    assert {ionic for _, ionic in fake_encoder.calls} == {DEFAULT_IONIC_STRENGTH}
    assert any("q0" in names for names, _ in fake_encoder.calls)


# ---------------------------------------------------------------------------
# SearchEngine: sequence filters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [{"exclude_exact": True}, {"sequence_identity_max": 0.9}])
def test_sequence_filters_require_query_sequences(built_indexes, kwargs):
    """These filters used to be silent no-ops without query_sequences."""
    engine = SearchEngine.load(built_indexes["l2"], verbose=False)
    with pytest.raises(ValueError, match="require query_sequences"):
        engine.search(torch.zeros(1, EMBED_DIM), k=TOP_K, **kwargs)


def test_query_sequences_must_align_with_queries(built_indexes):
    """A query_sequences list of the wrong length is refused rather than misaligned."""
    engine = SearchEngine.load(built_indexes["l2"], verbose=False)
    with pytest.raises(ValueError, match="aligned one-to-one"):
        engine.search(torch.zeros(2, EMBED_DIM), k=TOP_K, query_sequences=["ACDE"], exclude_exact=True)


def test_sequence_identity_max_is_inclusive():
    """A candidate exactly at sequence_identity_max is kept (it is the maximum allowed)."""

    class StubStore:
        """Minimal store returning a fixed sequence."""

        def get_seq(self, gid: int) -> str:
            """Return a fixed candidate sequence."""
            return "AAAA"

    identity_filter = SequenceIdentityFilter(
        max_identity=0.5,
        denominator="query",
        seq_store=StubStore(),
        identity_func=lambda query, target, denom: 0.5,
    )
    candidate = Candidate(score=0.0, gid=0, header=None, length=4)
    assert identity_filter.apply(candidate, query_seq="AAAA")


# ---------------------------------------------------------------------------
# IndexBuilder
# ---------------------------------------------------------------------------
def test_duplicate_shard_ids_are_rejected(tmp_path):
    """Two copies of a shard under root used to be indexed with one shared header list."""
    _write_shards(str(tmp_path / "root" / "old"), None, shard_dirs=(DEFAULT_SHARD_DIRS[0],))
    _write_shards(str(tmp_path / "root" / "new"), None, shard_dirs=(DEFAULT_SHARD_DIRS[0],))
    with pytest.raises(ValueError, match="Duplicate shard ids"):
        IndexBuilder(root=str(tmp_path / "root"), metric="l2", verbose=False)


def test_tokens_filename_follows_shard_regex(tmp_path):
    """Tokens files are named after the matched shard directory, not a hard-coded pattern."""
    root, tokens = str(tmp_path / "root"), str(tmp_path / "tokens")
    header_to_seq = _write_shards(root, tokens, shard_dirs=("myshard_7", "myshard_8"))

    builder = IndexBuilder(root=root, metric="l2", verbose=False, shard_id_regex=r"myshard_(\d+)")
    index_path = str(tmp_path / "custom.faiss")
    builder.build_index(index_path=index_path, tokens_dir=tokens, **_build_kwargs())

    store = SequenceStore.open_reader(index_path + ".seqs.sqlite")
    for gid in range(N_TOTAL):
        header, _ = store.get_header_len(gid)
        assert store.get_seq(gid) == header_to_seq[header]


def test_bad_tokens_fail_before_training(tmp_path, monkeypatch):
    """A missing tokens file is reported before training and leaves nothing behind."""
    root, tokens = str(tmp_path / "root"), str(tmp_path / "tokens")
    _write_shards(root, tokens)
    os.remove(os.path.join(tokens, f"{DEFAULT_SHARD_DIRS[1]}.tokens.pt"))

    def _training_must_not_start(*args, **kwargs):
        """Fail the test if the builder reaches training."""
        raise AssertionError("training started before the tokens were validated")

    monkeypatch.setattr(IndexBuilder, "_create_and_train_index", _training_must_not_start)
    builder = IndexBuilder(root=root, metric="l2", verbose=False)
    index_path = str(tmp_path / "idx.faiss")
    with pytest.raises(FileNotFoundError, match="Tokens file missing"):
        builder.build_index(index_path=index_path, tokens_dir=tokens, **_build_kwargs())

    assert not os.path.exists(index_path)
    assert _building_files(str(tmp_path)) == []


def test_failed_sequence_store_build_removes_temp_file(tmp_path, monkeypatch):
    """A failure while writing the store used to leave an orphaned *.building.* file."""
    root, tokens = str(tmp_path / "root"), str(tmp_path / "tokens")
    _write_shards(root, tokens)
    builder = IndexBuilder(root=root, metric="l2", verbose=False)

    def _failing_insert(self, rows):
        """Simulate a write failure part-way through the build."""
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(SequenceStore, "insert_rows", _failing_insert)
    db_path = str(tmp_path / "idx.faiss.seqs.sqlite")
    with pytest.raises(RuntimeError, match="simulated write failure"):
        builder.build_sequence_store(db_path, tokens, batch=10)

    assert not os.path.exists(db_path)
    assert _building_files(str(tmp_path)) == []


@pytest.mark.skipif(faiss.get_num_gpus() > 0, reason="only meaningful without FAISS GPUs")
def test_use_gpu_without_gpus_fails_clearly(tmp_path, monkeypatch):
    """use_gpu=True on faiss-cpu used to die with an AttributeError after sampling."""
    root = str(tmp_path / "root")
    _write_shards(root, None)
    builder = IndexBuilder(root=root, metric="l2", verbose=False)
    with pytest.raises(RuntimeError, match="no GPUs"):
        builder.build_index(index_path=str(tmp_path / "idx.faiss"), **_build_kwargs(use_gpu=True))


# ---------------------------------------------------------------------------
# starling-search CLI
# ---------------------------------------------------------------------------
def test_cli_seq_accepts_repeated_and_multiple_values():
    """`--seq A --seq B` used to keep only B."""
    repeated = starling_search.parse_args(["query", "--seq", "AAA", "--seq", "CCC"])
    grouped = starling_search.parse_args(["query", "--seq", "AAA", "CCC"])
    assert repeated.seq == ["AAA", "CCC"]
    assert grouped.seq == ["AAA", "CCC"]


def test_cli_defaults_match_api():
    """Query metric/device default to auto; build GPU defaults to auto and OPQ to on."""
    query = starling_search.parse_args(["query", "--seq", "AAA"])
    assert query.metric is None
    assert query.device is None
    assert query.rerank_ionic_strength is None

    build = starling_search.parse_args(["build", "--root", "r", "--index", "i", "--tokens", "t"])
    assert build.use_gpu is None
    assert build.opq is True
    no_opq = starling_search.parse_args(["build", "--root", "r", "--index", "i", "--tokens", "t", "--no-opq"])
    assert no_opq.opq is False


def test_cli_query_uses_index_metric_and_query_ionic_strength(built_indexes, fake_encoder, tmp_path):
    """
    End to end: both repeated --seq queries are searched, the L2 index is queried as L2
    (the old CLI default was cosine), and the rerank runs at the query ionic strength.
    """
    engine = SearchEngine.load(built_indexes["l2"], verbose=False)
    assert engine.seq_store is not None
    query_seqs = [engine.seq_store.get_seq(3), engine.seq_store.get_seq(400)]
    out_base = str(tmp_path / "hits")

    rc = starling_search.main(
        [
            "query",
            "--index", built_indexes["l2"],
            "--seq", query_seqs[0],
            "--seq", query_seqs[1],
            "--ionic-strength", str(QUERY_IONIC_STRENGTH_MM),
            "--device", "cpu",
            "--k", str(TOP_K),
            "--nprobe", str(NLIST),
            "--out", out_base,
            "--quiet",
        ]
    )
    assert rc == 0

    with open(out_base + ".csv") as fh:
        rows = fh.read().splitlines()[1:]
    query_indices = {int(row.split(",")[0]) for row in rows}
    assert query_indices == {0, 1}
    # L2 results carry no similarity column
    assert all(row.split(",")[5] == "" for row in rows)
    assert {ionic for _, ionic in fake_encoder.calls} == {QUERY_IONIC_STRENGTH_MM}


def test_cli_query_rejects_metric_mismatch(built_indexes, fake_encoder):
    """An explicit --metric that disagrees with the index is an error, not wrong hits."""
    with pytest.raises(ValueError, match="built for 'l2'"):
        starling_search.main(["query", "--index", built_indexes["l2"], "--metric", "cosine", "--seq", "ACDE"])


def test_cli_build_defaults_run_on_faiss_cpu(tmp_path):
    """The CLI build used to default to use_gpu=True (crash on faiss-cpu) and to no OPQ."""
    root, tokens = str(tmp_path / "root"), str(tmp_path / "tokens")
    _write_shards(root, tokens)
    index_path = str(tmp_path / "cli.faiss")

    rc = starling_search.main(
        [
            "build",
            "--root", root,
            "--index", index_path,
            "--tokens", tokens,
            "--sample-size", str(N_TOTAL),
            "--nlist", str(NLIST),
            "--m", str(PQ_M),
            "--nbits", str(PQ_NBITS),
            "--quiet",
        ]
    )
    assert rc == 0
    with open(index_path + ".manifest.json") as fh:
        assert json.load(fh)["opq"] == {"m": PQ_M}
    assert os.path.exists(index_path + ".seqs.sqlite")


# ---------------------------------------------------------------------------
# Converter CLIs
# ---------------------------------------------------------------------------
def _distance_map(spacing_A: float, n_residues: int) -> np.ndarray:
    """
    Build a symmetric distance map with a fixed per-bond spacing.

    Parameters
    ----------
    spacing_A : float
        Distance between consecutive residues in Angstrom.
    n_residues : int
        Number of residues.

    Returns
    -------
    np.ndarray
        Shape (n_residues, n_residues).
    """
    separation = np.abs(np.subtract.outer(np.arange(n_residues), np.arange(n_residues)))
    return (separation * spacing_A).astype(np.float32)


@pytest.fixture
def ca_trajectory(tmp_path) -> Tuple[str, str]:
    """
    Write a tiny CA-only PDB topology and XTC trajectory of ``TEST_SEQUENCE``.

    Returns
    -------
    tuple of str
        (pdb_path, xtc_path).
    """
    from starling.structure.coordinates import create_ca_topology_from_coords

    rng = np.random.default_rng(DATA_SEED)
    n_residues = len(TEST_SEQUENCE)
    extended = np.zeros((n_residues, 3))
    extended[:, 0] = np.arange(n_residues) * CA_CA_SPACING_NM
    coords = extended + rng.normal(scale=COORD_NOISE_NM, size=(N_TRAJ_FRAMES, n_residues, 3))

    traj = create_ca_topology_from_coords(TEST_SEQUENCE, coords.astype(np.float32))
    pdb_path, xtc_path = str(tmp_path / "traj.pdb"), str(tmp_path / "traj.xtc")
    traj[0].save_pdb(pdb_path)
    traj.save_xtc(xtc_path)
    return pdb_path, xtc_path


def _trajectory_distance_maps(pdb_path: str, xtc_path: str) -> np.ndarray:
    """
    Return the symmetrized per-frame distance maps of a trajectory.

    Parameters
    ----------
    pdb_path : str
        Topology file.
    xtc_path : str
        Trajectory file.

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_residues, n_residues).
    """
    from soursop.sstrajectory import SSTrajectory

    from starling.utilities import symmetrize_distance_maps

    protein = SSTrajectory(xtc_path, pdb_path).proteinTrajectoryList[0]
    maps, _ = protein.get_distance_map(return_instantaneous_maps=True)
    return symmetrize_distance_maps(maps)


def test_starling2starling_remove_errors_implies_check_and_prefixes_basename(tmp_path, monkeypatch):
    """
    --remove-errors without --error-check used to do nothing, and the 'fixed_' prefix
    was put in front of the whole -o path ('fixed_outdir/...').
    """
    n_residues = len(TEST_SEQUENCE)
    maps = np.stack(
        [_distance_map(GOOD_SPACING_A, n_residues)] * 2 + [_distance_map(BAD_SPACING_A, n_residues)]
    )
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    ensemble.Ensemble(maps, TEST_SEQUENCE).save(str(in_dir / "ens"), verbose=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["starling2starling", str(in_dir / "ens.starling"), "-o", str(out_dir), "--remove-errors"]
    )
    starling_converter.starling2starling()

    fixed_path = out_dir / "fixed_ens.starling"
    assert fixed_path.exists()
    assert len(ensemble.load_ensemble(str(fixed_path))) == 2
    assert not (tmp_path / "fixed_out").exists()


def test_numpy2starling_rejects_frame_count_mismatch(tmp_path, monkeypatch, ca_trajectory):
    """The XTC and the distance maps must have the same number of frames."""
    pdb_path, xtc_path = ca_trajectory
    maps = _trajectory_distance_maps(pdb_path, xtc_path)
    npy_path = str(tmp_path / "maps.npy")
    np.save(npy_path, maps[:-1])
    out_path = tmp_path / "from_numpy.starling"

    monkeypatch.setattr(
        sys,
        "argv",
        ["numpy2starling", npy_path, "-x", xtc_path, "-p", pdb_path, "-o", str(out_path)],
    )
    with pytest.raises(SystemExit) as excinfo:
        starling_converter.numpy2starling()
    assert excinfo.value.code == 1
    assert not out_path.exists()


def test_xtc2starling_keeps_input_coordinates(tmp_path, monkeypatch, ca_trajectory):
    """xtc2starling used to drop the trajectory and keep only the distance maps."""
    pdb_path, xtc_path = ca_trajectory
    out_prefix = tmp_path / "from_xtc"
    monkeypatch.setattr(
        sys, "argv", ["xtc2starling", "--xtc", xtc_path, "--pdb", pdb_path, "-o", str(out_prefix)]
    )
    starling_converter.xtc2starling()

    loaded = ensemble.load_ensemble(str(out_prefix) + ".starling")
    assert loaded.has_structures
    assert len(loaded) == N_TRAJ_FRAMES


# ---------------------------------------------------------------------------
# starling-pretokenize
# ---------------------------------------------------------------------------
def _write_fasta(path, records: Dict[str, str]) -> str:
    """
    Write a FASTA file.

    Parameters
    ----------
    path : pathlib.Path
        Output path (parent directories are created).
    records : dict
        Mapping header -> sequence.

    Returns
    -------
    str
        The path as a string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{header}\n{seq}\n" for header, seq in records.items()))
    return str(path)


def test_pretokenize_rejects_duplicate_output_names(tmp_path):
    """a/shard.fasta and b/shard.fasta used to overwrite the same shard.tokens.pt."""
    first = _write_fasta(tmp_path / "a" / "shard.fasta", {"h1": "ACDE"})
    second = _write_fasta(tmp_path / "b" / "shard.fasta", {"h2": "KLMN"})
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit, match="same output file"):
        starling_pretokenize.main([first, second, "-o", str(out_dir), "--no-progress"])
    assert not any(name.endswith(".tokens.pt") for name in os.listdir(out_dir))


def test_pretokenize_rejects_positional_and_list_inputs(tmp_path):
    """Positional FASTAs used to be silently ignored when --sequences was given."""
    first = _write_fasta(tmp_path / "a" / "one.fasta", {"h1": "ACDE"})
    second = _write_fasta(tmp_path / "b" / "two.fasta", {"h2": "KLMN"})
    list_file = tmp_path / "fastas.txt"
    list_file.write_text(second + "\n")

    with pytest.raises(SystemExit, match="not both"):
        starling_pretokenize.main([first, "--sequences", str(list_file), "-o", str(tmp_path / "out")])


def test_pretokenize_unique_names_still_work(tmp_path):
    """Distinct basenames each get their own tokens file."""
    first = _write_fasta(tmp_path / "a" / "one.fasta", {"h1": "ACDE"})
    second = _write_fasta(tmp_path / "b" / "two.fasta", {"h2": "KLMN"})
    out_dir = tmp_path / "out"

    assert starling_pretokenize.main([first, second, "-o", str(out_dir), "--no-progress"]) == 0
    assert sorted(os.listdir(out_dir)) == ["one.tokens.pt", "two.tokens.pt"]
