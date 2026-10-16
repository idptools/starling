"""
FAISS Index Builder
===================

Build high-performance FAISS indexes with OPQ+IVF-PQ compression and integrated
SQLite sequence metadata storage.

Overview
--------
The IndexBuilder creates FAISS indexes optimized for billion-scale similarity search:

* **OPQ (Optimized Product Quantization)**: Learned rotation for better quantization
* **IVF (Inverted File)**: Clustering for fast approximate search
* **PQ (Product Quantization)**: Compression to ~64-128 bytes per vector
* **Sequence Store**: SQLite database with sequences, headers, and metadata

Basic Usage
-----------
>>> from starling.search import IndexBuilder
>>>
>>> # Initialize and discover shards
>>> builder = IndexBuilder(
...     root="/path/to/shards",
...     metric="cosine",
...     verbose=True
... )
>>>
>>> # Build index with sequence store
>>> builder.build_index(
...     index_path="my_index.faiss",
...     tokens_dir="/path/to/tokens",
...     sample_size=655360,     # Training samples
...     nlist=16384,            # IVF clusters
...     m=64,                   # PQ subquantizers
...     nbits=8,                # Bits per subquantizer
...     use_gpu=True,           # GPU acceleration
...     use_opq=True,           # Enable OPQ
...     compress_sequences=True # Compress sequences with zstd
... )

Index Configuration
-------------------
**Training Parameters:**

* ``sample_size``: Number of vectors for training (default: 655,360)

  - Larger = better quantization quality
  - Minimum: 39 * nlist vectors required
  - Recommended: 100-1000 vectors per IVF cluster

* ``nlist``: Number of IVF clusters (default: 16,384)

  - More clusters = faster search but lower recall
  - Rule of thumb: sqrt(N) to N/1000 where N is total vectors
  - Auto-adjusted if training samples insufficient

* ``m``: PQ subquantizers (default: 64)

  - Must divide vector dimension evenly
  - More = better quality but larger memory footprint
  - Typical: 32-128 depending on dimension

* ``nbits``: Bits per subquantizer (default: 8)

  - Controls codebook size: 2^nbits entries per subquantizer
  - 8 bits = 256 entries (standard)
  - 16 bits = 65536 entries (higher quality, more memory)

**GPU Options:**

* ``use_gpu``: Train on GPU (default: auto, i.e. only if FAISS can see a GPU)
* ``gpu_device``: CUDA device ID
* ``gpu_fp16_lut``: Use float16 lookup tables (to address 48KB SMEM limit)

**Sequence Store:**

* ``tokens_dir``: Directory with tokenized sequences (.tokens.pt files)
* ``compress_sequences``: Use zstd compression

File Structure
--------------
The builder expects sharded feature files in this structure::

    root/
    ├── uniref50_idrs_only_000000/
    │   └── sequence_features.pt
    ├── uniref50_idrs_only_000001/
    │   └── sequence_features.pt
    └── ...

Feature files are found recursively under ``root``. Each shard directory name
must match ``shard_id_regex`` (default ``uniref50_idrs_only_(\\d{6})``), whose
first capture group is the numeric shard id. Shard ids must be unique: two
copies of the same shard anywhere under ``root`` is an error. Shards are
indexed in increasing shard-id order, and global ids (gids) are assigned
consecutively in that order.

If ``tokens_dir`` is given, it must contain one tokens file per shard, named
after the shard directory (i.e. the path component in which the shard id was
matched)::

    tokens_dir/
    ├── uniref50_idrs_only_000000.tokens.pt
    ├── uniref50_idrs_only_000001.tokens.pt
    └── ...

This is exactly what ``starling-pretokenize`` writes for a FASTA file named
``<shard directory name>.fasta``. Each tokens file is a list of
``{"header", "sequence", ...}`` records.

Vectors and sequences are paired by header, not by position, so every feature
shard must carry headers that match its tokens file one-to-one. Supported
feature layouts that carry headers are ``{"features": {header: vector}}``,
``{"features": (N, D) tensor, "headers": [...]}``, ``{header: vector}``, or a
list of ``{"header": ..., "embedding": vector}`` records. All tokens files are
checked (existence, record count and headers) before any training starts.

And produces these outputs::

    my_index.faiss              # FAISS index
    my_index.faiss.manifest.json # Metadata
    my_index.faiss.seqs.sqlite  # Sequence store (if tokens_dir provided)


See Also
--------
* :class:`SearchEngine`: Query the built index
* :class:`SequenceStore`: Direct database access
* :mod:`starling.scripts.starling_search`: Command-line interface (``starling-search build``)

Notes
-----
* Use ``verbose=True`` for progress tracking during long builds
* GPU training dramatically faster but index quality identical
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from typing import Any, List, Optional, cast

import numpy as np
import torch
from tqdm import tqdm

from starling.search.store import SequenceStore

try:
    import faiss
except Exception as e:
    raise ImportError(
        "FAISS is required. Install with 'pip install faiss-cpu', or a GPU build of FAISS from conda."
    ) from e

logger = logging.getLogger(__name__)

# Default shard ID regex (captures 6-digit zero-padded numeric id)
_SHARD_ID_RE_DEFAULT = re.compile(r"uniref50_idrs_only_(\d{6})")

# Suffix of the per-shard tokens files. This matches what starling-pretokenize writes
# (<FASTA basename>.tokens.pt), so a shard directory called X pairs with X.tokens.pt.
_TOKENS_SUFFIX = ".tokens.pt"


def _parse_shard_id_from_path(path: str) -> int:
    """
    Extract numeric shard id from a features path like:
    .../uniref50_idrs_only_000123/sequence_features.pt

    Returns an int (e.g., 123). Raises if not found.
    """
    m = _SHARD_ID_RE_DEFAULT.search(path)
    if not m:
        raise ValueError(f"Could not parse shard id from path: {path}")
    return int(m.group(1))


class IndexBuilder:
    """
    Builds FAISS OPQ+IVF-PQ indexes from sharded feature files.
    """

    def __init__(
        self,
        root: str,
        metric: str = "cosine",
        verbose: bool = True,
        shard_id_regex: Optional[str] = None,
    ):
        if metric not in {"l2", "cosine"}:
            raise ValueError("metric must be 'l2' or 'cosine'")

        self.root = root
        self.metric = metric
        self.verbose = verbose
        if verbose:
            self._log = lambda msg, *a, **k: logger.info(str(msg))
        else:
            self._log = lambda *a, **k: None

        # Compile shard id regex (user override or default)
        try:
            self._shard_id_re = (
                re.compile(shard_id_regex) if shard_id_regex else _SHARD_ID_RE_DEFAULT
            )
        except re.error as e:
            raise ValueError(f"Invalid shard_id_regex pattern: {shard_id_regex}: {e}")
        self._shard_id_regex_str = shard_id_regex or _SHARD_ID_RE_DEFAULT.pattern

        self.files: List[str] = []
        self.shard_ids: List[int] = []
        self.shard_dirs: List[str] = []
        self.counts: List[int] = []
        self.feature_headers: dict[int, list[str] | None] = {}
        self.total: int = 0
        self.dim: Optional[int] = None

        self._discover_files()

    def _parse_shard_id(self, path: str) -> int:
        """Extract shard id using configured regex. Raises ValueError if not found."""
        m = self._shard_id_re.search(path)
        if not m:
            raise ValueError(
                f"Could not parse shard id with pattern '{self._shard_id_regex_str}' from path: {path}"
            )
        # Expect first capture group is the numeric id
        try:
            return int(m.group(1))
        except Exception as e:
            raise ValueError(
                f"Shard id capture group not numeric in path: {path}"
            ) from e

    def _shard_dir_name(self, path: str) -> str:
        """
        Return the name of the path component in which the shard id was matched.

        The tokens file for a shard is named after this directory
        (``<shard dir name>.tokens.pt``), so a custom ``shard_id_regex`` works
        for both the feature shards and their tokens files.

        Parameters
        ----------
        path : str
            Path to a ``sequence_features.pt`` file.

        Returns
        -------
        str
            The shard directory name, e.g. ``uniref50_idrs_only_000123`` for the
            default pattern.

        Raises
        ------
        ValueError
            If the pattern does not match the path, or if the shard id was matched
            in the file name rather than in a directory name.
        """
        m = self._shard_id_re.search(path)
        if not m:
            raise ValueError(
                f"Could not parse shard id with pattern '{self._shard_id_regex_str}' from path: {path}"
            )

        # Find the path component that contains the captured shard id
        id_start = m.start(1)
        component_start = path.rfind(os.sep, 0, id_start) + 1
        component_end = path.find(os.sep, id_start)
        if component_end == -1:
            raise ValueError(
                f"Shard id pattern '{self._shard_id_regex_str}' matched the file name rather than a "
                f"shard directory in: {path}"
            )
        return path[component_start:component_end]

    def _extract_features_from_data(self, data: Any, *, path: str) -> torch.Tensor:
        """Accepts various formats; returns (N, D) float32."""
        if isinstance(data, torch.Tensor):
            t = data.float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            if t.dim() != 2:
                raise ValueError(
                    f"Tensor in {path} must be 1D or 2D, got {tuple(t.shape)}"
                )
            return t
        if isinstance(data, dict):
            # Case: data['features'] is a dict of header -> 1D/2D tensor
            if "features" in data and isinstance(data["features"], dict):
                feats_dict = data["features"]
                rows = []
                for h, v in feats_dict.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    vt = v.float()
                    if vt.dim() == 1:
                        vt = vt.unsqueeze(0)
                    if vt.dim() != 2:
                        raise ValueError(
                            f"Feature tensor for header '{h}' in {path} must be 1D or 2D"
                        )
                    rows.append(vt)
                if not rows:
                    raise ValueError(
                        f"No tensor features found in features dict for {path}"
                    )
                return torch.cat(rows, dim=0)
            # Direct tensor under 'features'
            if "features" in data and isinstance(data["features"], torch.Tensor):
                return self._extract_features_from_data(data["features"], path=path)
            # Dict of header -> 1D tensor (legacy case)
            kv = [(k, v) for k, v in data.items() if isinstance(v, torch.Tensor)]
            if kv:
                vecs = []
                for _, v in kv:
                    v = v.float()
                    if v.dim() == 1:
                        v = v.unsqueeze(0)
                    if v.dim() != 2 or v.size(0) != 1:
                        raise ValueError(f"All feature entries must be 1D in {path}")
                    vecs.append(v)
                return torch.cat(vecs, dim=0)
        if isinstance(data, list):
            # List of tensors -> stack
            if all(isinstance(x, torch.Tensor) for x in data) and data:
                proc = []
                for v in data:
                    v = v.float()
                    if v.dim() == 1:
                        v = v.unsqueeze(0)
                    if v.dim() != 2 or v.size(0) != 1:
                        raise ValueError(
                            f"All tensors in list must be 1D; offending shape {tuple(v.shape)} in {path}"
                        )
                    proc.append(v)
                return torch.cat(proc, dim=0)
            # List of dicts containing embedding tensor under common keys
            emb_keys = ("embedding", "feature", "tensor")
            rows = []
            for el in data:
                if isinstance(el, dict):
                    tensor = None
                    for k in emb_keys:
                        if k in el and isinstance(el[k], torch.Tensor):
                            tensor = el[k]
                            break
                    if tensor is not None:
                        t = tensor.float()
                        if t.dim() == 1:
                            t = t.unsqueeze(0)
                        if t.dim() != 2 or t.size(0) != 1:
                            raise ValueError(
                                f"Embedded tensor must be 1D in list element of {path}"
                            )
                        rows.append(t)
                    else:
                        # If dict has exactly one tensor value, use it
                        tv = [v for v in el.values() if isinstance(v, torch.Tensor)]
                        if len(tv) == 1:
                            t = tv[0].float()
                            if t.dim() == 1:
                                t = t.unsqueeze(0)
                            if t.dim() != 2 or t.size(0) != 1:
                                raise ValueError(
                                    f"Tensor must be 1D in list element of {path}"
                                )
                            rows.append(t)
            if rows:
                return torch.cat(rows, dim=0)
        raise ValueError(f"Unsupported .pt format in {path}")

    def _load_features(self, path: str) -> torch.Tensor:
        data = torch.load(path, map_location="cpu")
        return self._extract_features_from_data(data, path=path)

    @staticmethod
    def _feature_headers(data: Any, row_count: int) -> list[str] | None:
        """Return stable feature identifiers when the shard stores them."""
        if isinstance(data, dict):
            features = data.get("features")
            if isinstance(features, dict):
                entries = [
                    (header, value)
                    for header, value in features.items()
                    if isinstance(value, torch.Tensor)
                ]
                if len(entries) == row_count and all(
                    value.ndim == 1 or (value.ndim == 2 and value.shape[0] == 1)
                    for _, value in entries
                ):
                    headers = [str(header) for header, _ in entries]
                    return headers
            elif isinstance(features, torch.Tensor):
                headers = data.get("headers")
                if isinstance(headers, (list, tuple)) and len(headers) == row_count:
                    return [str(header) for header in headers]
            else:
                values = [
                    (header, value)
                    for header, value in data.items()
                    if isinstance(value, torch.Tensor)
                ]
                if len(values) == row_count and all(
                    value.ndim == 1 or (value.ndim == 2 and value.shape[0] == 1)
                    for _, value in values
                ):
                    return [str(header) for header, _ in values]
        elif isinstance(data, list):
            entries = []
            for record in data:
                if not isinstance(record, dict) or record.get("header") is None:
                    return None
                embedding = next(
                    (
                        record[key]
                        for key in ("embedding", "feature", "tensor")
                        if isinstance(record.get(key), torch.Tensor)
                    ),
                    None,
                )
                if embedding is None:
                    tensors = [
                        value
                        for value in record.values()
                        if isinstance(value, torch.Tensor)
                    ]
                    if len(tensors) != 1:
                        return None
                    embedding = tensors[0]
                if not (
                    embedding.ndim == 1
                    or (embedding.ndim == 2 and embedding.shape[0] == 1)
                ):
                    return None
                entries.append(str(record["header"]))
            if len(entries) == row_count:
                return entries
        return None

    def _discover_files(self) -> None:
        """Discover and inventory all feature shard files."""
        t0 = time.time()
        pattern = os.path.join(self.root, "**", "sequence_features.pt")
        found_paths = glob.glob(pattern, recursive=True)

        # Parse shard ids first so duplicates are caught before loading any features
        paths_by_sid: dict[int, List[str]] = {}
        for p in found_paths:
            try:
                sid = self._parse_shard_id(p)
            except ValueError:
                if self.verbose:
                    self._log(f"[DISCOVER][SKIP] Could not parse shard id: {p}")
                continue
            paths_by_sid.setdefault(sid, []).append(p)

        # Two feature files with the same shard id (e.g. an old and a new copy picked up by
        # the recursive glob) would both be indexed but share one header list and one tokens
        # file, which mislabels rows. We refuse rather than guess which copy is the right one.
        duplicates = {sid: ps for sid, ps in paths_by_sid.items() if len(ps) > 1}
        if duplicates:
            details = "; ".join(
                f"shard {sid}: {sorted(ps)}" for sid, ps in sorted(duplicates.items())
            )
            raise ValueError(
                f"Duplicate shard ids found under {self.root} (each shard id must appear once): {details}"
            )

        records = []
        for sid, (p,) in paths_by_sid.items():
            shard_dir = self._shard_dir_name(p)
            data = torch.load(p, map_location="cpu")
            arr = self._extract_features_from_data(data, path=p)
            n, d = arr.shape
            headers = self._feature_headers(data, int(n))
            records.append((sid, p, shard_dir, int(n), int(d), headers))

        # Sort strictly by numeric shard id to keep pairing stable
        records.sort(key=lambda t: t[0])

        if self.verbose:
            self._log(f"[DISCOVER] {len(records)} shard files")

        # Populate class fields in the sorted order
        self.files, self.shard_ids, self.shard_dirs, self.counts = [], [], [], []
        self.dim = None
        for sid, p, shard_dir, n, d, headers in records:
            if self.dim is None:
                self.dim = d
            elif self.dim != d:
                raise ValueError(f"Dim mismatch {self.dim} vs {d} in {p}")
            self.files.append(p)
            self.shard_ids.append(sid)
            self.shard_dirs.append(shard_dir)
            self.counts.append(n)
            self.feature_headers[sid] = headers

        self.total = sum(self.counts)
        self._log(
            f"[DISCOVER] total_vectors={self.total} dim={self.dim} time={time.time() - t0:.2f}s"
        )

    def sample_vectors(
        self, sample_size: int = 100_000, seed: int = 1234
    ) -> torch.Tensor:
        """Randomly sample vectors across all shards for training.

        Parameters
        ----------
        sample_size : int, optional
            Number of vectors to sample, by default 100_000
        seed : int, optional
            Random seed for reproducibility, by default 1234

        Returns
        -------
        torch.Tensor
            Sampled vectors of shape (sample_size, dim)

        Raises
        ------
        RuntimeError
            If no samples are collected
        """
        self._log(f"[SAMPLE] target={sample_size}")
        import random

        # a local generator, so building an index never reseeds the caller's
        # global random module; it draws exactly what random.seed(seed) did
        rng = random.Random(seed)
        picks = []
        remaining = sample_size

        for p, n in zip(self.files, self.counts):
            if remaining <= 0:
                break
            take = min(n, max(1, int(sample_size * (n / max(1, self.total)))))
            if take <= 0:
                continue

            arr = self._load_features(p)

            if take >= n:
                picks.append(arr)
            else:
                idxs = rng.sample(range(n), take)
                picks.append(arr[idxs])

            remaining -= take

        if not picks:
            raise RuntimeError("No samples collected")

        out = torch.cat(picks, dim=0)

        # Shuffle if we over-sampled
        if out.size(0) > sample_size:
            # seeded too, so the whole training sample is reproducible
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(out.size(0), generator=generator)[:sample_size]
            out = out[perm]

        self._log(f"[SAMPLE] actual={out.size(0)}")
        return out.float()

    def _create_and_train_index(
        self,
        train_np: np.ndarray,
        nlist: int,
        m: int,
        nbits: int,
        use_gpu: bool,
        gpu_device: int,
        gpu_fp16_lut: bool,
        use_opq: bool,
    ) -> faiss.Index:
        """Create and train the FAISS index.

        Parameters
        ----------
        train_np : np.ndarray
            Training data as a NumPy array of shape (n_samples, dim).
        nlist : int
            Number of inverted file list (IVF) partitions.
        m : int
            Number of subquantizers.
        nbits : int
            Number of bits per subvector.
        use_gpu : bool
            Whether to use GPU for training.
        gpu_device : int
            GPU device ID to use (if use_gpu is True).
        gpu_fp16_lut : bool
            Whether to use float16 lookup tables (if use_gpu is True).
        use_opq : bool
            Whether to use Optimized Product Quantization (OPQ).

        Returns
        -------
        faiss.Index
            The trained FAISS index.
        """
        # quantizer + metric
        quantizer = (
            faiss.IndexFlatIP(self.dim)
            if self.metric == "cosine"
            else faiss.IndexFlatL2(self.dim)
        )
        metric_type = (
            faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        )

        if use_opq:
            opq = faiss.OPQMatrix(self.dim, m)
            core = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits, metric_type)
            cpu_template = faiss.IndexPreTransform(opq, core)
        else:
            cpu_template = faiss.IndexIVFPQ(
                quantizer, self.dim, nlist, m, nbits, metric_type
            )

        if not use_gpu:
            cast(Any, cpu_template).train(train_np)
            return cpu_template

        faiss_gpu = cast(Any, faiss)
        res = faiss_gpu.StandardGpuResources()
        co = faiss_gpu.GpuClonerOptions()
        co.useFloat16 = gpu_fp16_lut
        gpu_index = faiss_gpu.index_cpu_to_gpu(res, gpu_device, cpu_template, co)
        gpu_index.train(train_np)
        return gpu_index

    def build_index(
        self,
        index_path: str,
        sample_size: int = 655_360,
        nlist: int = 16384,
        m: int = 64,
        nbits: int = 8,
        use_gpu: Optional[bool] = None,
        add_batch_size: int = 100_000,
        nprobe: int = 16,
        gpu_device: int = 0,
        gpu_fp16_lut: bool = True,
        use_opq: bool = True,
        tokens_dir: Optional[str] = None,
        compress_sequences: bool = False,
    ) -> faiss.Index:
        """Build an OPQ+IVF-PQ index.

        If ``tokens_dir`` is given, every shard's tokens file is checked
        (existence, record count, and one-to-one header match with the feature
        rows) before any sampling or training, so a bad tokens directory fails
        in seconds rather than after the index has been trained. Tokens files are
        expected at ``<tokens_dir>/<shard dir name>.tokens.pt``, where the shard
        directory name is the path component matched by ``shard_id_regex`` (see
        the module docstring for the full layout).

        Parameters
        ----------
        index_path : str
            Path to save the built index (e.g., "my_index.faiss")
        sample_size : int, optional
            Number of samples to use for training, by default 655_360
        nlist : int, optional
            Number of inverted file list (IVF) partitions, by default 16384
        m : int, optional
            Number of subquantizers, by default 64
        nbits : int, optional
            Number of bits per subvector, by default 8
        use_gpu : bool, optional
            Whether to use GPU for training. By default (None) a GPU is used only
            if FAISS reports one (``faiss.get_num_gpus() > 0``), so the CPU-only
            ``faiss-cpu`` build works out of the box.
        add_batch_size : int, optional
            Batch size for adding vectors to the index, by default 100_000
        nprobe : int, optional
            Number of probes for the IVFPQ index, by default 16
        gpu_device : int, optional
            GPU device ID to use (if use_gpu is True), by default 0
        gpu_fp16_lut : bool, optional
            Whether to use float16 lookup tables (if use_gpu is True), by default True
        use_opq : bool, optional
            Whether to use Optimized Product Quantization (OPQ), by default True
        tokens_dir : Optional[str], optional
            Directory containing one ``<shard dir name>.tokens.pt`` file per shard
            (as written by ``starling-pretokenize``). If given, a sequence store is
            written next to the index. By default None.
        compress_sequences : bool, optional
            Whether to compress sequences, by default False

        Returns
        -------
        faiss.Index
            The built FAISS index.

        Raises
        ------
        RuntimeError
            If the index cannot be built, or if ``use_gpu=True`` but FAISS reports
            no GPUs.
        FileNotFoundError
            If a shard's tokens file is missing.
        ValueError
            If a tokens file does not align one-to-one (by header) with its
            feature shard.
        """
        if self.dim is None or self.total == 0:
            raise RuntimeError("No vectors discovered under root; cannot build")

        # Resolve GPU use: auto-detect by default, fail clearly if a GPU was requested
        # but this FAISS build cannot see one (e.g. faiss-cpu)
        n_faiss_gpus = cast(Any, faiss).get_num_gpus()
        if use_gpu is None:
            use_gpu = n_faiss_gpus > 0
        elif use_gpu and n_faiss_gpus == 0:
            raise RuntimeError(
                "use_gpu=True but FAISS reports no GPUs (is faiss-cpu installed?). "
                "Use use_gpu=False (CLI: --no-gpu) or install a GPU build of FAISS."
            )

        # Validate the tokens for every shard before spending time on training
        if tokens_dir:
            self.validate_tokens(tokens_dir)

        self._log(
            f"[BUILD] start total={self.total} dim={self.dim} sample={sample_size} "
            f"nlist={nlist} m={m} nbits={nbits} gpu={use_gpu}"
        )
        t0 = time.time()

        # Sample and prepare training data
        train = self.sample_vectors(sample_size)
        train_np = train.detach().cpu().numpy().astype("float32", copy=False)
        if self.metric == "cosine":
            faiss.normalize_L2(train_np)

        # Adapt nlist based on training size
        needed = 39
        max_nlist = max(1, int(train_np.shape[0] // needed))
        if nlist > max_nlist:
            pow2 = 1
            while (pow2 << 1) <= max_nlist:
                pow2 <<= 1
            self._log(
                f"[BUILD] reducing nlist {nlist} -> {pow2} (train={train_np.shape[0]})"
            )
            nlist = max(1, pow2)

        index = self._create_and_train_index(
            train_np=train_np,
            nlist=nlist,
            m=m,
            nbits=nbits,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_fp16_lut=gpu_fp16_lut,
            use_opq=use_opq,
        )

        # Set nprobe on the core IVFPQ index
        ivfpq_index = faiss.downcast_index(
            index.index if isinstance(index, faiss.IndexPreTransform) else index
        )
        ivfpq_index.nprobe = min(int(nprobe), int(nlist))

        self._log("[BUILD] Populating index with all vectors...")
        gid_base = 0

        batch_ids = np.empty(add_batch_size, dtype=np.int64)

        for fi, (p, n) in tqdm(enumerate(zip(self.files, self.counts), start=1)):
            arr = (
                self._load_features(p)
                .detach()
                .cpu()
                .numpy()
                .astype("float32", copy=False)
            )
            if self.metric == "cosine":
                faiss.normalize_L2(arr)

            s = 0
            while s < n:
                e = min(s + add_batch_size, n)
                batch_size = e - s

                # Use view instead of copy when possible
                if s == 0 and e == n:
                    batch = arr
                else:
                    batch = np.ascontiguousarray(arr[s:e], dtype=np.float32)

                # Reuse pre-allocated ID array
                batch_ids[:batch_size] = np.arange(
                    gid_base + s, gid_base + e, dtype=np.int64
                )
                cast(Any, index).add_with_ids(batch, batch_ids[:batch_size])
                s = e
            gid_base += n

            if fi % 25 == 0 or fi == len(self.files):
                self._log(
                    f"[BUILD][ADD] {fi}/{len(self.files)} gid={gid_base}/{self.total} ({gid_base / self.total:.1%})"
                )

        # Save index and manifest
        self._log("[BUILD] Saving index to disk...")
        self.save_index(index, index_path, on_gpu=use_gpu)
        self.save_manifest(
            index_path + ".manifest.json",
            nlist=nlist,
            m=m,
            nbits=nbits,
            sample_size=sample_size,
            use_opq=use_opq,
        )

        if tokens_dir:
            self._log("[SEQSTORE] Building sequences.sqlite from tokens_dir...")
            self.build_sequence_store(
                index_path + ".seqs.sqlite", tokens_dir, compress=compress_sequences
            )
            self._log("[SEQSTORE] Done")

        self._log(f"[BUILD] done in {time.time() - t0:.1f}s")
        return index

    def save_index(self, index: faiss.Index, index_path: str, on_gpu: Optional[bool] = None) -> None:
        """
        Save an index to disk, copying it to the CPU first if it lives on a GPU.

        Parameters
        ----------
        index : faiss.Index
            The index to save.
        index_path : str
            Output file.
        on_gpu : bool, optional
            Whether the index lives on a GPU. None (the default) assumes it
            might whenever this FAISS build supports GPUs. Passing False
            avoids index_gpu_to_cpu, which deep-copies a CPU index (doubling
            its memory) when given one.

        Raises
        ------
        RuntimeError
            If copying a GPU index to the CPU fails. The error is raised
            rather than swallowed, which used to surface as an unrelated
            "cannot serialize" error from write_index.
        """
        # CPU builds of FAISS have no index_gpu_to_cpu, and their indexes are
        # always on the CPU already
        index_gpu_to_cpu = getattr(faiss, "index_gpu_to_cpu", None)
        if on_gpu is None:
            on_gpu = index_gpu_to_cpu is not None
        cpu_index = index_gpu_to_cpu(index) if on_gpu and index_gpu_to_cpu is not None else index
        t0 = time.time()
        faiss.write_index(cpu_index, index_path)
        if self.verbose:
            self._log(f"[SAVE] Wrote {index_path} in {time.time() - t0:.2f}s")

    def save_manifest(
        self,
        path: str,
        *,
        nlist: int,
        m: int,
        nbits: int,
        sample_size: int,
        use_opq: bool = True,
    ) -> None:
        """Save a JSON manifest with index metadata.

        Parameters
        ----------
        path : str
            Output path for the JSON manifest.
        nlist : int
            Number of IVF partitions actually used (after any reduction).
        m : int
            Number of PQ subquantizers (also the OPQ subspace count when OPQ is used).
        nbits : int
            Bits per PQ subquantizer.
        sample_size : int
            Requested number of training samples.
        use_opq : bool, optional
            Whether the index was built with an OPQ rotation. When False the
            manifest records ``"opq": null``. By default True.
        """
        data = {
            "version": 1,
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "metric": self.metric,
            "dim": int(self.dim) if self.dim is not None else None,
            "total": int(self.total),
            "ivfpq": {"nlist": nlist, "m": m, "nbits": nbits},
            "opq": {"m": m} if use_opq else None,
            "train": {"sample_size": sample_size},
            "build_date": time.strftime("%Y-%m-%d"),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        if self.verbose:
            self._log(f"[MANIFEST] Wrote {path}")

    def _tokens_path(self, tokens_dir: str, shard_index: int) -> str:
        """
        Return the expected tokens file path for one shard.

        Parameters
        ----------
        tokens_dir : str
            Directory containing the per-shard tokens files.
        shard_index : int
            Position of the shard in ``self.files`` (not the shard id).

        Returns
        -------
        str
            ``<tokens_dir>/<shard dir name>.tokens.pt``.
        """
        return os.path.join(tokens_dir, self.shard_dirs[shard_index] + _TOKENS_SUFFIX)

    def _load_aligned_token_records(self, tokens_dir: str, shard_index: int) -> List[dict]:
        """
        Load one shard's token records, reordered to match its feature rows.

        Vectors and sequences are paired by header, so the tokens file must
        contain exactly one record per feature row, with the same set of headers.

        Parameters
        ----------
        tokens_dir : str
            Directory containing the per-shard tokens files.
        shard_index : int
            Position of the shard in ``self.files`` (not the shard id).

        Returns
        -------
        list of dict
            Token records in feature-row order.

        Raises
        ------
        FileNotFoundError
            If the tokens file is missing.
        ValueError
            If the record count differs from the feature count, if the feature
            shard has no headers, or if the headers are missing, duplicated or
            do not match one-to-one.
        """
        feat_path = self.files[shard_index]
        feat_count = self.counts[shard_index]
        shard_id = self.shard_ids[shard_index]
        tok_path = self._tokens_path(tokens_dir, shard_index)

        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"Tokens file missing: {tok_path}")

        recs = torch.load(tok_path, map_location="cpu")
        tok_count = len(recs)

        if tok_count != feat_count:
            raise ValueError(
                f"Length mismatch for {tok_path}: tokens={tok_count} vs features={feat_count} (shard {shard_id:06d})"
            )

        feature_headers = self.feature_headers[shard_id]
        if feature_headers is None:
            raise ValueError(
                f"Feature shard {feat_path} has no stable sequence identifiers; "
                "cannot safely align vectors with token records. Store headers "
                "alongside features or provide a keyed feature mapping."
            )
        token_by_header = {}
        for record in recs:
            header = record.get("header")
            if header is None or header in token_by_header:
                raise ValueError(
                    f"Missing or duplicate token header in {tok_path}; "
                    "feature-to-sequence alignment must be one-to-one."
                )
            token_by_header[header] = record
        if set(token_by_header) != set(feature_headers):
            raise ValueError(
                f"Feature and token identifiers do not match for shard {shard_id:06d}."
            )
        return [token_by_header[header] for header in feature_headers]

    def validate_tokens(self, tokens_dir: str) -> None:
        """
        Check that every shard has a tokens file that aligns with its features.

        This runs the same checks as ``build_sequence_store`` (file exists, record
        count matches, headers match one-to-one) without writing anything, so
        ``build_index`` can fail before training rather than after.

        Parameters
        ----------
        tokens_dir : str
            Directory containing one ``<shard dir name>.tokens.pt`` file per shard.

        Raises
        ------
        FileNotFoundError
            If any shard's tokens file is missing.
        ValueError
            If any shard's tokens do not align with its feature rows.
        """
        for shard_index in range(len(self.files)):
            self._load_aligned_token_records(tokens_dir, shard_index)
        self._log(f"[SEQSTORE] Validated tokens for {len(self.files)} shards in {tokens_dir}")

    def _insert_sequence_rows(
        self, store: SequenceStore, tokens_dir: str, compress: bool, batch: int
    ) -> int:
        """
        Insert one row per indexed vector into an open writer store.

        Parameters
        ----------
        store : SequenceStore
            Writer store opened with ``SequenceStore.open_writer``.
        tokens_dir : str
            Directory containing one ``<shard dir name>.tokens.pt`` file per shard.
        compress : bool
            Whether to zstd-compress sequences and headers.
        batch : int
            Number of rows per ``insert_rows`` call.

        Returns
        -------
        int
            Total number of rows written (equal to the number of indexed vectors).
        """
        gid_base = 0
        rows = []

        for shard_index, (feat_count, shard_id) in enumerate(zip(self.counts, self.shard_ids)):
            recs = self._load_aligned_token_records(tokens_dir, shard_index)

            for local_idx, r in enumerate(recs):
                seq = r["sequence"]
                header = r.get("header")
                rows.append(
                    (
                        gid_base + local_idx,
                        len(seq),
                        SequenceStore.hash8(seq),
                        SequenceStore.encode_seq(seq, compress),
                        shard_id,
                        local_idx,
                        SequenceStore.encode_header(header, compress),
                    )
                )
                if len(rows) >= batch:
                    store.insert_rows(rows)
                    rows.clear()

            # advance by the features count for this shard
            gid_base += feat_count

        if rows:
            store.insert_rows(rows)
        return gid_base

    def build_sequence_store(
        self, db_path: str, tokens_dir: str, compress: bool = False, batch: int = 50_000
    ) -> None:
        """Build a SequenceStore SQLite database from tokenized sequence files.

        Tokens files are expected at ``<tokens_dir>/<shard dir name>.tokens.pt``
        (see the module docstring). If anything fails, the writer's temporary
        ``*.building.<pid>.<ts>`` file is deleted before the error is re-raised.

        Parameters
        ----------
        db_path : str
            The path to the SQLite database file to create.
        tokens_dir : str
            The directory containing the tokenized sequence files.
        compress : bool, optional
            Whether to compress the stored sequences, by default False.
        batch : int, optional
            The number of rows to insert in each batch, by default 50_000.

        Raises
        ------
        FileNotFoundError
            If a tokenized sequence file is missing.
        ValueError
            If there is a length mismatch between tokens and features.
        """
        store = SequenceStore.open_writer(db_path)
        try:
            n_rows = self._insert_sequence_rows(store, tokens_dir, compress, batch)
        except BaseException:
            # Don't leave an orphaned temp database behind on failure
            store.close_discard()
            raise

        store.close_publish()
        self._log(f"[SEQSTORE] Wrote {db_path} with {n_rows} entries")
