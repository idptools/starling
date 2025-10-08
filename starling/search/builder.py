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

* ``use_gpu``: Train on GPU
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

And produces these outputs::

    my_index.faiss              # FAISS index
    my_index.faiss.manifest.json # Metadata
    my_index.faiss.seqs.sqlite  # Sequence store (if tokens_dir provided)


See Also
--------
* :class:`SearchEngine`: Query the built index
* :class:`SequenceStore`: Direct database access
* :mod:`starling.search.cli`: Command-line interface

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
from typing import Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from starling.search.store import SequenceStore

try:
    import faiss  # type: ignore
except Exception as e:
    raise ImportError(
        "FAISS is required. Install with 'pip install faiss-gpu' (CUDA) or 'pip install faiss-cpu'."
    ) from e

logger = logging.getLogger(__name__)

# Default shard ID regex (captures 6-digit zero-padded numeric id)
_SHARD_ID_RE_DEFAULT = re.compile(r"uniref50_idrs_only_(\d{6})")


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
        self.counts: List[int] = []
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

    def _discover_files(self) -> None:
        """Discover and inventory all feature shard files."""
        t0 = time.time()
        pattern = os.path.join(self.root, "**", "sequence_features.pt")
        found_paths = glob.glob(pattern, recursive=True)

        records = []
        for p in found_paths:
            try:
                sid = self._parse_shard_id(p)
            except ValueError:
                if self.verbose:
                    self._log(f"[DISCOVER][SKIP] Could not parse shard id: {p}")
                continue
            arr = self._load_features(p)
            n, d = arr.shape
            records.append((sid, p, int(n), int(d)))

        # Sort strictly by numeric shard id to keep pairing stable
        records.sort(key=lambda t: t[0])

        if self.verbose:
            self._log(f"[DISCOVER] {len(records)} shard files")

        # Populate class fields in the sorted order
        self.files, self.shard_ids, self.counts = [], [], []
        self.dim = None
        for sid, p, n, d in records:
            if self.dim is None:
                self.dim = d
            elif self.dim != d:
                raise ValueError(f"Dim mismatch {self.dim} vs {d} in {p}")
            self.files.append(p)
            self.shard_ids.append(sid)
            self.counts.append(n)

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

        random.seed(seed)
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
                idxs = random.sample(range(n), take)
                picks.append(arr[idxs])

            remaining -= take

        if not picks:
            raise RuntimeError("No samples collected")

        out = torch.cat(picks, dim=0)

        # Shuffle if we over-sampled
        if out.size(0) > sample_size:
            perm = torch.randperm(out.size(0))[:sample_size]
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
            cpu_template.train(train_np)
            return cpu_template

        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = gpu_fp16_lut
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_device, cpu_template, co)
        gpu_index.train(train_np)
        return gpu_index

    def build_index(
        self,
        index_path: str,
        sample_size: int = 655_360,
        nlist: int = 16384,
        m: int = 64,
        nbits: int = 8,
        use_gpu: bool = True,
        add_batch_size: int = 100_000,
        nprobe: int = 16,
        gpu_device: int = 0,
        gpu_fp16_lut: bool = True,
        use_opq: bool = True,
        tokens_dir: Optional[str] = None,
        compress_sequences: bool = False,
    ) -> faiss.Index:
        """Build an OPQ+IVF-PQ index.

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
            Whether to use GPU for training, by default True
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
            Directory containing tokenized sequences, by default None
        compress_sequences : bool, optional
            Whether to compress sequences, by default False

        Returns
        -------
        faiss.Index
            The built FAISS index.

        Raises
        ------
        RuntimeError
            If the index cannot be built.
        """
        if self.dim is None or self.total == 0:
            raise RuntimeError("No vectors discovered under root; cannot build")

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
                index.add_with_ids(batch, batch_ids[:batch_size])
                s = e
            gid_base += n

            if fi % 25 == 0 or fi == len(self.files):
                self._log(
                    f"[BUILD][ADD] {fi}/{len(self.files)} gid={gid_base}/{self.total} ({gid_base / self.total:.1%})"
                )

        # Save index and manifest
        self._log("[BUILD] Saving index to disk...")
        self.save_index(index, index_path)
        self.save_manifest(
            index_path + ".manifest.json",
            nlist=nlist,
            m=m,
            nbits=nbits,
            sample_size=sample_size,
        )

        if tokens_dir:
            self._log("[SEQSTORE] Building sequences.sqlite from tokens_dir...")
            self.build_sequence_store(
                index_path + ".seqs.sqlite", tokens_dir, compress=compress_sequences
            )
            self._log("[SEQSTORE] Done")

        self._log(f"[BUILD] done in {time.time() - t0:.1f}s")
        return index

    def save_index(self, index: faiss.Index, index_path: str) -> None:
        """Save index to disk (converts GPU to CPU if needed)."""
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            cpu_index = index
        t0 = time.time()
        faiss.write_index(cpu_index, index_path)
        if self.verbose:
            self._log(f"[SAVE] Wrote {index_path} in {time.time() - t0:.2f}s")

    def save_manifest(
        self, path: str, *, nlist: int, m: int, nbits: int, sample_size: int
    ) -> None:
        """Save a JSON manifest with index metadata."""
        data = {
            "version": 1,
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "metric": self.metric,
            "dim": int(self.dim) if self.dim is not None else None,
            "total": int(self.total),
            "ivfpq": {"nlist": nlist, "m": m, "nbits": nbits},
            "opq": {"m": m},
            "train": {"sample_size": sample_size},
            "build_date": time.strftime("%Y-%m-%d"),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        if self.verbose:
            self._log(f"[MANIFEST] Wrote {path}")

    def build_sequence_store(
        self, db_path: str, tokens_dir: str, compress: bool = False, batch: int = 50_000
    ) -> None:
        """Build a SequenceStore SQLite database from tokenized sequence files.

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

        gid_base = 0
        rows = []

        for feat_path, feat_count, shard_id in zip(
            self.files, self.counts, self.shard_ids
        ):
            tok_path = os.path.join(
                tokens_dir, f"uniref50_idrs_only_{shard_id:06d}.tokens.pt"
            )

            if not os.path.exists(tok_path):
                raise FileNotFoundError(f"Tokens file missing: {tok_path}")

            recs = torch.load(tok_path, map_location="cpu")
            tok_count = len(recs)

            if tok_count != feat_count:
                raise ValueError(
                    f"Length mismatch for {tok_path}: tokens={tok_count} vs features={feat_count} (shard {shard_id:06d})"
                )

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

        store.close_publish()
        self._log(f"[SEQSTORE] Wrote {db_path} with {gid_base} entries")
