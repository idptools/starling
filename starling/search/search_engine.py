"""
FAISS Search Engine
===================

High-performance similarity search with flexible filtering, length gating, and exact reranking.

Overview
--------
The SearchEngine provides fast ANN (Approximate Nearest Neighbor) search with:

* **Multi-level Filtering**: Embedding distance, sequence length, exact matches, identity
* **Length Gating**: Pre-filter candidates by sequence length using indexed lookups
* **Reranking**: Exact rescoring of top-k candidates using full encoder
* **Batch Processing**: Efficient handling of multiple queries
* **Flexible Metrics**: Cosine similarity or L2 distance


Basic Usage
-----------
>>> from starling.search import SearchEngine
>>> import torch
>>> engine = SearchEngine.load("my_index.faiss", metric="cosine")
>>> queries = torch.randn(10, 768)
>>> queries = torch.nn.functional.normalize(queries, dim=1)
>>> results = engine.search(queries=queries, k=100, nprobe=128, return_similarity=True)
>>> for qi, hits in enumerate(results):
...     for score, gid, header, length in hits[:5]:
...         print(qi, score, gid, length)

Advanced Usage
--------------
**Filtering by Length:**
>>> engine.search(queries, k=100, nprobe=128, length_min=50, length_max=500)

**Excluding Exact Matches:**
>>> engine.search(queries, query_sequences=["MKTLLIL..."], k=100, exclude_exact=True)

**Exact Reranking:**
>>> engine.search(queries, k=100, nprobe=128, rerank=True, rerank_device="cuda:0")

Search Parameters
-----------------
See ``search()`` docstring for the full parameter list.

Common Patterns
---------------
Pattern 1: Near duplicates (filter exact + very similar)
>>> engine.search(queries, k=100, nprobe=256, exclude_exact=True, max_cosine_similarity=0.99)

Pattern 2: Length-focused neighborhood
>>> L = 200
>>> engine.search(queries, k=1000, nprobe=128, length_min=L-50, length_max=L+50)

Pattern 3: Diverse similar sequences
>>> engine.search(queries, k=500, nprobe=128, max_cosine_similarity=0.80, length_min=50, length_max=500)

Notes
-----
* Normalize queries for cosine: ``torch.nn.functional.normalize(q, dim=1)``
* Higher ``nprobe`` improves recall at cost of latency
* Use ``rerank`` for improved precision after coarse ANN stage
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from starling.search.search_utils import (
    Candidate,
    CandidateFilter,
    CosineSimFilter,
    ExactMatchFilter,
    L2DistanceFilter,
    LengthFilter,
    ScoreConverter,
    SequenceIdentityFilter,
    ValidGidFilter,
)
from starling.search.store import SequenceStore

try:
    import faiss
except Exception as e:
    raise ImportError(
        "FAISS is required. Please install a CPU or GPU compatible version"
    ) from e


class SearchEngine:
    """FAISS-backed similarity search with rich post-filtering.

    The :class:`SearchEngine` wraps a trained ``faiss.Index`` and optional
    :class:`~starling.search.store.SequenceStore` to provide high-level search
    primitives used across STARLING. In addition to ANN lookups, it supports
    length gating, identity thresholds, exact-match suppression, and optional
    reranking with the encoder.

    Parameters
    ----------
    index : faiss.Index
        Trained FAISS index (e.g., IVFPQ, HNSW) matching the embedding metric.
    metric : {'cosine', 'l2'}, default 'cosine'
        Similarity/distance metric used during index construction.
    seq_store : SequenceStore, optional
        Sequence/metadata store generated alongside the index. Required for
        filters that need sequence lengths or raw sequences, and for reranking.
    verbose : bool, default True
        When ``True`` the engine logs key operations (loading, filters, rerank).

    Attributes
    ----------
    index : faiss.Index
        Underlying FAISS index used for coarse ANN search.
    metric : str
        Metric string supplied during construction (``"cosine"`` or ``"l2"``).
    seq_store : SequenceStore or None
        Attached sequence store used for metadata-aware filtering.
    total : int
        Number of vectors stored in the index.
    dim : int
        Embedding dimensionality of the index.

    Examples
    --------
    >>> from starling.search import SearchEngine
    >>> engine = SearchEngine.load("/data/index.faiss")
    >>> hits = engine.search(queries, k=50, nprobe=128, rerank=True)
    >>> len(hits[0])
    50
    """

    def __init__(
        self,
        index: faiss.Index,
        metric: str = "cosine",
        seq_store: Optional[SequenceStore] = None,
        verbose: bool = True,
    ):
        if metric not in {"l2", "cosine"}:
            raise ValueError("metric must be 'l2' or 'cosine'")

        self.index = index
        self.metric = metric
        self.seq_store = seq_store
        self.verbose = verbose
        self._log = print if verbose else (lambda *a, **k: None)

        self.total = int(index.ntotal)
        self.dim = int(index.d)

    @classmethod
    def load(
        cls,
        index_path: str,
        metric: str = "cosine",
        verbose: bool = True,
    ) -> "SearchEngine":
        """Load a serialized FAISS index (and optional sequence store).

        This attempts to read ``index_path`` (a .faiss file) and, if present,
        a sibling ``.seqs.sqlite`` file with the same base name. The sequence
        store is optional; absence only disables filters / reranking that need it.

        Parameters
        ----------
        index_path : str
            Path to the serialized FAISS index (typically ends with ``.faiss``).
        metric : {'cosine', 'l2'}, default 'cosine'
            Metric that will be used for queries. Must match how embeddings were prepared.
        verbose : bool, default True
            Whether to emit log messages during loading.

        Returns
        -------
        SearchEngine
            A ready-to-query search engine instance.
        """
        index = faiss.read_index(index_path)

        # Try to load sequence store if available
        seq_store = None
        seq_db = index_path + ".seqs.sqlite"
        if os.path.exists(seq_db):
            try:
                seq_store = SequenceStore.open_reader(seq_db)
                if verbose:
                    print(f"[SEQSTORE] Attached {seq_db}")
            except Exception as e:
                if verbose:
                    print(f"[SEQSTORE] Failed to open {seq_db}: {e}")

        return cls(index=index, metric=metric, seq_store=seq_store, verbose=verbose)

    def _compute_fetch_k(
        self,
        k: int,
        overfetch: Optional[int],
        any_filters: bool,
    ) -> int:
        """Compute internal candidate fetch size prior to filtering.

        Overfetching helps preserve recall when downstream filters prune results.

        Parameters
        ----------
        k : int
            Desired number of final results per query.
        overfetch : int, optional
            User-provided multiplicative factor. If ``None``, it is auto-derived.
        any_filters : bool
            Indicator that at least one filter (besides ValidGid) is active.

        Returns
        -------
        int
            Number of candidates to request from FAISS per query.
        """
        if overfetch is None:
            overfetch = 5 if any_filters else 1
        fetch_k = k * max(1, overfetch)
        return fetch_k

    def _build_filters(
        self,
        return_similarity: bool,
        exclude_exact: bool,
        sequence_identity_max: Optional[float],
        identity_denominator: str,
        min_l2_distance: Optional[float],
        max_cosine_similarity: Optional[float],
        length_min: Optional[int],
        length_max: Optional[int],
    ) -> List[CandidateFilter]:
        """Instantiate the base (query-agnostic) filter objects.

        Sequence-dependent filters (exact match / identity) are created per-query
        later because they depend on the query sequence content.

        Parameters
        ----------
        return_similarity : bool
            Whether outward facing scores are similarities (True) or distances.
        exclude_exact : bool
            If True, exact sequence matches (by hash + content) will be removed.
        sequence_identity_max : float, optional
            Maximum allowed identity; candidates exceeding are filtered.
        identity_denominator : {'query','target','max','min','avg'}
            Identity denominator mode (passed along to the identity function).
        min_l2_distance : float, optional
            Minimum allowed L2 distance (applies only for ``metric='l2'``).
        max_cosine_similarity : float, optional
            Maximum allowed cosine similarity (applies only for ``metric='cosine'``).
        length_min : int, optional
            Minimum sequence length (inclusive).
        length_max : int, optional
            Maximum sequence length (inclusive).

        Returns
        -------
        list of CandidateFilter
            Base filters (ValidGid + optional embedding/length filters).
        """
        filters: List[CandidateFilter] = [ValidGidFilter()]

        # Embedding-level filters
        if self.metric == "l2" and min_l2_distance is not None:
            filters.append(L2DistanceFilter(min_l2_distance))

        if self.metric == "cosine" and max_cosine_similarity is not None:
            filters.append(CosineSimFilter(max_cosine_similarity, return_similarity))

        # Length filter
        if length_min is not None or length_max is not None:
            filters.append(LengthFilter(length_min, length_max))

        # Sequence-based filters will be instantiated per-query
        return filters

    def _set_nprobe(self, nprobe: Optional[int]) -> None:
        """Safely set ``nprobe`` on the underlying IVF index if available.

        Parameters
        ----------
        nprobe : int, optional
            Number of IVF lists to probe. ``None`` leaves index default unchanged.

        Notes
        -----
        * Silently no-ops if the index has no IVF component.
        * Removes any active ``max_codes`` cap (sets to 0 / unlimited).
        """
        if nprobe is None:
            return

        # Try to set on inner IVF index (works for IndexPreTransform too)
        inner = None
        try:
            core = self.index.index if hasattr(self.index, "index") else self.index
            inner = faiss.downcast_index(core)
        except Exception as e:
            self._log(f"[WARN] Failed to downcast index: {e}")

        if inner is not None and hasattr(inner, "nprobe"):
            inner.nprobe = int(nprobe)
            # Remove scan caps if present
            if hasattr(inner, "max_codes"):
                try:
                    inner.max_codes = 0  # 0 => unlimited
                except Exception as e:
                    self._log(f"[WARN] Failed to set max_codes: {e}")
            self._log(
                "[SEARCH] Set nprobe={} nlist={} max_codes={}".format(
                    getattr(inner, "nprobe", None),
                    getattr(inner, "nlist", None),
                    getattr(inner, "max_codes", None),
                )
            )
        elif hasattr(self.index, "nprobe"):
            # Fallback for non-wrapped IVF
            self.index.nprobe = int(nprobe)
            self._log(f"[SEARCH] Set nprobe={nprobe} (direct)")
        else:
            self._log("[WARN] Could not set nprobe, index has no nprobe attribute")

    def _ann_search(
        self,
        queries_np: np.ndarray,
        fetch_k: int,
        nprobe: Optional[int],
        selector_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute ANN search against FAISS with optional ID restriction.

        Parameters
        ----------
        queries_np : ndarray, shape (Q, D)
            Query vectors (float32) already normalized if cosine metric.
        fetch_k : int
            Number of candidates to request from FAISS per query.
        nprobe : int, optional
            IVF probe count override.
        selector_ids : sequence of int, optional
            If provided, attempt to restrict search to these IDs (uses IDSelectorBatch).

        Returns
        -------
        D : ndarray, shape (Q, fetch_k)
            Raw FAISS scores (inner products or squared distances).
        idxs : ndarray, shape (Q, fetch_k)
            Candidate global IDs (``-1`` for invalid slots).

        Notes
        -----
        Falls back to standard search if selector path fails or unsupported.
        """
        self._set_nprobe(nprobe)

        if selector_ids is not None:
            present = sum(1 for g in selector_ids if 0 <= int(g) < int(self.total))
            self._log(
                f"[SEARCH] Using selector: {present}/{len(selector_ids)} IDs in range "
                f"(total={self.total})"
            )

            try:
                params = faiss.SearchParametersIVF()
                if nprobe is not None:
                    params.nprobe = int(nprobe)

                sel = faiss.IDSelectorBatch(np.asarray(selector_ids, dtype=np.int64))
                params.sel = sel

                # note: you *must* use params as a kwarg otherwise faiss python wrappers fail
                # and you experience pain....
                D, idxs = self.index.search(queries_np, fetch_k, params=params)

                self._log("[SEARCH] IDSelectorBatch search completed successfully.")
                return D, idxs
            except Exception as e:
                self._log(
                    f"[WARN] IDSelectorBatch failed: {e}, falling back to standard search"
                )

        self._log("[SEARCH] Using standard search (no selector).")
        D, idxs = self.index.search(queries_np, fetch_k)
        return D, idxs

    def _collect_meta(self, idxs: np.ndarray) -> dict:
        """Fetch metadata for unique candidate IDs present in result indices.

        Parameters
        ----------
        idxs : ndarray, shape (Q, K)
            Candidate ID matrix (``-1`` entries ignored).

        Returns
        -------
        dict
            Mapping ``gid -> (header, length, hash8)``. Empty if no sequence store.
        """
        gids = {int(g) for g in idxs.flatten() if g >= 0}
        if not (self.seq_store and gids):
            return {}
        fetched = self.seq_store.get_many_meta(gids)
        return {gid: (hdr, length, h8) for gid, hdr, length, h8 in fetched}

    def _seq_identity(self, seq1: str, seq2: str, denom: str = "query") -> float:
        """Compute approximate (ungapped) sequence identity.

        This is a fast heuristic: counts exact character matches over the
        overlapping prefix (``min(len(seq1), len(seq2))``) and divides by a
        denominator selected by ``denom``. Any overhang beyond the overlap is
        implicitly treated as mismatched when the denominator is larger than the
        overlap (e.g. ``max`` / ``avg`` cases).

        Parameters
        ----------
        seq1 : str
            First (query) sequence.
        seq2 : str
            Second (target) sequence.
        denom : {'query','target','max','min','avg'}, default 'query'
            Denominator policy:
              * query  -> len(seq1)
              * target -> len(seq2)
              * max    -> max(len1, len2)
              * min    -> min(len1, len2)
              * avg    -> 0.5 * (len1 + len2)

        Returns
        -------
        float
            Identity fraction in [0,1]. Returns 0.0 if any sequence is empty or
            denominator resolves to 0.

        Notes
        -----
        This does NOT perform alignment (no gap handling). For short indels the
        value may underestimate true identity. Suitable for coarse filtering.
        """
        if not seq1 or not seq2:
            return 0.0
        if seq1 == seq2:
            return 1.0

        l1 = len(seq1)
        l2 = len(seq2)
        overlap = min(l1, l2)
        matches = 0
        # Fast loop over overlap
        for a, b in zip(seq1[:overlap], seq2[:overlap]):
            if a == b:
                matches += 1

        if denom == "query":
            denom_len = l1
        elif denom == "target":
            denom_len = l2
        elif denom == "max":
            denom_len = max(l1, l2)
        elif denom == "min":
            denom_len = overlap  # == min(l1, l2)
        elif denom == "avg":
            denom_len = 0.5 * (l1 + l2)
        else:  # Fallback to query length
            denom_len = l1

        if denom_len <= 0:
            return 0.0
        return matches / float(denom_len)

    def _filter_candidates(
        self,
        D: np.ndarray,
        idxs: np.ndarray,
        k: int,
        converter: ScoreConverter,
        query_sequences: Optional[List[str]],
        exclude_exact_sequence: bool,
        sequence_identity_max: Optional[float],
        identity_denominator: str,
        base_filters: List[CandidateFilter],
        meta_map: dict,
    ) -> Tuple[List[List[Tuple[float, int, Optional[str], Optional[int]]]], set]:
        """Apply filters for all queries and build preliminary top-k lists.

        Parameters
        ----------
        D : ndarray, shape (Q, F)
            Raw FAISS scores.
        idxs : ndarray, shape (Q, F)
            Candidate IDs (aligned with ``D``), ``-1`` marks empty slots.
        k : int
            Final desired results per query.
        converter : ScoreConverter
            Handles score conversion (cosine IP -> similarity / distance, etc.).
        query_sequences : list of str, optional
            Raw query sequences (needed for exact/identity filters).
        exclude_exact : bool
            If True, remove exact (sequence-equal) matches.
        sequence_identity_max : float, optional
            Identity threshold; above this candidates are removed.
        identity_denominator : str
            Denominator mode passed to identity filter.
        base_filters : list of CandidateFilter
            Filters that are query-agnostic.
        meta_map : dict
            Metadata map from ``gid`` to (header, length, hash8).

        Returns
        -------
        preliminary : list of list of tuple
            Per-query list of (score, gid, header, length) tuples after filtering.
        rerank_gid_set : set of int
            Unique GIDs that survived at least one query (union for reranking).
        """
        Q = D.shape[0]

        # Pre-compute query hashes if needed
        query_hashes = {}
        if exclude_exact_sequence and query_sequences and self.seq_store:
            query_hashes = {
                i: self.seq_store.hash8(s) for i, s in enumerate(query_sequences)
            }

        out: List[List[Tuple[float, int, Optional[str], Optional[int]]]] = []
        rerank_gid_set: set = set()

        for qi in range(Q):
            # Build per-query filters (includes sequence-specific filters)
            filters = list(base_filters)
            qseq = query_sequences[qi] if query_sequences is not None else None

            if (
                exclude_exact_sequence
                and qseq
                and self.seq_store
                and qi in query_hashes
            ):
                filters.append(ExactMatchFilter(query_hashes[qi], self.seq_store))

            if sequence_identity_max is not None and qseq and self.seq_store:
                filters.append(
                    SequenceIdentityFilter(
                        sequence_identity_max,
                        identity_denominator,
                        self.seq_store,
                        self._seq_identity,
                    )
                )

            # Filter candidates for this query
            row, filter_counts = self._filter_query_candidates(
                D[qi], idxs[qi], k, converter, qseq, filters, meta_map
            )

            self._log(
                f"[FILTER][q{qi}] total={filter_counts['total']} kept={len(row)} | "
                + " ".join(
                    f"filt_{k}={v}" for k, v in filter_counts.items() if k != "total"
                )
            )

            out.append(row)
            for _, gid, _, _ in row:
                rerank_gid_set.add(gid)

        return out, rerank_gid_set

    def _filter_query_candidates(
        self,
        scores: np.ndarray,
        gids: np.ndarray,
        k: int,
        converter: ScoreConverter,
        query_seq: Optional[str],
        filters: List[CandidateFilter],
        meta_map: dict,
    ) -> Tuple[List[Tuple[float, int, Optional[str], Optional[int]]], dict]:
        """Filter candidates for a single query row.

        Parameters
        ----------
        scores : ndarray, shape (F,)
            Raw FAISS scores for one query.
        gids : ndarray, shape (F,)
            Candidate IDs aligned with ``scores``.
        k : int
            Desired number of outputs (early stop once reached).
        converter : ScoreConverter
            Converter to transform raw score to outward facing metric.
        query_seq : str, optional
            Raw query sequence (for exact / identity filters); may be None.
        filters : list of CandidateFilter
            Active filters to apply in order.
        meta_map : dict
            Mapping gid -> (header, length, hash8) metadata.

        Returns
        -------
        row : list of tuple
            Filtered (score, gid, header, length) tuples (<= k items).
        filter_counts : dict
            Per-filter rejection counts plus total candidates examined.
        """
        row: List[Tuple[float, int, Optional[str], Optional[int]]] = []
        filter_counts = {f.get_name(): 0 for f in filters}
        filter_counts["total"] = 0

        for raw_score, gid in zip(scores, gids):
            filter_counts["total"] += 1

            # Build candidate
            header, length, stored_hash = meta_map.get(int(gid), (None, None, None))
            candidate = Candidate(
                score=converter.convert(raw_score),
                gid=int(gid),
                header=header,
                length=length,
                stored_hash=stored_hash,
            )

            # Apply filters
            passed = True
            for f in filters:
                if not f.apply(candidate, query_seq):
                    filter_counts[f.get_name()] += 1
                    passed = False
                    break

            if passed:
                row.append(candidate.as_tuple())
                if len(row) >= k:
                    break

        return row, filter_counts

    def _encode_sequences_for_rerank(
        self,
        gids: Iterable[int],
        device: Optional[str],
        batch_size: int,
        ionic_strength: Optional[int],
    ) -> dict:
        """Encode candidate sequences for exact reranking.

        Parameters
        ----------
        gids : iterable of int
            Unique candidate GIDs to materialize/encode.
        device : str, optional
            Device string passed to encoder (e.g., 'cuda:0', 'cpu'). Auto if None.
        batch_size : int
            Encoder batch size.
        ionic_strength : int, optional
            Ionic strength forwarded to the encoder (controls deterministic dropout etc.).

        Returns
        -------
        dict
            Mapping ``gid -> embedding (torch.Tensor, shape (D,))``. Empty if no seq store.
        """
        if not self.seq_store:
            return {}

        seq_map = {}
        requested = 0
        have_sequences = 0
        for gid in gids:
            requested += 1
            s = self.seq_store.get_seq(gid)
            if s is not None:
                have_sequences += 1
                seq_map[f"g{gid}"] = s

        if not seq_map:
            self._log(
                f"[RERANK] requested_gids={requested} have_sequences=0 encoded_vecs=0"
            )
            return {}

        from starling.inference.generation import sequence_encoder_backend

        chosen_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        ionic_strength = 150 if ionic_strength is None else ionic_strength
        emb_dict = sequence_encoder_backend(
            sequence_dict=seq_map,
            device=chosen_device,
            batch_size=batch_size,
            ionic_strength=ionic_strength,
            aggregate=True,
            output_directory=None,
        )

        # convert to gid -> tensor
        out: dict[int, torch.Tensor] = {}
        for k, v in emb_dict.items():
            gid = int(k[1:])  # strip leading 'g'
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            out[gid] = t.float()

        # normalize if cosine metric
        if self.metric == "cosine":
            for gid, t in out.items():
                out[gid] = torch.nn.functional.normalize(t.unsqueeze(0), dim=1).squeeze(
                    0
                )

        self._log(
            f"[RERANK] requested_gids={requested} have_sequences={have_sequences} encoded_vecs={len(out)}"
        )
        return out

    def _rerank(
        self,
        preliminary: List[List[Tuple[float, int, Optional[str], Optional[int]]]],
        queries: torch.Tensor,
        gid_to_vec: dict,
        k: int,
        converter: ScoreConverter,
    ) -> List[List[Tuple[float, int, Optional[str], Optional[int]]]]:
        """Rerank preliminary candidates using freshly encoded exact vectors.

        Parameters
        ----------
        preliminary : list of list
            Per-query preliminary candidate tuples (score, gid, header, length).
        queries : torch.Tensor, shape (Q, D)
            Original query embeddings (pre-normalized if cosine).
        gid_to_vec : dict
            Mapping gid -> candidate embedding for reranking.
        k : int
            Final number of results to keep.
        converter : ScoreConverter
            Score conversion utility (determines ascending/descending ordering logic).

        Returns
        -------
        list of list
            Per-query reranked list of (score, gid, header, length) tuples.
        """
        if not gid_to_vec:
            self._log(
                "[RERANK] No vectors available for rerank; returning prelim top-k"
            )
            return [row[:k] for row in preliminary]

        q_emb = queries.detach().cpu().float()
        if self.metric == "cosine":
            q_emb = torch.nn.functional.normalize(q_emb, dim=1)

        final: List[List[Tuple[float, int, Optional[str], Optional[int]]]] = []

        for qi, row in enumerate(preliminary):
            if not row:
                final.append([])
                self._log(f"[RERANK][q{qi}] prelim=0 have_vecs=0 final=0")
                continue

            # Find candidates with available vectors
            cand = [gid for _, gid, _, _ in row if gid in gid_to_vec]
            if not cand:
                final.append(row[:k])
                self._log(
                    f"[RERANK][q{qi}] prelim={len(row)} have_vecs=0 final={min(len(row), k)}"
                )
                continue

            # Compute exact scores
            mat = torch.stack([gid_to_vec[g] for g in cand])  # (N,D)
            qv = q_emb[qi]

            if self.metric == "cosine":
                sims = (qv.unsqueeze(0) @ mat.T).squeeze(0)
                raw_scores = sims if converter.return_similarity else (1.0 - sims)
                desc = converter.return_similarity
            else:  # l2 distance
                diffs = mat - qv
                raw_scores = torch.sqrt(torch.sum(diffs * diffs, dim=1))
                desc = False

            # Sort and build result
            order = torch.argsort(raw_scores, descending=desc)
            meta_lookup = {gid: (hdr, ln) for _, gid, hdr, ln in row}

            new_row: List[Tuple[float, int, Optional[str], Optional[int]]] = []
            for oi in order[:k]:
                gid = cand[int(oi)]
                hdr, ln = meta_lookup.get(gid, (None, None))
                new_row.append((float(raw_scores[int(oi)].item()), gid, hdr, ln))

            final.append(new_row)
            self._log(
                f"[RERANK][q{qi}] prelim={len(row)} have_vecs={len(cand)} final={len(new_row)}"
            )

        return final

    def search(
        self,
        queries: torch.Tensor,
        k: int = 10,
        nprobe: Optional[int] = None,
        return_similarity: bool = False,
        query_sequences: Optional[List[str]] = None,
        exclude_exact: bool = False,
        sequence_identity_max: Optional[float] = None,
        identity_denominator: str = "query",
        min_l2_distance: Optional[float] = None,
        max_cosine_similarity: Optional[float] = None,
        overfetch: Optional[int] = None,
        rerank: bool = False,
        rerank_device: Optional[str] = None,
        rerank_batch_size: int = 64,
        rerank_ionic_strength: Optional[int] = None,
        length_min: Optional[int] = None,
        length_max: Optional[int] = None,
    ) -> List[List[Tuple[float, int, Optional[str], Optional[int]]]]:
        """Approximate nearest neighbor search with optional filtering and reranking.

        Parameters
        ----------
        queries : torch.Tensor, shape (Q, D)
            Query embeddings. For cosine metric they should be L2-normalized externally
            (the method re-normalizes defensively before search).
        k : int, default 10
            Number of final results per query.
        nprobe : int, optional
            IVF probe count override; higher => better recall, slower.
        return_similarity : bool, default False
            If True (cosine), return similarity in [0,1]; otherwise distance.
        query_sequences : list of str, optional
            Raw sequences aligned with queries (enables exact / identity filters & rerank).
        exclude_exact : bool, default False
            Remove candidates whose sequence matches the query exactly.
        sequence_identity_max : float, optional
            Maximum allowed identity (0-1). Requires ``query_sequences`` and seq store.
        identity_denominator : {'query','target','max','min','avg'}, default 'query'
            Mode for identity normalization (forwarded to identity function).
        min_l2_distance : float, optional
            Minimum allowed L2 distance (metric='l2').
        max_cosine_similarity : float, optional
            Maximum allowed cosine similarity (metric='cosine').
        overfetch : int, optional
            Multiply ``k`` by this before filtering. Auto: 5 if filters active else 1.
        rerank : bool, default False
            If True, re-embed surviving candidates (exact scoring) and reorder.
        rerank_device : str, optional
            Device for reranking encoder; auto-select if None.
        rerank_batch_size : int, default 64
            Batch size for rerank embedding pass.
        rerank_ionic_strength : int, optional
            Ionic strength forwarded to encoder (if None uses model default).
        length_min : int, optional
            Minimum acceptable candidate length.
        length_max : int, optional
            Maximum acceptable candidate length.

        Returns
        -------
        list of list of tuple
            Per-query results: ``[(score, gid, header, length), ...]`` up to ``k``.

        Raises
        ------
        RuntimeError
            If filters requiring a sequence store are requested but none is attached.
        """
        # Validate requirements
        seq_filters_active = exclude_exact or sequence_identity_max is not None
        if (
            seq_filters_active
            or rerank
            or length_min is not None
            or length_max is not None
        ) and self.seq_store is None:
            raise RuntimeError(
                "SequenceStore required for filters/length gating/rerank"
            )

        # Prepare queries
        q_np = queries.detach().cpu().float().numpy()
        if self.metric == "cosine":
            faiss.normalize_L2(q_np)

        # Setup score converter and filters
        converter = ScoreConverter(self.metric, return_similarity)
        base_filters = self._build_filters(
            return_similarity,
            exclude_exact,
            sequence_identity_max,
            identity_denominator,
            min_l2_distance,
            max_cosine_similarity,
            length_min,
            length_max,
        )

        # Compute fetch_k
        any_filters = (
            len(base_filters) > 1 or exclude_exact or sequence_identity_max is not None
        )
        fetch_k = self._compute_fetch_k(k, overfetch, any_filters)
        self._log(
            f"[SEARCH] k={k} fetch_k={fetch_k} nprobe={nprobe} queries={q_np.shape[0]}"
        )

        # ANN search with optional length selector
        selector_ids = None
        if self.seq_store and (length_min is not None or length_max is not None):
            selector_ids = self.seq_store.get_gids_by_length_range(
                length_min, length_max
            )
        D, idxs = self._ann_search(q_np, fetch_k, nprobe, selector_ids=selector_ids)

        # Collect metadata and filter
        meta_map = self._collect_meta(idxs)
        preliminary, rerank_gid_set = self._filter_candidates(
            D=D,
            idxs=idxs,
            k=k,
            converter=converter,
            query_sequences=query_sequences,
            exclude_exact_sequence=exclude_exact,
            sequence_identity_max=sequence_identity_max,
            identity_denominator=identity_denominator,
            base_filters=base_filters,
            meta_map=meta_map,
        )

        if not rerank:
            return [row[:k] for row in preliminary]

        total_prelim = sum(len(r) for r in preliminary)
        self._log(
            f"[RERANK] total_prelim={total_prelim} unique_candidates={len(rerank_gid_set)}"
        )

        gid_to_vec = self._encode_sequences_for_rerank(
            gids=rerank_gid_set,
            device=rerank_device,
            batch_size=rerank_batch_size,
            ionic_strength=rerank_ionic_strength,
        )

        return self._rerank(
            preliminary=preliminary,
            queries=queries,
            gid_to_vec=gid_to_vec,
            k=k,
            converter=converter,
        )
