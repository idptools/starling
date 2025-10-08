"""
Search Utilities
================

Core utility classes for similarity search: score conversion, candidate representation, and extensible filters.

Overview
--------
This module provides building blocks for the search pipeline:

* **ScoreConverter**: Handles metric-specific score transformations
* **Candidate**: Immutable representation of search results
* **CandidateFilter**: Abstract base for custom filtering logic
* **Built-in Filters**: ValidGid, L2Distance, CosineSim, Length, ExactMatch, SequenceIdentity

These utilities are used internally by SearchEngine but can also be used directly for custom search pipelines.

Score Conversion
----------------

The ScoreConverter handles conversions between FAISS raw scores and user-facing outputs:

**For Cosine Similarity:**

* FAISS returns inner product scores (higher = more similar)
* ``return_similarity=True``: Output as-is [0, 1]
* ``return_similarity=False``: Convert to distance (1 - similarity)

**For L2 Distance:**

* FAISS returns squared L2 distance (lower = more similar)
* Always output as distance (no conversion)

Usage::

    >>> converter = ScoreConverter(metric="cosine", return_similarity=True)
    >>> output_score = converter.convert(raw_faiss_score=0.95)
    >>> output_score
    0.95

Candidate Representation
-------------------------

The Candidate dataclass provides a clean interface for search results:

Attributes:
    score (float): Converted score/similarity
    gid (int): Global sequence ID
    header (str | None): Sequence header from database
    length (int | None): Sequence length
    stored_hash (int | None): 8-byte sequence hash for deduplication

Usage::

    >>> candidate = Candidate(
    ...     score=0.95,
    ...     gid=12345,
    ...     header="sp|P12345|PROT_HUMAN",
    ...     length=234,
    ...     stored_hash=123456789
    ... )
    >>> candidate.as_tuple()
    (0.95, 12345, "sp|P12345|PROT_HUMAN", 234)

Custom Filters
--------------

Extend CandidateFilter to create custom filtering logic:

Example - Filter by minimum score::

    class MinScoreFilter(CandidateFilter):
        def __init__(self, min_score: float):
            self.min_score = min_score

        def apply(self, candidate: Candidate, query_seq: str = None) -> bool:
            return candidate.score >= self.min_score

        def get_name(self) -> str:
            return "min_score"


Built-in Filters
----------------

**ValidGidFilter**
    Filters out invalid GIDs (< 0). Always active in search pipeline.

    Usage::

        filter = ValidGidFilter()
        passes = filter.apply(candidate)  # False if gid < 0

**L2DistanceFilter**
    Filters by minimum L2 distance (for L2 metric).

    Parameters:
        min_distance (float): Minimum distance threshold

    Usage::

        filter = L2DistanceFilter(min_distance=0.5)
        passes = filter.apply(candidate)  # True if distance >= 0.5

**CosineSimFilter**
    Filters by maximum cosine similarity (for cosine metric).

    Parameters:
        max_similarity (float): Maximum similarity threshold
        return_similarity (bool): Whether scores are similarities or distances

    Usage::

        filter = CosineSimFilter(max_similarity=0.99, return_similarity=True)
        passes = filter.apply(candidate)  # True if similarity <= 0.99

**LengthFilter**
    Filters by sequence length range.

    Parameters:
        min_len (int | None): Minimum length (inclusive)
        max_len (int | None): Maximum length (inclusive)

    Usage::

        filter = LengthFilter(min_len=50, max_len=500)
        passes = filter.apply(candidate)  # True if 50 <= length <= 500

**ExactMatchFilter**
    Filters out exact sequence matches using hash + full comparison.

    Parameters:
        query_hash (int): Hash of query sequence
        seq_store (SequenceStore): Database for sequence lookup

    Usage::

        query_hash = SequenceStore.hash8(query_seq)
        filter = ExactMatchFilter(query_hash, seq_store)
        passes = filter.apply(candidate, query_seq)  # False if exact match

**SequenceIdentityFilter**
    Filters by maximum sequence identity.

    Parameters:
        max_identity (float): Maximum identity threshold (0-1)
        denominator (str): Identity denominator ("query", "target", "min", "max", "avg")
        seq_store (SequenceStore): Database for sequence lookup
        identity_func (callable): Function computing identity

    Usage::

        def compute_identity(seq1, seq2, denom="query"):
            # Your alignment logic here
            return identity_score

        filter = SequenceIdentityFilter(
            max_identity=0.95,
            denominator="query",
            seq_store=seq_store,
            identity_func=compute_identity
        )
        passes = filter.apply(candidate, query_seq)  # True if identity < 0.95

Filter Pipeline
---------------

Filters are applied sequentially in SearchEngine. First failed filter stops evaluation:

Pipeline order:
    1. ValidGidFilter (always first)
    2. L2DistanceFilter or CosineSimFilter (embedding-level)
    3. LengthFilter (metadata-level)
    4. ExactMatchFilter (sequence-level, per-query)
    5. SequenceIdentityFilter (alignment-level, per-query)

This ordering minimizes expensive operations (sequence fetches, alignments).

**Optimization Tips:**

1. Use length_min/max to pre-filter via SQL index (much faster than post-filter)
2. Place cheap filters before expensive ones
3. Use hash comparison before full sequence comparison
4. Consider overfetch parameter when using aggressive filters

Integration with SearchEngine
------------------------------

SearchEngine automatically builds and applies filters based on search parameters::

    results = engine.search(
        queries=queries,
        k=100,
        nprobe=128,
        # These parameters create filters internally:
        length_min=50,              # -> LengthFilter
        length_max=500,             # -> LengthFilter
        max_cosine_similarity=0.99, # -> CosineSimFilter
        exclude_exact_sequence=True,# -> ExactMatchFilter (per-query)
        sequence_identity_max=0.95  # -> SequenceIdentityFilter (per-query)
    )

See Also
--------
* :class:`SearchEngine`: Main search interface using these utilities
* :class:`SequenceStore`: Database for sequence lookups in filters
* :mod:`starling.search`: Main search module
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from starling.search.store import SequenceStore


@dataclass
class Candidate:
    """Represents a search result candidate."""

    score: float
    gid: int
    header: Optional[str]
    length: Optional[int]
    stored_hash: Optional[int] = None

    def as_tuple(self) -> Tuple[float, int, Optional[str], Optional[int]]:
        return (self.score, self.gid, self.header, self.length)


# ========== Filter Classes ==========
class CandidateFilter(ABC):
    """Base class for candidate filters."""

    @abstractmethod
    def apply(self, candidate: Candidate, query_seq: Optional[str] = None) -> bool:
        """Return True if candidate passes filter."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return filter name for logging."""
        pass


class ValidGidFilter(CandidateFilter):
    """Filter out invalid GIDs."""

    def apply(self, candidate: Candidate, query_seq: Optional[str] = None) -> bool:
        return candidate.gid >= 0

    def get_name(self) -> str:
        return "gid"


class L2DistanceFilter(CandidateFilter):
    """Filter by minimum L2 distance."""

    def __init__(self, min_distance: float):
        self.min_distance = min_distance

    def apply(self, candidate: Candidate, query_seq: Optional[str] = None) -> bool:
        return candidate.score >= self.min_distance

    def get_name(self) -> str:
        return "l2"


class CosineSimFilter(CandidateFilter):
    """Filter by maximum cosine similarity."""

    def __init__(self, max_similarity: float, return_similarity: bool):
        self.max_similarity = max_similarity
        self.return_similarity = return_similarity

    def apply(self, candidate: Candidate, query_seq: Optional[str] = None) -> bool:
        cos_sim = candidate.score if self.return_similarity else (1.0 - candidate.score)
        return cos_sim <= self.max_similarity

    def get_name(self) -> str:
        return "cos"


class LengthFilter(CandidateFilter):
    """Filter by sequence length range."""

    def __init__(self, min_len: Optional[int], max_len: Optional[int]):
        self.min_len = min_len
        self.max_len = max_len

    def apply(self, candidate: Candidate, query_seq: Optional[str] = None) -> bool:
        if candidate.length is None:
            return True
        if self.min_len is not None and candidate.length < self.min_len:
            return False
        if self.max_len is not None and candidate.length > self.max_len:
            return False
        return True

    def get_name(self) -> str:
        return "len"


class ExactMatchFilter(CandidateFilter):
    """Filter out exact sequence matches."""

    def __init__(self, query_hash: int, seq_store: SequenceStore):
        self.query_hash = query_hash
        self.seq_store = seq_store

    def apply(self, candidate: Candidate, query_seq: Optional[str] = None) -> bool:
        if candidate.stored_hash is None or candidate.stored_hash != self.query_hash:
            return True
        if query_seq is None:
            return True
        seq_val = self.seq_store.get_seq(candidate.gid)
        return seq_val != query_seq

    def get_name(self) -> str:
        return "exact"


class SequenceIdentityFilter(CandidateFilter):
    """Filter by maximum sequence identity."""

    def __init__(
        self,
        max_identity: float,
        denominator: str,
        seq_store: SequenceStore,
        identity_func,
    ):
        self.max_identity = max_identity
        self.denominator = denominator
        self.seq_store = seq_store
        self.identity_func = identity_func

    def apply(self, candidate: Candidate, query_seq: Optional[str] = None) -> bool:
        if query_seq is None:
            return True
        seq_val = self.seq_store.get_seq(candidate.gid)
        if seq_val is None:
            return True
        ident = self.identity_func(query_seq, seq_val, denom=self.denominator)
        return ident < self.max_identity

    def get_name(self) -> str:
        return "ident"


class ScoreConverter:
    """Handles score/similarity conversion for different metrics."""

    def __init__(self, metric: str, return_similarity: bool):
        self.metric = metric
        self.return_similarity = return_similarity

    def convert(self, raw_score: float) -> float:
        """Convert raw FAISS score to output format."""
        if self.metric == "cosine":
            return float(raw_score if self.return_similarity else 1.0 - raw_score)
        return float(raw_score)

    def to_similarity(self, score: float) -> float:
        """Convert score to similarity (for output formatting)."""
        if self.metric == "cosine":
            return score if self.return_similarity else (1.0 - score)
        return score

    def to_score(self, similarity: float) -> float:
        """Convert similarity to score (for output formatting)."""
        if self.metric == "cosine":
            return similarity if self.return_similarity else (1.0 - similarity)
        return similarity
