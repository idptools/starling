"""
Starling search module: FAISS-backed similarity search with filtering and reranking.
"""

from __future__ import annotations

from starling.search.builder import IndexBuilder
from starling.search.search_engine import SearchEngine
from starling.search.store import SequenceStore

__all__ = [
    "IndexBuilder",
    "SearchEngine",
    "SequenceStore",
    "build_index",
    "load_engine",
]


def build_index(
    *, root: str, index_path: str, tokens_dir: str, metric: str = "cosine", **kwargs
):
    builder = IndexBuilder(
        root=root,
        metric=metric,
        verbose=kwargs.pop("verbose", True),
        shard_id_regex=kwargs.pop("shard_id_regex", None),
    )
    return builder.build_index(index_path=index_path, tokens_dir=tokens_dir, **kwargs)


def load_engine(
    index_path: str, metric: str = "cosine", verbose: bool = True
) -> SearchEngine:
    return SearchEngine.load(index_path=index_path, metric=metric, verbose=verbose)
