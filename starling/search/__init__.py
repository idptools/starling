"""
Starling search module: FAISS-backed similarity search with filtering and reranking.
"""

from __future__ import annotations

import os
import sys
from typing import Final, Optional

# On macOS, faiss-cpu and torch each ship their own OpenMP runtime (libomp), and
# a process that loads both crashes: the second runtime aborts at start-up
# ("OMP: Error #15") or segfaults in its worker threads. Allowing the duplicate
# runtime and running FAISS's OpenMP single-threaded avoids both. torch has
# already been imported by starling and fixed its own thread count, so only
# FAISS runs single-threaded. Intel documents duplicate OpenMP runtimes as
# unsupported; this is the standard workaround for FAISS with PyTorch on macOS.
# Values the user has set themselves are left alone. These must be set before
# faiss is imported (by the submodules imported below).
MACOS_OPENMP_ENVIRONMENT: Final[dict[str, str]] = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
}
if sys.platform == "darwin":
    for variable, value in MACOS_OPENMP_ENVIRONMENT.items():
        os.environ.setdefault(variable, value)

from starling.search.builder import IndexBuilder  # noqa: E402
from starling.search.search_engine import SearchEngine  # noqa: E402
from starling.search.store import SequenceStore  # noqa: E402

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
    index_path: str, metric: Optional[str] = None, verbose: bool = True
) -> SearchEngine:
    """
    Load a FAISS index (and its sibling sequence store) into a SearchEngine.

    Parameters
    ----------
    index_path : str
        Path to the serialized FAISS index (typically ends with ``.faiss``).
    metric : {'cosine', 'l2'}, optional
        Query metric. By default (None) the metric the index was built with is
        used; if given, it must match the index or a ValueError is raised.
    verbose : bool, default True
        Whether to emit log messages during loading.

    Returns
    -------
    SearchEngine
        A ready-to-query search engine instance.
    """
    return SearchEngine.load(index_path=index_path, metric=metric, verbose=verbose)
