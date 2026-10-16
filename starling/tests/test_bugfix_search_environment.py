"""starling.search must make FAISS and PyTorch safe to use together on macOS."""

import os
import subprocess
import sys

import pytest

from starling.search import MACOS_OPENMP_ENVIRONMENT

# what a fresh interpreter reports for each variable after importing starling.search
PROBE = (
    "import os, starling.search; "
    "print(os.environ.get('KMP_DUPLICATE_LIB_OK'), os.environ.get('OMP_NUM_THREADS'))"
)


def _probe(environment: dict[str, str]) -> list[str]:
    """Import starling.search in a fresh interpreter and return the two OpenMP variables it sees."""
    result = subprocess.run([sys.executable, "-c", PROBE], env=environment, capture_output=True, text=True, check=True)
    return result.stdout.split()


def _environment_without_openmp_settings() -> dict[str, str]:
    environment = dict(os.environ)
    for variable in MACOS_OPENMP_ENVIRONMENT:
        environment.pop(variable, None)
    return environment


@pytest.mark.skipif(sys.platform != "darwin", reason="the workaround is only applied on macOS")
def test_search_sets_the_openmp_workaround_on_macos():
    assert _probe(_environment_without_openmp_settings()) == ["TRUE", "1"]


@pytest.mark.skipif(sys.platform != "darwin", reason="the workaround is only applied on macOS")
def test_search_keeps_the_users_own_openmp_settings():
    environment = _environment_without_openmp_settings()
    environment["OMP_NUM_THREADS"] = "4"
    assert _probe(environment) == ["TRUE", "4"]
