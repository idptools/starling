"""
Unit and regression test for the starling package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import starling


def test_starling_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "starling" in sys.modules
