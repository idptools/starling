"""
Tests that every console entry point imports using only declared dependencies.

A fresh install only gets the distributions listed in ``[project.dependencies]``
in ``pyproject.toml``. If a module reachable from a console script imports
something outside that set, ``starling`` dies with a ``ModuleNotFoundError`` at
startup for every new user, while continuing to work for developers who happen
to have the package installed transitively. This suite guards that boundary.
"""

import ast
from importlib import import_module
import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Distributions whose import name differs from the name on PyPI.
DIST_TO_IMPORT = {
    "pyyaml": "yaml",
    "faiss-cpu": "faiss",
    "faiss-gpu": "faiss",
    "hydra-core": "hydra",
    "pytorch-lightning": "pytorch_lightning",
}


def _read_pyproject():
    if not PYPROJECT.is_file():
        raise FileNotFoundError(PYPROJECT)
    try:
        tomllib = import_module("tomllib")
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        tomllib = import_module("tomli")
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def declared_import_names():
    """Return the set of import names provided by declared runtime dependencies."""
    pyproject = _read_pyproject()
    names = set()
    for dep in pyproject["project"]["dependencies"]:
        dist = re.split(r"[<>=!\[;\s]", dep.strip())[0].lower()
        names.add(DIST_TO_IMPORT.get(dist, dist.replace("-", "_")))
    return names


def entry_point_modules():
    """Return the module portion of every ``[project.scripts]`` entry point."""
    if not PYPROJECT.is_file():
        return []
    pyproject = _read_pyproject()
    scripts = pyproject["project"].get("scripts", {})
    return sorted({target.split(":")[0] for target in scripts.values()})


def module_path(module_name):
    """Map a dotted module name to its file inside the repo, or None."""
    candidate = REPO_ROOT / pathlib.Path(*module_name.split(".")).with_suffix(".py")
    return candidate if candidate.is_file() else None


def top_level_third_party_imports(path):
    """
    Return third-party modules imported at module scope in ``path``.

    Only module-scope imports matter here: an import inside a function is not
    executed at startup, so it cannot break the entry point on launch.
    """
    tree = ast.parse(path.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m != "starling" and m}


def extra_import_names(extra):
    """Return the import names provided by a given optional-dependency group."""
    pyproject = _read_pyproject()
    group = pyproject["project"].get("optional-dependencies", {}).get(extra, [])
    names = set()
    for dep in group:
        dist = re.split(r"[<>=!\[;\s]", dep.strip())[0].lower()
        names.add(DIST_TO_IMPORT.get(dist, dist.replace("-", "_")))
    return names


# Entry points that are documented as requiring an optional-dependency group.
# Anything not listed here must work on a bare ``pip install``.
ENTRY_POINT_EXTRAS = {
    "starling.training.vae_train": "train",
    "starling.training.diffusion_train": "train",
}


@pytest.mark.parametrize("module_name", entry_point_modules())
def test_entry_point_only_imports_declared_dependencies(module_name):
    """Every entry-point module must import only declared dependencies."""
    path = module_path(module_name)
    if path is None:
        raise pytest.skip.Exception(f"{module_name} not found in checkout")
    allowed = declared_import_names()
    extra = ENTRY_POINT_EXTRAS.get(module_name)
    if extra is not None:
        allowed |= extra_import_names(extra)
    undeclared = top_level_third_party_imports(path) - allowed
    assert not undeclared, (
        f"{module_name} imports {sorted(undeclared)} at module scope, which "
        f"is not in [project.dependencies] (or its declared extra). A fresh "
        f"install will fail with "
        f"ModuleNotFoundError. Either add the dependency to pyproject.toml or "
        f"remove the import."
    )


def test_main_cli_imports_without_psutil():
    """
    CLI help must work without psutil or loading the model-training stack.

    ``starling_main_cli`` carried an unused ``import psutil`` while psutil was
    never a declared dependency, so the ``starling`` command failed at startup
    on any fresh install.
    """
    code = (
        "import sys, builtins, importlib\n"
        "sys.modules.pop('psutil', None)\n"
        "_real = builtins.__import__\n"
        "def _fake(name, *a, **k):\n"
        "    if name.split('.')[0] == 'psutil':\n"
        "        raise ModuleNotFoundError(\"No module named 'psutil'\")\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _fake\n"
        "cli = importlib.import_module('starling.scripts.starling_main_cli')\n"
        "assert 'pytorch_lightning' not in sys.modules\n"
        "sys.argv = ['starling', '--help']\n"
        "cli.main()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        "starling.scripts.starling_main_cli failed to import without psutil:\n" + result.stderr
    )
    assert "--sampler" in result.stdout
