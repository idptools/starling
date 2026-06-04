# Changelog

This file contains our changelog for 



## June 3rd 2026

### Bug Fixes

- **`check_distance_map_for_error`: fix incorrect physical-distance bound** (`starling/utilities.py`).
  The previous implementation collapsed every sequence separation onto a single global threshold (`4.5 * ij_abs`) and only checked separations 1–4, so short-range unphysical distances (e.g. a sequence-adjacent pair several-fold too far apart) were never caught, and the threshold could falsely flag valid long-range pairs depending on `min_separation`. The check now applies a per-pair bound — two residues separated by `|i - j|` positions can be at most `|i - j| * max_bond_length` apart (a fully extended chain) — across all pairs by default. This is a hard physical maximum, so it never produces false positives. `Ensemble.check_for_errors` now scans every residue pair (`max_separation=None`).

### New

- **`max_bond_length` parameter on `check_distance_map_for_error`** (`starling/utilities.py`).
  Replaces the hard-coded bond length. Default is `4.81` Å (the 3.81 Å Mpipi bond length plus a +1 Å per-bond error margin). Added a square-matrix guard and a `max_separation=None` option to check all residue pairs.

- **`Ensemble.check_for_errors_trajectory()`** (`starling/structure/ensemble.py`).
  New method that mirrors `Ensemble.check_for_errors` but inspects the reconstructed 3D ensemble (the SMACOF/`SSProtein` trajectory) rather than the raw STARLING distance maps. It derives per-frame CA–CA distance maps from the trajectory and flags frames with physically impossible inter-residue distances — useful for catching reconstruction artefacts. When `remove_errors=True`, flagged frames are removed from both the trajectory and the distance maps (which stay in sync), and cached Rg/Rh values are invalidated. Raises `RuntimeError` if no trajectory is associated with the ensemble.

- **`--remove-errors` flag for `starling2xtc` and `starling2pdb`** (`starling/scripts/starling_converter.py`).
  When set, the reconstructed trajectory is scanned with `check_for_errors_trajectory(remove_errors=True)` and erroneous frames are removed *before* the XTC/PDB trajectory is written to disk.

### Documentation

- Documented the `--remove-errors` flag for the conversion utilities in `docs/usage/cli.rst`.
- Added `check_for_errors_trajectory` to the `Ensemble` autosummary stub.

### Build / Packaging

- **Fixed setuptools deprecation warnings on `uv build`** (`pyproject.toml`).
  Migrated the license metadata to the PEP 639 format: `license = { text = "MIT" }` is now the SPDX string `license = "MIT"`, added `license-files = ["LICENSE"]`, and removed the deprecated `License :: OSI Approved :: MIT License` classifier. Bumped `build-system.requires` from `setuptools>=61.0` to `setuptools>=77.0` (the minimum version supporting the SPDX license expression and `license-files` key).


## April 15th 2026

### Bug Fixes

- **`DDPMSampler`: fix `AttributeError` on initialization** (`starling/samplers/ddpm_sampler.py`).
  `self.device` was referenced before assignment when constructing the `ionic_strength` tensor. Moved `self.device = ddpm_model.device` before its first use.

- **`PLMSSampler`: fix NumPy 2.0 deprecation warning** (`starling/samplers/plms_sampler.py`).
  `alphacums[ddim_timesteps]` returned a torch tensor that was mixed with NumPy arithmetic, triggering `__array_wrap__` deprecation warnings under NumPy 2.0. Wrapped the result with `np.asarray()` to keep all computation in pure NumPy.

- **`test_starling.py`: fix `FileNotFoundError` in `save_trajectory` calls**.
  Tests passed `'outdata/test.pdb'` as the filename prefix, causing `save_trajectory` to append `.pdb` again (`test.pdb.pdb`). Changed all prefixes to `'outdata/test'`.

- **`test_starling.py`: remove duplicate `test_ensemble_generation` function**.
  Two functions with the same name existed; Python silently shadowed the first. Removed the duplicate, keeping the more complete version that includes the `save_trajectory` check.

- **`test_starling.py`: fix relative `outdata/` paths breaking when CWD ≠ tests dir**.
  All `'outdata/...'` paths were relative to CWD, so tests failed when run from the project root. Resolved paths relative to `__file__` via `_OUTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outdata")`.

- **`test_sequence_encoder_backend.py`: add missing `aggregate=False`** to four tests (`test_sequence_encoder_backend_basic`, `_remainder_batch`, `_saves_files`, `_pretokenized`).
  Tests asserted per-residue embedding shapes `(L, D)` but relied on the default `aggregate=True`, which collapses output to `(D,)`.

### New

- **Comprehensive ensemble generation test suite** (`starling/tests/test_ensemble_generation.py`).
  148 tests covering input handling, `generate()` parameter validation, `Ensemble` construction/properties/serialization, constraint classes (`Bond`, `StericClash`, `Distance`, `Rg`, `Re`, `Helicity`, `Multi`), distance map symmetrization, config defaults, and integration tests (marked `@pytest.mark.slow`).

- **Dockerfile and Docker instructions** (`docker/Dockerfile`, `docker/readme.md`).
  Added a multi-stage Docker build for creating a self-contained STARLING image based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04` with Python 3.11. The builder stage installs PyTorch (CUDA 12.4), STARLING, and pre-downloads model weights and FAISS search artifacts. The runtime stage copies only the venv, cached weights, and search artifacts into a slim image. The entrypoint is the `starling` CLI. Includes a companion `docker/readme.md` with build, run, and GPU usage instructions.

### Documentation

- Updated `README.md` with expanded content (+785/−206 lines).
- Updated Sphinx configuration (`docs/conf.py`) and restructured API docs (`docs/api.rst`).
- Revised user-facing documentation pages: installation, ensemble generation, ensemble analysis, CLI usage, search, sequence encoder, and possible issues.
- Refreshed autosummary stubs across all public modules (constraints, BME, search, tokenizer, model loading, etc.).
- Removed obsolete BME autosummary stubs (`BMEResult`, `ExperimentalObservable`, `diagnose_bme_result`, `print_bme_diagnostics`) following refactor into `bme_utils`.
- Added new autosummary stubs for newly exposed modules and classes.

### Code Quality

- **Cleaned up `pyproject.toml`**: removed `pytest`, `jupyter`, and `ipython` from runtime dependencies (pytest belongs in test extras; jupyter/ipython aren't needed at runtime); removed deprecated `pytest-runner`; removed commented-out dependency lines; populated `[project.urls]` with GitHub and ReadTheDocs links; trimmed verbose setuptools comments; registered custom pytest marks (`slow`, `integration`) and added a `filterwarnings` entry for torch deprecation warnings; updated `requires-python` from `>=3.8` to `>=3.10`.

- Fixed NumPy-style docstrings across multiple modules to comply with Sphinx/numpydoc parsing (`starling/__init__.py`, `starling/frontend/ensemble_generation.py`, `starling/samplers/plms_sampler.py`, `starling/search/store.py`, `starling/structure/bme.py`, `starling/structure/bme_utils.py`, `starling/structure/coordinates.py`, `starling/structure/ensemble.py`, `starling/utilities.py`).
- Added missing one-line docstrings to `training_step` and `validation_step` in `starling/models/diffusion.py` and `starling/models/continuous_diffusion.py`.
