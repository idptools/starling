# Changelog

This file contains our changelog for STARLING

## October 4th 2026

### Bug Fixes

- **Corrected generation and search edge cases.** Invalid maps and stale ensemble state are handled safely; constraint guidance and encoder batching preserve their intended axes. Search aligns features with token records, keeps a larger rerank candidate pool, and reports consistent L2 scores.
- **Fixed training and utility behavior.** The sigmoid schedule and min-SNR weighting are valid, BME honors an explicit theta, repeated search hits export correctly, and CLI flags match their names. Converter output paths, VAE evaluation batching/mode, training dependencies, and external-ensemble weight provenance are also corrected.

## October 3rd 2026

### New

- **DPM-Solver++(2M)** (`starling/samplers/dpmpp_sampler.py`).
  Added deterministic second-order sampling through `generate(..., sampler="dpmpp", steps=12)` and CLI `--sampler dpmpp`. Uses existing models and decoding; constraints are explicitly unsupported. DDIM-30 remains the default pending final validation of dpm++

### Improvements

- **Shared tensor distance-map symmetrization** (`starling/utilities.py`).
  Generation, constraints, and VAE training use one differentiable operation that reflects the upper triangle and zeros the diagonal without modifying inputs. Device and dtype are preserved; generation retains its CPU output contract. Restraint reference normalization is owned by `DistanceRestraints` rather than repeated by relaxation.

- **Reconstruction and physical relaxation use float32** (`starling/structure/coordinates.py`, `starling/minimizer/relax.py`, `starling/minimizer/fused_forces.py`).
  The eager CUDA force reduction avoids TF32 matrix multiplication, regardless of model inference precision settings.
  MDS initialization, FIRE, and thermalization compute in float32 on every device. FIRE rescales norms and power calculations to avoid overflow from clash forces. MPS eigensolver and pseudoinverse fallbacks run in float32 on CPU; iterative reconstruction and relaxation use native device tensors. CUDA FIRE and thermalization share a fixed-tile force kernel reused across sequence lengths without per-sequence specialization; unavailable fusion warns and falls back to eager float32 before drawing noise. GPU MDS checks batch completion every eight iterations while freezing converged maps immediately. Force-field parameters, convergence tolerance, and the 250-step default are unchanged. Seeded results remain reproducible but differ from earlier float64 reconstruction and relaxation.

- **Eager physical relaxation avoids redundant pair-matrix operations** (`starling/minimizer/forcefield.py`, `starling/minimizer/restraints.py`).
  Wang-Frenkel coefficients and charged-pair indices are cached per sequence. Integer powers use squares, force derivatives are algebraically factored, and restraints use signed clamps. CUDA distances use direct coordinate differences; CPU and MPS retain `torch.cdist`. Force-field parameters and cutoffs are unchanged.

- **Optional whole-force fusion for CUDA thermalization** (`compile_forces=True`, CLI `--compile-forces`).
  PyTorch compiles the native float32 force calculation instead of using the reusable Triton kernel; FIRE stays eager. The force field and 250-step default are unchanged. Compilation adds startup time and is intended for sustained throughput. Failures warn and fall back before thermalization draws noise; relaxation results record the number of compiled batches.

- **Reconstruction-to-relaxation avoids intermediate CPU transfers** (`starling/structure/coordinates.py`, `starling/inference/generation.py`, `starling/minimizer/relax.py`).
  Coordinates stay in Angstroms on-device between MDS and relaxation, with one conversion to nanometres at the generation boundary. Sampling uses batched symmetrization and one loop for full and partial batches. FIRE uses tensor masks for uphill steps and failed forces; Langevin checks accumulated non-finite-force flags at the output boundary. Public return types, convergence criteria, and reconstruction weights are unchanged.

- **Relaxation now thermalizes structures after minimizing them** (`starling/minimizer/langevin.py`, `starling/minimizer/relax.py`).
  Energy minimization leaves every conformation at a zero-temperature minimum, so relaxed bonds were far narrower than in Mpipi-GG simulations (0.380 ± 0.003 nm against 0.386 ± 0.018 nm, pooled over 200 natural IDRs). `relax_conformations()` now follows minimization with 250 steps (5 ps) of BAOAB Langevin dynamics at 300 K under the same force field and restraints. On 20 natural IDRs this restores the thermal bond spread (0.0175 nm against 0.0177 nm in simulations) and brings the angle and dihedral distributions 5x and 3x closer to the simulations, at the cost of a slightly larger change in Rg (0.3% on average, at most ~3%). Longer runs (1000 or 2500 steps) changed none of this. Pass `thermalization_steps=0` for the previous minimized-only structures; `temperature_K` and `seed` control the run. This applies to `generate(relax=True)` and `--relax`.

### Bug Fixes

- **Model caching respects checkpoint paths and device** (`starling/inference/model_loading.py`).
  Requests for different checkpoints or devices reload the single cached model pair instead of silently reusing incompatible models.

- **Relaxation preserves result dtypes and rejects invalid controls** (`starling/minimizer`).
  Energies retain the documented NumPy float64 output type while device calculations stay float32. Non-finite controls and invalid step counts are rejected before force evaluation.

- **CLI startup no longer loads the model-training stack** (`starling/inference/model_loading.py`, `starling/samplers`).
  Lightning model imports are deferred until checkpoint loading. CLI help and ensemble utilities no longer import optional text-metric dependencies or require `psutil` at startup; checkpoint loading and sampler equations are unchanged.

- **Ensembles retain ionic strength for later relaxation** (`starling/structure/ensemble.py`, `starling/minimizer/relax.py`).
  Generated and saved ensembles record ionic strength. Later relaxation uses an explicit override, the recorded value, or 150 mM for legacy files without this metadata. Zero ionic strength is preserved.

- **Save-only generation returns `None`** (`starling/frontend/ensemble_generation.py`).
  `generate(return_data=False)` now follows its documented return contract instead of returning an empty dictionary. Ensemble files are still saved normally.

- **Inference models were left in training mode** (`starling/inference/model_loading.py`).
  Loaded diffusion and VAE models now enter evaluation mode. This disables the sequence encoder's random 20% ionic-strength masking during generation. Predictions change relative to the incorrect training-mode behavior; model weights and sampler equations are unchanged.

- **DDIM high-step numerical instability** (`starling/samplers/ddim_sampler.py`).
  Cap near-zero-SNR timesteps to bound noise-to-data error amplification and remove duplicate steps. Safe legacy schedules, including DDIM-30, are preserved. Invalid step counts and noise levels now raise clear errors.


## October 2nd 2026

### Improvements

- **Weighted coordinate reconstruction** (`starling/structure/coordinates.py`, `starling/structure/weighted_stress.py`).
  Reconstruction now refines distance maps with inverse squared empirical map-error weights, giving accurately predicted short-range distances greater influence. Classical MDS supplies a deterministic initial structure before batched Torch SMACOF refinement. CPU, CUDA, and MPS use the same reconstruction path; diffusion samplers and model weights are unchanged.
  Production reconstruction coefficients use the reproducible posterior-sample fit on 192 sequences, with 48 additional validation sequences. This good-faith calibration serves as a first-attempt solution to address an issue identified by Davide Mercadante (thanks!).

### Bug Fixes

- **Reconstruction failed when batches converged at different iterations** (`starling/structure/coordinates.py`).
  Stress histories now retain their documented width, allowing independently converged batches to concatenate correctly.

- **Float64 distance maps crashed during reconstruction** (`starling/structure/coordinates.py`).
  NumPy and Torch inputs are converted to the solver's float32 dtype per batch, avoiding mixed-dtype matrix multiplication failures.

### Removed

- **Legacy sklearn and Adam reconstruction routes**, their unused helpers, the MDS comparison benchmark, and obsolete API pages.
  Removed the scikit-learn runtime dependency and unused CPU-worker/random-restart configuration. Reconstruction uses one Torch implementation rather than separate device-specific algorithms.

### Documentation

- Added the calibration runner, family-separated validation, frozen selection manifest, raw-input/source provenance, and per-sequence results. Removed exploratory weighting implementations and ensemble-mean-bias fitting; unrelated devtools are unchanged.

- Updated README, API, usage, and Docker guides to describe classical initialization and weighted Torch SMACOF, removing obsolete CPU-worker and restart options. Added calibration methods in `devtools/scripts/weighted_reconstruction.md`. MDS is coordinate reconstruction, not a change to diffusion sampling.

## September 28th 2026

### New

- **Mpipi-GG relaxation of 3D conformations** (`starling/minimizer/`).
  STARLING's distance maps have good local geometry (bonds are 3.85 ± 0.02 Å), but the unweighted MDS reconstruction used to build 3D structures before October 2nd 2026 did not: because unweighted stress fits every distance with equal absolute weight, bonds came out compressed (~3.1-3.2 ± 0.5-0.7 Å), non-bonded beads routinely overlapped (e.g. ~10 pairs closer than 0.6σ per FUS LCD conformation), and in 5-20% of frames at least one bond exceeded 6 Å. The weighted reconstruction added on October 2nd 2026 removes the broken chains and much of the compression, but bonds still come out short and spread out and some non-bonded beads still sit too close (α-synuclein: 3.52 ± 0.17 Å bonds and ~2.5 pairs closer than 4 Å per conformation, which relaxation takes to 3.80 ± 0.02 Å and none). The new `starling.minimizer` package fixes this by energy-minimizing each conformation under a local reimplementation of the Mpipi-GG force field (harmonic bonds, Wang-Frenkel with a 3σ cutoff, Debye-Hückel with a 35 Å cutoff, bonded neighbours excluded) while flat-bottomed harmonic restraints on every residue pair at least 4 apart in sequence hold the global shape to the STARLING distance map. On real ensembles this restores bonds to 3.80 Å and removes every clash while changing Rg by ~0.1% per conformation, and repairs the broken-chain frames rather than requiring them to be discarded.

  Use `relax_ensemble(ensemble)` to get back a new `Ensemble` with relaxed structures plus a `RelaxationResult` (call `.summary()` for a before/after report), or `relax_conformations(coordinates, sequence, reference_distances=...)` for raw coordinates in Å. Runs batched in PyTorch on CUDA, MPS or CPU using a locally written FIRE 2.0 minimizer; 400 conformations of a 163-residue IDR take ~25 s on MPS, including the MDS build.

- **`relax` option for `generate()` and `--relax` flag for the `starling` CLI** (`starling/frontend/ensemble_generation.py`, `starling/inference/generation.py`, `starling/scripts/starling_main_cli.py`).
  Relaxes the MDS-reconstructed structures with Mpipi-GG as they are generated, restrained to the distance map each was built from, so the `.xtc`/`.pdb` and `.starling` outputs carry the relaxed structures. **`generate()` now relaxes by default (`relax=True`)** whenever `return_structures=True`; pass `relax=False` for the previous raw MDS output. The CLI keeps the old behaviour unless `--relax` is given, and `--relax` without `-r`/`--return_structures` is an error rather than silently doing nothing. With `remove_errors`/`--remove-errors`, relaxation runs before the 3D error screen, so conformations MDS broke are repaired and kept rather than discarded and resampled (on α-synuclein with the earlier unweighted reconstruction, 50 conformations went from a worst bond of 19.8 Å to 3.95 Å with none discarded).

- **Mpipi-GG parameter file** (`starling/minimizer/data/mpipi_gg_pairs.tsv`).
  Plain-text copy of the FINCHES GGv1 Wang-Frenkel parameters for the 20 amino acids, so STARLING does not depend on FINCHES. Epsilon is stored in kcal/mol (the native LAMMPS units, which is also what the FINCHES pickles contain despite their docstrings saying kJ/mol) and converted to kJ/mol on load.

- **Minimizer test suite** (`starling/tests/test_minimizer.py`).
  31 tests covering the parameter set (including an exact comparison against FINCHES when it is installed), agreement of the PyTorch energies with a plain NumPy reference implementation, analytic forces against autograd, bonded-neighbour exclusions, restraint behaviour, relaxation repairing local geometry while keeping global dimensions, and the `generate()`/CLI wiring (including that the 3D error screen sees relaxed structures, plus two slow tests against the real model: one end-to-end through `generate()`, and one checking that relaxation still repairs structures built with the earlier unweighted SMACOF).

## August 21st 2026

### Bug Fixes

- **`starling` command failed on every fresh install with `ModuleNotFoundError: No module named 'psutil'`** (`starling/scripts/starling_main_cli.py`).
  The CLI module carried a module-scope `import psutil`, but `psutil` was never listed in `[project.dependencies]` and the name was never used anywhere in the package. Developers did not see this because `psutil` is commonly present as a transitive dependency, but any clean environment broke at startup before `main()` could run. Removed the unused import.

- **`starling-vae-train` and `starling-ddpm-train` failed on a fresh install** (`pyproject.toml`).
  Both entry points import `hydra`, `omegaconf`, and `wandb` at module scope, none of which were declared anywhere. They are now declared in a new `train` optional-dependency group rather than as hard requirements, since they are not needed to generate ensembles: install with `pip install "idptools-starling[train]"`.

- **Locally placed model weights were ignored, breaking offline/air-gapped installs** (`starling/configs.py`, `starling/inference/model_loading.py`).
  `DEFAULT_ENCODER_WEIGHTS_PATH` and `DEFAULT_DDPM_WEIGHTS_PATH` were computed twice: first as paths inside `DEFAULT_MODEL_DIR` (`~/.starling_weights`), and then immediately overwritten with the GitHub release URLs. The `~/.starling_weights` values were dead by the time anything read them, so weights placed there were never found. `ModelManager.load_models` then keyed its cache lookup on the URL, checking only `$TORCH_HOME/hub/checkpoints/<basename>` and downloading if that exact file was absent. Weight resolution now lives in `configs.resolve_weights_path()`, which searches `~/.starling_weights/` first and the torch hub cache second, and only downloads when the file is genuinely missing from both.

### New

- **`STARLING_OFFLINE` environment variable** (`starling/configs.py`).
  When set to a truthy value (`1`, `true`, `yes`, `on`), STARLING never attempts a network connection. If a required checkpoint or search artifact is missing, it raises a `FileNotFoundError` listing every location that was searched and how to fix it, instead of trying to download and failing on a dropped connection. Exposed as `configs.is_offline()`.

- **`configs.candidate_weights_paths()` and `configs.torch_hub_checkpoint_dir()`**.
  Helpers exposing the local search order for a weights file. `torch_hub_checkpoint_dir()` resolves `TORCH_HOME` (and `XDG_CACHE_HOME`) lazily at call time rather than at import, so setting `TORCH_HOME` after importing `starling` is honoured.

- **Offline support for search artifacts** (`starling/configs.py`).
  `_download_if_missing()` now respects `STARLING_OFFLINE`: a missing artifact raises a `FileNotFoundError` naming the relevant `STARLING_FAISS_INDEX_PATH` / `STARLING_SEQSTORE_PATH` / `STARLING_FAISS_MANIFEST_PATH` override, and an artifact that is present but fails its MD5 check is used with a warning rather than triggering a re-download.

- **`--remove-errors` flag for the `starling` CLI** (`starling/scripts/starling_main_cli.py`, `starling/inference/generation.py`, `starling/frontend/ensemble_generation.py`).
  Discards conformations containing physically impossible inter-residue distances and generates replacements, so the number of conformations returned still matches `-c`/`--conformations`. Screening runs in two stages: `Ensemble.check_for_errors()` on the raw distance maps, then — when `-r`/`--return_structures` is also set — `Ensemble.check_for_errors_trajectory()` on the reconstructed 3D conformations. The cheap distance-map screen runs first so MDS reconstruction is only spent on conformations that already look plausible. All filtering happens before anything is written to disk, so the `.xtc`, `.pdb` and `.starling` outputs contain only conformations that passed both screens. Exposed on the Python API as `generate(..., remove_errors=True)`.

  Sampling and screening repeat until the requested number of clean conformations has been accumulated, capped at `DEFAULT_MAX_ERROR_FILTER_ROUNDS` (10) rounds; if the target still cannot be met a `RuntimeError` is raised rather than silently returning fewer conformations than requested. Note this differs from the existing `--remove-errors` flag on `starling2xtc`/`starling2pdb`, which operates on an already-generated archive and so can only remove frames, not replace them.

- **Error-filtering test suite** (`starling/tests/test_remove_errors.py`).
  14 tests driving the filter loop with a stubbed sampler, covering discard-and-replace, multi-round top-up, rounds where every conformation fails, overshoot trimming, the max-rounds backstop and its `RuntimeError`, and — for the 3D path — that coordinates stay aligned 1:1 with the surviving distance maps across top-up rounds.

- **Entry-point dependency test suite** (`starling/tests/test_entry_point_imports.py`).
  Parses `[project.scripts]` and asserts that every console entry point imports only modules provided by the declared dependencies (or by an entry point's documented extra), so a missing runtime dependency fails in CI rather than on a new user's first run. Includes a specific regression test that `starling.scripts.starling_main_cli` imports with `psutil` unavailable.

- **Offline resolution test suite** (`starling/tests/test_offline_weights.py`).
  20 tests covering local-path passthrough, `~` expansion, the `~/.starling_weights`-over-hub-cache precedence, `STARLING_OFFLINE` parsing, the contents of the offline error messages, and the search-artifact offline paths. None of them touch the network.

### Improvements

- **Factored the sampling batch loop out of `generate_backend()`** into `_sample_distance_maps()` (`starling/inference/generation.py`), so it can be invoked repeatedly by the error-filtering loop. Behaviour of the default (unfiltered) path is unchanged.

- **Fixed a latent bug in the verbose per-sequence summary** (`starling/inference/generation.py`).
  It computed `n_conformers = len(sym_distance_maps)`, reading a variable that is deleted in the `return_data=False` cleanup path. The count is now captured before the ensemble is built and freed.

- **`starling --info` now reports the weights file that will actually be loaded** (`starling/scripts/starling_main_cli.py`).
  It previously printed `DEFAULT_ENCODER_WEIGHTS_PATH` / `DEFAULT_DDPM_WEIGHTS_PATH` verbatim, which — because of the bug above — was always a GitHub URL and told you nothing about what was on disk. It now prints the resolved local file (or `NOT FOUND LOCALLY`), whether offline mode is on, and the directories searched, making it the natural first check when diagnosing a deployment.

### Documentation

- Documented `--remove-errors` in `README.md` and added a *Removing erroneous conformations* section to `docs/usage/cli.rst`, including how it differs from the same-named converter flag (which is now cross-referenced).

- **Documented the `train` extra** across `README.md`, `docs/usage/installation.rst` and `docs/usage/cli.rst`.
  The installation page gains a cross-referenceable *Installing the training dependencies* section, and `cli.rst` gains a *Training tools (advanced)* section (it previously did not mention the training entry points at all) that links to it. Covers installing the extra from PyPI, directly from GitHub via the PEP 508 `package[extra] @ url` form (including pinning a branch/tag/commit), and from a local clone (plain and editable), with a note that the brackets must be quoted in `zsh`.

- **Documented installing from GitHub without cloning** (`docs/usage/installation.rst`).
  The *Install from GitHub* section previously only described the clone-then-install route, even though the README documented the one-line `pip install git+...` form. Both are now shown, along with `pip install -e .` for editable installs.

- **Removed `starling-sample` and `ae-train` from the README's training command table.**
  Both console scripts were deleted in July 2026 because they pointed at modules that do not exist, but the README still advertised them.

- Added an **Offline / air-gapped installation** section to `docs/usage/installation.rst` covering the weights search order, required filenames, `STARLING_OFFLINE`, shared read-only weight directories, the Zenodo search artifacts, the `torch.hub` protein language model cache (which `STARLING_OFFLINE` does not govern), Docker, and a summary table of the relevant environment variables.

## August 17th 2026

This release is a cleanup of the model code: several exploratory and superseded model implementations that were never used by the released STARLING models have been removed from `main`, along with the training-config plumbing and autosummary stubs that referenced them. No user-facing behaviour of `starling`, `generate()`, or `Ensemble` changes.

### Removed

- **Continuous-time diffusion formulation** (`starling/models/continuous_diffusion.py`).
  Removed the `ContinuousDiffusion` model and its log-SNR noise schedules (`beta_linear_log_snr`, `alpha_cosine_log_snr`, `karras_log_snr`). The released models use the discrete DDPM formulation in `starling/models/diffusion.py`. The `continuous:` block has been dropped from `starling/configs/diffusion/diffusion.yaml`, and `diffusion.type` now only accepts `discrete`.

- **Pre-transformer UNet backbone** (`starling/models/unet.py`, `starling/configs/unet/unet.yaml`).
  Removed `UNetConditional` and its building blocks (`ResnetLayer`, `CrossAttentionResnetLayer`, `Downsample`, `ConditionalSequential`). The diffusion model is built on the ViT backbone (`starling/models/vit.py`), so the UNet was dead code. The `unet` entry was removed from the Hydra defaults in `starling/configs/configs.yaml`, and `starling/models/vit.py` now imports `SinusoidalPosEmb` from `starling/models/transformer.py` rather than from the UNet module.

- **Orphaned exploratory modules**: the VQ-VAE quantizer (`starling/models/quantize.py`, `VectorQuantizer2`), the original ResNet encoder/decoder family (`starling/models/resnets_original.py`, `Resnet18`–`Resnet152` encoders/decoders), and the EMA helper (`starling/models/ema.py`). None of these were imported anywhere in the package.

- **Broken console entry points** (`pyproject.toml`).
  Removed the `starling-sample` and `ae-train` scripts, which pointed at `starling.training.vae_generate` and `starling.training.ae_train`; neither module exists, so the entry points failed on invocation.

### Bug Fixes

- **`starling-ddpm-train` failed at import time** (`starling/training/diffusion_train.py`).
  Removed a stale `import starling.data.ddpm_loader`; that module no longer exists (only `ddpm_loader_tar` is present), so the training entry point raised `ModuleNotFoundError` before doing anything. The alias was never referenced.

### Code Quality

- **`setup_models()` in `starling/training/diffusion_train.py`** now returns only the diffusion model instead of a `(UNet, diffusion_model)` tuple, and the `model_architecture.txt` written at the start of training now records the diffusion model (i.e. the ViT backbone plus sequence encoder) rather than the unused UNet.

### Documentation

- Removed the `models.unet`, `models.ema`, `models.continuous_diffusion`, `models.quantize`, and `models.resnets_original` entries from `docs/api.rst` and deleted their autosummary stubs under `docs/autosummary/`.
- Fixed the description line in this changelog, which was previously missing the package name.

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
