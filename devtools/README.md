# Development, testing, and data-preparation tools

This directory contains the conda test environment, the lab's data-preparation and evaluation scripts, and the tests for those scripts. None of it is installed with the `starling` package.


## Manifest

### Conda environment

* `conda-envs`: directory containing the YAML file(s) which describe Conda environments and their dependencies
  * `test_env.yaml`: simple test environment with base dependencies (`python`, `pip`, `pytest`, `pytest-cov`, `codecov`). It specifies the `conda-forge` and `defaults` channels.

### Scripts (`scripts/`)

These scripts were written for the lab's cluster and most contain hard-coded paths (see [Hard-coded cluster paths](#hard-coded-cluster-paths) below).

* `append_ionic_strength.py`: Append an `ionic_strength` dataset (in mM, set with `--salt`) to every HDF5 file matching `--input_pattern`.
* `calibrate_weighted_stress.py`: Select a calibration panel, fit paired MD/VAE decoder errors per residue separation and validate weighted coordinate reconstruction. See `weighted_reconstruction.md`.
* `create_conda_env.py`: Helper program for spinning up new conda environments based on a starter file with Python Version and Env. Name command-line options.
* `extract_finches_matrix.py`: Compute the FINCHES pairwise homotypic interaction matrix (Mpipi-GG or CALVADOS2 at the salt concentration given by `--salt`) from the sequence stored in each HDF5 file, and write the raw, row-normalized and Frobenius-normalized matrices back into that file.
* `extract_latents.py`: Encode the distance maps in each simulation HDF5 file with a VAE (posterior means, in batches of 64) and save the latents, their average and the sequence to a `*_adamW_vae_encoded.h5` file next to the input.
* `extract_latents_calvados.py`: As `extract_latents.py`, for the CALVADOS dataset; distance maps are symmetrized and padded to 384x384 first.
* `extract_latents_tar.py`: Encode the distance maps in webdataset tar shards with a VAE and write the latents, sequences and ionic strengths to new tar shards (10,000 samples each) for diffusion training.
* `fit_weighted_stress.py`: Helper functions used by `calibrate_weighted_stress.py` (per-separation error fitting, leave-one-family-out validation, paired-record loading).
* `get_ground_truth.py`: Compute the mean radius of gyration and end-to-end distance of every simulation HDF5 file in a directory and write them to a CSV.
* `vae_test.py`: Evaluate a VAE's reconstruction loss (decoding the posterior mean) as a function of sequence length over validation tar shards, and write per-length statistics to a text file.
* `20mM_300mM_dm.ipynb`: Notebook comparing VAE latents of the same sequences simulated at different ionic strengths (KL divergence, PCA/t-SNE).
* `large_dm_VAE_test.ipynb`: Notebook exploring VAE checkpoints on large distance maps (learned decoder variance, loss weights, per-latent KL).
* `latent_PCA.ipynb`: Notebook running PCA and clustering on average latent vectors and relating them to sequence properties.
* `sequence_embeddings.ipynb`: Notebook relating sequence-encoder embeddings to radius of gyration and end-to-end distance (linear regression, PCA, nearest neighbours).
* `weighted_reconstruction.md`, `weighted_stress_calibration_panel.csv`, `weighted_stress_expanded_study.json`: notes, the frozen panel and results from `calibrate_weighted_stress.py`.
* `loss_by_length_preprint_VAE.txt`: example output from `vae_test.py` for the preprint VAE.

The following scripts are described in this README or the documentation but are **not yet in the repository**:

* `benchmark_generation.py` (not yet in the repository): Benchmark `generate()` with local weights, relaxed structures, and STARLING/PDB/XTC saving. Defaults to DPM++-12; `--device cuda` or `--device mps` selects the accelerator. Reports loading, first-use and warmed totals, and separate phase timings. CUDA reports peak allocation and refuses shared-GPU timings; MPS peak allocation is not reported. Uses one warmup and three measured repeats. Output includes sequences, weight and source hashes, and software versions. Model and force `torch.compile` are disabled; reusable CUDA Triton forces are allowed.
* `profile_relaxation.py` (not yet in the repository): Measure MDS-only reconstruction and MDS with physical relaxation on cached distance maps. Reports first-use and warmed timings, FIRE and thermalization costs, and peak CUDA memory; refuses shared-GPU timings. Default: float32 MDS and FIRE, with a reusable Triton thermalization kernel and no `torch.compile`. `--eager-forces`: genuinely eager float32. `--reference-fp64`: eager float64 FIRE and thermalization with the current MDS implementation. `--compile-forces`: optional `torch.compile` path. `--operators 10`: operator trace.
* `validate_reconstruction_precision.py` (not yet in the repository): Compare float32 weighted MDS and FIRE with float64 initialization and minimization. Reports convergence, force residuals, bond lengths, angles, clashes, Rg, end-to-end distances, distance-map errors, and stress differences on cached maps. Isolates FIRE precision from the combined change. The MPS reference runs FIRE in float64 on CPU; candidate computations remain float32.
* `benchmark_fused_forces.py` (not yet in the repository): Validate the shared production force kernel against float64 forces and 250-step trajectories, including determinism and reuse across sequence lengths. Reports thermalization timings and peak memory. `--precision mixed` tests float32 pair calculations with float64 accumulation/integration; `--precision float32` tests fully float32 thermalization. Precision comparisons use coupled noise and report bonds, angles, dihedral moments, Rg, end-to-end distance, and mean-map error. Requires CUDA and Triton; does not change production dispatch or FIRE precision.
* `validate_samplers.py` (not yet in the repository): the comparative sampler validation runner referred to in `docs/usage/ensemble_generation.rst`.

### Tests (`tests/`)

Fast CPU tests for the scripts above; run them with `pytest devtools/tests`.


## Hard-coded cluster paths

The scripts and the Hydra training configs point at the lab's cluster file system. Anyone else must change these before running them.

Scripts in `scripts/`:

* `append_ionic_strength.py`: `--input_pattern` defaults to `/work/bnovak/projects/sequence2ensemble/lammps_data/300mM_data/mPIPIgg_300mM/*encoded.h5`.
* `calibrate_weighted_stress.py`: `--md-root` defaults to `/work/j.lotthammer/projects`, and the `md_xtc`/`md_topology` columns of `weighted_stress_calibration_panel.csv` point under `/work/j.lotthammer/`.
* `extract_finches_matrix.py`: `--input_pattern` defaults to `/work/bnovak/projects/sequence2ensemble/lammps_data/300mM_data/mPIPIgg_300mM/*encoded.h5`.
* `extract_latents.py`: the VAE checkpoint (`/work/j.lotthammer/projects/starling/VAEs/...`) and input glob (`/work/bnovak/projects/sequence2ensemble/lammps_data/300mM_data/...`) are set in `main()`.
* `extract_latents_calvados.py`: the VAE checkpoint, output directory (`/work/bnovak/...`) and input glob (`/work/j.lotthammer/...`) are set in `main()`.
* `extract_latents_tar.py`: `TARDIR`, `OUTDIR` and `MODEL` at the top of the file (`/work/bnovak/...`).
* `get_ground_truth.py`: the input directory (`/work/bnovak/projects/sequence2ensemble/lammps_data/20mM_data/mPIPIgg_20mM`) is set in `main()`.
* `vae_test.py`: `TARDIR` (`/home/borna/starling_data/validation`) and `MODEL` (`/work/bnovak/...`) at the top of the file.
* The notebooks also load data and checkpoints from `/work/bnovak/...` and `/home/borna/...`.

Hydra configs in `starling/configs/` (used by `starling-vae-train` and `starling-ddpm-train`; any value can also be overridden on the command line, e.g. `starling-vae-train dataloader.tar.dataset=/path/to/shards`):

* `trainer/vae_trainer.yaml`: `fine_tune: true` with `checkpoint: /work/bnovak/projects/sequence2ensemble/vae_training/resnet18_KLD_1e8_warmup_0_1_Adam_1e4/model-kernel-epoch=31-epoch_val_loss=1.75.ckpt` and `num_epochs: 1`. As shipped, `starling-vae-train` therefore fine-tunes that checkpoint for a single epoch rather than training a VAE from scratch; set `trainer.fine_tune=false trainer.checkpoint=null` and a larger `trainer.num_epochs` to train from scratch.
* `trainer/trainer.yaml`: a commented-out `checkpoint: /home/borna/.starling_weights/ddpm_no_encoder_weights.ckpt`.
* `dataloader/dataloader.yaml` and `dataloader/vae_dataloader.yaml`: `tar.dataset` is `/work/bnovak/projects/sequence2ensemble/lammps_data/combined_data/data`, and the `h5` blocks point at `/home/borna/ionic_strength_data/*.csv`. The `h5` dataloader type is not supported by either training script.
* `diffusion/diffusion.yaml`: `discrete.distance_map_encoder` (the VAE used to encode distance maps during diffusion training) is `/work/bnovak/projects/sequence2ensemble/vae_training/preprint_VAE_cont_training_KLD_1e4/model-kernel-epoch=09-epoch_val_loss=2.04.ckpt`.

Both training scripts resume automatically from `<trainer.output_path>/last.ckpt` when it exists, unless `trainer.checkpoint` is set or `trainer.fine_tune` is true. `starling/training/config.yaml` is an unused legacy config; the training scripts never read it.


## How to contribute changes
- Clone the repository if you have write access to the main repo, fork the repository if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code (`pytest starling/tests devtools/tests`)
- Ensure that the test environment dependencies (`conda-envs/test_env.yaml`) line up with the dependencies declared in `pyproject.toml`
- Push the branch to the repo (either the main or your fork) with `git push -u origin {your branch name}`
  * Note that `origin` is the default name assigned to the remote, yours may be different
- Make a PR on GitHub with your changes
- We'll review the changes and get your code into the repo after lively discussion!

There are currently no continuous integration workflows in the repository (no `.github/workflows`), so run the tests locally before opening a PR.


## Checklist for updates
- [ ] Make sure there is an/are issue(s) opened for your specific update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR as needed until tests pass
- [ ] Tag the final, debugged version 
   *  `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged in

## Versioning
The package version is set by [versioningit](https://github.com/jwodder/versioningit) (configured under `[tool.versioningit]` in `pyproject.toml`), which infers it from the most recent `git` tag and writes it to `starling/_version.py`. If the current commit is tagged and the working tree is clean, the version is the tag, e.g. `1.2.0`. Otherwise it is `{tag}+{distance}.g{commit hash}`, where `distance` is the number of commits since the tag, with `.dirty` appended if the working tree has uncommitted changes. Without any tags, versions are based on `1.0.0`.
