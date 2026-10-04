# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests, 
conda installation, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

You should test your code, but do not feel compelled to use these specific programs. You also may not need Unix and 
Windows testing if you only plan to deploy on specific platforms. These are just to help you get started.

### Conda Environment:

This directory contains the files to setup the Conda environment for testing purposes

* `conda-envs`: directory containing the YAML file(s) which fully describe Conda Environments, their dependencies, and those dependency provenance's
  * `test_env.yaml`: Simple test environment file with base dependencies. Channels are not specified here and therefore respect global Conda configuration
  
### Additional Scripts:

This directory contains OS agnostic helper scripts which don't fall in any of the previous categories
* `scripts`
  * `benchmark_generation.py`: Benchmark `generate()` with local weights, relaxed structures, and STARLING/PDB/XTC saving. Defaults to DPM++-12; `--device cuda` or `--device mps` selects the accelerator. Reports loading, first-use and warmed totals, and separate phase timings. CUDA reports peak allocation and refuses shared-GPU timings; MPS peak allocation is not reported. Uses one warmup and three measured repeats. Output includes sequences, weight and source hashes, and software versions. Model and force `torch.compile` are disabled; reusable CUDA Triton forces are allowed.
  * `create_conda_env.py`: Helper program for spinning up new conda environments based on a starter file with Python Version and Env. Name command-line options
  * `profile_relaxation.py`: Measure MDS-only reconstruction and MDS with physical relaxation on cached distance maps. Reports first-use and warmed timings, FIRE and thermalization costs, and peak CUDA memory; refuses shared-GPU timings. Default: float32 MDS and FIRE, with a reusable Triton thermalization kernel and no `torch.compile`. `--eager-forces`: genuinely eager float32. `--reference-fp64`: eager float64 FIRE and thermalization with the current MDS implementation. `--compile-forces`: optional `torch.compile` path. `--operators 10`: operator trace.
  * `validate_reconstruction_precision.py`: Compare float32 weighted MDS and FIRE with float64 initialization and minimization. Reports convergence, force residuals, bond lengths, angles, clashes, Rg, end-to-end distances, distance-map errors, and stress differences on cached maps. Isolates FIRE precision from the combined change. The MPS reference runs FIRE in float64 on CPU; candidate computations remain float32.
  * `benchmark_fused_forces.py`: Validate the shared production force kernel against float64 forces and 250-step trajectories, including determinism and reuse across sequence lengths. Reports thermalization timings and peak memory. `--precision mixed` tests float32 pair calculations with float64 accumulation/integration; `--precision float32` tests fully float32 thermalization. Precision comparisons use coupled noise and report bonds, angles, dihedral moments, Rg, end-to-end distance, and mean-map error. Requires CUDA and Triton; does not change production dispatch or FIRE precision.


## How to contribute changes
- Clone the repository if you have write access to the main repo, fork the repository if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code
- Ensure that the test environment dependencies (`conda-envs`) line up with the build and deploy dependencies (`conda-recipe/meta.yaml`)
- Push the branch to the repo (either the main or your fork) with `git push -u origin {your branch name}`
  * Note that `origin` is the default name assigned to the remote, yours may be different
- Make a PR on GitHub with your changes
- We'll review the changes and get your code into the repo after lively discussion!


## Checklist for updates
- [ ] Make sure there is an/are issue(s) opened for your specific update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR as needed until tests pass
- [ ] Tag the final, debugged version 
   *  `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged in

## Versioneer Auto-version
[Versioneer](https://github.com/warner/python-versioneer) will automatically infer what version 
is installed by looking at the `git` tags and how many commits ahead this version is. The format follows 
[PEP 440](https://www.python.org/dev/peps/pep-0440/) and has the regular expression of:
```regexp
\d+.\d+.\d+(?\+\d+-[a-z0-9]+)
```
If the version of this commit is the same as a `git` tag, the installed version is the same as the tag, 
e.g. `starling-0.1.2`, otherwise it will be appended with `+X` where `X` is the number of commits 
ahead from the last tag, and then `-YYYYYY` where the `Y`'s are replaced with the `git` commit hash.
