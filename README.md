STARLING - prediction of disordered protein ensembles from sequence
=============================================

[//]: # (Badges)
[![PyPI Version](https://img.shields.io/pypi/v/idptools-starling.svg)](https://pypi.org/project/idptools-starling/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Docs Status](https://readthedocs.org/projects/idptools-starling/badge/?version=latest)](https://idptools-starling.readthedocs.io)
[![Python Versions](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://pypi.org/project/idptools-starling/)
[![GitHub stars](https://img.shields.io/github/stars/idptools/starling.svg?style=social&label=Star)](https://github.com/idptools/starling/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/idptools/starling)](https://github.com/idptools/starling/commits)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idptools/idpcolab/blob/main/STARLING/STARLING_demo.ipynb)
<br>
<img src="starling_logo-1.png" alt="My Image" width="150"/>
<br>

# About
##### Last updated October 10th 2026

STARLING (con**ST**ruction of intrinsic**A**lly diso**R**dered proteins ensembles efficient**L**y v**I**a multi-dime**N**sional **G**enerative models) is a latent-space probabilistic denoising diffusion model for predicting coarse-grained ensembles of intrinsically disordered regions.  

STARLING was developed by **Borna Novak** and **Jeff Lotthammer** in the [Holehouse lab](https://www.holehouselab.com/) (with some occasional help from Ryan and Alex, as is their wont). 

For more information, please take a look at our paper!

Novak, B., Lotthammer, J. M., Emenecker, R. J. & Holehouse, A. S. 
[**Accurate predictions of disordered protein ensembles with STARLING.** ](https://www.nature.com/articles/s41586-026-10141-2)
*Nature* **652,** 240–250 (2026).  

# Documentation
Detailed documentation is provided on readthedocs, although this readme is probably enough to do most things.

[https://idptools-starling.readthedocs.io/en/latest/](https://idptools-starling.readthedocs.io/en/latest/)


# Colab notebook
A Google Colab notebook for predicting ensembles and performing rudimentary analysis [is available here](https://colab.research.google.com/github/idptools/idpcolab/blob/main/STARLING/STARLING_demo.ipynb).

---

# Installation

STARLING is available on GitHub (bleeding edge) and on PyPi (stable). 

We recommend creating a fresh conda environment for STARLING (although in principle, there's nothing special about the STARLING environment)

```bash
conda create -n starling  python=3.11 -y
conda activate starling
```

You can then install STARLING from PyPI using pip (or uv):

```bash
pip install idptools-starling
```

Or you can install the bleeding-edge version from GitHub:
```bash
pip install git+https://github.com/idptools/starling.git
```

To check that STARLING has been installed correctly, run

	starling --help

A Docker image is also available — see the [Docker documentation](docker/readme.md) for details.

---

# Quickstart
The easiest way to use STARLING for ensemble generation is with the `starling` command-line tool.

	starling <amino acid sequence> -c 400 --outname my_cool_idr -r

Example:

	starling MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA --outname synuclein -r

Will generate three files:	

* `synuclein.starling` — the full STARLING ensemble file. This holds all the information associated with the ensemble.
* `synuclein_STARLING.pdb` — the topology file for the ensemble.
* `synuclein_STARLING.xtc` — the trajectory file for the ensemble.

By default, STARLING generates 400 conformations — to change the number of conformations, use the `-c` flag (e.g., `-c 1000` would generate an ensemble with 1000 conformations). 

With `-r`, the 3D structures are relaxed with the Mpipi-GG force field by default (see [Structure relaxation](#structure-relaxation)); add `--relax fast` for a faster reconstruction before relaxation, or `--no-relax` to get the raw reconstructed structures instead.

# Performance
STARLING is VERY fast on GPUs and — honestly — VERY fast on Apple Silicon as well. It is a bit slower on CPUs, but we're talking minutes instead of seconds for ensemble generation. 

---

# Command-Line Interface (CLI)

STARLING installs several command-line tools. Below is a complete reference for all of them.

## `starling` — Ensemble Generation

The main CLI tool. Generates conformational ensembles from amino acid sequences.

```bash
starling <input> [options]
```

**Input formats:** a raw amino acid sequence, a `.fasta` file, a `.tsv` file (`name<TAB>sequence`), or a `.seq.in` file.

**Examples:**

```bash
# Single sequence, 400 conformations with 3D structures
starling MKVIFLAVLGLGIVVTTVLY -c 400 -r --outname my_protein

# From a FASTA file, using GPU (the output directory must already exist)
mkdir -p results
starling proteins.fasta -c 200 -d cuda:0 -r -o ./results

# Print STARLING configuration info
starling --info
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `user_input` | positional | — | Sequence string, FASTA file, TSV file, or `.seq.in` file |
| `-c, --conformations` | int | 400 | Number of conformations to generate |
| `-s, --steps` | int | 30 | Number of sampler steps (DDIM with 30 steps runs 31 model evaluations and warns below 10 steps; PLMS needs at least 20; DDPM ignores this and always uses all 1000 timesteps) |
| `--sampler` | str | `ddim` | Sampler: `ddim`, `ddpm`, `plms` or `dpmpp` |
| `-d, --device` | str | auto | Device: `cpu`, `cuda:0`, `cuda:1`, `mps`, etc. |
| `-b, --batch_size` | int | 100 | Batch size for sampling |
| `-o, --output_directory` | str | `.` | Output directory for saving results (must already exist) |
| `--outname` | str | `sequence_1` | Output filename prefix (single sequence only) |
| `-r, --return_structures` | flag | off | Generate PDB + XTC 3D structures (relaxed with Mpipi-GG unless `--no-relax` is given) |
| `--relax [standard\|fast]` / `--no-relax` | str | `standard` with `-r` | Relax the 3D structures with Mpipi-GG (see [Structure relaxation](#structure-relaxation)); `--relax` alone means `standard`, and `--relax fast` runs a shorter MDS step first. Relaxation needs structures, so `--relax` without `-r` is an error |
| `--compile-forces` | flag | off | Fuse the CUDA thermalization forces for faster relaxation of large jobs (adds compilation time at startup; needs `-r`, relaxation and a CUDA device) |
| `--seed` | int | unseeded | Random seed; the same seed reproduces the same ensembles on the same hardware and settings |
| `--ionic_strength` | float | 150 | Solvent ionic strength in mM (STARLING was trained at 20, 150 and 300 mM) |
| `--remove-errors` | flag | off | Discard physically impossible conformations and regenerate replacements, so the output still contains `-c` conformations |
| `-v, --verbose` | flag | off | Enable verbose output |
| `--disable_progress_bar` | flag | off | Hide both progress bars (overall and per-step) |
| `--info` | flag | — | Print STARLING configuration and exit |
| `--version` | flag | — | Print version and exit |

### Output files

| File | Description |
|------|-------------|
| `<name>.starling` | Binary ensemble archive (distance maps + metadata, plus the structures when `-r` is used). `<name>` is the full sequence name, including any dots |
| `<name>_STARLING.pdb` | PDB topology (when `-r` is used) |
| `<name>_STARLING.xtc` | XTC trajectory with all conformations (when `-r` is used) |

Because sequence names become file names, names containing `/` (common in FASTA headers) are rejected before any sampling starts; rename them first.

---

## `starling-benchmark` — Performance Benchmarking

Profile model throughput and measure performance across different configurations.

```bash
starling-benchmark [options]
```

**Examples:**

```bash
# Default benchmark sweep (10 to 1000 conformations)
starling-benchmark --device cuda:0

# Single run with 500 conformations and model compilation
starling-benchmark --device cuda:0 --single-run 500 --compile
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--device` | str | auto | Device for benchmarking |
| `--batch-size` | int | 100 | Batch size |
| `--steps` | int | 30 | Diffusion steps |
| `--sequence` | str | alpha-synuclein | Test sequence (default: 140 aa) |
| `--cooltime` | int | 20 | Cooldown seconds between runs |
| `--single-run` | int | 0 | Single test with N conformations (0 = sweep series) |
| `--compile` | flag | off | Enable PyTorch model compilation (CUDA only) |

---

## File Conversion Tools

STARLING ships with several converters for working with `.starling` ensemble archives.

By default each converter writes to the current directory, naming its output after the input file (e.g. `my_ensemble.pdb`). `-o` takes either an existing directory (output is written there under the input's name) or a filename prefix. Converters never re-save the input `.starling` archive. If the archive holds no structures, `starling2pdb` and `starling2xtc` reconstruct them from the distance maps, but these structures are **not** relaxed with Mpipi-GG; use `starling.minimizer.relax_ensemble()` (see [Structure relaxation](#structure-relaxation)) if you need relaxed structures for an existing archive.

### `starling2pdb` — Convert to PDB

```bash
starling2pdb my_ensemble.starling -o ./output
```

Generates a multi-model PDB trajectory file.

### `starling2xtc` — Convert to XTC

```bash
starling2xtc my_ensemble.starling -o ./output
```

Generates a PDB topology file plus a compressed XTC trajectory file.

### `starling2numpy` — Convert to NumPy

```bash
starling2numpy my_ensemble.starling -o ./output
```

Exports the raw distance maps as a NumPy `.npy` array with shape `(n_conformations, n_residues, n_residues)`.

### `starling2sequence` — Print sequence

```bash
starling2sequence my_ensemble.starling
```

Prints the amino acid sequence stored in the `.starling` archive to stdout (there is no `-o`; redirect with `>` to save it, e.g. `starling2sequence my_ensemble.starling > my_ensemble.txt`).

### `starling2info` — Print ensemble metadata

```bash
starling2info my_ensemble.starling
```

Displays metadata about the ensemble, including creation date, sequence, number of conformations, radius of gyration, end-to-end distance, and model weights used.

### `starling2starling` — Repair/validate an archive

```bash
# Check for errors (nothing is written)
starling2starling my_ensemble.starling --error-check

# Check and remove problematic conformations; writes ./fixed_my_ensemble.starling
starling2starling my_ensemble.starling --remove-errors

# ...or write fixed_my_ensemble.starling into an existing directory
starling2starling my_ensemble.starling --remove-errors -o ./cleaned

# Overwrite the original file
starling2starling my_ensemble.starling --remove-errors --overwrite
```

`--remove-errors` implies `--error-check`. A new file is only written when `--remove-errors` is set and at least one erroneous conformation was found and removed; any structures in the archive are kept (the bad frames are sliced out).

### `numpy2starling` — Restore from NumPy

```bash
numpy2starling distance_maps.npy -s MKVIFLAVLGLGIVVTTVLY -o ./output
```

Converts a NumPy distance map array and a sequence back into a `.starling` archive. Supports optional `--build-structures` to reconstruct 3D coordinates (unrelaxed), and `-x` / `-p` to attach existing XTC/PDB trajectories (an XTC must have one frame per distance map).

### `xtc2starling` — Convert XTC trajectory to STARLING

```bash
xtc2starling --xtc trajectory.xtc --pdb topology.pdb -o ./output
```

Converts an existing XTC trajectory and PDB topology into a `.starling` archive. The distance maps are computed from the trajectory, and the trajectory's coordinates are kept as the archive's structures.


### Converter summary

| Command | Input | Output | Description |
|---------|-------|--------|-------------|
| `starling2pdb` | `.starling` | `.pdb` | Multi-model PDB trajectory |
| `starling2xtc` | `.starling` | `.pdb` + `.xtc` | Topology + compressed trajectory |
| `starling2numpy` | `.starling` | `.npy` | Raw distance maps as NumPy array |
| `starling2sequence` | `.starling` | stdout | Print amino-acid sequence |
| `starling2info` | `.starling` | stdout | Print metadata (version, date, Rg, etc.) |
| `starling2starling` | `.starling` | `.starling` | Error check; with `--remove-errors`, write `fixed_<name>.starling` |
| `numpy2starling` | `.npy` | `.starling` | Restore archive from NumPy |
| `xtc2starling` | `.xtc` + `.pdb` | `.starling` | Convert MD trajectory to STARLING |

---

## `starling-search` — Sequence Search

STARLING includes a FAISS-based similarity search engine that uses ensemble-aware sequence embeddings. It has two subcommands: `build` and `query`.

### `starling-search query` — Find similar sequences

Search the pre-built FAISS index for sequences with similar ensemble properties.

```bash
starling-search query \
  --seq MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKT \
  --k 20 \
  --nprobe 128 \
  --exclude-exact \
  --out search_results
```

```bash
# Two queries, with filtering by sequence identity and length
starling-search query \
  --seq MKVIFLAVLGLGIVVTTVLY MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKT \
  --k 50 \
  --sequence-identity-max 0.9 \
  --length-min 40 \
  --length-max 800 \
  --out-format csv \
  --out filtered_results
```

Several query sequences can be given after one `--seq` (as above) or by repeating the flag (`--seq SEQ1 --seq SEQ2`). Results are written to `<out>.csv` (or `<out>.jsonl`) plus a FASTA file of the hits, `<out>.fasta`.

#### Query options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--index` | str | `default` | FAISS index path; `default` (or a path that doesn't exist) downloads and caches the pre-built index |
| `--seq` | str | — | Query sequence(s); several can follow one `--seq`, and the flag can be repeated |
| `--k` | int | 10 | Number of nearest neighbors to return |
| `--nprobe` | int | 64 | FAISS probe count (higher = slower but more accurate) |
| `--metric` | str | the index's metric | `cosine` or `l2`; must match the metric the index was built with |
| `--exclude-exact` / `--include-exact` | flag | exclude | Skip hits whose sequence exactly matches the query |
| `--sequence-identity-max` | float | — | Maximum sequence identity to the query (0–1, inclusive) |
| `--identity-denominator` | str | `query` | How to compute identity: `query`, `target`, `max`, `min`, `avg` |
| `--length-min` | int | — | Minimum target sequence length |
| `--length-max` | int | — | Maximum target sequence length |
| `--max-cosine-similarity` | float | — | Upper bound on cosine similarity (inclusive; applied to the exact rerank scores when reranking) |
| `--min-l2-distance` | float | — | Lower bound on L2 distance (inclusive; applied to the exact rerank scores when reranking) |
| `--rerank` / `--no-rerank` | flag | rerank | Re-encode the top hits and the queries with the full encoder and rescore them exactly |
| `--rerank-batch-size` | int | 64 | Batch size for reranking |
| `--rerank-device` | str | same as `--device` | Device for reranking |
| `--rerank-ionic-strength` | int | same as `--ionic-strength` | Ionic strength (mM) at which hits and queries are re-encoded for reranking |
| `--device` | str | auto | Device for query embedding (auto picks `cuda`, then `mps`, then `cpu`) |
| `--batch-size` | int | 256 | Batch size for embedding |
| `--ionic-strength` | int | 150 | Ionic strength in mM for encoding the queries |
| `--out` | str | `nearest_neighbors` | Output file basename (the extension is set by `--out-format`) |
| `--out-format` | str | `csv` | Output format: `csv` or `jsonl` |
| `--verbose` / `--quiet` | flag | verbose | Verbose logging |

### `starling-search build` — Build a custom FAISS index

Build a FAISS index from sequence-feature shards and pre-tokenized sequences (advanced usage). Each shard is a `sequence_features.pt` file found recursively under `--root`, in a directory whose name matches `--shard-regex`; its tokens must be in `<tokens>/<shard directory name>.tokens.pt`, which is what `starling-pretokenize` writes for a FASTA file named `<shard directory name>.fasta`. Feature vectors are matched to sequences by header, so every feature shard must carry headers that match its tokens file one-to-one, and shard ids must be unique. All tokens files are checked before training starts.

```bash
starling-search build \
  --root /data/corpus \
  --tokens /data/corpus/tokens \
  --index /indexes/my_index.faiss \
  --sample-size 1000000 \
  --nlist 32768
```

#### Build options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--root` | str | required | Root directory searched recursively for `sequence_features.pt` shards |
| `--index` | str | required | Output FAISS index path |
| `--tokens` | str | required | Directory with one `<shard directory name>.tokens.pt` file per shard |
| `--metric` | str | `cosine` | Distance metric: `cosine` or `l2` |
| `--sample-size` | int | 655360 | Training sample size |
| `--nlist` | int | 16384 | FAISS IVF nlist parameter |
| `--m` | int | 64 | Number of PQ subquantizers (must divide the embedding dimension) |
| `--nbits` | int | 8 | Bits per PQ subquantizer |
| `--add-batch-size` | int | 100000 | Batch size for adding vectors |
| `--nprobe` | int | 16 | Default FAISS probe count stored with the index |
| `--use-gpu` / `--no-gpu` | flag | auto | Train on GPU; by default a GPU is used only if FAISS reports one |
| `--gpu-device` | int | 0 | GPU device ID |
| `--gpu-fp16-lut` / `--no-gpu-fp16-lut` | flag | on | Use FP16 lookup tables on GPU |
| `--opq` / `--no-opq` | flag | on | Learn an Optimized Product Quantization rotation before IVF-PQ |
| `--compress` | flag | off | Compress the stored sequences |
| `--shard-regex` | str | `uniref50_idrs_only_(\d{6})` | Regex whose first capture group is the numeric shard id |
| `--verbose` / `--quiet` | flag | verbose | Verbose output |

GPU FAISS is not installed by pip (the default install uses `faiss-cpu`); see the [installation docs](https://idptools-starling.readthedocs.io/en/latest/usage/installation.html#gpu-installation-cuda) for installing it from conda.

---

## `starling-pretokenize` — Pre-tokenize Sequences

Pre-encode FASTA files for rapid FAISS index construction (used before `starling-search build`).

```bash
starling-pretokenize sequences/*.fasta \
  --output tokens_dir \
  --combined \
  --workers 4
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `fastas` | positional | — | Input FASTA file(s); each is written to `<output>/<basename>.tokens.pt` (two inputs with the same basename are an error) |
| `-o, --output` | str | required | Output directory for token files |
| `--combined` | flag | off | Merge all into a single `<prefix>.pt` file |
| `--prefix` | str | `pretokenized` | Prefix for combined output file |
| `--sequences` | str | — | Text file with FASTA paths (one per line), used instead of positional FASTA files (giving both is an error) |
| `--workers` | int | 1 | Number of parallel tokenizer workers |
| `--no-progress` | flag | off | Hide progress bars |

---

## Training CLIs (Advanced)

These tools are primarily used for model development and retraining.

| Command | Description |
|---------|-------------|
| `starling-vae-train` | Train the VAE encoder model |
| `starling-ddpm-train` | Train the diffusion model |

These commands depend on Hydra, OmegaConf, Weights & Biases, WebDataset and tabulate, which are **not** part of the default install because they are not needed to generate ensembles. To use them, install STARLING with the `train` extra.

From PyPI:

```bash
pip install "idptools-starling[train]"
```

Straight from GitHub, without cloning — note that the extra attaches to the *package name*, not to the URL:

```bash
pip install "idptools-starling[train] @ git+https://github.com/idptools/starling.git"
```

To pin a branch, tag or commit, append it to the URL with `@`:

```bash
pip install "idptools-starling[train] @ git+https://github.com/idptools/starling.git@main"
```

From a local clone:

```bash
pip install ".[train]"

# or, for an editable development install
pip install -e ".[train]"
```

> **Keep the quotes.** In `zsh` (the default shell on macOS) an unquoted `pip install idptools-starling[train]` is treated as a glob and fails with `zsh: no matches found`. Quoting works in every shell.

---

# Python Library

As well as the command-line tools, STARLING provides a powerful Python API for generating and analyzing ensembles programmatically.

## Supported input formats

All main API functions accept sequences in multiple formats:

| Format | Example |
|--------|---------|
| Single sequence string | `'MKVIFLAVLGLGIVVTTVLY'` |
| List (or tuple) of sequences | `['MKVIFLA...', 'MDVFMKG...']` |
| Dictionary of name→sequence | `{'protein_a': 'MKVIFLA...', 'protein_b': 'MDVFMKG...'}` |
| Path (str or `pathlib.Path`) to a `.fasta` file | `'proteins.fasta'` |
| Path (str or `pathlib.Path`) to a `.tsv` / `.seq.in` file | `'sequences.tsv'` (tab-separated `name\tsequence`; blank lines are skipped) |

Sequences may only contain the 20 standard amino acids, and empty sequences are rejected. In FASTA files, non-canonical residues are converted (X→G, B→N, U→C, Z→Q; `*`, `-` and spaces removed) with a warning naming each sequence that was changed; every other input type raises a `ValueError` instead. `generate()` skips sequences longer than 380 residues with a warning, and raises a `ValueError` (before loading the models) if no sequences are left.

---

## `generate()` — Generate Ensembles

The `generate` function is the main entry point for generating conformational ensembles using the STARLING model. It accepts various input types, generates conformations with a diffusion sampler (DDIM by default), and optionally returns 3D structures (relaxed with Mpipi-GG by default).

```python
from starling import generate
```

### Basic usage

```python
# Single sequence → single Ensemble object
E = generate('MKVIFLAVLGLGIVVTTVLY', return_single_ensemble=True)

# List of sequences → dict of Ensemble objects
E_dict = generate(['MKVIFLAVLGLGIVVTTVLY', 'MDVFMKGLSKAKEGVVAAAEKTKQGVAE'])

# Dictionary of sequences → dict of Ensemble objects
E_dict = generate({'seq1': 'MKVIFLAVLGLGIVVTTVLY', 'seq2': 'MDVFMKGLSKAKEGVVAAAEKTKQGVAE'})

# From a FASTA file, with 3D structures, saved to disk (./results must already exist)
E_dict = generate('proteins.fasta', conformations=500, return_structures=True, output_directory='./results')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_input` | str / list / tuple / dict / path | — | Input sequences (see supported formats above) |
| `conformations` | int | 400 | Number of conformations to generate |
| `ionic_strength` | float | 150 | Solvent ionic strength in mM; must be finite and >= 0 (STARLING was trained at 20, 150 and 300 mM) |
| `device` | str | `None` (auto) | Device: `'cpu'`, `'cuda:0'`, `'mps'`, etc. |
| `steps` | int | 30 | Number of denoising steps (DDIM with 30 steps runs 31 model evaluations and warns below 10 steps, because very large deterministic steps narrow the ensemble; PLMS needs at least 20 steps and raises a `ValueError` otherwise; ignored by DDPM, which always uses all 1000 timesteps) |
| `sampler` | str | `'ddim'` | Sampler: `'ddim'`, `'ddpm'`, `'plms'` or `'dpmpp'` (DPM-Solver++(2M), which does not support constraints) |
| `return_structures` | bool | `False` | Generate 3D structures (PDB/XTC), relaxed with Mpipi-GG unless `relax=False` |
| `batch_size` | int | 100 | Batch size for sampling |
| `output_directory` | str | `None` | Existing directory to save results in. If set, each ensemble is written as `<name>.starling` and, with `return_structures=True`, `<name>_STARLING.pdb` + `<name>_STARLING.xtc`. Sequence names containing `/` are then rejected |
| `output_name` | str | `None` | Override the sequence name (and so the filename prefix) when a single sequence string is passed |
| `return_data` | bool | `True` | Return `Ensemble` objects (set `False` for fire-and-forget disk saves; then returns `None`) |
| `verbose` | bool | `False` | Print status messages |
| `show_progress_bar` | bool | `True` | Show global progress bar |
| `show_per_step_progress_bar` | bool | `True` | Show per-step denoising progress bar |
| `pdb_trajectory` | bool | `False` | Write the structures as one multi-model `<name>_STARLING.pdb` instead of a PDB topology + XTC trajectory (only applies with `return_structures=True` and an `output_directory`) |
| `return_single_ensemble` | bool | `False` | Return a single `Ensemble` instead of a dict (only one sequence allowed) |
| `constraint` | Constraint or list of Constraints | `None` | Constraint for guided generation; a list or tuple is combined into a `MultiConstraint` (see [Constrained Generation](#constrained-generation)) |
| `encoder_path` | str | `None` | Custom encoder model checkpoint |
| `ddpm_path` | str | `None` | Custom diffusion model checkpoint |
| `remove_errors` | bool | `False` | Discard conformations with physically impossible distances and generate replacements, so exactly `conformations` clean conformations are returned |
| `relax` | str or bool | `"standard"` | With `return_structures=True`, relax the 3D structures with Mpipi-GG while restrained to their STARLING distance maps: `"standard"`, `"fast"` (a shorter MDS step first) or `False` (see below). `True` means `"standard"` |
| `compile_forces` | bool | `False` | Fuse the CUDA thermalization forces for faster relaxation of large jobs (adds compilation time; needs relaxation, `return_structures=True` and a CUDA device) |
| `seed` | int | `None` | Seed for a reproducible run (see [Reproducibility](#reproducibility)); leaves PyTorch's global random state untouched |

### Returns

- **`dict[str, Ensemble]`** — by default (one entry per input sequence, keyed by sequence name; a single string gets the name `sequence_1`)
- **`Ensemble`** — when `return_single_ensemble=True` and a single sequence is provided
- **`None`** — when `return_data=False`

### Structure relaxation

STARLING predicts distance maps; 3D structures are reconstructed from them by classical MDS followed by weighted SMACOF. This gets the global shape right but leaves bonds compressed and beads too close together. With `return_structures=True` (or `starling -r`), `generate()` therefore relaxes each structure with the Mpipi-GG force field while restraining it to its STARLING distance map: a FIRE energy minimization followed by 250 steps of Langevin thermalization at 300 K. This gives bond, angle and dihedral distributions close to those of Mpipi-GG simulations while leaving global dimensions essentially unchanged. Pass `relax=False` (or `--no-relax`) for the raw reconstructed structures.

There are two relaxation modes, which relax each structure in exactly the same way and differ only in the reconstruction before it. `relax="standard"` (the default) reconstructs each structure as STARLING always has: the classical-MDS starting point from a full eigendecomposition, then up to 300 SMACOF iterations. `relax="fast"` (`--relax fast`) computes only the three eigenvectors the starting point needs and runs 100 SMACOF iterations, leaving relaxation to finish the fit; SMACOF does most of its work in its first few dozen iterations, and relaxation refits every structure to its distance map anyway. This makes the reconstruction about 4x faster for long sequences. Across 20-380 residues and compact to expanded ensembles, the relaxed fast ensembles match the standard ones in fit to the distance maps, radius of gyration, bond lengths and closest contacts, but individual conformations differ, because each lands on a different structure that fits its map about equally well. Structures returned without relaxation always use the standard reconstruction.

Structures built later from the distance maps — via `Ensemble.trajectory`, `Ensemble.build_ensemble_trajectory()`, `starling2xtc` or `starling2pdb` — are **not** relaxed. To relax the structures of an existing ensemble, use `starling.minimizer.relax_ensemble()`, which returns a new `Ensemble` with relaxed structures plus a `RelaxationResult` with diagnostics:

```python
from starling import load_ensemble
from starling.minimizer import relax_ensemble

ensemble = load_ensemble('my_ensemble.starling')
relaxed, relaxation_result = relax_ensemble(ensemble)
relaxed.save_trajectory('my_ensemble_relaxed')
```

### Reproducibility

Pass `seed` (or `--seed` on the command line) to make a run reproducible. With the same seed, device, batch size and settings, `generate()` returns the same ensembles, and it leaves PyTorch's global random state untouched. Without a seed, sampling draws from PyTorch's global generator, so seeding PyTorch before the call also works:

```python
E = generate('MKVIFLAVLGLGIVVTTVLY', seed=42, return_single_ensemble=True)

# or, equivalently for reproducibility
import torch
torch.manual_seed(42)
E = generate('MKVIFLAVLGLGIVVTTVLY', return_single_ensemble=True)
```

Either way, relaxation draws its own random numbers, so changing `relax` or `return_structures` never changes the distance maps. 3D reconstruction (SMACOF) always runs at full float32 precision, whatever `torch.set_float32_matmul_precision()` is set to, so structures rebuilt from saved distance maps are the same in any process. Constraint-guided sampling runs cuDNN deterministically, so it is reproducible too.

---

## `sequence_encoder()` — Ensemble-Aware Sequence Embeddings

STARLING jointly trains a transformer-based sequence encoder that produces embeddings optimized for ensemble generation. Sequences with similar ensemble properties tend to have similar embeddings, making them useful for search and design applications.

```python
from starling import sequence_encoder
```

### Basic usage

```python
# Residue-level embeddings (returns dict of name → tensor with shape (L, D))
embeddings = sequence_encoder('proteins.fasta')

# Protein-level embeddings via mean pooling
embeddings = sequence_encoder('proteins.fasta', aggregate=True)

# With custom settings
embeddings = sequence_encoder(
    {'prot_a': 'MKVIFLA...', 'prot_b': 'MDVFMKG...'},
    ionic_strength=150,
    batch_size=64,
    aggregate=True,
    device='cuda:0',
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence_dict` | str / list / dict | — | Input sequences (same formats as `generate()`) |
| `ionic_strength` | int | 150 | Ionic strength in mM |
| `batch_size` | int | 32 | Sequences per batch |
| `aggregate` | bool | `False` | Return protein-level (mean-pooled) embeddings instead of residue-level |
| `device` | str | `None` (auto) | Target device |
| `output_directory` | str | `None` | Optional directory to save embeddings in, one `<name>.pt` file per sequence; nothing is returned in this case |
| `encoder_path` | str | `None` | Custom encoder checkpoint |
| `ddpm_path` | str | `None` | Custom diffusion model checkpoint |
| `pretokenized` | bool | `False` | Skip tokenization if inputs are already tokenized |
| `bucket` | bool | `False` | Sequences are always batched longest first; with `bucket=True` batches are also split wherever the length bucket (`L // bucket_size`) changes, so no sequence is padded by `bucket_size` residues or more |
| `bucket_size` | int | 32 | Width of each length bucket in residues (only used with `bucket=True`) |
| `free_cuda_cache` | bool | `False` | Release CUDA memory after each batch |
| `return_on_cpu` | bool | `True` | Move tensors to CPU before returning |

### Returns

- **`dict[str, torch.Tensor]`** — keys are sequence names, values are tensors with shape `(L, D)` (residue-level) or `(D,)` (aggregated)
- **`None`** — when `output_directory` is set (the embeddings are written to disk instead)

---

## `load_ensemble()` — Load a Saved Ensemble

Reload a previously generated and saved STARLING ensemble from disk.

```python
from starling import load_ensemble

ensemble = load_ensemble('path/to/my_favorite_ensemble.starling')

# Load without the stored 3D structures
ensemble = load_ensemble('my_ensemble.starling', ignore_structures=True)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | — | Path to a `.starling` file |
| `ignore_structures` | bool | `False` | Discard any 3D structures stored in the file. The whole archive is still read, so this does not make loading faster |

### Returns

- **`Ensemble`** object

---

## `set_compilation_options()` — PyTorch Model Compilation

If you intend to use STARLING repeatedly (e.g., in loops or batch processing), enable `torch.compile` to optimize model kernels. This adds overhead during the first call but improves subsequent runs by approximately 40% (tested on NVIDIA A5000).

```python
import starling

# Enable compilation
starling.set_compilation_options(enabled=True)

# Enable with custom options
starling.set_compilation_options(
    enabled=True,
    mode='max-autotune',
    backend='inductor',
    fullgraph=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `None` | Enable or disable compilation (`None` keeps the current setting) |
| `mode` | str | `'default'` | Compilation mode: `'default'`, `'reduce-overhead'`, `'max-autotune'` |
| `backend` | str | `'inductor'` | Compilation backend |
| `fullgraph` | bool | `True` | Compile full graph |
| `dynamic` | bool | `None` | Handle dynamic shapes |
| `options` | dict | — | Backend-specific options, e.g. `{'triton.cudagraphs': True}` |

Any other `torch.compile` keyword argument is passed through. As with `torch.compile`, `mode` and `options` cannot be combined: passing both raises a `ValueError`, and setting one removes the other from the stored settings.

### Returns

- **`dict`** with the current compilation settings

---

## `Ensemble` Class

The `Ensemble` class represents an ensemble of conformations for a protein chain. It stores distance maps (in Å) from which all structural parameters can be derived. If you build one yourself with `Ensemble(distance_maps, sequence)`, the maps must be full symmetric matrices with a zero diagonal (use `starling.utilities.symmetrize_distance_maps()` on maps that hold only one triangle).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.sequence` | str | Amino acid sequence |
| `.number_of_conformations` | int | Total number of conformations |
| `.sequence_length` | int | Number of residues |
| `.has_structures` | bool | Whether 3D structures are available |
| `.trajectory` | SSProtein | 3D trajectory object (built from the distance maps on first access if the ensemble has no structures; such structures are not relaxed) |

### Structural analysis methods

#### `.rij()` — Inter-residue distance
```python
Ensemble.rij(i, j, return_mean=False, use_bme_weights=False)
```
Returns the distance between residues `i` and `j` across all conformations, or the mean distance if `return_mean=True`.

#### `.end_to_end_distance()` — End-to-end distance
```python
Ensemble.end_to_end_distance(return_mean=False, use_bme_weights=False)
```
Returns the end-to-end distance across all conformations, or the mean.

#### `.radius_of_gyration()` — Radius of gyration
```python
Ensemble.radius_of_gyration(return_mean=False, force_recompute=False, use_bme_weights=False)
```
Returns the radius of gyration across all conformations, or the mean.

#### `.hydrodynamic_radius()` — Hydrodynamic radius
```python
Ensemble.hydrodynamic_radius(return_mean=False, force_recompute=False, mode='nygaard', alpha1=0.216, alpha2=4.06, alpha3=0.821)
```
Computes the hydrodynamic radius (Å) of each conformation with the Nygaard equation (default) or, with `mode='kr'`, the Kirkwood-Riseman equation, `1/Rh = (1/N^2) Σ_{i≠j} 1/r_ij`. The Nygaard equation is not defined for very short chains (where Rg/Rh would be zero or negative) and raises a `ValueError` there; use `mode='kr'` instead.

#### `.local_radius_of_gyration()` — Local Rg for a sub-region
```python
Ensemble.local_radius_of_gyration(start, end, return_mean=False, use_bme_weights=False)
```
Returns the radius of gyration for the sub-region `start` to `end - 1` (0-based, `end` exclusive, like a Python slice). The region must satisfy `0 <= start < end <= sequence_length`.

#### `.distance_maps()` — Pairwise distance maps
```python
Ensemble.distance_maps(return_mean=False, use_bme_weights=False)
```
Returns the raw distance maps as `(n, L, L)` NumPy arrays, or the average distance map if `return_mean=True`.

#### `.contact_map()` — Contact maps
```python
Ensemble.contact_map(contact_thresh=11, return_mean=False, return_summed=False)
```
Returns binary contact maps using a distance threshold. If `return_mean=True`, returns the contact probability (0–1) for each residue pair. If `return_summed=True`, returns summed contacts instead.

### 3D structure reconstruction

#### `.build_ensemble_trajectory()`
```python
Ensemble.build_ensemble_trajectory(
    batch_size=100,
    device=None,
    force_recompute=False,
    progress_bar=True,
)
```
Initializes 3D coordinates with classical MDS, then refines weighted distance stress using batched Torch SMACOF on CPU, CUDA, or MPS. Accurately predicted residue separations receive greater weight. SMACOF runs up to 300 iterations and stops a conformation early only if its stress stops changing; most conformations use all 300 (`starling.structure.coordinates.smacof_converged()` reports which stopped early). Returns an `SSProtein` trajectory object. There are no random restarts or CPU-worker controls. If the ensemble already has structures, nothing is rebuilt unless `force_recompute=True`. Structures built here are not relaxed with Mpipi-GG; use `starling.minimizer.relax_ensemble()` for that (see [Structure relaxation](#structure-relaxation)).

### Error checking

#### `.check_for_errors()`
```python
Ensemble.check_for_errors(remove_errors=False, verbose=True, rebuild_trajectory=False)
```
Scans for problematic conformations (e.g., impossible distances). Returns a list of bad frame indices. If `remove_errors=True`, removes them in place; any attached structures are sliced to match, so relaxed or user-supplied coordinates are kept. With `rebuild_trajectory=True` the structures are instead rebuilt from the remaining distance maps (unrelaxed).

### Bayesian Maximum Entropy (BME) reweighting

#### `.reweight_bme()`
```python
Ensemble.reweight_bme(observables, calculated_values, theta=None, theta_range=(0.01, 10.0), theta_n_points=15, max_iterations=50000, optimizer='L-BFGS-B', initial_weights=None, force_recompute=False, verbose=True, show=False, save_theta_scan_plot=None, theta_method='perpendicular')
```
Performs BME reweighting against a list of `ExperimentalObservable`s, given the per-conformation values of each observable (`calculated_values`, shape `(n_conformations, n_observables)`). The weights minimize `(1/2) χ² + θ · D_KL(w || w0)`. With `theta=None`, θ is chosen automatically from an L-curve scan: a `(min, max)` tuple for `theta_range` scans `theta_n_points` log-spaced values, while a list or array is used as the exact grid. Returns a `BMEResult` (weights in `result.weights`; with automatic θ, the scan is in `result.metadata['theta_scan_result']`). A successful fit is cached and reused only for identical inputs; a failed fit gives a warning and is not cached. After reweighting, `rij()`, `end_to_end_distance()`, `radius_of_gyration()`, `local_radius_of_gyration()` and `distance_maps()` accept `use_bme_weights=True` for reweighted statistics.

### File I/O

#### `.save()` — Save an ensemble to disk
```python
Ensemble.save(filename_prefix, compress=False, reduce_precision=None, compression_algorithm='lzma', verbose=True)
```
Saves the ensemble (distance maps, metadata and any structures) as `<filename_prefix>.starling`, or as `.starling.xz` / `.starling.gzip` with `compress=True`. Anything after the last `.` in `filename_prefix` is treated as an extension and removed, so pass e.g. `'NP_000537.3.starling'` to keep a dotted name. `reduce_precision` is a flag: when true, distances are rounded to 0.1 Å and stored as float16 (it defaults to the value of `compress`).

#### `.save_trajectory()` — Save 3D trajectory
```python
Ensemble.save_trajectory(filename_prefix, pdb_trajectory=False)
```
Saves the 3D trajectory as `<filename_prefix>.pdb` (topology) plus `<filename_prefix>.xtc`, or as a single multi-model `<filename_prefix>.pdb` if `pdb_trajectory=True`. If the ensemble has no structures yet, they are built (unrelaxed) from the distance maps first.

---

## Constrained Generation

STARLING allows you to generate structural ensembles with constraints — such as experimentally measured distances or local/global shape features. These are passed to `generate()` via the `constraint` parameter. Constraints work with the DDIM (default), PLMS and DDPM samplers, but not with DPM++. Residue indices are 0-based (the first residue is 0) and are checked against the sequence length.

### Available constraint types

```python
from starling.inference.constraints import (
    DistanceConstraint,
    RgConstraint,
    ReConstraint,
    HelicityConstraint,
    BondConstraint,
    StericClashConstraint,
    MultiConstraint,
)
```

#### `DistanceConstraint` — target distance between two residues
```python
constraint = DistanceConstraint(resid1=10, resid2=200, target=50)
```

#### `RgConstraint` — target radius of gyration
```python
constraint = RgConstraint(target=50)
```

#### `ReConstraint` — target end-to-end distance
```python
constraint = ReConstraint(target=100)
```

#### `HelicityConstraint` — enforce helical structure in a range
```python
# residues 10 to 99 (0-based); resid_end is exclusive, like a Python slice
constraint = HelicityConstraint(resid_start=10, resid_end=100)
```

#### `BondConstraint` — maintain consecutive residue spacing
```python
constraint = BondConstraint(bond_length=3.81)
```

#### `StericClashConstraint` — prevent steric clashes
```python
constraint = StericClashConstraint(steric_clash_definition=5.0)
```

#### `MultiConstraint` — combine multiple constraints
```python
constraint = MultiConstraint([
    DistanceConstraint(resid1=10, resid2=200, target=50),
    RgConstraint(target=30),
])
```

You can also pass a plain list or tuple of constraints to `generate()`; it is wrapped in a `MultiConstraint` for you. Each constraint inside a `MultiConstraint` keeps its own `constraint_weight` and guidance window, but the time schedule is the `MultiConstraint`'s own (`'cosine'` by default; build the `MultiConstraint` yourself to choose another). The constraints' own `schedule` settings are ignored, with a warning if they differ.

### Applying constraints

```python
ensemble = generate(sequence, constraint=constraint, return_single_ensemble=True)
```

### Tuning constraint parameters

The individual constraints accept the following keyword arguments (`MultiConstraint` takes only `constraints`, `schedule` and `verbose`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_constant` | float | 2.0 | Strength of the constraint |
| `tolerance` | float | 0.0 | Tolerance around the target value (not accepted by `StericClashConstraint`) |
| `constraint_weight` | float | 1.0 | Overall weight of the constraint's guidance |
| `schedule` | str or None | `'cosine'` | How guidance strength changes during denoising: `'cosine'`, `'bell_shaped'`, `'linear'`, or `None` for a constant weight |
| `guidance_start` | float | 0.0 | When to start applying the constraint (0.0 = start of denoising) |
| `guidance_end` | float | 1.0 | When to stop applying the constraint (1.0 = end of denoising) |

Guidance strength does not depend on the batch size: each conformation is guided by the gradient of its own loss, calibrated so that existing settings behave as they did at the default batch size of 100.

**Guidance timing reference:**

| Window | `guidance_start` | `guidance_end` | What's being denoised |
|--------|-------------------|-----------------|----------------------|
| Early | 0.0 | 0.3 | Mostly noise, minimal structural information |
| Mid | 0.3 | 0.7 | Emerging structure, useful features begin to form |
| Late | 0.7 | 1.0 | Fine details, near-final structural refinement |

Experimenting with these parameters for your particular application is recommended.

---

# FAQs/Help

#### I get a NumPy compilation warning error!?

Oh no! You get the following error message:

	A module that was compiled using NumPy 1.x cannot be run in
	NumPy 2.2.3 as it may crash. To support both 1.x and 2.x
	versions of NumPy, modules must be compiled with NumPy 2.0.
	Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
	
	If you are a user of the module, the easiest solution will be to
	downgrade to 'numpy<2' or try to upgrade the affected module.
	We expect that some modules will need time to support NumPy 2.

We have seen this if folks are trying to install on Intel Macs because (Py)Torch stopped supporting Intel Macs after torch=2.2.2. If you're NOT on an Intel mac, the recommended way to resolve us by upgrading torch:

	# recommended, but ANY version above 2.2.2 should work
	pip install torch==2.6.0	

or if you're on an Intel mac and torch > 2.2.2 is not available, downgrade numpy:

	pip install numpy==1.26.1	

#### Potential PyTorch / CUDA version issues
If you are on an older version of CUDA, a torch version that *does not have the correct CUDA version* will be installed. This can cause a segfault when running STARLING. To fix this, you need to install torch for your specific CUDA version. For example, to install PyTorch on Linux using pip with a CUDA version of 12.1, you would run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

To figure out which version of CUDA you currently have (assuming you have a CUDA-enabled GPU that is set up correctly), you need to run:
```bash
nvidia-smi
```
This should return information about your GPU, NVIDIA driver version, and your CUDA version at the top.

Please see the [PyTorch install instructions](https://pytorch.org/get-started/locally/) for more info. 

#### Search on macOS (OpenMP)
On macOS, `faiss-cpu` and PyTorch each ship their own OpenMP runtime, and a Python process that uses both can abort with `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized` (or segfault). STARLING handles this automatically: importing `starling.search` (which `starling-search` does) on macOS sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before FAISS is loaded, keeping any values you have already set. As a result FAISS runs single-threaded on macOS unless you set `OMP_NUM_THREADS` yourself. If your own code imports `faiss` before `starling.search`, export the variables yourself before starting Python:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

#### Maximum sequence length
STARLING currently supports sequences up to **380 residues** in length. `generate()` (and the `starling` command) skips longer sequences with a warning.

---

## Copyright
Copyright (c) 2024-2026, Borna Novak, Jeffrey Lotthammer, Alex Holehouse
