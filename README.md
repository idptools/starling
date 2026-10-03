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
##### Last updated August 17th 2026

STARLING (con**ST**ruction of intrinsic**A**lly diso**R**dered proteins ensembles efficient**L**y v**I**a multi-dime**N**sional **G**enerative models) is a latent-space probabilistic denoising diffusion model for predicting coarse-grained ensembles of intrinsically disordered regions.  

STARLING was developed by **Borna Novak** and **Jeff Lotthammer** in the [Holehouse lab](https://www.holehouselab.com/) (with some occasional help from Ryan and Alex, as is their wont). 

For more information, please take a look at our paper!

Novak, B., Lotthammer, J. M., Emenecker, R. J. & Holehouse, A. S. 
[**Accurate predictions of disordered protein ensembles with STARLING.** ](https://www.nature.com/articles/s41586-026-10141-2)
*Nature* **652,** 240–250 (2026).  

# Documentation
Detailed documentation is provdied on readthedocs, although this readme is probably enough to do most things.

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

# From a FASTA file, using GPU
starling proteins.fasta -c 200 -d cuda:0 -r -o ./results

# Print STARLING configuration info
starling --info
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `user_input` | positional | — | Sequence string, FASTA file, TSV file, or `.seq.in` file |
| `-c, --conformations` | int | 400 | Number of conformations to generate |
| `--steps` | int | 30 | Number of DDIM denoising steps |
| `-d, --device` | str | auto | Device: `cpu`, `cuda:0`, `cuda:1`, `mps`, etc. |
| `-b, --batch_size` | int | 100 | Batch size for sampling |
| `-o, --output_directory` | str | `.` | Output directory for saving results |
| `--outname` | str | auto | Override output filename prefix (single sequence only) |
| `-r, --return_structures` | flag | off | Generate PDB + XTC 3D structures |
| `--ionic_strength` | int | 150 | Solvent ionic strength in mM (20, 150, or 300) |
| `--remove-errors` | flag | off | Discard physically impossible conformations and regenerate replacements, so the output still contains `-c` conformations |
| `--num-cpus` | int | auto | Max CPUs for MDS reconstruction |
| `--num-mds-init` | int | 4 | Number of parallel MDS initializations |
| `-v, --verbose` | flag | off | Enable verbose output |
| `--disable_progress_bar` | flag | off | Hide progress bars |
| `--info` | flag | — | Print STARLING configuration and exit |
| `--version` | flag | — | Print version and exit |

### Output files

| File | Description |
|------|-------------|
| `*.starling` | Binary ensemble archive (distance maps + metadata) |
| `*_STARLING.pdb` | PDB topology (when `-r` is used) |
| `*_STARLING.xtc` | XTC trajectory with all conformations (when `-r` is used) |

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

Prints the amino acid sequence stored in the `.starling` archive to stdout.

### `starling2info` — Print ensemble metadata

```bash
starling2info my_ensemble.starling
```

Displays metadata about the ensemble, including creation date, sequence, number of conformations, radius of gyration, end-to-end distance, and model weights used.

### `starling2starling` — Repair/validate an archive

```bash
# Check for errors
starling2starling my_ensemble.starling --error-check

# Check and remove problematic conformations
starling2starling my_ensemble.starling --error-check --remove-errors -o fixed_

# Overwrite the original file
starling2starling my_ensemble.starling --error-check --remove-errors --overwrite
```

### `numpy2starling` — Restore from NumPy

```bash
numpy2starling distance_maps.npy -s MKVIFLAVLGLGIVVTTVLY -o ./output
```

Converts a NumPy distance map array and a sequence back into a `.starling` archive. Supports optional `--build-structures` to reconstruct 3D coordinates, and `-x` / `-p` to attach existing XTC/PDB trajectories.

### `xtc2starling` — Convert XTC trajectory to STARLING

```bash
xtc2starling --xtc trajectory.xtc --pdb topology.pdb -o ./output
```

Converts an existing XTC trajectory and PDB topology into a `.starling` archive.


### Converter summary

| Command | Input | Output | Description |
|---------|-------|--------|-------------|
| `starling2pdb` | `.starling` | `.pdb` | Multi-model PDB trajectory |
| `starling2xtc` | `.starling` | `.pdb` + `.xtc` | Topology + compressed trajectory |
| `starling2numpy` | `.starling` | `.npy` | Raw distance maps as NumPy array |
| `starling2sequence` | `.starling` | stdout | Print amino-acid sequence |
| `starling2info` | `.starling` | stdout | Print metadata (version, date, Rg, etc.) |
| `starling2starling` | `.starling` | `.starling` | Re-save with optional error removal |
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
# With filtering by sequence identity and length
starling-search query \
  --seq MKVIFLAVLGLGIVVTTVLY \
  --k 50 \
  --sequence-identity-max 0.9 \
  --length-min 40 \
  --length-max 800 \
  --rerank \
  --out-format csv \
  --out filtered_results
```

#### Query options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--index` | str | `default` | FAISS index path; `default` auto-downloads the pre-built index |
| `--seq` | str | — | Query sequence(s), can be specified multiple times |
| `--k` | int | 10 | Number of nearest neighbors to return |
| `--nprobe` | int | 64 | FAISS probe count (higher = slower but more accurate) |
| `--metric` | str | `cosine` | Distance metric: `cosine` or `l2` |
| `--exclude-exact` | flag | on | Skip exact sequence matches in results |
| `--sequence-identity-max` | float | — | Maximum sequence identity threshold |
| `--identity-denominator` | str | `query` | How to compute identity: `query`, `target`, `max`, `min`, `avg` |
| `--length-min` | int | — | Minimum target sequence length |
| `--length-max` | int | — | Maximum target sequence length |
| `--max-cosine-similarity` | float | — | Pre-filter upper bound on cosine similarity |
| `--min-l2-distance` | float | — | Pre-filter lower bound on L2 distance |
| `--rerank` | flag | on | Re-embed top hits with full encoder for more accurate ranking |
| `--rerank-batch-size` | int | 64 | Batch size for reranking |
| `--rerank-device` | str | auto | Device for reranking |
| `--rerank-ionic-strength` | int | auto | Ionic strength for reranking |
| `--device` | str | `cuda:0` | Device for query embedding |
| `--batch-size` | int | 256 | Batch size for embedding |
| `--ionic-strength` | int | 150 | Ionic strength in mM for encoding |
| `-o, --out` | str | `nearest_neighbors` | Output file basename |
| `--out-format` | str | `csv` | Output format: `csv` or `jsonl` |
| `--verbose` | flag | on | Verbose logging |

### `starling-search build` — Build a custom FAISS index

Build a FAISS index from pre-tokenized sequences (advanced usage).

```bash
starling-search build \
  --root /data/corpus \
  --tokens /data/corpus/tokens \
  --index /indexes/my_index.faiss \
  --sample-size 1000000 \
  --nlist 32768 \
  --use-gpu
```

#### Build options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--root` | str | required | Root data directory |
| `--index` | str | required | Output FAISS index path |
| `--tokens` | str | required | Directory with pre-tokenized sequences |
| `--metric` | str | `cosine` | Distance metric: `cosine` or `l2` |
| `--sample-size` | int | 655360 | Training sample size |
| `--nlist` | int | 16384 | FAISS IVF nlist parameter |
| `--m` | int | 64 | HNSW M parameter |
| `--nbits` | int | 8 | Quantization bits |
| `--add-batch-size` | int | 100000 | Batch size for adding vectors |
| `--nprobe` | int | 16 | FAISS probe count |
| `--use-gpu` | flag | on | Use GPU for index building |
| `--gpu-device` | int | 0 | GPU device ID |
| `--gpu-fp16-lut` | flag | on | Use FP16 lookup tables on GPU |
| `--opq` | flag | off | Enable Optimized Product Quantization |
| `--compress` | flag | off | Compress sequences |
| `--shard-regex` | str | — | Regex filter for shard files |
| `--verbose` | flag | on | Verbose output |

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
| `fastas` | positional | — | Input FASTA file(s) |
| `-o, --output` | str | required | Output directory for token files |
| `--combined` | flag | off | Merge all into a single `.pt` file |
| `--prefix` | str | `pretokenized` | Prefix for combined output file |
| `--sequences` | str | — | Text file with FASTA paths (one per line) |
| `--workers` | int | 1 | Number of parallel tokenizer workers |
| `--no-progress` | flag | off | Hide progress bars |

---

## Training CLIs (Advanced)

These tools are primarily used for model development and retraining.

| Command | Description |
|---------|-------------|
| `starling-vae-train` | Train the VAE encoder model |
| `starling-ddpm-train` | Train the diffusion model |

These commands depend on Hydra, OmegaConf and Weights & Biases, which are **not** part of the default install because they are not needed to generate ensembles. To use them, install STARLING with the `train` extra.

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
| List of sequences | `['MKVIFLA...', 'MDVFMKG...']` |
| Dictionary of name→sequence | `{'protein_a': 'MKVIFLA...', 'protein_b': 'MDVFMKG...'}` |
| Path to a `.fasta` file | `'proteins.fasta'` |
| Path to a `.tsv` / `.seq.in` file | `'sequences.tsv'` (tab-separated `name\tsequence`) |

---

## `generate()` — Generate Ensembles

The `generate` function is the main entry point for generating conformational ensembles using the STARLING model. It accepts various input types, generates conformations using DDIM/DDPM, and optionally returns 3D structures.

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

# From a FASTA file, with 3D structures, saved to disk
E_dict = generate('proteins.fasta', conformations=500, return_structures=True, output_directory='./results')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_input` | str / list / dict | — | Input sequences (see supported formats above) |
| `conformations` | int | 400 | Number of conformations to generate |
| `ionic_strength` | int | 150 | Solvent ionic strength in mM (20, 150, or 300) |
| `device` | str | `None` (auto) | Device: `'cpu'`, `'cuda:0'`, `'mps'`, etc. |
| `steps` | int | 30 | Number of denoising steps |
| `sampler` | str | `'ddim'` | Sampler backend |
| `return_structures` | bool | `False` | Generate 3D structures (PDB/XTC) |
| `batch_size` | int | 100 | Batch size for sampling |
| `num_cpus_mds` | int | auto | Max CPUs for MDS reconstruction |
| `num_mds_init` | int | 4 | Number of parallel MDS initializations |
| `output_directory` | str | `None` | Save directory (if set, writes `.starling` files to disk) |
| `output_name` | str | `None` | Override filename prefix (single-sequence mode) |
| `return_data` | bool | `True` | Return `Ensemble` objects (set `False` for fire-and-forget disk saves) |
| `verbose` | bool | `False` | Print status messages |
| `show_progress_bar` | bool | `True` | Show global progress bar |
| `show_per_step_progress_bar` | bool | `True` | Show per-step denoising progress bar |
| `pdb_trajectory` | bool | `False` | Save PDB trajectory alongside XTC |
| `return_single_ensemble` | bool | `False` | Return a single `Ensemble` instead of a dict (single-sequence mode) |
| `constraint` | Constraint | `None` | Constraint object for guided generation |
| `encoder_path` | str | `None` | Custom encoder model checkpoint |
| `ddpm_path` | str | `None` | Custom diffusion model checkpoint |

### Returns

- **`dict[str, Ensemble]`** — by default (one entry per input sequence)
- **`Ensemble`** — when `return_single_ensemble=True` and a single sequence is provided
- **`None`** — when `return_data=False`

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
| `output_directory` | str | `None` | Optional directory to save embeddings |
| `encoder_path` | str | `None` | Custom encoder checkpoint |
| `ddpm_path` | str | `None` | Custom diffusion model checkpoint |
| `pretokenized` | bool | `False` | Skip tokenization if inputs are already tokenized |
| `bucket` | bool | `False` | Adaptive bucketing by sequence length (improves throughput for variable-length inputs) |
| `bucket_size` | int | 32 | Max unique lengths per bucket |
| `free_cuda_cache` | bool | `False` | Release CUDA memory after each batch |
| `return_on_cpu` | bool | `True` | Move tensors to CPU before returning |

### Returns

- **`dict[str, torch.Tensor]`** — keys are sequence names, values are tensors with shape `(L, D)` (residue-level) or `(D,)` (aggregated)

---

## `load_ensemble()` — Load a Saved Ensemble

Reload a previously generated and saved STARLING ensemble from disk.

```python
from starling import load_ensemble

ensemble = load_ensemble('path/to/my_favorite_ensemble.starling')

# Load without 3D structures (faster)
ensemble = load_ensemble('my_ensemble.starling', ignore_structures=True)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | — | Path to a `.starling` file |
| `ignore_structures` | bool | `False` | Skip loading 3D structures for faster loading |

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
| `enabled` | bool | `None` | Enable or disable compilation |
| `mode` | str | `'default'` | Compilation mode: `'default'`, `'reduce-overhead'`, `'max-autotune'` |
| `backend` | str | `'inductor'` | Compilation backend |
| `fullgraph` | bool | `True` | Compile full graph |
| `dynamic` | bool | `None` | Handle dynamic shapes |

### Returns

- **`dict`** with the current compilation settings

---

## `Ensemble` Class

The `Ensemble` class represents an ensemble of conformations for a protein chain. It stores distance maps from which all structural parameters can be derived.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.sequence` | str | Amino acid sequence |
| `.number_of_conformations` | int | Total number of conformations |
| `.sequence_length` | int | Number of residues |
| `.has_structures` | bool | Whether 3D structures are available |
| `.trajectory` | SSProtein | 3D trajectory object (lazy-built on first access) |

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
Computes the hydrodynamic radius from the ensemble.

#### `.local_radius_of_gyration()` — Local Rg for a sub-region
```python
Ensemble.local_radius_of_gyration(start, end, return_mean=False, use_bme_weights=False)
```
Returns the radius of gyration for a sub-region defined by residues `start` to `end`.

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
    num_cpus_mds=configs.DEFAULT_CPU_COUNT_MDS,
    num_mds_init=configs.DEFAULT_MDS_NUM_INIT,
    device=None,
    force_recompute=False,
    progress_bar=True,
)
```
Reconstructs 3D coordinates from distance maps using multidimensional scaling (MDS). Returns an `SSProtein` trajectory object.

### Error checking

#### `.check_for_errors()`
```python
Ensemble.check_for_errors(remove_errors=False, verbose=True, rebuild_trajectory=False)
```
Scans for problematic conformations (e.g., impossible distances). Returns a list of bad frame indices. If `remove_errors=True`, removes them in place.

### Bayesian Maximum Entropy (BME) reweighting

#### `.reweight_bme()`
```python
Ensemble.reweight_bme(experimental_data, ensemble_properties, weights=None, verbose=True)
```
Performs BME reweighting against experimental data. After reweighting, structural property methods accept `use_bme_weights=True` for reweighted statistics.

### File I/O

#### `.save()` — Save an ensemble to disk
```python
Ensemble.save(filename_prefix, compress=False, reduce_precision=None, compression_algorithm='lzma', verbose=True)
```
Saves the ensemble as a `.starling` archive.

#### `.save_trajectory()` — Save 3D trajectory
```python
Ensemble.save_trajectory(filename_prefix, pdb_trajectory=False)
```
Saves the 3D trajectory as XTC (or PDB if `pdb_trajectory=True`).

---

## Constrained Generation

STARLING allows you to generate structural ensembles with constraints — such as experimentally measured distances or local/global shape features. These are passed to `generate()` via the `constraint` parameter.

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

### Applying constraints

```python
ensemble = generate(sequence, constraint=constraint)
```

### Tuning constraint parameters

All constraints accept the following keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_constant` | float | 2.0 | Strength of the constraint |
| `tolerance` | float | 0.0 | Tolerance around the target value |
| `schedule` | str | `'cosine'` | Weight schedule: `'cosine'` or `'bell_shaped'` |
| `guidance_start` | float | 0.0 | When to start applying the constraint (0.0 = start of denoising) |
| `guidance_end` | float | 1.0 | When to stop applying the constraint (1.0 = end of denoising) |

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

#### Maximum sequence length
STARLING currently supports sequences up to **380 residues** in length.

---

## Copyright
Copyright (c) 2024-2026, Borna Novak, Jeffrey Lotthammer, Alex Holehouse
