# STARLING on Docker

We provide a Dockerfile for building a self-contained STARLING Docker image. The image includes:

- Python 3.11 with CUDA 12.4 support
- All STARLING dependencies (including PyTorch)
- Pre-downloaded model weights (encoder + DDPM)
- Pre-downloaded search artifacts (FAISS index, sequence store, manifest)

> **Note:** We recommend using the Docker image primarily on compute infrastructure with GPU access. The image is Linux-based, so running it on macOS will **not** use MPS acceleration and will fall back to CPU.

---

## Building the Image

The Dockerfile is located in the `docker/` directory and uses the repository root as the build context. From the `docker/` directory, run:

```bash
cd docker/
docker build -f Dockerfile -t starling ..
```

The build is a multi-stage process:

1. **Builder stage** — installs Python, PyTorch (CUDA 12.4), and STARLING; downloads model weights and search artifacts.
2. **Runtime stage** — copies only the virtual environment, cached weights, and search artifacts into a slim CUDA runtime image.

The first build will take a while as it downloads model weights (~hundreds of MB) and search artifacts (~2.4 GB). Subsequent builds use Docker layer caching and will be much faster unless `starling/` source code changes.

---

## Running the Container

The image uses `starling` as its entrypoint, so you can pass any STARLING CLI arguments directly.

### Check the version

```bash
docker run --rm starling --version
```

### Print help

```bash
docker run --rm starling --help
```

### With GPU access

To use GPU acceleration, pass the `--gpus` flag (requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker run --rm --gpus all starling --help
```

---

## Generating Ensembles

The primary use of STARLING is generating conformational ensembles from protein sequences. Output files are written inside the container at `/work` by default, so mount a local directory to retrieve them.

### Single sequence

```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/work \
  starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
  -c 200 \
  --ionic_strength 150
```

This generates a `.starling` archive containing 200 distance-map conformations.

### Single sequence with 3D structures

Add `-r` / `--return_structures` to also produce PDB topology and XTC trajectory files:

```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/work \
  starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
  -c 200 \
  --return_structures \
  --ionic_strength 150
```

**Output files:**

| File | Description |
|------|-------------|
| `*.starling` | Binary ensemble archive (distance maps + metadata) |
| `*.pdb` | PDB topology (when `--return_structures` is used) |
| `*.xtc` | XTC trajectory with all conformations (when `--return_structures` is used) |

### From a FASTA file

Mount the directory containing your FASTA file and pass its path inside the container:

```bash
docker run --rm --gpus all \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/work \
  starling /input/sequences.fasta \
  -c 500 \
  --return_structures \
  --output_directory /work
```

### Key generation options

| Flag | Default | Description |
|------|---------|-------------|
| `-c, --conformations` | 400 | Number of conformations to generate |
| `--steps` | 25 | DDPM diffusion steps (lower = faster, higher = more diverse) |
| `-d, --device` | auto | Device: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `-b, --batch_size` | auto | Batch size for generation |
| `-r, --return_structures` | off | Generate PDB + XTC 3D structures |
| `--ionic_strength` | 150 | Solvent ionic strength: 20, 150, or 300 mM |
| `-o, --output_directory` | `.` | Output directory |
| `--outname` | auto | Override output filename prefix (single-sequence mode) |
| `-v, --verbose` | off | Verbose output |

---

## Converting Ensembles

STARLING provides several conversion utilities. Since the Docker entrypoint is `starling`, use `docker run --entrypoint <command>` to access them.

### Convert to PDB

```bash
docker run --rm \
  -v $(pwd)/output:/work \
  --entrypoint starling2pdb \
  starling /work/my_ensemble.starling -o /work
```

### Convert to XTC (topology + trajectory)

```bash
docker run --rm \
  -v $(pwd)/output:/work \
  --entrypoint starling2xtc \
  starling /work/my_ensemble.starling -o /work
```

### Convert to NumPy

```bash
docker run --rm \
  -v $(pwd)/output:/work \
  --entrypoint starling2numpy \
  starling /work/my_ensemble.starling -o /work
```

### Print sequence from archive

```bash
docker run --rm \
  -v $(pwd)/output:/work \
  --entrypoint starling2sequence \
  starling /work/my_ensemble.starling
```

### Print ensemble metadata

```bash
docker run --rm \
  -v $(pwd)/output:/work \
  --entrypoint starling2info \
  starling /work/my_ensemble.starling
```

### Error-check and repair an archive

```bash
docker run --rm \
  -v $(pwd)/output:/work \
  --entrypoint starling2starling \
  starling /work/my_ensemble.starling --error-check --remove-errors -o /work/fixed_
```

### All conversion commands

| Command | Input | Output | Description |
|---------|-------|--------|-------------|
| `starling2pdb` | `.starling` | `.pdb` | Multi-model PDB trajectory |
| `starling2xtc` | `.starling` | `.pdb` + `.xtc` | Topology + compressed trajectory |
| `starling2numpy` | `.starling` | `.npy` | Raw distance maps as NumPy array |
| `starling2sequence` | `.starling` | stdout | Print amino-acid sequence |
| `starling2info` | `.starling` | stdout | Print metadata (version, date, Rg, etc.) |
| `starling2starling` | `.starling` | `.starling` | Re-save with optional error removal |
| `numpy2starling` | `.npy` | `.starling` | Restore archive from NumPy (legacy) |

---

## Sequence Search

The Docker image ships with pre-downloaded FAISS search artifacts, so sequence search works out of the box.

### Query by sequence

```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/work \
  --entrypoint starling-search \
  starling query \
  --seq MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
  --k 20 \
  --nprobe 128 \
  --exclude-exact \
  --out /work/search_results
```

### Query with filtering

```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/work \
  --entrypoint starling-search \
  starling query \
  --seq MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKT \
  --k 50 \
  --sequence-identity-max 0.9 \
  --length-min 40 \
  --length-max 800 \
  --rerank \
  --out /work/filtered_results \
  --out-format csv
```

### Key search options

| Flag | Default | Description |
|------|---------|-------------|
| `--seq` | — | Query sequence(s), can be repeated |
| `--k` | 10 | Number of neighbors to return |
| `--nprobe` | 64 | FAISS probe count (higher = slower but more accurate) |
| `--exclude-exact` | on | Skip exact sequence matches |
| `--sequence-identity-max` | — | Filter by max sequence identity |
| `--length-min` / `--length-max` | — | Filter by target sequence length |
| `--rerank` | off | Re-embed top hits for more accurate ranking |
| `-o, --out` | `nearest_neighbors` | Output file basename |
| `--out-format` | `csv` | Output format: `csv` or `jsonl` |

---

## Benchmarking

Run performance benchmarks inside the container:

```bash
docker run --rm --gpus all \
  --entrypoint starling-benchmark \
  starling \
  --device cuda:0 \
  --batch-size 512 \
  --steps 25
```

---

## CPU-Only Usage

If no GPU is available, the container falls back to CPU automatically. Simply omit the `--gpus` flag:

```bash
docker run --rm \
  -v $(pwd)/output:/work \
  starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
  -c 50 \
  --return_structures \
  --device cpu
```

Note that CPU inference is significantly slower than GPU.

---

## Tips

- **Volume mounts are required** to access output files. The container's working directory is `/work`.
- **Input files** (FASTA, TSV) must also be mounted into the container. Use a read-only mount (`:ro`) for inputs.
- **GPU memory:** For long sequences or large batch sizes, you may need to reduce `-b` / `--batch_size` to avoid OOM errors.
- **Image size:** The image is large (~8–10 GB) due to bundled PyTorch, CUDA runtime, model weights, and search artifacts. This is expected.
- **Rebuilding:** If you modify STARLING source code, only the `COPY starling/` and subsequent layers will rebuild. Earlier layers (system packages, PyTorch install) are cached.


---

## Running Offline / Air-Gapped

The image downloads model weights and search artifacts at **build** time, so a container built on a networked machine needs no network access at runtime. To make that explicit — and to get a clear error rather than a hung connection if anything is missing — set `STARLING_OFFLINE`:

```bash
docker run --rm -e STARLING_OFFLINE=1 --gpus all \
  -v $(pwd)/output:/work \
  starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK -c 200
```

To move the image to a machine with no internet access, export and load it:

```bash
# on the networked machine
docker save starling | gzip > starling.tar.gz

# on the offline machine
gunzip -c starling.tar.gz | docker load
```

If you would rather mount weights from the host than bake them into the image, place the two checkpoint files in a directory and mount it at `/root/.starling_weights`:

```bash
docker run --rm -e STARLING_OFFLINE=1 --gpus all \
  -v /path/to/weights:/root/.starling_weights:ro \
  -v $(pwd)/output:/work \
  starling --info
```

`starling --info` prints the weights file that will actually be loaded, so it is the quickest way to confirm an offline setup is wired up correctly. See the [offline installation docs](https://idptools-starling.readthedocs.io/en/latest/usage/installation.html#offline-air-gapped-installation) for the full list of environment variables.
