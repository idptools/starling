# STARLING throughput benchmarks

These scripts measure how fast STARLING generates ensembles on your computer, in conformers per minute. Each one times `starling.generate()` end to end, from sequence in to ensemble out, in one of three modes:

| Script | Mode | What `generate()` is asked for |
|---|---|---|
| `benchmark_distance_maps.py` | Distance maps only | `return_structures=False` |
| `benchmark_mds_standard.py` | Maps + 3D structures, standard MDS, relaxed | `return_structures=True, relax="standard"` (the default when structures are requested) |
| `benchmark_mds_fast.py` | Maps + 3D structures, fast MDS, relaxed | `return_structures=True, relax="fast"` |
| `run_all.py` | All three, then a comparison table | |

Both structure modes relax every structure the same way (Mpipi-GG minimization then a short thermalization); they differ only in the MDS reconstruction before it.

## Running

With STARLING installed (`pip install idptools-starling`, or `pip install -e .` from this repository), run from this directory:

```bash
python run_all.py
```

or one mode at a time:

```bash
python benchmark_distance_maps.py
python benchmark_mds_standard.py
python benchmark_mds_fast.py
```

The first run downloads the model weights if they are not already cached.

By default, every script uses alpha-synuclein (140 residues), 400 conformations per call, 3 timed calls per mode, and STARLING's defaults for everything else (DDIM with 30 steps, batch size 100, 150 mM), on the fastest device available (CUDA, then Apple MPS, then CPU). The options are the same for every script:

| Option | Default | |
|---|---|---|
| `--sequence` | alpha-synuclein | any amino acid sequence |
| `--conformations` | 400 | conformations per timed call |
| `--repeats` | 3 | timed calls per mode; the median is reported |
| `--device` | fastest available | `cpu`, `mps`, `cuda` or `cuda:N` |
| `--sampler` | `ddim` | `ddim`, `ddpm`, `plms` or `dpmpp` |
| `--steps` | 30 | denoising steps |
| `--batch-size` | 100 | conformations sampled at once |
| `--seed` | 0 | seeds sampling and relaxation |

On a CPU the structure modes are slow (MDS reconstruction takes minutes for 400 conformations), so a quicker first look is:

```bash
python run_all.py --conformations 100 --repeats 1
```

## What is timed

For each mode, the script first makes one untimed warm-up call (one batch of conformations), which loads the models and sets up the device; only the first mode run in a session pays for loading the models. It then times each call from start to return, so the numbers include everything a user waits for: sampling, decoding the distance maps, and, in the structure modes, MDS reconstruction and relaxation. Writing files is not included, since nothing is saved.

The output starts with the STARLING, PyTorch, and Python versions, the device, and the settings, so a pasted result says what it was measured on. For example:

```
STARLING 2.0.2, PyTorch 2.10.0, Python 3.12.13
Device: cuda:0 (NVIDIA RTX A4000)
Sequence: 140 residues; 400 conformations per call, timed calls per mode: 3; ddim with 30 steps, batch size 100

Distance maps only
  warm-up: ...
  call 1: ...
  median: ... conformers/min
```

Throughput depends on sequence length (longer sequences are slower in every mode, and the structure modes more so), so compare numbers measured on the same sequence.
