<!-- ![STARLING_LOGO_FULL](Lamprotornis_hildebrandti_-Tanzania-8-2c.jpg) -->
STARLING
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/starling/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/starling/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/starling/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/starling/branch/main)


Construction of intrinsically disordered protein ensembles through multiscale generative models

# Installation
STARLING is currently only available on Github. 

We recommend creating a fresh conda environment for STARLING (although in principle there's nothing special about the STARLING environment)

```bash
conda create -n starling  python=3.11 -y
conda activate starling
```

You can then install STARLING from GitHub directly using pip:

```bash
pip install idptools-starling
```

Or you can clone and install the bleeding-edge version locally as
	
```bash
git clone git@github.com:idptools/starling.git
cd starling
pip install .
```
### NB: Potential Pytorch / CUDA version issues
If you are on an older version of CUDA, a torch version that *does not have the correct CUDA version* will be installed. This can cause a segfault when running STARLING. To fix this, you need to install torch for your specific CUDA version. For example, to install PyTorch on Linux using pip with a CUDA version of 12.1, you would run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
  
To figure out which version of CUDA you currently have (assuming you have a CUDA-enabled GPU that is set up correctly), you need to run:
```bash
nvidia-smi
```
Which should return information about your GPU, NVIDIA driver version, and your CUDA version at the top.

Please see the [PyTorch install instructions](https://pytorch.org/get-started/locally/) for more info. 
  

# Quickstart
The easiest way to use STARLING is the `starling` command-line tool.

	starling <amino acid sequence> -c <number of confomers> --outname my_cool_idr
	
This will generate an output file call `my_cool_idr.starling`. To convert this to a PDB trajectory run

	starling2pdb my_cool_idr.starling
	
Or to convert to an xtc/pdb combo run:

	starling2xtc my_cool_idr.starling	

### starling tool documentation

	`usage: starling [-h] [-c CONFORMATIONS] [-d DEVICE] [-s STEPS] [-m METHOD] [-b BATCH_SIZE] [-o OUTPUT_DIRECTORY] [--outname OUTNAME] [-r] [-v] [--num-cpus NUM_CPUS]
	                [--num-mds-init NUM_MDS_INIT] [--no-ddim] [--disable_progress_bar] [--info] [--version]
	                [user_input]
	
	Generate distance maps using STARLING.
	
	positional arguments:
	  user_input            Input sequences in various formats (file, string, list, or dict)
	
	options:
	  -h, --help            show this help message and exit
	  -c CONFORMATIONS, --conformations CONFORMATIONS
	                        Number of conformations to generate (default: 200)
	  -d DEVICE, --device DEVICE
	                        Device to use for predictions (default: None, auto-detected)
	  -s STEPS, --steps STEPS
	                        Number of steps to run the DDPM model (default: 25)
	  -b BATCH_SIZE, --batch_size BATCH_SIZE
	                        Batch size to use for sampling (default: 100)
	  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
	                        Directory to save output (default: '.')
	  --outname OUTNAME     If provided and a single sequence is provided, defines the prefix ahead of .pdb/.xtc/.npy extensions (default: None)
	  -r, --return_structures
	                        Return the 3D structures (default: False)
	  -v, --verbose         Enable verbose output (default: False)
	  --num-cpus NUM_CPUS   Sets the max number of CPUs to use. Default: 4.
	  --num-mds-init NUM_MDS_INIT
	                        Sets the number of MDS jobs to be run in parallel. More may give better reconstruction but requires 1:1 with #CPUs to avoid performance penalty. Default: 4.
	  --no-ddim             Disable DDIM for sampling.
	  --disable_progress_bar
	                        Disable progress bar during generation (default: False)
	  --info                Print STARLING information only
	  --version             Print STARLING version o`nly



# Python library 
STARLING can generate Ensemble objects which enable deep investigation into ensemble properties using the ``generate`` function.

## `generate` function documentation
The `generate` function is the main entry point for generating distance maps using the STARLING model. This function accepts various input types, generates conformations using DDPM, and optionally returns the 3D structures. You can customize several parameters for batch size, device, number of steps, and more.

To get started, first import the function:
```python
from starling import generate
```

The ``generate`` function is flexible and can take in sequences in multiple formats. Here are a few examples:

```python
# Example 1: Provide a single sequence as a string
sequence = 'MKVIFLAVLGLGIVVTTVLY'

# E is an Ensemble() object
E = generate(sequence, return_single_ensemble=True)


# Example 2: Provide a list of sequences
sequences = ['MKVIFLAVLGLGIVVTTVLY', 'MKVIFLAVLGLGIVVTTVLY']

# returns a dictionary of the Ensemble() objects
E_dict = generate(sequences)

# Example 3: Provide a dictionary of sequences

# returns a dictionary of the Ensemble() objects
sequences = {'seq1': 'MKVIFLAVLGLGIVVTTVLY', 'seq2': 'MKVIFLAVLGLGIVVTTVLY'}

E_dict = generate(sequences)
```

### ``generate`` function parameters:

- **`user_input`** : `str`, `list`, `dict`  
    The input sequences for the model, which can be provided in multiple formats:
    - `str`: Path to a `.fasta` file.
    - `str`: Path to a `.tsv` or `seq.in` file (formatted as `name\tsequence`).
    - `str`: A single sequence as a string.
    - `list`: A list of sequences.
    - `dict`: A dictionary of sequences (`name: sequence` pairs).

- **`conformations`** : `int`  
    The number of conformations to generate. Default is `200`. The default is defined in `configs.DEFAULT_NUMBER_CONFS`.

- **`device`** : `str`  
    The device to use for prediction. Default is `None`, which selects the optimal device:
    - 'gpu' (CUDA or MPS)
    - Falls back to CPU if GPU is unavailable.

- **`return_single_ensemble`** : `bool`      
	Flag which, if set to true, means we return a STARLING Ensemble() object instead of a dictionary of ID:Ensemble mapping.

- **`steps`** : `int`  
    The number of steps for the DDPM model. Default is `10`. The default is defined in `configs.DEFAULT_STEPS`.

- **`return_structures`** : `bool`  
    If `True`, returns the generated 3D structures. Default is `False`.

- **`batch_size`** : `int`  
    The batch size for sampling. Default is `100` (uses approximately 20 GB memory).
    The default is defined in `configs.DEFAULT_BATCH_SIZE`.

- **`verbose`** : `bool`  
    If `True`, prints verbose output during execution. Default is `False`.

- **`show_progress_bar`** : `bool`  
    If `True`, displays a progress bar during generation. Default is `True`.

## Using an Ensemble class object

### Overview
The `Ensemble` class represents an ensemble of conformations for a protein chain. It is designed to store and manipulate multiple distance maps, and from distance maps, all other structural parameters can be derived


### Methods

#### `.rij()`
```python
Ensemble.rij(i, j, return_mean=False)
```
Returns the distance between residues i, and j, either all instantaneous values or the average in `return_mean=False` is set to True.

#### `.end_to_end_distance()`
```python
Ensemble.end_to_end_distance(return_mean=False)
```
Returns the end-to-end distance, either all instantaneous values or the average in `return_mean=False` is set to True.


#### `.radius_of_gyration()`
```python
Ensemble.radius_of_gyration(return_mean=False)
```
Returns the radius of gyration, either all instantaneous values or the average in `return_mean=False` is set to True.

#### `.loacl_radius_of_gyration()`
```python
Ensemble.local_radius_of_gyration(start, end, return_mean=False)
```
Returns the radius of gyration between two specific residues, either all instantaneous values or the average in `return_mean=False` is set to True.


#### `.distance_maps()`
```python
Ensemble.distance_maps(return_mean=False)
```
Returns the ensemble distance maps, either all instantaneous values (as n x n np.arrays) or the average distance map `return_mean=False` is set to True.


#### `.contact_map()`
```python
Ensemble.contact_map(contact_thresh=11, 
						return_mean=False, 
						return_summed=False):
```
Using a threshold for contacts defines residues that are in direct contact. If return_mean and return_summed are set to False, the function returns a 3D array of instantaneous contact maps for each conformation. If `return_mean` is set, the average contract value for each i-j contact is returned (i.e., a value between 0 and 1). If `return_summed` is set to true, then the summed values are returned instead of the average value.

#### `.build_ensemble_trajectory()`
```python
Ensemble.build_ensemble_trajectory(batch_size=100,
        num_cpus_mds=configs.DEFAULT_CPU_COUNT_MDS,
        num_mds_init=configs.DEFAULT_MDS_NUM_INIT,
        device=None,
        force_recompute=False,
        progress_bar=True,
```
Allows you to build either a pdb file with multiple model entries or a trajectory file (.xtc file) 


## `load_ensemble` Function Documentation
STARLING can also easily reload previously generated and saved STARLING ensembles

```python
from starling import load_ensemble
ensemble = load_ensemble('path/to/my_favorite_ensemble.starling')
```

### Copyright
Copyright (c) 2024-2025, Borna Novak, Jeffrey Lotthammer, Alex Holehouse


#### Acknowledgements 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
