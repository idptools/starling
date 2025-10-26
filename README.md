<!-- ![STARLING_LOGO_FULL](Lamprotornis_hildebrandti_-Tanzania-8-2c.jpg) -->
STARLING - prediction of disordered protein ensembles from sequence
=============================================

[//]: # (Badges)
[![PyPI Version](https://img.shields.io/pypi/v/idptools-starling.svg)](https://pypi.org/project/idptools-starling/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Docs Status](https://readthedocs.org/projects/idptools-starling/badge/?version=latest)](https://idptools-starling.readthedocs.io)
[![Python Versions](https://img.shields.io/pypi/pyversions/idptools-starling.svg)](https://pypi.org/project/idptools-starling/)
[![GitHub stars](https://img.shields.io/github/stars/idptools/starling.svg?style=social&label=Star)](https://github.com/idptools/starling/stargazers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idptools/idpcolab/blob/main/STARLING/STARLING_demo.ipynb)
<br>
<img src="starling_logo-1.png" alt="My Image" width="150"/>
<br>

# About
STARLING (con**ST**ruction of intrinsic**A**lly diso**R**dered proteins ensembles efficient**L**y v**I**a multi-dime**N**sional **G**enerative models) is a latent-space probabilistic denoising diffusion model for predicting coarse-grained ensembles of intrinsically disordered regions.  

STARLING was developed by **Borna Novak** and **Jeff Lotthammer** in the [Holehouse lab](https://www.holehouselab.com/) (with some occasional help from Ryan and Alex, as is their wont). 

For more information, please take a look at our preprint!<br>
**Accurate predictions of conformational ensembles of disordered proteins with STARLING.** Novak, B., Lotthammer, J. M., Emenecker, R. J. & Holehouse, A. S. ***bioRxiv*** [2025.02.14.638373 Preprint at https://doi.org/10.1101/2025.02.14.638373 (2025)](https://www.biorxiv.org/content/10.1101/2025.02.14.638373v1)

### A note on STARLING version/manuscript
The STARLING manuscript [was preprinted and submitted in February 2025](https://doi.org/10.1101/2025.02.14.638373). Reviews were received in May 2025, and a revised version was resubmitted (but not updated as a preprint, in compliance with the publisher's requirements) in October 2025. 

However, STARLING itself has been completely updated as part of the revision process, moving from a UNet-based diffusion model to a Vision Transformer. The notebooks in this repository reflect the revised version. We are happy to share our revised version of the manuscript with people—please just email alex (alex dot holehouse at wustl dot edu), and he can share it!

# Documentation
[https://idptools-starling.readthedocs.io/en/latest/](https://idptools-starling.readthedocs.io/en/latest/)

# Performance
STARLING is VERY fast on GPUs and – honestly – VERY fast on Apple Silicon as well. It is a bit slower than CPUs, but we're talking minutes instead of seconds for ensemble generation. 

# Quick installation
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

Or you can clone and install the bleeding-edge version from GitHub:
	
```bash
pip install git+https://github.com/idptools/starling.git
```

To check that STARLING has been installed correctly, run

	starling --help

# Quickstart
The easiest way to use STARLING for ensemble generation is with the `starling` command-line tool.

	starling <amino acid sequence> -c --outname my_cool_idr -r
	
Example:

	starling MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA --outname synuclein -r
	
Will generate three files:	

* `synuclein.starling` - the full STARLING ensemble file. This holds all the information associated with the ensemble.
* `synuclein_STARLING.pdb` - the topology file for the ensemble.
* `synuclein_STARLING.xtc - the trajectory file for the ensemble. 

By default, STARLING generates 400 conformations - to change the number of conformations, you can use the `-c` flag (e.g., `-c 1000` would generate an ensemble with 1000 conformations. 


### Querying a STARLING ensemble

The command `starling2info` provides information about a .starling file, including when it was generated, the sequence, the number of conformations, and model weights.


### Converting a STARLING ensemble
STARLING comes with several tools for converting .starling files to other types of files, including:

* `starling2pdb` - convert a .staring file to a PDB ensemble
* `starling2xtc` - convert a .staring file to a PDB file and an XTC ensemble.
		
### Benchmarking STARLING
You can benchmark STARLING's performance using

	starling-benchmark
		

# Python library 
As well as a command-line tool, STARLING provides a simple but powerful Python API for generating ensembles.

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

## Ensemble-aware sequence embeddings

STARLING jointly trains a Vision Transformer (ViT) that operates in the VAE latent space alongside a transformer-based sequence encoder designed to produce sequence embeddings optimized for ensemble generation. These embeddings can be extracted independently of the ensemble generation process. We observe that sequences with similar ensemble properties tend to have similar embeddings, making them useful for search and design applications. To extract these sequence embeddings, use the following code block:

```python
from starling.frontend.ensemble_generation import sequence_encoder

sequence_embeddings = sequence_encoder(fasta)
```

The output of this function consists of residue-level embeddings, which can be aggregated into a protein-level embedding using a mean-pooling operation.

## Constrained generation of ensembles

STARLING allows you to generate structural ensembles with constraints or restraints — such as experimentally measured distances or local/global shape features. These can be directly incorporated into sampling by passing them as arguments to the `generate` function. To do this, simply include the following code block as part of your generate call:

### Distance constraint

```python
constraint = DistanceConstraint(resid1=10, resid2=200, target=50)
```

### Radius of gyration constraint
```python
constraint = RgConstraint(target=50)
```

### End-to-end distance constraint
```python
constraint = ReConstraint(target=100)
```

### Helicity constraint
```python
constraint = HelicityConstraint(start=10, end=100)
```
These constraints are applied during generation by including them directly in the `generate` function call, as shown below:

```python
# constraint = any of the above constraints
ensemble = generate(sequence, constraint=constraint)
```

Additionally, the strength of the constraints can be controlled using the `force_constant` keyword, which takes a float value. The timing of when these constraints are applied during the denoising process is also important and can be specified using the `guidance_start` and `guidance_end` keywords — both of which should be values between 0 and 1, representing the fractional progress through the denoising steps. 

| Timing      | `guidance_start` | `guidance_end` | What’s being denoised during this time              |
|-------------|------------------|----------------|-----------------------------------------------------|
| Early       | 0.0              | 0.3            | Mostly noise, minimal structural information        |
| Mid         | 0.3              | 0.7            | Emerging structure, useful features begin to form   |
| Late        | 0.7              | 1.0            | Fine details, near-final structural refinement       |

Experimenting with the above parameters for your own purpose is recommended. 


## Speed up STARLING by compiling the torch models

If you intend to use STARLING repeatedly in your computational workflow, consider calling torch.compile to optimize the kernels within the STARLING models. While this adds some overhead during the initial model loading, it improves the performance of subsequent runs by approximately 40% (tested on an NVIDIA A5000 GPU). To compile STARLING, include the following code block in your script:

```python
import starling

# The following line enables compiling
starling.set_compilation_options(enabled=True)

# To change any compilation options use 
starling.set_compilation_options(mode='',...)
```

Utilizing the above functionality is particularly useful in scenarios where models are repeatedly called within loops, batch processing, or iterative inference, as it enhances execution speed, reduces overhead, and ensures more efficient memory usage.

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

#### `.local_radius_of_gyration()`
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

## FAQS/Help

#### I get a numpy compilation warning error!?
Oh no! You get the following error message:
 
	A module that was compiled using NumPy 1.x cannot be run in
	NumPy 2.2.3 as it may crash. To support both 1.x and 2.x
	versions of NumPy, modules must be compiled with NumPy 2.0.
	Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
	
	If you are a user of the module, the easiest solution will be to
	downgrade to 'numpy<2' or try to upgrade the affected module.
	We expect that some modules will need time to support NumPy 2.
	
We have seen this if folks are trying to install on Intel Macs because (Py)Torch stopped supporting Intel Macs after torch=2.2.2. If you're NOT on an Intel mac, the recommended way to resolve us by upgrading torch:

	# recomemended, but ANY version above 2.2.2 should work
	pip install torch==2.6.0	
	
or if you're on an Intel mac and torch > 2.2.2 is not available, downgrade numpy:

	pip install numpy==1.26.1	

#### Potential Pytorch / CUDA version issues
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

## Copyright
Copyright (c) 2024-2025, Borna Novak, Jeffrey Lotthammer, Alex Holehouse


### Acknowledgements 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
