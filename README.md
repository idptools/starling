<!-- ![STARLING_LOGO_FULL](Lamprotornis_hildebrandti_-Tanzania-8-2c.jpg) -->
STARLING
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/starling/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/starling/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/starling/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/starling/branch/main)


Construction of intrinsically disordered proteins ensembles through multiscale generative models

# Installation
STARLING is currently only available on Github. You should be able to pull STARLING and install it locally using pip:

```bash
git clone git@github.com:idptools/starling.git
cd starling
pip install -e .
```
## Potential Pytorch / CUDA version issues
If you are on an older version of CUDA, a torch version that *does not have the correct CUDA version* will be installed. This can cause a segfault when running metapredict. To fix this, you need to install torch for your specific CUDA version. For example, to install PyTorch on Linux using pip with a CUDA version of 12.1, you would run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
  
To figure out which version of CUDA you currently have (assuming you have a CUDA-enabled GPU that is set up correctly), you need to run:
```bash
nvidia-smi
```
Which should return information about your GPU, NVIDIA driver version, and your CUDA version at the top.

Please see the [PyTorch install instructions](https://pytorch.org/get-started/locally/) for more info. 
  

# Usage in Python
STARLING can generate distance maps and structures in Python using the ``generate`` function.

## `generate` Function Documentation
The `generate` function is the main entry point for generating distance maps using the STARLING model. This function accepts a variety of input types, generates conformations using DDPM, and optionally returns the 3D structures. You can customize several parameters for batch size, device, number of steps, and more.

To get started, first import the function:
```python
from starling.frontend.ensemble_generation import generate
```

The ``generate`` function is flexible and can take in sequences in multiple formats. Here are a few examples:

```python
# Example 1: Provide a single sequence as a string
sequence = 'MKVIFLAVLGLGIVVTTVLY'
dist_maps=generate(sequence)

# Example 2: Provide a list of sequences
sequences = ['MKVIFLAVLGLGIVVTTVLY', 'MKVIFLAVLGLGIVVTTVLY']
dist_maps=generate(sequences)

# Example 3: Provide a dictionary of sequences
sequences = {'seq1': 'MKVIFLAVLGLGIVVTTVLY', 'seq2': 'MKVIFLAVLGLGIVVTTVLY'}
dist_maps=generate(sequences)

# Example 4: Provide a path to a .fasta file
dist_maps=generate('path/to/sequences.fasta')

# Example 5: Provide a path to a .tsv file
dist_maps=generate('path/to/sequences.tsv')

# Example 6: Provide a path to a .in file
dist_maps=generate('path/to/sequences.in')
```

By default, the function returns a dictionary of distance maps for each sequence. If you want to return the generated 3D structures, you can set `return_structures=True`:

```python
dist_maps=generate('path/to/sequences.fasta', return_structures=True)
```

You can also specify the output directory to save the distance maps and structures if return_structures=True:

```python
dist_maps=generate('path/to/sequences.fasta', output_directory='path/to/output', return_structures=True)
```

### Additional Usage of the ``generate`` Function

#### ``generate`` function parameters:

- **`user_input`** : `str`, `list`, `dict`  
    The input sequences for the model, which can be provided in multiple formats:
    - `str`: Path to a `.fasta` file.
    - `str`: Path to a `.tsv` or `seq.in` file (formatted as `name\tsequence`).
    - `str`: A single sequence as a string.
    - `list`: A list of sequences.
    - `dict`: A dictionary of sequences (`name: sequence` pairs).

- **`conformations`** : `int`  
    The number of conformations to generate. Default is `200`. The default is defined in `configs.DEFAULT_NUMBER_CONFS`.


- **`encoder`** : `str`  
    The file path to the encoder model. Default uses the model defined in `configs.DEFAULT_ENCODER_WEIGHTS_PATH`.

- **`ddpm`** : `str`  
    The file path to the DDPM model. Default uses the model defined in `configs.DEFAULT_DDPM_WEIGHTS_PATH`.

- **`device`** : `str`  
    The device to use for prediction. Default is `None`, which selects the optimal device:
    - 'gpu' (CUDA or MPS)
    - Falls back to CPU if GPU is unavailable.

- **`steps`** : `int`  
    The number of steps for the DDPM model. Default is `10`. The default is defined in `configs.DEFAULT_STEPS`.

- **`method`** : `str`  
    The method for generating the 3D structure. Options include:
    - `'mds'` (default)
    - `'gd'`

- **`ddim`** : `bool`  
    Whether to use DDIM for sampling. Default is `True`.

- **`return_structures`** : `bool`  
    If `True`, returns the generated 3D structures. Default is `False`.

- **`batch_size`** : `int`  
    The batch size for sampling. Default is `100` (uses approximately 20 GB memory).
    The default is defined in `configs.DEFAULT_BATCH_SIZE`.

- **`output_directory`** : `str`  
    The directory where output files will be saved. Default is `None`. If specified, the function saves:
    - Distance maps as `.npy` files (`<sequence_name>_STARLING_DM.npy`).
    - Structures (if generated) as `.xtc` and `.pdb` files (`<sequence_name>_STARLING.xtc`, `<sequence_name>_STARLING.pdb`).

- **`return_data`** : `bool`  
    If `True`, the function returns distance maps and (optionally) structures as a dictionary. Default is `True`.

- **`verbose`** : `bool`  
    If `True`, prints verbose output during execution. Default is `False`.

- **`show_progress_bar`** : `bool`  
    If `True`, displays a progress bar during generation. Default is `True`.

#### ``generate`` function returns:

- **`dict`** or **`None`**  
    - If `return_data=True`, Default is ``True`` returns a dictionary where:
      - Keys are sequence names.
      - Values are `np.ndarray` distance maps.
    - If `return_structures=True`, the dictionary also includes:
      - Keys with `'_traj'` suffix containing `mdtraj.Trajectory` objects for each structure.
    - If `output_directory` is specified, results are saved to the directory.
    - If `return_data=False`, the function returns `None`.

## `load_ensemble` Function Documentation
STARLING can also easily reload previously generated and saved STARLING ensembles

```python
from starling.structure.ensemble import load_ensemble
ensemble = load_ensemble('path/to/my_favorite_ensemble.starling')
```

# Usage from the command-line
STARLING can also be used from the command-line using the ``starling`` command.

## `starling` command-line Documentation
The `starling` command allows generating distance maps using the STARLING model. Sequences can be input using a variety of input methods. The command generates conformations using DDPM, and optionally returns the 3D structures. You can customize several parameters for batch size, device, number of steps, and more.

``starling`` by default will save the distance maps to your current working directory as a `.npy` file. The ``starling`` command is flexible and can take in sequences in multiple formats. Here are a few examples:

To generate distance maps for a single sequence, you can input the sequence directly.
```bash
starling MKVIFLAVLGLGIVVTTVLY
```

To geneated distance maps for multiple sequences, you need to specify a file path to a `.fasta` file or a `.tsv` or `.in` file formatted as `name\tsequence`.
```bash
starling path/to/sequences.fasta
```

### Additional usage
``starling`` has several optional arguments that can be used to customize the output and behavior of the command.

- **`-c`, `--conformations`** 
    Number of conformations to generate. Default is 10.

- **`-d`, `--device`**
    Device to use for predictions (e.g., "cpu", "cuda"). By default, the tool will auto-detect the device.

- **`-s`, `--steps`**
    Number of steps to run the Diffusion-based model (DDPM). Default is 10.

- **`-m`, `--method`**
    Method to use for generating 3D structures. 
    Options include "gd" (Gradient Descent) and "mds" (Multidimensional Scaling). 
    Default is "mds".

- **`-b`, `--batch_size`**
    Batch size to use for sampling during model inference. Default is 100.

- **`-o`, `--output_directory`**
    Directory to save the generated output files. Default is the current working directory.

- **`-r`, `--return_structures`**
    Whether to return the generated 3D structures in addition to distance maps. 
    If this flag is set, the tool will return the 3D structure output.

- **`-v`, `--verbose`**
    Enable verbose output to display detailed processing information. Default is False.

- **`--no-ddim`**
    Disable the DDIM (Denoising Diffusion Implicit Models) for sampling. DDIM is enabled by default, and using this option will turn it off.

- **`--disable_progress_bar`**
    Disable the progress bar during generation. By default, the progress bar is enabled.

- **`--encoder`**
    Path to the pre-trained encoder model. The encoder is used for feature extraction in the generation process.

- **`--ddpm`**
    Path to the pre-trained DDPM model used for distance map generation.

#### Example Usage

```bash
starling path/to/sequences.fasta -c 100 -d cuda -s 5 -m gd -b 50 -o path/to/output -r -v
```
This would generate 100 conformations for the sequences in the `.fasta` file using the CUDA device, 5 steps for the DDPM model, the Gradient Descent method for 3D structure generation, a batch size of 50, and save the output to the specified directory. The command will also return the generated 3D structures ``-r`` and display verbose output ``-v``.


### Copyright

Copyright (c) 2024-2025, Borna Novak, Jeffrey Lotthammer, Alex Holehouse


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
