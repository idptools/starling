<!-- ![STARLING_LOGO_FULL](Lamprotornis_hildebrandti_-Tanzania-8-2c.jpg) -->
STARLING
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/starling/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/starling/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/starling/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/starling/branch/main)


Construction of intrinsically disordered proteins ensembles through multiscale generative models

## Installation
Right now some of the dependencies for STARLING are not pep517 compliant, so we can't make those dependencies installable using the pyprogect.toml file. For now, follow the below steps to install all dependencies to use STARLING.

1. Install FINCHES
```bash
conda install numpy pytorch scipy cython matplotlib jupyter  -c pytorch
conda install mdtraj
pip install metapredict
pip install git+https://git@github.com/idptools/finches.git
```
2. Install additional dependencies 
```bash
pip install ipython pytorch-lightning scikit-learn einops esm tqdm PyYAML h5py pandas pytest
```
3. Install SPARROW
```bash
pip install git+https://git@github.com/idptools/sparrow.git
```
4. Install STARLING
```bash
git clone git@github.com:idptools/starling.git
cd starling
pip install -e .
```


### Copyright

Copyright (c) 2024, Borna Novak


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
