Installation
=============

STARLING is available on GitHub (bleeding edge) and on PyPi (stable).

Creating an Environment
------------------------

We recommend creating a fresh conda environment for STARLING:

.. code-block:: bash

    conda create -n starling python=3.11 -y
    conda activate starling

Installation Options
----------------------

Install from PyPi (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install STARLING from PyPi using pip:

.. code-block:: bash

    pip install idptools-starling

Install from GitHub (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Or you can clone and install the bleeding-edge version from GitHub:

.. code-block:: bash

    git clone git@github.com:idptools/starling.git
    cd starling
    pip install .

GPU Installation (CUDA)
-----------------------

For GPU-accelerated search with FAISS, you need to install PyTorch and FAISS-GPU 
via conda to match your CUDA version. As of October 14th, 2025, the pip package 
for ``faiss-gpu`` is not available, so conda is required. There is currently a 
roadmap to bring support for faiss-gpu wheels back to PyPi you can see more at 
the following `GitHub issue <https://github.com/facebookresearch/faiss/issues/3152#issuecomment-3172876462>`_.
Until then, we must use conda for the GPU components.

**Step 1: Create Environment**

.. code-block:: bash

    conda create -y -n starling python=3.11
    conda activate starling

**Step 2: Install PyTorch with CUDA Support**

Install PyTorch that matches your GPU's CUDA version (example for CUDA 12.x):

.. code-block:: bash

    conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.4

For other CUDA versions, visit the `PyTorch installation page <https://pytorch.org/get-started/locally/>`_.

**Step 3: Install FAISS-GPU**

Install FAISS-GPU matching your CUDA version:

.. code-block:: bash

    conda install -y -c pytorch "faiss-gpu=1.8.*" cuda-version=12.4

**Step 4: Install Other Dependencies**

Install the remaining dependencies via conda (preferred) or pip:

.. code-block:: bash

    conda install -y -c conda-forge lightning numpy scipy cython matplotlib \
      jupyter ipython scikit-learn einops tqdm hdf5plugin mdtraj

**Step 5: Install Pure-Python Packages**

Install packages not available on conda-forge:

.. code-block:: bash

    pip install protfasta soursop "metapredict>=3.0"

**Step 6: Install STARLING**

Finally, install STARLING without auto-installing dependencies:

.. code-block:: bash

    # From PyPI:
    pip install --no-deps idptools-starling
    
    # Or from source:
    cd /path/to/starling
    pip install --no-deps .

**Verification**

Verify GPU support is working:

.. code-block:: bash

    python -c "import faiss; print(f'FAISS GPUs available: {faiss.get_num_gpus()}')"
    python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

Verification
-------------

To verify that STARLING has installed correctly, run:

.. code-block:: bash

    starling --help