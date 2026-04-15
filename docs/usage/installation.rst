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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Docker
------

STARLING ships with a Dockerfile that produces a self-contained image with
Python 3.11, CUDA 12.4, all dependencies, pre-downloaded model weights, and
pre-built FAISS search artifacts. This is the easiest way to run STARLING on
GPU-equipped compute infrastructure without managing local environments.

.. note::

   The Docker image is Linux-based, so running it on macOS will **not** use
   MPS acceleration and will fall back to CPU.

Building the image
~~~~~~~~~~~~~~~~~~

The Dockerfile lives in the ``docker/`` directory and uses the repository root
as the build context. From inside the ``docker/`` directory, run:

.. code-block:: bash

    cd docker/
    docker build -f Dockerfile -t starling ..

The build is a multi-stage process:

1. **Builder stage** — installs Python, PyTorch (CUDA 12.4), and STARLING;
   downloads model weights and search artifacts.
2. **Runtime stage** — copies only the virtual environment, cached weights, and
   search artifacts into a slim CUDA runtime image.

The first build downloads model weights and search artifacts (~2.4 GB) and may
take a while. Subsequent builds use Docker layer caching and are much faster
unless ``starling/`` source code changes.

Running the container
~~~~~~~~~~~~~~~~~~~~~

The image uses ``starling`` as its entrypoint, so CLI arguments are passed
directly:

.. code-block:: bash

    # Print version
    docker run --rm starling --version

    # Print help
    docker run --rm starling --help

To use GPU acceleration, pass the ``--gpus`` flag (requires the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_):

.. code-block:: bash

    docker run --rm --gpus all starling --help

Generating ensembles
~~~~~~~~~~~~~~~~~~~~

Output files are written to ``/work`` inside the container. Mount a local
directory to retrieve them:

.. code-block:: bash

    # Single sequence
    docker run --rm --gpus all \
      -v $(pwd)/output:/work \
      starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
      -c 200 \
      --ionic_strength 150

    # With 3D structures (PDB + XTC)
    docker run --rm --gpus all \
      -v $(pwd)/output:/work \
      starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
      -c 200 \
      --return_structures \
      --ionic_strength 150

    # From a FASTA file
    docker run --rm --gpus all \
      -v $(pwd)/input:/input:ro \
      -v $(pwd)/output:/work \
      starling /input/sequences.fasta \
      -c 500 \
      --return_structures \
      --output_directory /work

Conversion utilities
~~~~~~~~~~~~~~~~~~~~

Since the entrypoint is ``starling``, use ``--entrypoint`` to access other
commands:

.. code-block:: bash

    # Convert to PDB
    docker run --rm -v $(pwd)/output:/work \
      --entrypoint starling2pdb \
      starling /work/my_ensemble.starling -o /work

    # Convert to XTC (topology + trajectory)
    docker run --rm -v $(pwd)/output:/work \
      --entrypoint starling2xtc \
      starling /work/my_ensemble.starling -o /work

    # Convert to NumPy
    docker run --rm -v $(pwd)/output:/work \
      --entrypoint starling2numpy \
      starling /work/my_ensemble.starling -o /work

    # Print sequence or metadata
    docker run --rm -v $(pwd)/output:/work \
      --entrypoint starling2info \
      starling /work/my_ensemble.starling

Sequence search
~~~~~~~~~~~~~~~

Search artifacts are baked into the image, so FAISS search works out of the
box:

.. code-block:: bash

    docker run --rm --gpus all \
      -v $(pwd)/output:/work \
      --entrypoint starling-search \
      starling query \
      --seq MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
      --k 20 \
      --nprobe 128 \
      --exclude-exact \
      --out /work/search_results

CPU-only usage
~~~~~~~~~~~~~~

If no GPU is available, omit the ``--gpus`` flag. The container falls back to
CPU automatically:

.. code-block:: bash

    docker run --rm \
      -v $(pwd)/output:/work \
      starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEK \
      -c 50 \
      --device cpu

Docker tips
~~~~~~~~~~~

* **Volume mounts are required** to access output files. The container's working
  directory is ``/work``.
* **Input files** (FASTA, TSV) must also be mounted into the container. Use a
  read-only mount (``:ro``) for inputs.
* **GPU memory:** For long sequences or large batch sizes, reduce
  ``-b`` / ``--batch_size`` to avoid OOM errors.
* **Image size:** The image is ~8–10 GB due to bundled PyTorch, CUDA runtime,
  model weights, and search artifacts.
* **Rebuilding:** Modifying STARLING source code only invalidates the
  ``COPY starling/`` layer and later; earlier layers (system packages, PyTorch)
  are cached.