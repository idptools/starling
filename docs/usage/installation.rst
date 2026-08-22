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

You can install the bleeding-edge version straight from GitHub without cloning:

.. code-block:: bash

    pip install git+https://github.com/idptools/starling.git

Or clone first, which is what you want if you plan to edit the code:

.. code-block:: bash

    git clone git@github.com:idptools/starling.git
    cd starling
    pip install .

Use ``pip install -e .`` instead for an editable (development) install.

.. _training-dependencies:

Installing the training dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default install contains everything needed to generate and analyse
ensembles. The model-training entry points (``starling-vae-train`` and
``starling-ddpm-train``) additionally need Hydra, OmegaConf and Weights &
Biases, which are kept out of the default install. If you intend to train
models, install the ``train`` extra.

From PyPI:

.. code-block:: bash

    pip install "idptools-starling[train]"

Straight from GitHub, without cloning. Note that this uses the PEP 508
``package[extra] @ url`` form — the extra goes on the *package name*, not on the
URL:

.. code-block:: bash

    pip install "idptools-starling[train] @ git+https://github.com/idptools/starling.git"

To pin a branch, tag or commit, append it to the URL with ``@``:

.. code-block:: bash

    pip install "idptools-starling[train] @ git+https://github.com/idptools/starling.git@main"

From a local clone:

.. code-block:: bash

    pip install ".[train]"

    # or, for an editable development install
    pip install -e ".[train]"

.. note::

   Keep the quotes. In ``zsh`` (the default shell on macOS) an unquoted
   ``pip install idptools-starling[train]`` is treated as a glob pattern and
   fails with ``zsh: no matches found``. Quoting works in every shell, so it is
   the safest habit.

Without this extra, ``starling-vae-train`` and ``starling-ddpm-train`` fail
immediately with a ``ModuleNotFoundError``. Nothing else in STARLING is
affected — ensemble generation, analysis, the conversion utilities and search
all work with the default install.

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

.. _offline-installation:

Offline / air-gapped installation
---------------------------------

STARLING downloads two model weight files (the VAE encoder/decoder and the DDPM
weights) the first time it runs, and — if you use ``starling_search`` — three
FAISS search artifacts. On a machine with no outbound internet access you can
pre-place all of these by hand and tell STARLING never to attempt a download.

Where STARLING looks for model weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each of the two checkpoint files, STARLING checks the following locations
in order and uses the first one that exists:

1. The path given by ``STARLING_ENCODER_PATH`` / ``STARLING_DDPM_PATH``, or the
   ``encoder_path`` / ``ddpm_path`` arguments to ``generate()``.
2. ``~/.starling_weights/`` (change this by setting ``DEFAULT_MODEL_DIR`` in
   ``~/.starling_weights/configs.py``).
3. The torch hub checkpoint cache, ``$TORCH_HOME/hub/checkpoints/`` — which is
   ``~/.cache/torch/hub/checkpoints/`` unless ``TORCH_HOME`` is set. This is
   where STARLING puts weights it has downloaded itself.

Only if the files are absent from all of these does STARLING try to download
them. Note the files must keep their released names:

.. code-block:: text

    STARLING_v2.0.0_ViT_VAE_2025_10_14.ckpt
    STARLING_v2.0.0_ViT_DDPM_2025_10_14.ckpt

Forcing offline behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``STARLING_OFFLINE`` to make STARLING use only locally available files:

.. code-block:: bash

    export STARLING_OFFLINE=1

With this set, STARLING never opens a network connection. If a required file is
missing it raises a ``FileNotFoundError`` that lists every path it searched,
rather than hanging on a connection attempt that your firewall will drop.

Setting up an offline machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a machine that *does* have internet access, download the two checkpoints
from the `GitHub release <https://github.com/idptools/starling/releases/tag/v2.0.0>`_
and copy them to the offline machine:

.. code-block:: bash

    # on the offline machine
    mkdir -p ~/.starling_weights
    cp /path/to/STARLING_v2.0.0_ViT_VAE_2025_10_14.ckpt  ~/.starling_weights/
    cp /path/to/STARLING_v2.0.0_ViT_DDPM_2025_10_14.ckpt ~/.starling_weights/
    export STARLING_OFFLINE=1

Then confirm STARLING can see them — ``starling --info`` prints the file it will
actually load for each model, or tells you it could not find it locally:

.. code-block:: bash

    starling --info

If you would rather keep the weights in a shared read-only location (for
example ``/opt/starling/weights``, so all users on a cluster share one copy),
point the environment variables at the files directly:

.. code-block:: bash

    export STARLING_ENCODER_PATH=/opt/starling/weights/STARLING_v2.0.0_ViT_VAE_2025_10_14.ckpt
    export STARLING_DDPM_PATH=/opt/starling/weights/STARLING_v2.0.0_ViT_DDPM_2025_10_14.ckpt
    export STARLING_OFFLINE=1

Search artifacts
~~~~~~~~~~~~~~~~

Ensemble generation needs nothing beyond the two checkpoints above. The
``starling_search`` command additionally needs three FAISS artifacts (~2.4 GB
total), which are fetched from `Zenodo <https://zenodo.org/records/17342150>`_.
Pre-place these in ``~/.starling_search/``, keeping their released names:

.. code-block:: text

    ensemble_search_gpu_nlist_32768_m_64_nbits_8_use_opq_True_compressed_False.faiss
    ensemble_search_gpu_nlist_32768_m_64_nbits_8_use_opq_True_compressed_False.faiss.seqs.sqlite
    ensemble_search_gpu_nlist_32768_m_64_nbits_8_use_opq_True_compressed_False.faiss.manifest.json

or point ``STARLING_FAISS_INDEX_PATH``, ``STARLING_SEQSTORE_PATH`` and
``STARLING_FAISS_MANIFEST_PATH`` at them individually.

.. note::

   The protein language model used by the sequence encoder is downloaded by
   ``torch.hub`` into ``$TORCH_HOME/hub/checkpoints/`` and is **not** covered by
   ``STARLING_OFFLINE``. If you use the sequence-encoder features, copy that
   cache directory across from an online machine as well.

Docker
~~~~~~

The provided Dockerfile downloads weights and search artifacts at *build* time,
so an image built on a networked machine and exported with ``docker save``
needs no network at runtime. Set ``STARLING_OFFLINE=1`` in the container to make
that guarantee explicit:

.. code-block:: bash

    docker run --rm -e STARLING_OFFLINE=1 --gpus all -v $(pwd)/output:/work starling ...

Summary of environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Purpose
   * - ``STARLING_OFFLINE``
     - Never attempt a network download; fail with a clear error instead.
   * - ``STARLING_ENCODER_PATH``
     - Explicit path to the VAE encoder/decoder checkpoint.
   * - ``STARLING_DDPM_PATH``
     - Explicit path to the DDPM checkpoint.
   * - ``TORCH_HOME``
     - Root of the torch cache; weights are read from/written to
       ``$TORCH_HOME/hub/checkpoints/``.
   * - ``STARLING_FAISS_INDEX_PATH``
     - Explicit path to the FAISS index for ``starling_search``.
   * - ``STARLING_SEQSTORE_PATH``
     - Explicit path to the sequence store SQLite database.
   * - ``STARLING_FAISS_MANIFEST_PATH``
     - Explicit path to the search manifest JSON.
