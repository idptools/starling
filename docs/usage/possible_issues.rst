Possible Issues and Solutions
=============================

This page addresses common issues you might encounter when using Starling and provides solutions.

NumPy Compilation Warning Error
-------------------------------

**Issue:**

You encounter the following error message:

.. code-block:: text

    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.2.3 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.

**Solution:**

This issue commonly appears on Intel Macs because PyTorch stopped supporting Intel Macs after torch=2.2.2.

If you're **not** on an Intel Mac, upgrade PyTorch:

.. code-block:: bash

    # recommended, but ANY version above 2.2.2 should work
    pip install torch==2.6.0

If you're on an Intel Mac and torch > 2.2.2 is not available, downgrade NumPy:

.. code-block:: bash

    pip install numpy==1.26.1

PyTorch / CUDA Version Issues
----------------------------

**Issue:**

If you're using an older CUDA version, PyTorch might install without proper CUDA support, causing STARLING to segfault.

**Solution:**

Install PyTorch with the correct CUDA version for your system. For example, with CUDA 12.1:

.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cu121

To check your current CUDA version:

.. code-block:: bash

    nvidia-smi

This will display information about your GPU, NVIDIA driver version, and CUDA version at the top.

For more information, refer to the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

FAISS Installation Problems
---------------------------

**Issue:**

Import errors or missing GPU support when running ``starling-search`` or the
Python search API.

**Solution:**

* **CPU-only environments**: The default installation includes ``faiss-cpu`` which 
  works on all platforms. If you compiled FAISS manually, ensure ``FAISS_PATH`` 
  does not shadow the packaged version.

* **GPU acceleration**: FAISS-GPU requires conda installation as there is no pip 
  package available. Follow the detailed GPU installation instructions in 
  :doc:`installation` which includes:

  1. Installing PyTorch with CUDA support via conda
  2. Installing FAISS-GPU matching your CUDA version via conda
  3. Installing STARLING with ``--no-deps`` to avoid conflicts

* **Verification**: After installation, verify GPU support:

  .. code-block:: bash

       python -c "import faiss; print(f'FAISS GPUs: {faiss.get_num_gpus()}')"

  If this returns ``0``, check that:
  
  - Your CUDA version matches between PyTorch and FAISS-GPU
  - NVIDIA drivers are properly installed (``nvidia-smi`` works)
  - PyTorch can access the GPU (``torch.cuda.is_available()`` returns ``True``)

**Common Issues:**

* **CUDA version mismatch**: Ensure ``cuda-version`` in the FAISS-GPU install 
  command matches your PyTorch CUDA version
* **pip vs conda conflicts**: Always use ``--no-deps`` when pip installing STARLING 
  after conda-installing GPU packages
* **Multiple FAISS versions**: Run ``pip list | grep faiss`` and ``conda list faiss`` 
  to check for conflicting installations
