Possible Issues and Solutions
============================

This page addresses common issues you might encounter when using Starling and provides solutions.

NumPy Compilation Warning Error
------------------------------

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

