Installation
===========

STARLING is available on GitHub (bleeding edge) and on PyPi (stable).

Creating an Environment
----------------------

We recommend creating a fresh conda environment for STARLING:

.. code-block:: bash

    conda create -n starling python=3.11 -y
    conda activate starling

Installation Options
------------------

Install from PyPi (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install STARLING from PyPi using pip:

.. code-block:: bash

    pip install idptools-starling

Install from GitHub (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Or you can clone and install the bleeding-edge version from GitHub:

.. code-block:: bash

    git clone git@github.com:idptools/starling.git
    cd starling
    pip install .

Verification
-----------

To verify that STARLING has installed correctly, run:

.. code-block:: bash

    starling --help