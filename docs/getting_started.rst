Getting Started
===============

Welcome to STARLING! This guide gives you the fastest path from a fresh
installation to your first ensemble and highlights the primary documentation
entry points for deeper dives.

What you will do
----------------

* Configure a Python environment and install STARLING
* Generate an ensemble from the command line
* Reproduce the same workflow in Python and inspect key outputs


Install STARLING
----------------

Follow the step-by-step instructions in :doc:`usage/installation` to
create a new conda environment or install directly with ``pip``. After
installation, confirm the CLI is reachable:

.. code-block:: bash

	 starling --help

If the command returns the usage banner, you are ready to generate ensembles.

Quickstart: command line
------------------------

The ``starling`` executable is the fastest way to create conformational
ensembles.

.. code-block:: bash

	 starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK \
			 -c 100 \
			 --outname example_ensemble \
			 --output_directory results \
			 -r

Key flags:

* ``-c``/``--conformations`` – number of structures to sample (default 400)
* ``--ionic_strength`` – pick from 20, 150, or 300 mM conditions (default 150)
* ``-r``/``--return_structures`` – Flag which, if provided, means STARLING returns 3D coordinates alongside distance maps as a .pdb file (single topology file) and a .xtc file (compressed trajectory format).

The CLI writes ``.starling`` archives and (if ``r-`` is provided PDB/XTC files into the
selected output directory. A more detailed walkthrough lives in
:doc:`usage/cli`.

Quickstart: Python API
----------------------

The Python interface mirrors the CLI and exposes richer programmatic control.

.. code-block:: python

	 from starling import generate, load_ensemble

	 sequence = "MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK"
	 ensemble = generate(sequence, conformations=100, ionic_strength=150, return_single_ensemble=True)

	 mean_rg = ensemble.radius_of_gyration(return_mean=True)
	 print(f"Mean Rg: {mean_rg:.2f} Å")

	 ensemble.save("example_ensemble")
	 reloaded = load_ensemble("example_ensemble.starling")

The :doc:`usage/ensemble_generation` page explains batching, GPU control,
and performance tuning. :doc:`usage/ensemble` covers the analysis helpers
available on the ``Ensemble`` class.

Where to go next
----------------

* :doc:`usage/constraints` – steer sampling with experimental restraints and enable PyTorch compilation for repeat jobs.	
* :doc:`usage/sequence_encoder` – extract ensemble-aware sequence embeddings for downstream analysis.	
* :doc:`usage/search` – index large databases and retrieve related sequences with FAISS-powered search.
* :doc:`usage/possible_issues` – troubleshoot installation or runtime hiccups.
	
	
How to cite
----------------

If you find STARLING useful, please consider citing the following:

**Accurate predictions of conformational ensembles of disordered proteins with STARLING** Novak, B., Lotthammer, J. M., Emenecker, R. J. & Holehouse, A. S. bioRxiv (2025). doi:10.1101/2025.02.14.638373 (*Main STARLING preprint (under review)*)

**Physics-driven coarse-grained model for biomolecular phase separation with near-quantitative accuracy** Joseph, J. A., Reinhardt, A., Aguirre, A., Chew, P. Y., Russell, K. O., Espinosa, J. R., Garaizar, A. & Collepardo-Guevara, R. Nat. Comput. Sci. 1, 732–743 (2021) (*Coarse-grained model from which STARLING was trained - PLEASE cite this alongside STARLING*)


  

  

	
