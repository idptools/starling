Ensemble Generation
====================

STARLING provides powerful tools for generating conformational ensembles of intrinsically disordered proteins (IDPs). This guide covers everything from basic usage to advanced options.

.. seealso::

     * :doc:`usage/cli` for command-line generation and conversion helpers.
     * :doc:`usage/constraints` to steer sampling with experimental restraints and
         enable Torch compilation.

Getting Started
----------------

When to Use STARLING
~~~~~~~~~~~~~~~~~~~~

STARLING is designed for:

* Generating structural ensembles of intrinsically disordered proteins (IDPs)
* Predicting conformational properties of disordered regions
* Exploring the conformational space of proteins with significant disorder

Basic Ensemble Generation
--------------------------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to generate an ensemble is using the command-line interface:

.. code-block:: bash

    starling MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK -c 100 --outname example_ensemble

Common Parameters:

* ``-c``: Number of conformations to generate (default: 200)
* ``--outname``: Base name for output files
* ``--ionic_strength``: Ionic strength in mM (default: 150)
* ``--steps``: Number of diffusion steps (default: 30)
* ``--sampler``: Sampling algorithm to use (default: "ddim")

For a complete list of options, run:

.. code-block:: bash

    starling --help

Python API
~~~~~~~~~~

You can also generate ensembles programmatically:

.. code-block:: python

    from starling import generate
    
    # Basic usage with a single sequence
    sequence = "MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK"
    ensemble = generate(sequence, conformations=100)
    ensemble.save("example_ensemble.starling")
    
    # Process multiple sequences at once
    sequences = [
        "GSGSGSGSGSGS",
        "ACDEFGHIKLMNPQRSTVWY"
    ]
    ensembles = generate(sequences, conformations=50)
    
    # Access individual ensembles from the returned dictionary
    for name, ens in ensembles.items():
        print(f"Ensemble {name}: {len(ens)} conformations")
        ens.save(f"{name}_ensemble.starling")

Working with Multiple Input Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STARLING accepts various input formats:

.. code-block:: python

    # From a dictionary with custom names
    sequence_dict = {
        "protein_A": "GSGSGSGSGSGS",
        "protein_B": "ACDEFGHIKLMNPQRSTVWY"
    }
    ensembles = generate(sequence_dict, conformations=50)
    
    # From a FASTA file
    ensembles = generate("path/to/sequences.fasta", conformations=50)
    
    # From a TSV file (name, sequence format)
    ensembles = generate("path/to/sequences.tsv", conformations=50)

Environment Control
--------------------

Ionic Strength Control
~~~~~~~~~~~~~~~~~~~~~~

STARLING is trained on ensembles generated at three different ionic strengths (20mM, 150mM, 300mM).
You can adjust the ionic strength to model different environments:

Command Line Interface:

.. code-block:: bash

    starling SEQUENCE -c 100 --ionic_strength 150 --outname low_ionic_strength_ensemble

Python API:

.. code-block:: python

    # Generate at physiological ionic strength (150mM)
    ensemble = generate(sequence, conformations=100, ionic_strength=150)

    # Generate at low ionic strength (20mM)
    ensemble = generate(sequence, conformations=100, ionic_strength=20)

    # Generate at high ionic strength (300mM)
    ensemble = generate(sequence, conformations=100, ionic_strength=300)
    
    # Calculate and compare properties at different ionic strengths
    rg_150 = ensemble.radius_of_gyration(return_mean=True)
    print(f"Mean Rg at 150mM: {rg_150:.2f} Å")

Controlling Ensemble Size
~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance quality and performance by adjusting ensemble size:

.. code-block:: python

    # Small ensemble for quick analysis
    small_ensemble = generate(sequence, conformations=20)
    
    # Medium ensemble for standard analysis
    medium_ensemble = generate(sequence, conformations=100)
    
    # Large ensemble for detailed statistical analysis
    large_ensemble = generate(sequence, conformations=500)

Performance Tuning
------------------

Batch and Device Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance throughput and memory use by adjusting hardware-related options:

.. code-block:: python

    ensemble = generate(
        sequences,
        conformations=100,
        device="cuda:0",         # Pin generation to a specific accelerator
        batch_size=64,           # Increase to improve GPU utilisation
        num_cpus_mds=8,          # Allocate more CPUs for 3D reconstruction
        show_progress_bar=True,
        verbose=False,
    )

Remember that ``batch_size`` cannot exceed ``conformations`` and larger values
increase peak memory usage. For CPU-only runs, reduce ``batch_size`` or switch
``device`` to ``"cpu"`` for predictable performance.

Sampler Selection
~~~~~~~~~~~~~~~~~

STARLING supports multiple diffusion samplers so you can trade accuracy for
latency:

.. code-block:: python

    # Deterministic DDIM sampling – faster, deterministic trajectories
    ddim_ensemble = generate(sequence, conformations=100, sampler="ddim", steps=20)

    # Stochastic DDPM sampling – higher fidelity at the cost of runtime
    ddpm_ensemble = generate(sequence, conformations=100, sampler="ddpm", steps=50)

Model Compilation
~~~~~~~~~~~~~~~~~

For repeated predictions, compile the underlying PyTorch models once per
process:

.. code-block:: python

    import starling

    starling.set_compilation_options(enabled=True, mode="reduce-overhead")
    ensemble = generate(sequence, conformations=100)

The first invocation warms up kernels; subsequent calls reuse compiled graphs
and can reduce runtime by ~40% on supported GPUs. See :doc:`usage/constraints`
for advanced compilation options.

Guided Sampling
---------------

Constraint-driven Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

STARLING can enforce experimental restraints during diffusion. Pass any
constraint (or list of constraints) from
:mod:`starling.inference.constraints` to the ``constraint`` argument:

.. code-block:: python

    from starling.inference.constraints import DistanceConstraint

    constraint = DistanceConstraint(
        resid1=10,
        resid2=200,
        target=50.0,
        tolerance=2.0,
        force_constant=2.5,
    )
    ensemble = generate(sequence, conformations=200, constraint=constraint)

Combine multiple constraints or tune ``force_constant``/``guidance`` settings to
steer sampling toward experimental observables. Visit :doc:`usage/constraints`
for a catalogue of available restraints and tuning advice.

Saving and Loading Ensembles
------------------------------

Saving Ensembles
~~~~~~~~~~~~~~~~~

Save ensembles in STARLING format for later use:

.. code-block:: python

    # Save with default options
    ensemble.save("my_ensemble")
    
    # Save with compression for smaller file size
    ensemble.save("my_ensemble_compressed", compress=True)
    
    # Auto-save during generation
    ensemble = generate(
        sequence, 
        conformations=100, 
        output_directory="results"
    )

Loading Ensembles
~~~~~~~~~~~~~~~~~

Load previously generated ensembles:

.. code-block:: python

    from starling.structure.ensemble import load_ensemble
    
    # Load an ensemble
    ensemble = load_ensemble("my_ensemble.starling")
    
    # Load without 3D structures for faster loading
    ensemble = load_ensemble("my_ensemble.starling", ignore_structures=True)
    
    print(f"Loaded ensemble with {len(ensemble)} conformations")

Output Files and Conversion
---------------------------

STARLING generates output in its native format, which can be converted to common molecular formats:

.. code-block:: bash

    # Convert to PDB trajectory
    starling2pdb example_ensemble.starling
    
    # Convert to XTC/PDB for molecular dynamics software
    starling2xtc example_ensemble.starling

From Python:

.. code-block:: python

    # Save directly to PDB trajectory format
    ensemble.save_trajectory("my_structures", pdb_trajectory=True)
    
    # Save as PDB/XTC combination
    ensemble.save_trajectory("my_structures")

Tips and Troubleshooting
-------------------------

Common Issues
~~~~~~~~~~~~~~

* **Memory errors**: Reduce batch_size or conformations if you encounter CUDA out of memory errors
* **Long sequences**: STARLING has a maximum sequence length limit; consider dividing long proteins into domains
* **Performance**: Use GPU acceleration when available for significantly faster generation
* **Invalid amino acids**: Only standard 20 amino acids are supported; other characters will be rejected