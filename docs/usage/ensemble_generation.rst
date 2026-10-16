Ensemble Generation
====================

STARLING provides powerful tools for generating conformational ensembles of intrinsically disordered proteins (IDPs). This guide covers everything from basic usage to advanced options.

.. seealso::

     * :doc:`cli` for command-line generation and conversion helpers.
     * :doc:`constraints` to steer sampling with experimental restraints.
     * :doc:`performance` to enable Torch compilation.

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

* ``-c``: Number of conformations to generate (default: 400)
* ``--outname``: Base name for output files
* ``--ionic_strength``: Ionic strength in mM (default: 150)
* ``-s``/``--steps``: Number of diffusion steps (default: 30)
* ``--sampler``: Sampling algorithm to use (default: "ddim")
* ``-r``: Also write 3D structures (PDB + XTC), relaxed with Mpipi-GG unless
  ``--no-relax`` is given (see :ref:`relaxing-structures`; ``--relax fast``
  selects the faster relaxation mode)

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
    ensemble = generate(sequence, conformations=100, return_single_ensemble=True)
    ensemble.save("example_ensemble.starling")
    
    # Process multiple sequences at once; generate() returns a dict of
    # {name: Ensemble} unless return_single_ensemble=True
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Tuples of sequences and ``pathlib.Path`` file paths work too. Non-canonical
residues raise a ``ValueError``, except in FASTA files, where they are converted
(X→G, B→N, U→C, Z→Q; ``*``, ``-`` and spaces removed) with a warning naming each
sequence that changed. Empty sequences are rejected, blank lines in TSV files
are skipped, and sequences longer than 380 residues are skipped with a warning
(if none are left, ``generate()`` raises a ``ValueError`` before loading the
models).

Environment Control
--------------------

Ionic Strength Control
~~~~~~~~~~~~~~~~~~~~~~

STARLING is trained on ensembles generated at three different ionic strengths (20mM, 150mM, 300mM).
You can adjust the ionic strength (any finite value >= 0 mM is accepted) to model different environments:

Command Line Interface:

.. code-block:: bash

    starling SEQUENCE -c 100 --ionic_strength 20 --outname low_ionic_strength_ensemble

Python API:

.. code-block:: python

    # Generate at physiological ionic strength (150mM)
    ensemble = generate(sequence, conformations=100, ionic_strength=150, return_single_ensemble=True)

    # Generate at low ionic strength (20mM)
    ensemble = generate(sequence, conformations=100, ionic_strength=20, return_single_ensemble=True)

    # Generate at high ionic strength (300mM)
    ensemble = generate(sequence, conformations=100, ionic_strength=300, return_single_ensemble=True)
    
    # Calculate and compare properties at different ionic strengths
    rg_150 = ensemble.radius_of_gyration(return_mean=True)
    print(f"Mean Rg at 150mM: {rg_150:.2f} Å")

Controlling Ensemble Size
~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance quality and performance by adjusting ensemble size:

.. code-block:: python

    # Small ensemble for quick analysis
    small_ensemble = generate(sequence, conformations=20, return_single_ensemble=True)
    
    # Medium ensemble for standard analysis
    medium_ensemble = generate(sequence, conformations=100, return_single_ensemble=True)
    
    # Large ensemble for detailed statistical analysis
    large_ensemble = generate(sequence, conformations=500, return_single_ensemble=True)

Performance Tuning
------------------

Batch and Device Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance throughput and memory use by adjusting hardware-related options:

.. code-block:: python

    ensembles = generate(
        sequences,
        conformations=100,
        device="cuda:0",         # Pin generation to a specific accelerator
        batch_size=64,           # Increase to improve GPU utilisation
        show_progress_bar=True,
        verbose=False,
    )

``batch_size`` is capped at ``conformations``, and larger values increase peak
memory usage. For CPU-only runs, reduce ``batch_size`` or switch
``device`` to ``"cpu"`` for predictable performance.

Sampler Selection
~~~~~~~~~~~~~~~~~

STARLING supports multiple diffusion samplers so you can trade accuracy for
latency:

.. code-block:: python

    # Deterministic DDIM sampling – faster, deterministic trajectories
    ddim_ensemble = generate(sequence, conformations=100, sampler="ddim", steps=20, return_single_ensemble=True)

    # Pseudo linear multi-step (PLMS) sampling
    plms_ensemble = generate(sequence, conformations=100, sampler="plms", return_single_ensemble=True)

    # Second-order DPM-Solver++: fewer model evaluations
    dpmpp_ensemble = generate(sequence, conformations=100, sampler="dpmpp", steps=12, return_single_ensemble=True)

    # Stochastic DDPM reference: always uses the full training schedule
    ddpm_ensemble = generate(sequence, conformations=100, sampler="ddpm", return_single_ensemble=True)

DDIM-30 remains the default. DPM++ uses the existing model weights and decoder;
it does not support constraints (DDIM, PLMS and DDPM do). DDPM ignores
``steps`` and always uses all 1000 training timesteps. The deterministic
solvers still draw random initial noise.

PLMS needs at least 20 steps: ``generate(sampler="plms", steps=...)`` with fewer
raises a ``ValueError`` before any models are loaded (use DDIM or DPM++ for
fewer steps). DDIM issues a ``UserWarning`` below 10 steps, because such large
deterministic steps pull samples towards their mean and narrow the ensemble.

For reproducible ensembles, pass ``seed`` (``--seed`` on the command line) and
keep the device, batch size and other settings the same. A seeded call leaves
PyTorch's global random state untouched. Without a seed, sampling draws from
PyTorch's global generator, so seeding PyTorch before the call also works:

.. code-block:: python

    ensemble = generate(sequence, conformations=100, seed=42, return_single_ensemble=True)

    # or, equivalently for reproducibility
    import torch

    torch.manual_seed(42)
    ensemble = generate(sequence, conformations=100, return_single_ensemble=True)

Either way, relaxation draws its own random numbers, so changing ``relax`` or
``return_structures`` never changes the distance maps. SMACOF
reconstruction always runs at full float32 precision, whatever
``torch.set_float32_matmul_precision()`` is set to, so structures rebuilt from
saved distance maps are the same in any process.

DDIM preserves its legacy strided schedule: requesting 30 steps performs 31
model evaluations. Unsafe near-zero-SNR timesteps are capped and duplicates
removed. With few steps, where the strided grid would start well short of
pure noise (for example 10 steps, which would start at timestep 901), the
DDIM grid is instead stretched to start at the noisiest safe timestep; the
default 30 steps (and 100 steps) are unchanged. DPM++ uses rounded log-SNR
spacing; it warns if integer timestep collisions reduce the requested number
of evaluations.

A comparative validation runner for the samplers
(``devtools/scripts/validate_samplers.py``, which compares them against cached
DDPM reference ensembles) is not yet in the repository.

Model Compilation
~~~~~~~~~~~~~~~~~

For repeated predictions, compile the underlying PyTorch models once per
process:

.. code-block:: python

    import starling

    starling.set_compilation_options(enabled=True, mode="reduce-overhead")
    ensemble = generate(sequence, conformations=100)

The first invocation warms up kernels; subsequent calls reuse compiled graphs
and can reduce runtime by ~40% on supported GPUs. See :doc:`performance`
for advanced compilation options.

Guided Sampling
---------------

Constraint-driven Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

STARLING can enforce experimental restraints during diffusion. Pass any
constraint (or list of constraints, which is combined into a
:class:`~starling.inference.constraints.MultiConstraint`) from
:mod:`starling.inference.constraints` to the ``constraint`` argument:

.. code-block:: python

    from starling.inference.constraints import DistanceConstraint

    # residue indices are 0-based and must lie within the sequence
    constraint = DistanceConstraint(
        resid1=10,
        resid2=60,
        target=50.0,
        tolerance=2.0,
        force_constant=2.5,
    )
    ensemble = generate(sequence, conformations=200, constraint=constraint, return_single_ensemble=True)

Combine multiple constraints or tune ``force_constant``/``guidance`` settings to
steer sampling toward experimental observables. Constraints are not supported
by the ``dpmpp`` sampler. Visit :doc:`constraints`
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
    
    # Auto-save during generation; "results" must already exist, and each
    # ensemble is written as results/<name>.starling (here sequence_1.starling)
    ensembles = generate(
        sequence, 
        conformations=100, 
        output_directory="results"
    )

:meth:`~starling.structure.ensemble.Ensemble.save` treats anything after the
last ``.`` in its argument as an extension and replaces it, so include
``.starling`` yourself when a name contains dots (for example
``ensemble.save("NP_000537.3.starling")``). ``generate()`` does this for you
when it saves to ``output_directory``. Because sequence names become file
names, ``generate()`` rejects names containing ``/`` when ``output_directory``
is set.

Loading Ensembles
~~~~~~~~~~~~~~~~~

Load previously generated ensembles:

.. code-block:: python

    from starling.structure.ensemble import load_ensemble
    
    # Load an ensemble
    ensemble = load_ensemble("my_ensemble.starling")
    
    # Discard any stored 3D structures (the whole file is still read, so this
    # does not make loading faster)
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

    # Drop physically impossible reconstructed frames before writing
    starling2xtc example_ensemble.starling --remove-errors

Both ``starling2pdb`` and ``starling2xtc`` accept ``--remove-errors``, which
scans the reconstructed trajectory and discards any frames with physically
impossible inter-residue distances before the trajectory is written. If the
archive holds no structures, these commands reconstruct them from the distance
maps without relaxing them. See :doc:`cli` for details.

From Python:

.. code-block:: python

    # Save directly to PDB trajectory format
    ensemble.save_trajectory("my_structures", pdb_trajectory=True)
    
    # Save as PDB/XTC combination
    ensemble.save_trajectory("my_structures")

.. _relaxing-structures:

Relaxing structures
-------------------

STARLING predicts distance maps; 3D structures are reconstructed from them by
classical MDS followed by weighted SMACOF. This gets the global shape right but
leaves bonds compressed and beads too close together. When
``return_structures=True``, :func:`starling.generate` therefore relaxes each
structure by default (``relax="standard"``) with the Mpipi-GG force field while
restraining it to its STARLING distance map: a FIRE energy minimization
followed by 250 steps of Langevin thermalization at 300 K. This gives bond,
angle and dihedral distributions close to those of Mpipi-GG simulations while
leaving global dimensions essentially unchanged.

.. code-block:: python

    # relaxed structures (the default, relax="standard")
    ensemble = generate(sequence, return_structures=True, return_single_ensemble=True)

    # relaxed structures after a faster MDS step
    fast = generate(sequence, return_structures=True, relax="fast", return_single_ensemble=True)

    # raw MDS/SMACOF structures
    raw = generate(sequence, return_structures=True, relax=False, return_single_ensemble=True)

    # on CUDA, fuse the thermalization forces for faster relaxation of large jobs
    ensemble = generate(sequence, return_structures=True, compile_forces=True, device="cuda", return_single_ensemble=True)

There are two relaxation modes, which relax each structure in exactly the
same way and differ only in the reconstruction before it.
``relax="standard"`` (the default; ``True`` means the same) reconstructs each
structure as STARLING always has: the classical-MDS starting point from a full
eigendecomposition, then up to 300 SMACOF iterations. ``relax="fast"``
computes only the three eigenvectors the starting point needs and runs 100
SMACOF iterations, leaving relaxation to finish the fit. SMACOF does most of
its work in its first few dozen iterations, and relaxation refits every
structure to its distance map anyway, so this makes the reconstruction about
4x faster for long sequences without changing the relaxed ensembles: across
20-380 residues and compact to expanded ensembles they match the standard
ones in fit to the distance maps, radius of gyration, bond lengths and closest
contacts. Individual conformations do differ, because each lands on a
different structure that fits its map about equally well. Structures returned
without relaxation always use the standard reconstruction.

``relax`` has no effect without ``return_structures=True``. With
``remove_errors=True``, relaxation runs before the 3D error screen, so
conformations broken by MDS are repaired rather than discarded.

.. _relaxing-existing-ensembles:

Relaxing an existing ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structures built from the distance maps after generation, by
:attr:`~starling.structure.ensemble.Ensemble.trajectory`,
:meth:`~starling.structure.ensemble.Ensemble.build_ensemble_trajectory`,
``starling2xtc`` or ``starling2pdb``, are **not** relaxed. Use
:func:`starling.minimizer.relax_ensemble <starling.minimizer.relax.relax_ensemble>` to relax an ensemble you have
already generated or loaded. It returns a new ensemble with relaxed structures
(the original is left untouched) plus a ``RelaxationResult`` with diagnostics:

.. code-block:: python

    from starling import load_ensemble
    from starling.minimizer import relax_ensemble

    ensemble = load_ensemble("my_ensemble.starling")
    relaxed, relaxation_result = relax_ensemble(ensemble)
    relaxed.save("my_ensemble_relaxed")
    relaxed.save_trajectory("my_ensemble_relaxed")

Tips and Troubleshooting
-------------------------

Common Issues
~~~~~~~~~~~~~~

* **Memory errors**: Reduce batch_size or conformations if you encounter CUDA out of memory errors. The VAE decodes ``VAE_DECODE_CHUNK_SIZE`` (default 16) conformations at a time; lowering it in ``~/.starling_weights/configs.py`` also helps on small GPUs, especially with constraints
* **Long sequences**: STARLING has a maximum sequence length of 380 residues (longer sequences are skipped with a warning); consider dividing long proteins into domains
* **Performance**: Use GPU acceleration when available for significantly faster generation
* **Invalid amino acids**: Only standard 20 amino acids are supported; other characters raise a ``ValueError``, except in FASTA files, where non-canonical residues are converted with a warning
