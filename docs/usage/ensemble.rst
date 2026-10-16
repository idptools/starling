Working with Ensembles
======================

The ``Ensemble`` class is a core component of STARLING that represents multiple conformations of a protein chain. This guide covers how to load, analyze, and manipulate conformational ensembles.

.. seealso::

     * :doc:`constraints` for generating ensembles that satisfy
         experimental restraints.
     * :doc:`ensemble_generation` for details on controlling the diffusion
         sampler.

Loading Ensembles
-----------------

Ensembles can be loaded from STARLING format files:

.. code-block:: python

    from starling.structure.ensemble import load_ensemble
    
    # Load an ensemble from a file
    ensemble = load_ensemble("example_ensemble.starling")
    
    # Optionally discard any stored 3D structures (the whole file is still
    # read, so this does not make loading faster)
    ensemble = load_ensemble("example_ensemble.starling", ignore_structures=True)
    
    # Get basic ensemble information
    print(ensemble)  # Shows sequence length, ensemble size, and structure status
    print(len(ensemble))  # Number of conformations

Structural Analysis
-------------------

Calculating Ensemble Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Ensemble`` class provides methods to compute various biophysical properties:

.. code-block:: python

    # Get radius of gyration for all conformations
    rg_values = ensemble.radius_of_gyration()
    
    # Get mean radius of gyration
    mean_rg = ensemble.radius_of_gyration(return_mean=True)
    
    # Calculate end-to-end distance
    end_to_end = ensemble.end_to_end_distance(return_mean=True)
    
    # Local radius of gyration of residues 10-29 (0-based; end is exclusive,
    # like a Python slice, and must satisfy 0 <= start < end <= sequence length)
    local_rg = ensemble.local_radius_of_gyration(10, 30, return_mean=True)
    
    # Calculate hydrodynamic radius using different methods
    rh_nygaard = ensemble.hydrodynamic_radius(mode="nygaard", return_mean=True)
    rh_kr = ensemble.hydrodynamic_radius(mode="kr", return_mean=True)

``mode="kr"`` uses the Kirkwood-Riseman expression
:math:`1/R_h = (1/N^2) \sum_{i \neq j} 1/r_{ij}` (as in SOURSOP 2.0.6 and
later). The Nygaard equation (the default) is parameterised for chains much
longer than a few residues; for very short chains, where its Rg/Rh ratio is not
positive, it raises a ``ValueError`` and ``mode="kr"`` should be used instead.

If you construct an :class:`~starling.structure.ensemble.Ensemble` yourself
from distance maps, the maps must be full symmetric matrices with a zero
diagonal; use :func:`starling.utilities.symmetrize_distance_maps` on maps that
hold only one triangle.

Distance and Contact Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access distance maps and contact information:

.. code-block:: python

    # Get distance between specific residues (zero-indexed)
    distances = ensemble.rij(0, 10)  # Distance between first and 11th residue
    mean_distance = ensemble.rij(0, 10, return_mean=True)
    
    # Get distance maps for all conformations
    distance_maps = ensemble.distance_maps()
    
    # Get mean distance map
    mean_distance_map = ensemble.distance_maps(return_mean=True)
    
    # Calculate contact maps (residues within 11Å)
    contact_maps = ensemble.contact_map()
    
    # Get mean contact frequency
    mean_contacts = ensemble.contact_map(return_mean=True)

Working with 3D Structures
--------------------------

Accessing and Generating Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STARLING can generate 3D structures from distance maps:

.. code-block:: python

    # Check if the ensemble already has 3D structures
    has_structures = ensemble.has_structures
    
    # Access the trajectory (builds 3D structures from the distance maps if
    # the ensemble has none; these are not relaxed)
    trajectory = ensemble.trajectory
    
    # Explicitly build structures with custom parameters
    ensemble.build_ensemble_trajectory(
        batch_size=100,       # Number of maps reconstructed per batch
        device="cuda",        # Use GPU acceleration if available
        force_recompute=True  # Rebuild structures even if they exist
    )
    
    # Save trajectory to files
    ensemble.save_trajectory("my_structures", pdb_trajectory=True)  # Save as multi-model PDB
    ensemble.save_trajectory("my_structures")  # Save as PDB/XTC

Structures built by ``trajectory`` or ``build_ensemble_trajectory()`` come
straight from MDS/SMACOF reconstruction and are not relaxed. Structures made by
:func:`starling.generate` with ``return_structures=True`` are relaxed with the
Mpipi-GG force field by default; to relax the structures of an ensemble you
have loaded, use :func:`starling.minimizer.relax_ensemble <starling.minimizer.relax.relax_ensemble>` (see
:ref:`relaxing-existing-ensembles`).

Checking for errors
~~~~~~~~~~~~~~~~~~~

:meth:`~starling.structure.ensemble.Ensemble.check_for_errors` finds
conformations with physically impossible inter-residue distances:

.. code-block:: python

    bad_frames = ensemble.check_for_errors()

    # Remove them in place. Any attached structures are sliced to match, so
    # relaxed or user-supplied coordinates are kept; pass
    # rebuild_trajectory=True to rebuild (unrelaxed) structures from the
    # remaining distance maps instead.
    ensemble.check_for_errors(remove_errors=True)

Ensemble Reweighting with BME
-----------------------------

Optimize ensemble weights to match experimental data:

.. code-block:: python

    from starling.structure.bme import ExperimentalObservable
    import numpy as np
    
    # Define experimental observables
    obs1 = ExperimentalObservable(value=25.0, uncertainty=2.0, 
                                 constraint="lower", name="Rg")
    obs2 = ExperimentalObservable(value=30.0, uncertainty=3.0, 
                                 constraint="upper", name="End-to-end distance")
    
    # Calculate ensemble values for these observables
    rg_values = ensemble.radius_of_gyration()
    ete_values = ensemble.end_to_end_distance()
    calculated = np.column_stack([rg_values, ete_values])
    
    # Perform BME reweighting
    result = ensemble.reweight_bme(
        observables=[obs1, obs2],
        calculated_values=calculated,
        theta=0.5  # Balance between data fitting and ensemble diversity
    )
    
    # Use BME-reweighted values in calculations
    weighted_rg = ensemble.radius_of_gyration(use_bme_weights=True, return_mean=True)

Saving Ensembles
----------------

Save ensembles in STARLING format:

.. code-block:: python

    # Basic save
    ensemble.save("my_ensemble")
    
    # Save with compression and reduced precision for smaller file size
    ensemble.save("my_ensemble_compressed", compress=True, reduce_precision=True)
