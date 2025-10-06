Working with Ensembles
=====================

The ``Ensemble`` class is a core component of STARLING that represents multiple conformations of a protein chain. This guide covers how to load, analyze, and manipulate conformational ensembles.

Loading Ensembles
----------------

Ensembles can be loaded from STARLING format files:

.. code-block:: python

    from starling.structure.ensemble import load_ensemble
    
    # Load an ensemble from a file
    ensemble = load_ensemble("example_ensemble.starling")
    
    # Optionally ignore 3D structures for faster loading
    ensemble = load_ensemble("example_ensemble.starling", ignore_structures=True)
    
    # Get basic ensemble information
    print(ensemble)  # Shows sequence length, ensemble size, and structure status
    print(len(ensemble))  # Number of conformations

Structural Analysis
------------------

Calculating Ensemble Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Ensemble`` class provides methods to compute various biophysical properties:

.. code-block:: python

    # Get radius of gyration for all conformations
    rg_values = ensemble.radius_of_gyration()
    
    # Get mean radius of gyration
    mean_rg = ensemble.radius_of_gyration(return_mean=True)
    
    # Calculate end-to-end distance
    end_to_end = ensemble.end_to_end_distance(return_mean=True)
    
    # Calculate hydrodynamic radius using different methods
    rh_nygaard = ensemble.hydrodynamic_radius(mode="nygaard", return_mean=True)
    rh_kr = ensemble.hydrodynamic_radius(mode="kr", return_mean=True)

Distance and Contact Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
-------------------------

Accessing and Generating Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STARLING can generate 3D structures from distance maps:

.. code-block:: python

    # Check if the ensemble already has 3D structures
    has_structures = ensemble.has_structures
    
    # Access the trajectory (generates 3D structures if needed)
    trajectory = ensemble.trajectory
    
    # Explicitly build structures with custom parameters
    ensemble.build_ensemble_trajectory(
        num_cpus_mds=4,       # Number of CPUs for structure generation
        num_mds_init=4,       # Number of MDS initializations
        device="cuda",        # Use GPU acceleration if available
        force_recompute=True  # Rebuild structures even if they exist
    )
    
    # Save trajectory to files
    ensemble.save_trajectory("my_structures", pdb_trajectory=True)  # Save as multi-model PDB
    ensemble.save_trajectory("my_structures")  # Save as PDB/XTC

Ensemble Reweighting with BME
---------------------------

Optimize ensemble weights to match experimental data:

.. code-block:: python

    from starling.inference.bme import ExperimentalObservable
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
--------------

Save ensembles in STARLING format:

.. code-block:: python

    # Basic save
    ensemble.save("my_ensemble")
    
    # Save with compression and reduced precision for smaller file size
    ensemble.save("my_ensemble_compressed", compress=True, reduce_precision=True)
