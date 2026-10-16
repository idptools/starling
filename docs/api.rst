API reference
=============

.. currentmodule:: starling

High-level functions
--------------------

.. autosummary::
   :toctree: autosummary
   :recursive:

   generate
   sequence_encoder
   load_ensemble
   set_compilation_options

Structure & Ensembles
---------------------

Coordinate reconstruction uses classical MDS initialization followed by
weighted Torch SMACOF on CPU, CUDA, or MPS.

.. autosummary::
   :toctree: autosummary
   :recursive:

   structure.ensemble
   structure.bme
   structure.bme_utils
   structure.coordinates
   structure.weighted_stress

Structure Relaxation (Mpipi-GG)
-------------------------------

:func:`starling.generate` with ``return_structures=True`` relaxes the
reconstructed structures with the Mpipi-GG force field by default, restrained
to their STARLING distance maps. Structures built later from the distance maps
(``Ensemble.trajectory`` or ``Ensemble.build_ensemble_trajectory()``) are not
relaxed; use
:func:`starling.minimizer.relax_ensemble <starling.minimizer.relax.relax_ensemble>`
to relax the structures of an ensemble you have loaded or built yourself.

.. autosummary::
   :toctree: autosummary

   minimizer
   minimizer.relax
   minimizer.forcefield
   minimizer.restraints
   minimizer.fire
   minimizer.langevin
   minimizer.parameters
   minimizer.potentials

Frontend
--------

.. autosummary::
   :toctree: autosummary
   :recursive:

   frontend.ensemble_generation
   frontend.starling_viz

Inference
---------

.. autosummary::
   :toctree: autosummary
   :recursive:

   inference
   inference.generation
   inference.constraints
   inference.model_loading
   inference.evaluate_vae

Models
------

.. autosummary::
   :toctree: autosummary
   :recursive:

   models.vae
   models.diffusion
   models.vit
   models.transformer
   models.attention
   models.blocks
   models.normalization
   models.vae_components

Samplers
--------

.. autosummary::
   :toctree: autosummary
   :recursive:

   samplers.base_sampler
   samplers.sampler_utilities
   samplers.ddpm_sampler
   samplers.ddim_sampler
   samplers.plms_sampler
   samplers.dpmpp_sampler

Data Processing
---------------

.. autosummary::
   :toctree: autosummary
   :recursive:

   data.tokenizer
   data.distributions
   data.positional_encodings
   data.schedulers
   data.data_wrangler
   data.argument_parser

Search & Indexing
-----------------

.. autosummary::
   :toctree: autosummary
   :recursive:

   search
   search.search_engine
   search.store
   search.builder
   search.search_utils

Configuration & Utilities
-------------------------

.. autosummary::
   :toctree: autosummary
   :recursive:

   configs
   utilities

Training
--------

.. autosummary::
   :toctree: autosummary
   :recursive:

   training
