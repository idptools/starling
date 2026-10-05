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
   search.similarity_search
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
