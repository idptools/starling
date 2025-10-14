Guided Sampling with Constraints
================================

STARLING exposes hooks for steering the diffusion process with physics-inspired
constraints. Use this page when you need to encode experimental restraints or
bias ensembles toward specific structural properties.

Constraint overview
-------------------

All restraints live in :mod:`starling.inference.constraints` and derive from a
common :class:`~starling.inference.constraints.Constraint` base class. They can
be passed directly to :func:`starling.generate` through the ``constraint``
argument.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Description
   * - :class:`~starling.inference.constraints.DistanceConstraint`
     - Maintain a target distance (Å) between two residues with optional
       tolerance and force constant control.
   * - :class:`~starling.inference.constraints.RgConstraint`
     - Bias ensembles toward a target radius of gyration.
   * - :class:`~starling.inference.constraints.ReConstraint`
     - Enforce an end-to-end distance between the first and last residues.
   * - :class:`~starling.inference.constraints.HelicityConstraint`
     - Encourage helix-like distance patterns within a residue range, useful for
       modelling local secondary structure.
   * - :class:`~starling.inference.constraints.BondConstraint`
     - Penalise deviations from ideal backbone bond lengths.
   * - :class:`~starling.inference.constraints.StericClashConstraint`
     - Discourage steric clashes by penalising distances below a threshold.

Any constraint accepts shared keyword arguments:

``constraint_weight``
   Scalar applied to the gradient update (default ``1.0``).

``guidance_start`` / ``guidance_end``
   Normalised diffusion timestep window (``0`` is the start, ``1`` the end) in
   which the constraint is active.

``schedule``
   Strategy for time-dependent scaling. ``"cosine"`` (default) fades the
   influence in smoothly; ``"bell_shaped"`` ramps up mid-way; pass ``None`` to
   keep a constant weight.

``verbose``
   Log per-step statistics when ``True``.

Basic example
-------------

.. code-block:: python

   from starling import generate
   from starling.inference.constraints import DistanceConstraint, RgConstraint

   constraint = DistanceConstraint(
       resid1=15,
       resid2=42,
       target=35.0,
       tolerance=2.0,
       force_constant=3.0,
       guidance_start=0.2,
       guidance_end=0.8,
   )

   # Combine multiple restraints by wrapping them in a list
   constraints = [
       constraint,
       RgConstraint(target=25.0, force_constant=1.5, schedule="bell_shaped"),
   ]

   ensemble = generate(
       "MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK",
       conformations=200,
       constraint=constraints,
       return_single_ensemble=True,
   )

Constraints receive the internal encoder and latent scaling factors
automatically, so no additional setup is required. If multiple constraints are
provided they are applied sequentially at each diffusion step.

Tuning gradients
----------------

The base class exposes helpers that make it easier to keep guidance stable:

* ``constraint_weight`` – increase to tighten the restraint, decrease to allow
  more variation.
* ``get_time_scale`` – schedules with strong mid-to-late emphasis tend to
  stabilise final structures without derailing early denoising.
* ``get_adaptive_clip_threshold`` – advanced users can subclass and override to
  change gradient clipping behaviour.

See Also
--------

* :doc:`ensemble_generation` – Complete sampling options and batching guidance
* :doc:`ensemble` – Analyzing constrained ensembles once they are saved
* :doc:`performance` – Accelerating generation with PyTorch compilation
* :mod:`starling.inference.constraints` – Full API reference for constraint classes
