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
argument, either one at a time or as a list/tuple (see below). Constraints work
with the DDIM (default), PLMS and DDPM samplers; the DPM++ sampler does not
support them.

Residue indices are 0-based (the first residue is 0) and are checked against
the sequence length when sampling starts.

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
       modelling local secondary structure. The range is ``resid_start`` to
       ``resid_end - 1`` (``resid_end`` is exclusive, like a Python slice).
   * - :class:`~starling.inference.constraints.BondConstraint`
     - Penalise deviations from ideal backbone bond lengths.
   * - :class:`~starling.inference.constraints.StericClashConstraint`
     - Discourage steric clashes by penalising distances below a threshold.
   * - :class:`~starling.inference.constraints.MultiConstraint`
     - Combine several constraints into a single guidance step.

Each individual constraint accepts ``force_constant`` (default ``2.0``) and,
except :class:`~starling.inference.constraints.StericClashConstraint`,
``tolerance`` (default ``0.0``), plus these shared keyword arguments:

``constraint_weight``
   Scalar applied to the gradient update (default ``1.0``). Guidance strength
   does not depend on the batch size: each conformation is moved by the
   gradient of its own loss, calibrated so that existing values behave as they
   did at the default batch size of 100.

``guidance_start`` / ``guidance_end``
   Normalised diffusion timestep window (``0`` is the start, ``1`` the end) in
   which the constraint is active.

``schedule``
   Strategy for time-dependent scaling. ``"cosine"`` (default) fades the
   influence in smoothly from zero at the start of sampling to full strength at
   the end; ``"bell_shaped"`` peaks about 43% of the way through sampling;
   ``"linear"`` ramps linearly from zero to full strength; pass ``None`` to
   keep a constant weight. Any other value raises a ``ValueError``.

``verbose``
   Log per-step statistics when ``True``.

Basic example
-------------

.. code-block:: python

   from starling import generate
   from starling.inference.constraints import DistanceConstraint, MultiConstraint, RgConstraint

   constraint = DistanceConstraint(
       resid1=15,
       resid2=42,
       target=35.0,
       tolerance=2.0,
       force_constant=3.0,
       guidance_start=0.2,
       guidance_end=0.8,
   )

   # Combine multiple restraints with a MultiConstraint (a plain list also
   # works and is wrapped in a MultiConstraint for you)
   constraints = MultiConstraint(
       [constraint, RgConstraint(target=25.0, force_constant=1.5)],
   )

   ensemble = generate(
       "MQDRVKRPMNAFIVWSRDQRRKMALENPRMRNSEISKQLGYQWKMLTEAEKWPFFQEAQKLQAMHREKYPNYKYRPRRKAKMLPK",
       conformations=200,
       constraint=constraints,
       return_single_ensemble=True,
   )

Constraints receive the internal encoder and latent scaling factors
automatically, so no additional setup is required.

Passing a plain list or tuple of constraints to ``generate()`` also works: it is
wrapped in a :class:`~starling.inference.constraints.MultiConstraint` with the
default ``"cosine"`` schedule. Inside a ``MultiConstraint`` the losses of all
constraints are combined into one guidance step at each diffusion step. Each
constraint keeps its own ``constraint_weight`` and
``guidance_start``/``guidance_end`` window, but the time schedule is the
``MultiConstraint``'s, set with its own ``schedule`` argument (for example
``MultiConstraint([...], schedule="bell_shaped")``). The constraints' own
``schedule`` settings are ignored, with a warning for any that differ from the
``MultiConstraint``'s.

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
