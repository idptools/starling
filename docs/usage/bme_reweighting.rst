BME Reweighting
===============

Bayesian Maximum Entropy (BME) reweighting adjusts the statistical weights of
conformations in a STARLING ensemble so the weighted average better reproduces
experimental observables while staying as close as possible to the prior
(uniform) distribution. This guide covers end-to-end reweighting workflows
using the :mod:`starling.structure.bme` module.

.. seealso::

   * :doc:`ensemble` – loading ensembles and computing structural properties.
   * :mod:`starling.structure.bme` – full API reference for the BME classes.
   * :mod:`starling.structure.bme_utils` – helper functions and constants.

Concepts
--------

BME works by solving the constrained optimisation problem:

.. math::

   \min_{\boldsymbol{w}} \; \tfrac{1}{2} \chi^2(\boldsymbol{w})
   + \theta \, D_{\mathrm{KL}}(\boldsymbol{w} \| \boldsymbol{w}_0)

where :math:`\chi^2` measures how well the reweighted ensemble matches
experiment, :math:`D_{\mathrm{KL}}` is the Kullback–Leibler divergence from the
prior weights :math:`\boldsymbol{w}_0`, and :math:`\theta` balances data
fidelity against ensemble diversity.

* **Low** :math:`\theta` → aggressive fitting, risk of overfitting.
* **High** :math:`\theta` → weights stay close to the prior, less data influence.

Defining Experimental Observables
---------------------------------

Wrap each measurement in an
:class:`~starling.structure.bme_utils.ExperimentalObservable`:

.. code-block:: python

   from starling.structure.bme import ExperimentalObservable

   # Equality restraint: measured Rg = 25 ± 2 Å
   rg_obs = ExperimentalObservable(
       value=25.0,
       uncertainty=2.0,
       constraint="equality",
       name="Rg",
   )

   # Upper-bound restraint: end-to-end distance ≤ 60 Å
   ete_obs = ExperimentalObservable(
       value=60.0,
       uncertainty=3.0,
       constraint="upper",
       name="End-to-end distance",
   )

   # Lower-bound restraint: Rh ≥ 15 Å
   rh_obs = ExperimentalObservable(
       value=15.0,
       uncertainty=1.5,
       constraint="lower",
       name="Rh",
   )

Supported ``constraint`` types are ``"equality"``, ``"upper"``, and
``"lower"``.

Running BME Reweighting
-----------------------

Via the Ensemble helper
~~~~~~~~~~~~~~~~~~~~~~~

The simplest path is to call
:meth:`~starling.structure.ensemble.Ensemble.reweight_bme` directly on an
``Ensemble`` object:

.. code-block:: python

   import numpy as np
   from starling import load_ensemble
   from starling.structure.bme import ExperimentalObservable

   ensemble = load_ensemble("my_ensemble.starling")

   # Compute per-conformation values for each observable
   rg_values = ensemble.radius_of_gyration()
   ete_values = ensemble.end_to_end_distance()
   calculated = np.column_stack([rg_values, ete_values])

   observables = [
       ExperimentalObservable(value=25.0, uncertainty=2.0, name="Rg"),
       ExperimentalObservable(value=55.0, uncertainty=3.0, name="Re"),
   ]

   result = ensemble.reweight_bme(
       observables=observables,
       calculated_values=calculated,
       theta=0.5,
       verbose=True,
   )

   print(f"χ² initial: {result.chi_squared_initial:.3f}")
   print(f"χ² final:   {result.chi_squared_final:.3f}")
   weights = result.weights  # one weight per conformation

If you leave ``theta=None`` (the default), :math:`\theta` is chosen
automatically by an L-curve scan. ``theta_range`` sets the values scanned: a
``(min, max)`` tuple scans ``theta_n_points`` (default 15) log-spaced values
between the two, while a list or array is used as the exact grid. The knee of
the L-curve is found with ``theta_method`` (``"perpendicular"``, the default,
or ``"curvature"``), and the scan is stored in
``result.metadata['theta_scan_result']`` (also available as
``ensemble.theta_scan_result``):

.. code-block:: python

   result = ensemble.reweight_bme(
       observables=observables,
       calculated_values=calculated,
       theta_range=(0.01, 10.0),
       theta_n_points=15,
   )
   scan = result.metadata["theta_scan_result"]
   print(f"Selected theta: {scan.optimal_theta:.3g}")

A successful fit is cached on the ensemble and reused only when
``reweight_bme`` is called again with identical inputs; any change to the
observables, calculated values, theta settings, optimizer settings or initial
weights triggers a new fit. If a fit fails, a ``RuntimeWarning`` is issued and
the result is not cached.

After reweighting, ``rij()``, ``end_to_end_distance()``,
``radius_of_gyration()``, ``local_radius_of_gyration()`` and
``distance_maps()`` accept ``use_bme_weights=True`` to compute weighted
averages:

.. code-block:: python

   weighted_rg = ensemble.radius_of_gyration(
       return_mean=True, use_bme_weights=True
   )

Other per-conformation values, such as ``hydrodynamic_radius()``, do not take
``use_bme_weights``; weight them yourself:

.. code-block:: python

   weighted_rh = np.average(ensemble.hydrodynamic_radius(), weights=result.weights)

Via the low-level BME class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control, instantiate :class:`~starling.structure.bme.BME` directly:

.. code-block:: python

   from starling.structure.bme import BME

   bme = BME(
       observables=observables,
       calculated_values=calculated,
       theta=0.5,
   )

   result = bme.fit(verbose=True)
   optimised_weights = result.weights  # also available as bme.result.weights

   # Predict reweighted values for new calculated data
   predicted = bme.predict(calculated)

Theta Scanning
--------------

Choosing the right :math:`\theta` is critical. Use
:func:`~starling.structure.bme_utils.theta_scan` to sweep a range and inspect the
trade-off:

.. code-block:: python

   from starling.structure.bme import theta_scan

   # an explicit grid of theta values...
   scan_result = theta_scan(
       observables=observables,
       calculated_values=calculated,
       theta_range=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
   )

   # ...or a (min, max) range scanned at n_points log-spaced values
   scan_result = theta_scan(
       observables=observables,
       calculated_values=calculated,
       theta_range=(0.01, 10.0),
       n_points=15,
   )

The returned :class:`~starling.structure.bme_utils.ThetaScanResult` contains
the scanned ``theta_values`` with the per-theta :math:`\chi^2`
(``chi_squared_values``), :math:`\phi` (``phi_values``, the effective fraction
of frames) and KL divergence (``kl_divergence_values``), plus the automatically
selected ``optimal_theta``, so you can select the best balance between fitting
and diversity.

Interpreting Results
--------------------

:class:`~starling.structure.bme_utils.BMEResult` provides several diagnostic helpers:

.. code-block:: python

   # Run diagnostics – warns if effective sample size is low
   result.diagnostics(warn_threshold=0.5)

   # Effective fraction of frames (0-1); multiply by the number of
   # conformations for the effective sample size
   phi = result.phi
   n_eff = phi * len(result.weights)

   # KL divergence from the prior
   kl = result.kl_divergence

A large KL divergence or very small ``phi`` suggests the reweighting had to
deviate substantially from the prior, which may indicate the ensemble is
incompatible with the data or :math:`\theta` is too low.

See Also
--------

* :doc:`ensemble` – Structural analyses on ensembles.
* :doc:`ensemble_generation` – Generating ensembles that can be reweighted.
* :doc:`constraints` – Steering sampling at generation time instead of
  post-hoc reweighting.
