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

   \min_{\boldsymbol{w}} \; \chi^2(\boldsymbol{w})
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
:class:`~starling.structure.bme.ExperimentalObservable`:

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

After reweighting, ensemble analysis methods accept ``use_bme_weights=True``
to compute weighted averages:

.. code-block:: python

   weighted_rg = ensemble.radius_of_gyration(
       return_mean=True, use_bme_weights=True
   )

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
   optimised_weights = bme.weights

   # Predict reweighted values for new calculated data
   predicted = bme.predict(calculated)

Theta Scanning
--------------

Choosing the right :math:`\theta` is critical. Use
:func:`~starling.structure.bme.theta_scan` to sweep a range and inspect the
trade-off:

.. code-block:: python

   from starling.structure.bme import theta_scan

   scan_result = theta_scan(
       observables=observables,
       calculated_values=calculated,
       theta_values=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
   )

The returned :class:`~starling.structure.bme_utils.ThetaScanResult` contains
per-theta :math:`\chi^2`, effective sample sizes, and KL divergence values so
you can select the best balance between fitting and diversity.

Interpreting Results
--------------------

:class:`~starling.structure.bme.BMEResult` provides several diagnostic helpers:

.. code-block:: python

   # Run diagnostics – warns if effective sample size is low
   result.diagnostics(warn_threshold=0.5)

   # Effective sample size (fraction of the original ensemble retained)
   n_eff = result.phi

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
