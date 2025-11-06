"""
Bayesian Maximum Entropy (BME) reweighting for ensemble refinement.

This module provides tools for reweighting molecular ensembles using experimental
observables through the Bayesian Maximum Entropy framework.

The BME method optimally reweights ensemble conformations to match experimental
data while minimizing the bias introduced (measured by relative entropy). This
provides a principled way to integrate experimental constraints into molecular
simulation ensembles.

Key Classes
-----------
ExperimentalObservable : dataclass
    Container for experimental data with value, uncertainty, and constraint type.

BME : class
    Main BME optimizer that performs the reweighting optimization.

BMEResult : dataclass
    Container for optimization results including weights, chi-squared, and metadata.

Example Usage
-------------

Integration with Ensemble
--------------------------
The BME functionality is integrated into the Ensemble class and is the
recommended way to use BME with STARLING.

>>> # Using with Ensemble
>>> from starling import generate, load_ensemble
>>> ensemble = generate("GS"*30, conformations=200) # or load_ensemble("path/to/ensemble")
>>> # Calculate observables
>>> rg_values = ensemble.radius_of_gyration()
>>> ete_values = ensemble.end_to_end_distance()
>>> calculated = np.column_stack([rg_values, ete_values])
>>> # Define experimental constraints
>>> obs_rg = ExperimentalObservable(23.0, 2.0, name="Rg", constraint="equality")
>>> obs_ete = ExperimentalObservable(55.0, 5.0, name="End-to-end", constraint="upper")
>>>
>>> # Perform BME reweighting
>>> result = ensemble.reweight_bme([obs_rg, obs_ete], calculated)
>>>
>>> # Use BME weights in calculations
>>> reweighted_rg = ensemble.radius_of_gyration(return_mean=True, use_bme_weights=True)
>>> reweighted_ete = ensemble.end_to_end_distance(return_mean=True, use_bme_weights=True)

BME Standalone
--------------------------
The BME class can also be used standalone without the Ensemble class. This is
useful for advanced workflows or when you want to apply BME
to observables not directly supported by STARLING.

>>> from starling.structure.bme import BME, ExperimentalObservable
>>> import numpy as np
>>>
>>> # Define experimental observables with different constraint types
>>> # Only three valid constraint strings: "equality", "upper", "lower"
>>>
>>> # Equality constraint: Rg should match 25.0 ± 2.0 Å (default)
>>> obs_rg = ExperimentalObservable(
...     value=25.0,
...     uncertainty=2.0,
...     name="Radius of gyration"
... )
>>>
>>> # Upper bound: End-to-end distance should not exceed 70 Å
>>> obs_ete = ExperimentalObservable(
...     value=70.0,
...     uncertainty=5.0,
...     constraint="upper",
...     name="End-to-end distance"
... )
>>>
>>> # Calculated values from ensemble (n_frames x n_observables)
>>> calculated = np.random.randn(1000, 3) * 2 + np.array([24, 65, 0.25])
>>>
>>> # Create and fit BME model
>>> bme = BME([obs_rg, obs_ete, obs_helix], calculated, theta=0.5)
>>> result = bme.fit(verbose=True)
>>>
>>> # Use optimized weights
>>> reweighted_means = bme.predict(calculated)
"""

from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, rel_entr

from starling.structure.bme_utils import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OPTIMIZER,
    DEFAULT_THETA,
    LAMBDA_INIT_SCALE,
    MIN_WEIGHT_THRESHOLD,
    VALID_CONSTRAINTS,
    BMEResult,
    ExperimentalObservable,
    ThetaScanResult,
    theta_scan,
)


class BME:
    """
    Bayesian Maximum Entropy reweighting. Refine ensembles with experimental data.
    Used to improve fit to experiments while minimizing bias in the ensemble.

    Parameters
    ----------
    observables : list[ExperimentalObservable]
        List of experimental observables to fit.
    calculated_values : np.ndarray
        Array of calculated observable values for each frame.
        Shape: (n_frames, n_observables)
    initial_weights : np.ndarray, optional
        Initial weights for ensemble frames. If None, uniform weights are used.
    """

    def __init__(
        self,
        observables: list,
        calculated_values: np.ndarray,
        theta: float = DEFAULT_THETA,
        initial_weights: Optional[np.ndarray] = None,
    ):
        # Validate inputs
        self._validate_inputs(observables, calculated_values, initial_weights)

        # Store observables
        self.observables = observables
        self.calculated_values = calculated_values
        # Avoid setting self.theta here to not shadow the @property
        # self.theta = None
        self.n_frames = calculated_values.shape[0]
        self.n_observables = len(observables)

        # Initialize weights
        if initial_weights is None:
            self.initial_weights = np.ones(self.n_frames) / self.n_frames
        else:
            self.initial_weights = initial_weights / np.sum(initial_weights)

        # Initialize Lagrange multipliers with small random values
        self._lambdas = np.random.normal(
            loc=0.0, scale=LAMBDA_INIT_SCALE, size=self.n_observables
        ).astype(np.float64)

        # Get bounds for optimization
        self._bounds = [obs.get_bounds() for obs in self.observables]

        # Results storage
        self._result: Optional[BMEResult] = None
        self._theta_scan_result: Optional["ThetaScanResult"] = None
        self._theta: Optional[float] = None

    def _validate_inputs(self, observables, calculated_values, initial_weights):
        """Validate input parameters."""
        if not isinstance(observables, list) or len(observables) == 0:
            raise ValueError("observables must be a non-empty list")

        if not all(isinstance(obs, ExperimentalObservable) for obs in observables):
            raise ValueError("All observables must be ExperimentalObservable instances")

        if not isinstance(calculated_values, np.ndarray):
            raise ValueError("calculated_values must be a numpy array")

        if calculated_values.ndim != 2:
            raise ValueError("calculated_values must be 2D (n_frames, n_observables)")

        if calculated_values.shape[1] != len(observables):
            raise ValueError(
                f"Number of observables ({len(observables)}) must match "
                f"calculated_values columns ({calculated_values.shape[1]})"
            )

        if initial_weights is not None:
            if not isinstance(initial_weights, np.ndarray):
                raise ValueError("initial_weights must be a numpy array")
            if len(initial_weights) != calculated_values.shape[0]:
                raise ValueError("initial_weights length must match number of frames")

    def _compute_chi_squared(self, weights: np.ndarray) -> float:
        """
        Compute chi-squared goodness of fit with constraint handling.

        How constraints work:
        - "equality": Always penalize deviations from experimental value
        - "upper": Only penalize if calculated > experimental (allow being below)
        - "lower": Only penalize if calculated < experimental (allow being above)

        Parameters
        ----------
        weights : np.ndarray
            Current ensemble weights.

        Returns
        -------
        float
            Chi-squared value.
        """
        chi_squared = 0.0

        for obs_idx, observable in enumerate(self.observables):
            # Compute weighted average of calculated values
            calculated_avg = np.sum(self.calculated_values[:, obs_idx] * weights)

            # Difference from experimental value
            diff = calculated_avg - observable.value

            # Determine if we should penalize this deviation
            should_penalize = False

            if observable.constraint == "equality":
                # Always penalize deviations from experimental value
                should_penalize = True
            elif observable.constraint == "upper":
                # Only penalize if calculated exceeds experimental
                should_penalize = diff > 0
            elif observable.constraint == "lower":
                # Only penalize if calculated is below experimental
                should_penalize = diff < 0

            if should_penalize:
                chi_squared += (diff / observable.uncertainty) ** 2

        return chi_squared

    def _objective_and_gradient(self, lambdas: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the BME objective function and its gradient.

        This implements the maximum entropy objective with Gaussian priors:
        L = lambda^T * O_exp + (theta/2) * lambda^T * Sigma^2 * lambda + log(Z)

        Parameters
        ----------
        lambdas : np.ndarray
            Current Lagrange multipliers.

        Returns
        -------
        tuple
            (objective_value, gradient) both scaled by 1/theta for numerical stability.
        """
        # shouldnt ever happen with fit logic.
        if self._theta is None:
            raise RuntimeError(
                "theta has not been set; call fit() with theta or auto_theta=True"
            )

        # Compute log weights: log(w_i) = -lambda^T * O_calc_i + log(w0_i)
        log_unnormalized_weights = -np.sum(
            lambdas * self.calculated_values, axis=1
        ) + np.log(self.initial_weights)

        # Compute log partition function Z = log(sum(exp(log_unnormalized_weights)))
        log_partition_function = logsumexp(log_unnormalized_weights)

        # Compute normalized weights
        reweighted_probabilities = np.exp(
            log_unnormalized_weights - log_partition_function
        )

        # Compute ensemble average of calculated observables: <O_calc>
        ensemble_avg_calculated = np.sum(
            reweighted_probabilities[:, np.newaxis] * self.calculated_values, axis=0
        )

        # Extract experimental values and uncertainties
        experimental_values = np.array([obs.value for obs in self.observables])
        experimental_uncertainties_squared = np.array(
            [obs.uncertainty**2 for obs in self.observables]
        )

        # Regularization term: (theta/2) * sum(lambda_i^2 * sigma_i^2)
        regularization_term = (
            self._theta / 2 * np.sum(lambdas**2 * experimental_uncertainties_squared)
        )

        # Constraint term: lambda^T * O_exp
        # Where O_exp are the experimental observable values
        constraint_term = np.dot(lambdas, experimental_values)

        # Objective function: gamma(lambda) = log(Z(lambda)) + sum(lambda_i F_i^(exp)) + (theta/2) sum(lambda_i^2 sigma_i^2)
        # Where:
        #   Z(lambda) = partition function (normalization constant)
        #   F_i^(exp) = experimental observable values
        #   sigma_i^2 = experimental uncertainties squared
        #   theta = regularization parameter
        objective = log_partition_function + constraint_term + regularization_term

        # Gradient: dgamma/dlambda = O_exp + theta * Sigma^2 * lambda - <O_calc>
        gradient = (
            experimental_values
            + self._theta * lambdas * experimental_uncertainties_squared
            - ensemble_avg_calculated
        )

        # Divide by theta to avoid numerical problems
        return objective / self._theta, gradient / self._theta

    def scan_theta(
        self,
        theta_range: Union[Tuple[float, float], np.ndarray] = (0.01, 10.0),
        n_points: int = 15,
        log_scale: bool = True,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        optimizer: str = DEFAULT_OPTIMIZER,
        verbose: bool = False,
        progress_callback: Optional[callable] = None,
        method: str = "perpendicular",
    ) -> "ThetaScanResult":
        """
        Scan a range of theta values and select the optimal regularization (θ).

        This helper runs a theta scan using this instance's observables,
        calculated_values, and initial_weights, returning a ThetaScanResult that
        contains one BME fit per theta, summary metrics, and the index/value of
        the recommended theta based on an L-curve knee criterion.

        Parameters
        ----------
        theta_range : tuple[float, float] | np.ndarray, optional
            - If a tuple (min_theta, max_theta) is provided, generate a grid of
              size n_points within this range (logarithmic if log_scale=True).
            - If a 1D array is provided, use it as the exact grid of theta values.
            Default is (0.01, 10.0).
        n_points : int, optional
            Number of theta samples used when theta_range is a tuple. Default 15.
        log_scale : bool, optional
            If True and theta_range is a tuple, sample theta values on a log scale.
            Default True.
        max_iterations : int, optional
            Maximum iterations for the internal optimizer in each single-theta fit.
            Default DEFAULT_MAX_ITERATIONS.
        optimizer : str, optional
            Optimizer name forwarded to scipy.optimize.minimize (e.g., "L-BFGS-B").
            Default DEFAULT_OPTIMIZER.
        verbose : bool, optional
            If True, print scan progress messages. Default False.
        progress_callback : Callable[[int, int, float], None] | None, optional
            Optional callback invoked for each theta with signature
            progress_callback(current_index, total, theta_value).
        method : str, optional
            Knee selection rule for the L-curve:
              - "perpendicular" (default): maximum perpendicular distance to the
                line connecting the endpoints (classic knee finding).
              - "curvature": maximum Menger curvature (3-point curvature estimate).

        Returns
        -------
        ThetaScanResult
            Object with:
            - theta_values: np.ndarray of scanned θ
            - chi_squared_values: np.ndarray of final χ² per θ
            - phi_values: np.ndarray of Φ (entropy-based effective fraction) per θ
            - kl_divergence_values: np.ndarray of KL divergence per θ
            - results: list[BMEResult] for each θ
            - optimal_theta: selected θ value
            - optimal_idx: index of the selected θ
            - method: human-readable name of the knee selection method used

        Notes
        -----
        - This method does not modify self._theta or self._result; it only stores
          the scan outcome in self._theta_scan_result for later access.
        - To run a full auto-θ fit and set the model state accordingly, call
          BME.fit(auto_theta=True) or use Ensemble.reweight_bme(..., theta=None).

        Examples
        --------
        >>> scan = bme.scan_theta(theta_range=(1e-2, 1e1), n_points=25, method="curvature")
        >>> scan.print_summary()
        >>> fig = scan.plot(show=True, figsize=(8, 4))
        """
        scan_result = theta_scan(
            observables=self.observables,
            calculated_values=self.calculated_values,
            theta_range=theta_range,
            n_points=n_points,
            log_scale=log_scale,
            max_iterations=max_iterations,
            optimizer=optimizer,
            verbose=verbose,
            progress_callback=progress_callback,
            initial_weights=self.initial_weights,
            method=method,
        )
        self._theta_scan_result = scan_result
        return scan_result

    def _run_single_theta_optimization(
        self,
        max_iterations: int,
        optimizer: str,
        verbose: bool,
    ) -> BMEResult:
        """
        Core BME optimization for a single theta value.

        This is the primary optimization routine that performs the Bayesian Maximum
        Entropy fit using scipy.optimize.minimize. It computes the objective function
        and gradient, applies bounds from observable constraints, and returns a
        BMEResult with optimized weights and diagnostics.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations passed to scipy.optimize.minimize.
        optimizer : str
            Optimization method name accepted by scipy.optimize.minimize
            (e.g., 'L-BFGS-B', 'SLSQP'). Must support `jac=True` and `bounds`.
        verbose : bool
            If True, prints progress information including initial/final chi-squared,
            theta, problem dimensions, and optimizer messages.

        Returns
        -------
        BMEResult
            Optimization result. See BMEResult for field documentation.

            If optimization succeeds, contains optimized weights, lambdas, and metrics.
            If optimization fails, weights revert to initial_weights and chi_squared_final
            is set to np.nan.

        Notes
        -----
        - Bounds on Lagrange multipliers enforce constraint types (equality, upper, lower).
        - If optimization fails, self._result is not modified; only the returned
          BMEResult reflects the failure.

        Examples
        --------
        >>> bme = BME(observables, calculated_values, theta=0.5)
        >>> result = bme._run_single_theta_optimization(
        ...     max_iterations=1000, optimizer="L-BFGS-B", verbose=True
        ... )
        >>> print(f"Success: {result.success}, Chi²: {result.chi_squared_final:.4f}")
        """
        chi_squared_initial = self._compute_chi_squared(self.initial_weights)

        if verbose:
            print("BME Optimization")
            print(f"  Theta: {self._theta}")
            print(f"  Observables: {self.n_observables}")
            print(f"  Frames: {self.n_frames}")
            print(f"  Chi-squared initial: {chi_squared_initial:.4f}")

        optimization_result = minimize(
            self._objective_and_gradient,
            self._lambdas,
            options={"maxiter": max_iterations, "disp": verbose},
            method=optimizer,
            jac=True,
            bounds=self._bounds,
        )

        if optimization_result.success:
            log_weights_opt = -np.sum(
                optimization_result.x[np.newaxis, :] * self.calculated_values, axis=1
            ) + np.log(self.initial_weights)
            weights_opt = np.exp(log_weights_opt - logsumexp(log_weights_opt))

            chi_squared_final = self._compute_chi_squared(weights_opt)
            relative_entropy = float(
                np.sum(rel_entr(weights_opt, self.initial_weights))
            )
            phi = float(np.exp(-relative_entropy))

            if verbose:
                print(
                    f"  Optimization successful (iterations: {optimization_result.nit})"
                )
                print(f"  Chi-squared final: {chi_squared_final:.4f}")
                print(f"  Effective number of frames: {phi:.4f}")

            result = BMEResult(
                weights=weights_opt,
                initial_weights=self.initial_weights.copy(),
                lambdas=optimization_result.x.copy(),
                chi_squared_initial=chi_squared_initial,
                chi_squared_final=chi_squared_final,
                phi=phi,
                n_iterations=optimization_result.nit,
                success=True,
                message=optimization_result.message,
                theta=self._theta,
                observables=self.observables,
                calculated_values=self.calculated_values,
                metadata={
                    "optimizer": optimizer,
                    "max_iterations": max_iterations,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        else:
            if verbose:
                print("  Optimization failed")
                print(f"  Message: {optimization_result.message}")

            result = BMEResult(
                weights=self.initial_weights.copy(),
                initial_weights=self.initial_weights.copy(),
                lambdas=optimization_result.x.copy(),
                chi_squared_initial=chi_squared_initial,
                chi_squared_final=np.nan,
                phi=np.nan,
                n_iterations=getattr(optimization_result, "nit", -1),
                success=False,
                message=optimization_result.message,
                theta=self._theta,
                observables=self.observables,
                calculated_values=self.calculated_values,
                metadata={
                    "optimizer": optimizer,
                    "max_iterations": max_iterations,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

        return result

    def fit(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        optimizer: str = DEFAULT_OPTIMIZER,
        verbose: bool = True,
        *,
        theta: Optional[float] = None,
        auto_theta: bool = True,
        theta_scan_kwargs: Optional[dict] = None,
    ) -> BMEResult:
        # 1) choose effective theta
        if theta is not None:
            if theta <= 0:
                raise ValueError(f"theta must be positive, got {theta}")
            self._theta = float(theta)
            self._theta_scan_result = None
            if verbose:
                print(f"[BME] Using manual theta = {self._theta:.4g} (no theta scan).")

        elif auto_theta:
            if verbose:
                print(
                    "[BME] Auto theta mode: running theta scan to select optimal θ..."
                )

            default_scan_kwargs = dict(
                theta_range=(0.01, 10.0),
                n_points=15,
                log_scale=True,
                max_iterations=max_iterations,
                optimizer=optimizer,
                verbose=False,
                progress_callback=None,
            )
            if theta_scan_kwargs is not None:
                default_scan_kwargs.update(theta_scan_kwargs)

            scan_result = self.scan_theta(**default_scan_kwargs)
            opt_idx = scan_result.optimal_idx

            self._theta = float(scan_result.optimal_theta)
            self._result = scan_result.results[opt_idx]
            self._theta_scan_result = scan_result
            self._lambdas = self._result.lambdas.copy()

            if verbose:
                print(
                    f"[BME] Selected θ={self._theta:.4g} via {scan_result.method} "
                    f"(Chi_squared_init={self._result.chi_squared_initial:.3f}, "
                    f"Chi_squared_final={self._result.chi_squared_final:.3f}, "
                    f"N_eff={self._result.phi:.3f})."
                )
            return self._result

        else:
            # auto_theta=False and no manual theta: choose a sane default
            if self._theta is None:
                self._theta = DEFAULT_THETA
            if verbose:
                print(
                    f"[BME] auto_theta=False and no manual theta; "
                    f"using θ={self._theta:.4g}."
                )

        # 2) run single-theta optimization for self._theta
        self._result = self._run_single_theta_optimization(
            max_iterations=max_iterations,
            optimizer=optimizer,
            verbose=verbose,
        )
        return self._result

    @property
    def result(self) -> Optional[BMEResult]:
        """
        Get the most recent optimization result.

        Returns
        -------
        BMEResult or None
            The optimization result, or None if fit() has not been called.
        """
        return self._result

    @property
    def theta_scan_result(self) -> Optional["ThetaScanResult"]:
        """Last theta scan run on this BME instance (if any)."""
        return self._theta_scan_result

    def predict(self, calculated_values: np.ndarray) -> np.ndarray:
        """
        Compute weighted averages of observables using optimized BME weights.

        This method applies the fitted BME weights to compute ensemble-averaged
        values for any set of observables calculated from the same frames. This
        is useful when you want to apply BME weights to observables that weren't
        used in the fitting process, or when using BME standalone without the
        Ensemble class.

        Parameters
        ----------
        calculated_values : np.ndarray
            Calculated observable values for each frame.
            Shape: (n_frames, n_observables)
            Note: n_frames must match the number of frames used in fit()

        Returns
        -------
        np.ndarray
            Weighted average of each observable. Shape: (n_observables,)

        Raises
        ------
        ValueError
            If fit() has not been called yet, or if the number of frames doesn't match.

        Examples
        --------
        >>> # Fit BME with some observables
        >>> bme = BME([obs1, obs2], calculated_training, theta=1)
        >>> result = bme.fit()
        >>>
        >>> # Apply weights to different observables from same frames
        >>> rg_values = calculate_rg(frames)  # Shape: (n_frames, 1)
        >>> weighted_rg = bme.predict(rg_values)
        >>>
        >>> # Or multiple new observables at once
        >>> new_observables = np.column_stack([rg, ete, contacts])  # (n_frames, 3)
        >>> weighted_means = bme.predict(new_observables)  # Returns 3 means

        Notes
        -----
        When using BME through the Ensemble class, you typically don't need this
        method - use `ensemble.rij(use_bme_weights=True)` or equivalent methods instead.
        This method is primarily for standalone BME usage or advanced workflows.
        """
        if self._result is None or not self._result.success:
            raise ValueError(
                "Model has not been successfully fitted yet. Call fit() first."
            )

        if calculated_values.shape[0] != self.n_frames:
            raise ValueError(
                f"Number of frames in calculated_values ({calculated_values.shape[0]}) "
                f"must match training data ({self.n_frames})"
            )

        # Compute weighted averages
        return np.sum(self._result.weights[:, np.newaxis] * calculated_values, axis=0)

    @property
    def theta(self) -> Optional[float]:
        """Theta value used in the most recent fit (if any)."""
        return self._theta
