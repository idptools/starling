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

>>> from starling.inference.bme import BME, ExperimentalObservable
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

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, rel_entr

# Constants
DEFAULT_THETA = 0.5
DEFAULT_MAX_ITERATIONS = 50000
DEFAULT_OPTIMIZER = "L-BFGS-B"
LAMBDA_INIT_SCALE = 1e-3
MIN_WEIGHT_THRESHOLD = 1e-50


# Valid constraint types
VALID_CONSTRAINTS = {"equality", "upper", "lower"}


@dataclass
class ExperimentalObservable:
    """
    Container for experimental observable data.

    Parameters
    ----------
    value : float
        The experimental value of the observable.
    uncertainty : float
        The experimental uncertainty (standard deviation).
    constraint : str, optional
        Type of constraint. Must be one of:
        - "equality" (default): Observable should match value ± uncertainty
        - "upper": Observable should not exceed value
        - "lower": Observable should not fall below value
    name : str, optional
        Optional name/description of the observable.

    Examples
    --------
    >>> # Equality constraint (default): Rg = 25 ± 2 Å
    >>> obs1 = ExperimentalObservable(25.0, 2.0, name="Rg")
    >>>
    >>> # Upper bound: distance should not exceed 70 Å
    >>> obs2 = ExperimentalObservable(70.0, 5.0, constraint="upper", name="Max distance")
    >>>
    >>> # Lower bound: helicity should be at least 30%
    >>> obs3 = ExperimentalObservable(0.3, 0.05, constraint="lower", name="Min helicity")
    """

    value: float
    uncertainty: float
    constraint: str = "equality"
    name: Optional[str] = None

    def __post_init__(self):
        """Validate the observable data."""
        if self.uncertainty <= 0:
            raise ValueError(f"Uncertainty must be positive, got {self.uncertainty}")

        # Validate constraint
        if not isinstance(self.constraint, str):
            raise TypeError(
                f"constraint must be a string ('equality', 'upper', or 'lower'), got {type(self.constraint).__name__}"
            )

        constraint_lower = self.constraint.lower().strip()
        if constraint_lower not in VALID_CONSTRAINTS:
            raise ValueError(
                f"Invalid constraint: '{self.constraint}'. Must be 'equality', 'upper', or 'lower'"
            )

        # Normalize to lowercase
        self.constraint = constraint_lower

    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the optimization bounds for the Lagrange multiplier.

        These bounds ensure the constraint type is enforced during optimization:
        - "equality": No bounds (lambda can be any value)
        - "upper": lambda >= 0 (positive lambda pushes observable down)
        - "lower": lambda <= 0 (negative lambda pushes observable up)

        Returns
        -------
        tuple
            (lower_bound, upper_bound) for the Lagrange multiplier.
        """
        if self.constraint == "equality":
            return (None, None)
        elif self.constraint == "upper":
            return (0.0, None)
        else:  # "lower"
            return (None, 0.0)


@dataclass
class BMEResult:
    """
    Container for BME optimization results.

    Attributes
    ----------
    weights : np.ndarray
        Optimized weights for each frame in the ensemble.
    initial_weights : np.ndarray
        Initial (uniform) weights before optimization.
    lambdas : np.ndarray
        Optimized Lagrange multipliers.
    chi_squared_initial : float
        Chi-squared before optimization.
    chi_squared_final : float
        Chi-squared after optimization.
    phi : float
        Fraction of effective frames (measure of ensemble diversity).
    n_iterations : int
        Number of optimization iterations.
    success : bool
        Whether the optimization succeeded.
    message : str
        Optimization status message.
    theta : float
        Theta parameter used in optimization.
    observables : list[ExperimentalObservable]
        List of experimental observables used.
    calculated_values : np.ndarray
        Calculated observable values for each frame.
    metadata : dict
        Additional metadata about the optimization.
    """

    weights: np.ndarray
    initial_weights: np.ndarray
    lambdas: np.ndarray
    chi_squared_initial: float
    chi_squared_final: float
    phi: float
    n_iterations: int
    success: bool
    message: str
    theta: float
    observables: list
    calculated_values: np.ndarray
    metadata: dict

    def __str__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"BME Result [{status}]\n"
            f"  Chi-squared initial: {self.chi_squared_initial:.4f}\n"
            f"  Chi-squared final:   {self.chi_squared_final:.4f}\n"
            f"  phi (effective fraction): {self.phi:.4f}\n"
            f"  Iterations: {self.n_iterations}\n"
            f"  Theta: {self.theta}"
        )

    def __repr__(self):
        return self.__str__()


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
    theta : float, optional
        Regularization parameter controlling the trade-off between fitting
        experimental data and maintaining ensemble diversity. Default is 0.5.
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
        self._validate_inputs(observables, calculated_values, theta, initial_weights)

        # Store observables
        self.observables = observables
        self.calculated_values = calculated_values
        self.theta = theta
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

    def _validate_inputs(self, observables, calculated_values, theta, initial_weights):
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

        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")

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
            self.theta / 2 * np.sum(lambdas**2 * experimental_uncertainties_squared)
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
            + self.theta * lambdas * experimental_uncertainties_squared
            - ensemble_avg_calculated
        )

        # TODO: I don't think this is needed? Legacy from before gradient fix and lgosumexp trick - but scaling shouldnt hurt optimum
        # Divide by theta to avoid numerical problems
        return objective / self.theta, gradient / self.theta

    def fit(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        optimizer: str = DEFAULT_OPTIMIZER,
        verbose: bool = True,
    ) -> BMEResult:
        """
        Perform BME optimization to find optimal ensemble weights.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of optimization iterations. Default is 50000.
        optimizer : str, optional
            Optimization method to use. Default is 'L-BFGS-B'.
        verbose : bool, optional
            If True, print optimization progress. Default is True.

        Returns
        -------
        BMEResult
            Object containing optimization results including weights, chi-squared,
            and convergence information.
        """
        chi_squared_initial = self._compute_chi_squared(self.initial_weights)

        if verbose:
            print("BME Optimization")
            print(f"  Theta: {self.theta}")
            print(f"  Observables: {self.n_observables}")
            print(f"  Frames: {self.n_frames}")
            print(f"  Chi-squared initial: {chi_squared_initial:.4f}")

        # Perform optimization
        optimization_result = minimize(
            self._objective_and_gradient,
            self._lambdas,
            options={"maxiter": max_iterations, "disp": verbose},
            method=optimizer,
            jac=True,
            bounds=self._bounds,
        )

        if optimization_result.success:
            # Compute optimized weights
            log_weights_opt = -np.sum(
                optimization_result.x[np.newaxis, :] * self.calculated_values, axis=1
            ) + np.log(self.initial_weights)
            weights_opt = np.exp(log_weights_opt - logsumexp(log_weights_opt))

            chi_squared_final = self._compute_chi_squared(weights_opt)

            relative_entropy = float(
                np.sum(rel_entr(weights_opt, self.initial_weights))
            )

            phi = np.exp(-relative_entropy)

            if verbose:
                print(
                    f"  Optimization successful (iterations: {optimization_result.nit})"
                )
                print(f"  Chi-squared final: {chi_squared_final:.4f}")
                print(f"  Effective number of frames: {phi:.4f}")

            self._result = BMEResult(
                weights=weights_opt,
                initial_weights=self.initial_weights.copy(),
                lambdas=optimization_result.x.copy(),
                chi_squared_initial=chi_squared_initial,
                chi_squared_final=chi_squared_final,
                phi=phi,
                n_iterations=optimization_result.nit,
                success=True,
                message=optimization_result.message,
                theta=self.theta,
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

            self._result = BMEResult(
                weights=self.initial_weights.copy(),
                initial_weights=self.initial_weights.copy(),
                lambdas=optimization_result.x.copy(),
                chi_squared_initial=chi_squared_initial,
                chi_squared_final=np.nan,
                phi=np.nan,
                n_iterations=optimization_result.nit
                if hasattr(optimization_result, "nit")
                else -1,
                success=False,
                message=optimization_result.message,
                theta=self.theta,
                observables=self.observables,
                calculated_values=self.calculated_values,
                metadata={
                    "optimizer": optimizer,
                    "max_iterations": max_iterations,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
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


# Helper function for analyzing BME results
def diagnose_bme_result(result: BMEResult, warn_threshold: float = 0.5) -> dict:
    """
    Diagnose BME reweighting results and identify potential issues.

    Parameters
    ----------
    result : BMEResult
        The BME optimization result to diagnose
    warn_threshold : float
        Phi threshold below which to warn about diversity loss. Default 0.5.

    Returns
    -------
    dict
        Dictionary containing diagnostic information and warnings
    """
    diagnostics = {}
    warnings = []

    # Calculate effective sample size (Neff)
    n_effective = 1 / np.sum(result.weights**2)
    diagnostics["n_effective"] = n_effective
    diagnostics["n_effective_fraction"] = n_effective / len(result.weights)

    # Weight distribution statistics
    diagnostics["weight_min"] = result.weights.min()
    diagnostics["weight_max"] = result.weights.max()
    diagnostics["weight_std"] = result.weights.std()
    diagnostics["weight_range_orders"] = np.log10(
        diagnostics["weight_max"] / diagnostics["weight_min"]
    )

    # Chi-squared improvement
    diagnostics["chi2_improvement"] = (
        result.chi_squared_initial - result.chi_squared_final
    )
    diagnostics["chi2_improvement_pct"] = (
        diagnostics["chi2_improvement"] / result.chi_squared_initial
    ) * 100

    # Check for issues
    if result.phi < warn_threshold:
        warnings.append(
            f"Low Phi ({result.phi:.3f} < {warn_threshold}): Significant loss of ensemble diversity. "
            f"Consider increasing theta or loosening observable uncertainties."
        )

    if diagnostics["weight_range_orders"] > 3:
        warnings.append(
            f"Large weight range ({diagnostics['weight_range_orders']:.1f} orders of magnitude): "
            f"A few frames dominate the reweighted ensemble."
        )

    if n_effective < 0.1 * len(result.weights):
        warnings.append(
            f"Low effective sample size ({n_effective:.1f} / {len(result.weights)}): "
            f"Only ~{diagnostics['n_effective_fraction'] * 100:.1f}% of frames are effectively used."
        )

    if result.chi_squared_final > 2 * len(result.observables):
        warnings.append(
            f"High final Chi-squared ({result.chi_squared_final:.2f}): Poor fit to experimental data. "
            f"Observables may be incompatible with ensemble."
        )

    diagnostics["warnings"] = warnings
    diagnostics["status"] = "OK" if len(warnings) == 0 else "WARNING"

    return diagnostics


def print_bme_diagnostics(result: BMEResult, warn_threshold: float = 0.5):
    """
    Print a formatted diagnostic report for BME results.

    Parameters
    ----------
    result : BMEResult
        The BME optimization result to diagnose
    warn_threshold : float
        Phi threshold below which to warn about diversity loss
    """
    diag = diagnose_bme_result(result, warn_threshold)

    print("\n" + "=" * 60)
    print("BME DIAGNOSTIC REPORT")
    print("=" * 60)

    print(f"\nOptimization Status: {result.message}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.n_iterations}")

    print(f"\nChi-squared:")
    print(f"  Initial:     {result.chi_squared_initial:.4f}")
    print(f"  Final:       {result.chi_squared_final:.4f}")
    print(
        f"  Improvement: {diag['chi2_improvement']:.4f} ({diag['chi2_improvement_pct']:.1f}%)"
    )

    print(f"\nEnsemble Diversity:")
    print(f"  Phi (Φ):                {result.phi:.4f}")
    print(
        f"  Effective sample size:  {diag['n_effective']:.1f} / {len(result.weights)} ({diag['n_effective_fraction'] * 100:.1f}%)"
    )
    print(f"  Theta (θ):              {result.theta}")

    print(f"\nWeight Distribution:")
    print(f"  Min:        {diag['weight_min']:.2e}")
    print(f"  Max:        {diag['weight_max']:.2e}")
    print(f"  Std Dev:    {diag['weight_std']:.2e}")
    print(f"  Range:      {diag['weight_range_orders']:.1f} orders of magnitude")

    if len(diag["warnings"]) > 0:
        print(f"\n⚠️  WARNINGS ({len(diag['warnings'])}):")
        for i, warning in enumerate(diag["warnings"], 1):
            print(f"  {i}. {warning}")
    else:
        print(f"\n✓ Status: {diag['status']} - No issues detected")

    print("=" * 60)
