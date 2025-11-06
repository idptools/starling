from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_THETA = 0.5
DEFAULT_MAX_ITERATIONS = 50000
DEFAULT_OPTIMIZER = "L-BFGS-B"
LAMBDA_INIT_SCALE = 1e-3
MIN_WEIGHT_THRESHOLD = 1e-50

# Valid constraint types
VALID_CONSTRAINTS = {"equality", "upper", "lower"}


# ----------------------------------------------------------------------
# ExperimentalObservable
# ----------------------------------------------------------------------
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
                "constraint must be a string ('equality', 'upper', or 'lower'), "
                f"got {type(self.constraint).__name__}"
            )

        constraint_lower = self.constraint.lower().strip()
        if constraint_lower not in VALID_CONSTRAINTS:
            raise ValueError(
                f"Invalid constraint: '{self.constraint}'. "
                "Must be 'equality', 'upper', or 'lower'"
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
        """
        if self.constraint == "equality":
            return (None, None)
        elif self.constraint == "upper":
            return (0.0, None)
        else:  # "lower"
            return (None, 0.0)


# ----------------------------------------------------------------------
# BMEResult
# ----------------------------------------------------------------------
@dataclass
class BMEResult:
    """
    Container for BME optimization results.
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
    observables: List[ExperimentalObservable]
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

    @property
    def kl_divergence(self) -> float:
        """
        returns the KL divergence (relative entropy) from phi.
        """
        if self.phi > 0:
            return -np.log(self.phi)
        else:
            return np.inf

    # ----- Integrated QC helpers -------------------------------------------------

    def diagnostics(self, warn_threshold: float = 0.5) -> dict:
        """
        Diagnose BME reweighting results and identify potential issues.

        Returns
        -------
        dict
            Dictionary containing diagnostic information and warnings.

        Notes
        -----
        We report two notions of effective sample size:

        - neff_entropy (N_eff^(S)): entropy-based, derived from Φ = exp(-D_KL).
        This is the standard BME measure: N_eff^(S) = N * Φ.

        - neff_renyi2 (N_eff^(2)): 1 / sum_i w_i^2 (Rényi-2 / participation ratio).
        This is more sensitive to a few large weights, so it is always
        <= neff_entropy for the same weights.
        """
        diagnostics: dict = {}
        warnings: List[str] = []

        N = len(self.weights)

        # ---- Effective sample sizes ---------------------------------------------
        # 1) Entropy-based Neff from phi (matches BME papers)
        neff_entropy = N * float(self.phi)
        diagnostics["neff_entropy"] = neff_entropy
        diagnostics["neff_entropy_fraction"] = float(self.phi)

        # 2) Rényi-2 / participation ratio: 1 / sum w^2
        neff_renyi2 = 1.0 / np.sum(self.weights**2)
        diagnostics["neff_renyi2"] = float(neff_renyi2)
        diagnostics["neff_renyi2_fraction"] = float(neff_renyi2 / N)

        # ---- Weight distribution statistics -------------------------------------
        weight_min = float(self.weights.min())
        weight_max = float(self.weights.max())
        weight_std = float(self.weights.std())
        diagnostics["weight_min"] = weight_min
        diagnostics["weight_max"] = weight_max
        diagnostics["weight_std"] = weight_std

        if weight_min > 0:
            diagnostics["weight_range_orders"] = float(
                np.log10(weight_max / weight_min)
            )
        else:
            diagnostics["weight_range_orders"] = np.inf

        # ---- Chi-squared improvement --------------------------------------------
        diagnostics["chi2_improvement"] = (
            self.chi_squared_initial - self.chi_squared_final
        )
        diagnostics["chi2_improvement_pct"] = (
            (diagnostics["chi2_improvement"] / self.chi_squared_initial) * 100.0
            if self.chi_squared_initial > 0
            else np.nan
        )

        # ---- Checks / warnings ---------------------------------------------------
        # Main diversity check uses the entropy-based fraction (Φ)
        if self.phi < warn_threshold:
            warnings.append(
                f"Low Phi ({self.phi:.3f} < {warn_threshold}): "
                "Significant loss of ensemble diversity. "
                "Consider increasing theta or loosening observable uncertainties."
            )

        # Secondary check: very concentrated weights (using Renyi-2 Neff)
        if neff_renyi2 < 0.1 * N:
            warnings.append(
                "Low effective sample size (1/Σw²) "
                f"({neff_renyi2:.1f} / {N}): "
                f"Only ~{diagnostics['neff_renyi2_fraction'] * 100:.1f}% of frames "
                "are effectively used (participation ratio)."
            )

        if diagnostics["weight_range_orders"] > 3:
            warnings.append(
                "Large weight range "
                f"({diagnostics['weight_range_orders']:.1f} orders of magnitude): "
                "A few frames dominate the reweighted ensemble."
            )

        if self.chi_squared_final > 2 * len(self.observables):
            warnings.append(
                "High final Chi-squared "
                f"({self.chi_squared_final:.2f}): Poor fit to experimental data. "
                "Observables may be incompatible with ensemble."
            )

        diagnostics["warnings"] = warnings
        diagnostics["status"] = "OK" if len(warnings) == 0 else "WARNING"

        return diagnostics

    def print_diagnostics(
        self,
        warn_threshold: float = 0.5,
    ):
        """
        Print a formatted diagnostic report for this BME result.
        """
        diag = self.diagnostics(warn_threshold)
        N = len(self.weights)

        print("\n" + "=" * 60)
        print("BME DIAGNOSTIC REPORT")
        print("=" * 60)

        print(f"\nOptimization Status: {self.message}")
        print(f"Success: {self.success}")
        print(f"Iterations: {self.n_iterations}")

        # Use fixed column widths for alignment
        key_w = 32
        val_w = 14

        print("\nChi-squared:")
        print(f"  {'Initial':<{key_w}} {self.chi_squared_initial:>{val_w}.4f}")
        print(f"  {'Final':<{key_w}} {self.chi_squared_final:>{val_w}.4f}")
        print(
            f"  {'Improvement':<{key_w}} {diag['chi2_improvement']:>{val_w}.4f} "
            f"({diag['chi2_improvement_pct']:.1f}%)"
        )

        print("\nEnsemble Diversity:")
        print(f"  {'Phi (Φ, entropy fraction)':<{key_w}} {self.phi:>{val_w}.4f}")
        print(
            f"  {'N_eff^(S) (entropy-based)':<{key_w}} "
            f"{diag['neff_entropy']:>{val_w}.1f}  / {N}"
        )
        print(
            f"  {'N_eff^(2) (1/Σw², Renyi-2)':<{key_w}} "
            f"{diag['neff_renyi2']:>{val_w}.1f}  / {N}"
        )
        print(f"  {'Theta (θ)':<{key_w}} {self.theta:>{val_w}.4f}")

        print("\nWeight Distribution:")
        print(f"  {'Min':<{key_w}} {diag['weight_min']:>{val_w}.2e}")
        print(f"  {'Max':<{key_w}} {diag['weight_max']:>{val_w}.2e}")
        print(f"  {'Std Dev':<{key_w}} {diag['weight_std']:>{val_w}.2e}")
        print(
            f"  {'Range (orders of magnitude)':<{key_w}} "
            f"{diag['weight_range_orders']:>{val_w}.1f}"
        )

        if len(diag["warnings"]) > 0:
            print(f"\n WARNINGS ({len(diag['warnings'])}):")
            for i, warning in enumerate(diag["warnings"], 1):
                print(f"  {i}. {warning}")
        else:
            print(f"\n✓ Status: {diag['status']} - No issues detected")

        print("\nNotes on effective sample size:")
        print("  - N_eff^(S): entropy-based (from Φ = exp(-D_KL));")
        print("    this matches the usual BME definition.")
        print("  - N_eff^(2): 1/Σw² (Rényi-2 / participation ratio);")
        print("    more sensitive to a few large weights, so it is always")
        print("    ≤ N_eff^(S) for the same weights.")

        print("=" * 60)


# ----------------------------------------------------------------------
# Theta scan support
# ----------------------------------------------------------------------
@dataclass
class ThetaScanResult:
    """
    Container for theta scan results.
    """

    theta_values: np.ndarray
    chi_squared_values: np.ndarray
    phi_values: np.ndarray
    kl_divergence_values: np.ndarray
    results: List[BMEResult]
    optimal_theta: float
    optimal_idx: int
    method: str

    # ----- QC / visualization integration -------------------------------

    def plot(
        self,
        figsize: Tuple[int, int] = (10, 4),
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot theta-scan diagnostics.

        Panels (1 x 2):
          [0]  Chi squared vs N_eff (effective frames), colored by log10(theta)
          [1]  Weight distribution at optimal theta
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        theta = self.theta_values
        chi2 = self.chi_squared_values

        # phi is our definition of the effective fraction
        phi = self.phi_values
        kl = self.kl_divergence_values
        opt_idx = self.optimal_idx

        # ------------------------------------------------------------------
        # Panel 1: Chi squared vs phi (L-curve style)
        # ------------------------------------------------------------------
        ax1 = axes[0]
        sc = ax1.scatter(
            phi,
            chi2,
            c=np.log10(theta),
            s=50,
            alpha=0.7,
        )
        ax1.plot(phi, chi2, "k-", alpha=0.3, linewidth=1)

        ax1.scatter(
            phi[opt_idx],
            chi2[opt_idx],
            color="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label=f"Optimal θ={self.optimal_theta:.3f}",
            zorder=5,
        )

        ax1.set_xlabel(rf"$N_{{eff}}$ (effective fraction)", fontsize=11)
        ax1.set_ylabel(rf"$\chi^2$", fontsize=11)
        # ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)

        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label(rf"$\log_{{10}}(\theta)$", fontsize=10)
        # ------------------------------------------------------------------

        ax2 = axes[1]
        optimal_result = self.results[opt_idx]
        sorted_weights = np.sort(optimal_result.weights)[::-1]

        ax2.plot(
            sorted_weights, "o-", markersize=3, color="darkred", label="BME weights"
        )
        ax2.axhline(
            1.0 / len(sorted_weights),
            color="blue",
            linestyle="--",
            label="Uniform weight",
        )
        ax2.set_xlabel("Conformation (sorted by weight)", fontsize=11)
        ax2.set_ylabel("Weight", fontsize=11)
        ax2.set_yscale("log")
        ax2.legend(frameon=False)
        # ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            # Prevent auto-rendering in Jupyter inline backend
            plt.close(fig)

        return fig

    def print_summary(self, n_show: int = 5):
        """
        Print a formatted summary of theta scan results.
        """
        print("\n" + "=" * 60)
        print("THETA SCAN SUMMARY")
        print("=" * 60)

        print(
            f"\nScan range: {self.theta_values[0]:.4f} to {self.theta_values[-1]:.4f}"
        )
        print(f"Number of points: {len(self.theta_values)}")
        print(f"Method: {self.method}\n")

        print(f"RECOMMENDED THETA: {self.optimal_theta:.4f}")
        opt_idx = self.optimal_idx
        print(f"Chi squared: {self.chi_squared_values[opt_idx]:.4f}")
        print(f"N_eff:   {self.phi_values[opt_idx]:.4f}")
        print(f"Relative Entropy:   {self.kl_divergence_values[opt_idx]:.4f}")

        # # Aligned table header
        # print("\n Sample of theta values:")
        # h_theta = "Theta"
        # h_chi = "Chi squared"
        # h_neff = "N_eff"
        # h_kl = "Rel. Entropy"
        # h_status = "Status"
        # print(f"{h_theta:>10}  {h_chi:>14}  {h_neff:>12}  {h_kl:>14}  {h_status:^8}")
        # print("-" * 60)

        # indices = np.linspace(0, len(self.theta_values) - 1, n_show, dtype=int)
        # for idx in indices:
        #     status = "✓" if self.results[idx].success else "✗"
        #     marker = " <- OPTIMAL" if idx == opt_idx else ""
        #     print(
        #         f"{self.theta_values[idx]:>10.4f}  "
        #         f"{self.chi_squared_values[idx]:>14.4f}  "
        #         f"{self.phi_values[idx]:>12.4f}  "
        #         f"{self.kl_divergence_values[idx]:>14.4f}  "
        #         f"{status:^8}{marker}"
        #     )

        print(
            "  • L-curve shows trade-off between fit quality (Chi squared) "
            "and effective sample size (N_eff)"
        )
        print("  • Optimal theta balances these competing objectives")
        print("  • N_eff < 0.5 indicates significant loss of ensemble diversity")
        print("  • Consider increasing theta if N_eff is too low at optimum")
        print("=" * 60)


# ----------------------------------------------------------------------
# Theta scan driver + knee finding
# ----------------------------------------------------------------------
def theta_scan(
    observables: List[ExperimentalObservable],
    calculated_values: np.ndarray,
    theta_range: Union[Tuple[float, float], np.ndarray] = (0.01, 10.0),
    n_points: int = 15,
    log_scale: bool = True,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    optimizer: str = DEFAULT_OPTIMIZER,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    initial_weights: Optional[np.ndarray] = None,
    method: str = "perpendicular",
) -> ThetaScanResult:
    """
    Scan a range of regularization parameters (theta) for BME reweighting.

    Parameters
    ----------
    observables : List[ExperimentalObservable]
        Experimental observables to use for reweighting.
    calculated_values : np.ndarray
        Calculated values from the ensemble. Shape typically (n_models, n_observables).
    theta_range : tuple or np.ndarray, optional
        If a tuple (min, max) is given, a grid of `n_points` thetas is created;
        if an array is supplied it is used directly. Default (0.01, 10.0).
    n_points : int, optional
        Number of theta samples when `theta_range` is a tuple. Default 15.
    log_scale : bool, optional
        If True and `theta_range` is a tuple, sample theta logarithmically.
        Default True.
    max_iterations : int, optional
        Maximum optimizer iterations forwarded to BME.fit. Default from module.
    optimizer : str, optional
        Optimizer name forwarded to BME.fit. Default from module.
    verbose : bool, optional
        If True, print progress messages. Default False.
    progress_callback : callable, optional
        Optional callback called as progress_callback(current_index, total, theta).
    initial_weights : np.ndarray or None, optional
        Optional initial weight vector for BME.
    method : str, optional
        Method used to pick optimal theta ('perpendicular' or 'curvature').

    Returns
    -------
    ThetaScanResult
        Object containing theta grid, metric arrays, individual BMEResult objects,
        and the selected optimal theta and index.

    Raises
    ------
    ValueError
        If n_points < 1 when a tuple theta_range is provided, or if log_scale is True
        and theta_range tuple contains non-positive endpoints.
    """
    from starling.structure.bme import BME

    if isinstance(theta_range, (tuple, list)):
        if n_points < 1:
            raise ValueError(
                f"n_points must be >= 1 when theta_range is a tuple/list. Received: {n_points}"
            )
        if log_scale:
            if theta_range[0] <= 0 or theta_range[1] <= 0:
                raise ValueError(
                    "theta_range endpoints must be positive when log_scale=True, Received: "
                    f"{theta_range}"
                )

    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")

    # Generate theta values
    if isinstance(theta_range, (tuple, list)):
        if log_scale:
            theta_values = np.logspace(
                np.log10(theta_range[0]), np.log10(theta_range[1]), n_points
            )
        else:
            theta_values = np.linspace(theta_range[0], theta_range[1], n_points)
    else:
        theta_values = np.asarray(theta_range)

    chi_squared_vals: List[float] = []
    phi_vals: List[float] = []
    kl_vals: List[float] = []
    bme_results: List[BMEResult] = []

    for i, theta in enumerate(theta_values):
        if progress_callback is not None:
            progress_callback(i + 1, len(theta_values), theta)
        if verbose:
            print(f"Processing theta {i + 1}/{len(theta_values)}: {theta:.4f}")

        bme = BME(
            observables=observables,
            calculated_values=calculated_values,
            initial_weights=initial_weights,
        )
        # Single-theta fit, no auto scan:
        result = bme.fit(
            max_iterations=max_iterations,
            optimizer=optimizer,
            verbose=False,
            theta=float(theta),
            auto_theta=False,
        )

        bme_results.append(result)
        chi_squared_vals.append(result.chi_squared_final)
        phi_vals.append(result.phi)
        kl_vals.append(result.kl_divergence)

    chi_squared_vals_arr = np.array(chi_squared_vals)
    phi_vals_arr = np.array(phi_vals)
    kl_vals_arr = np.array(kl_vals)

    optimal_idx, method = find_optimal_theta(
        chi_squared_vals_arr, kl_vals_arr, method=method
    )
    optimal_theta = float(theta_values[optimal_idx])

    return ThetaScanResult(
        theta_values=theta_values,
        chi_squared_values=chi_squared_vals_arr,
        phi_values=phi_vals_arr,
        kl_divergence_values=kl_vals_arr,
        results=bme_results,
        optimal_theta=optimal_theta,
        optimal_idx=optimal_idx,
        method=method,
    )


def find_optimal_theta(
    chi_squared_values: np.ndarray,
    kl_divergence_values: np.ndarray,
    method: str = "perpendicular",
) -> Tuple[int, str]:
    """
    Find optimal theta value using L-curve analysis.
    """
    if method == "curvature":
        idx = _find_knee_curvature(chi_squared_values, kl_divergence_values)
        return idx, "Menger curvature"
    elif method == "perpendicular":
        idx = _find_knee_perpendicular(chi_squared_values, kl_divergence_values)
        return idx, "Perpendicular distance"
    else:
        raise ValueError(
            f"Unknown method: {method}, must be 'curvature' or 'perpendicular'"
        )


def _find_knee_curvature(x_values: np.ndarray, y_values: np.ndarray) -> int:
    """
    Find knee point using Menger curvature (3-point formula).
    """
    x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min() + 1e-10)
    y_norm = (y_values - y_values.min()) / (y_values.max() - y_values.min() + 1e-10)

    n_points = len(x_norm)
    curvature = np.zeros(n_points)

    for i in range(1, n_points - 1):
        p0 = np.array([x_norm[i - 1], y_norm[i - 1]])
        p1 = np.array([x_norm[i], y_norm[i]])
        p2 = np.array([x_norm[i + 1], y_norm[i + 1]])

        v1 = p1 - p0
        v2 = p2 - p1

        area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p0 - p2)
        c = np.linalg.norm(p1 - p0)

        if a * b * c > 1e-10:
            curvature[i] = 4 * area / (a * b * c)

    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    return int(np.argmax(curvature))


def _find_knee_perpendicular(x_values: np.ndarray, y_values: np.ndarray) -> int:
    """
    Find knee point using perpendicular distance from line connecting endpoints.
    """
    x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min() + 1e-10)
    y_norm = (y_values - y_values.min()) / (y_values.max() - y_values.min() + 1e-10)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return len(x_values) // 2

    line_unit = line_vec / line_len

    distances = []
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        vec = point - p1
        proj_length = np.dot(vec, line_unit)
        proj_point = p1 + proj_length * line_unit
        perp_dist = np.linalg.norm(point - proj_point)
        distances.append(perp_dist)

    return int(np.argmax(distances))
