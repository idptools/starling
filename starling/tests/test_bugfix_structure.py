"""
Regression tests for fixes in the structure layer (Ensemble, BME, viz helpers).

Everything here runs on the CPU from synthetic random-walk coordinates and
distance maps, so no model weights are needed.
"""

import dataclasses
import importlib
import sys
import warnings

import matplotlib
import numpy as np
import pytest
from soursop.sstrajectory import SSTrajectory

from starling import configs
from starling.structure.bme import BME
from starling.structure.bme_utils import (
    BMEResult,
    ExperimentalObservable,
    build_theta_grid,
    find_optimal_theta,
    theta_scan,
)
from starling.structure.coordinates import create_ca_topology_from_coords
from starling.structure.ensemble import Ensemble

# Virtual CA-CA distance between consecutive residues in Angstroms (the standard
# trans-peptide value), used to build random-walk test chains.
CA_CA_BOND_LENGTH_A = 3.8

# Seed for the global numpy RNG, which BME uses to initialise its multipliers.
BME_LAMBDA_SEED = 0


def random_walk_coordinates(n_frames: int, n_residues: int, seed: int = 0) -> np.ndarray:
    """
    Build freely jointed random-walk chains with fixed CA-CA bond lengths.

    Parameters
    ----------
    n_frames : int
        Number of independent chains (conformations).
    n_residues : int
        Number of beads (residues) per chain.
    seed : int, optional
        Seed for the random number generator. Default 0.

    Returns
    -------
    np.ndarray
        Coordinates in Angstroms, shape (n_frames, n_residues, 3), with the
        first bead of every chain at the origin.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_frames, n_residues - 1, 3))
    steps = CA_CA_BOND_LENGTH_A * steps / np.linalg.norm(steps, axis=-1, keepdims=True)
    origin = np.zeros((n_frames, 1, 3))
    return np.concatenate([origin, np.cumsum(steps, axis=1)], axis=1)


def distance_maps_from_coordinates(coordinates_A: np.ndarray) -> np.ndarray:
    """
    Compute full, symmetric inter-residue distance maps.

    Parameters
    ----------
    coordinates_A : np.ndarray
        Coordinates in Angstroms, shape (n_frames, n_residues, 3).

    Returns
    -------
    np.ndarray
        Distance maps in Angstroms, shape (n_frames, n_residues, n_residues).
    """
    return np.linalg.norm(coordinates_A[:, :, None] - coordinates_A[:, None, :], axis=-1)


def random_walk_ensemble(n_frames: int = 12, n_residues: int = 20, seed: int = 0) -> Ensemble:
    """
    Build an Ensemble (no structures) from random-walk distance maps.

    Parameters
    ----------
    n_frames : int, optional
        Number of conformations. Default 12.
    n_residues : int, optional
        Chain length. Default 20.
    seed : int, optional
        Seed for the random walk. Default 0.

    Returns
    -------
    Ensemble
        Ensemble of a poly-glycine chain with the random-walk distance maps.
    """
    maps = distance_maps_from_coordinates(random_walk_coordinates(n_frames, n_residues, seed))
    return Ensemble(maps, "G" * n_residues)


def ssprotein_from_coordinates(sequence: str, coordinates_A: np.ndarray):
    """
    Wrap Angstrom coordinates as a SOURSOP SSProtein (mdtraj works in nm).

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    coordinates_A : np.ndarray
        Coordinates in Angstroms, shape (n_frames, n_residues, 3).

    Returns
    -------
    soursop.ssprotein.SSProtein
        Protein trajectory with one frame per conformation.
    """
    coordinates_nm = coordinates_A / configs.CONVERT_ANGSTROM_TO_NM
    return SSTrajectory(TRJ=create_ca_topology_from_coords(sequence, coordinates_nm)).proteinTrajectoryList[0]


def make_failed(result: BMEResult) -> BMEResult:
    """
    Return a copy of a BME result marked as a failed optimisation.

    Parameters
    ----------
    result : BMEResult
        A (successful) result to copy.

    Returns
    -------
    BMEResult
        Copy that looks like what BME returns when the optimizer fails:
        success False, initial weights, and NaN chi-squared and phi.
    """
    return dataclasses.replace(
        result,
        success=False,
        weights=result.initial_weights.copy(),
        chi_squared_final=np.nan,
        phi=np.nan,
        message="forced failure for testing",
    )


@pytest.fixture
def bme_problem():
    """Two observables over 300 frames that the ensemble can be reweighted towards."""
    rng = np.random.default_rng(1)
    calculated = np.column_stack([rng.normal(25, 3, 300), rng.normal(70, 8, 300)])
    observables = [
        ExperimentalObservable(23.0, 1.0, name="Rg"),
        ExperimentalObservable(65.0, 3.0, constraint="upper", name="ETE"),
    ]
    return observables, calculated


# ---------------------------------------------------------------------------
# Ensemble: hydrodynamic radius
# ---------------------------------------------------------------------------


def test_rh_cache_is_keyed_on_alphas():
    ensemble = random_walk_ensemble(n_residues=40)
    default_rh = ensemble.hydrodynamic_radius(return_mean=True)
    changed_rh = ensemble.hydrodynamic_radius(return_mean=True, alpha1=0.3)
    forced_rh = ensemble.hydrodynamic_radius(return_mean=True, alpha1=0.3, force_recompute=True)

    assert changed_rh == pytest.approx(forced_rh)
    assert changed_rh != pytest.approx(default_rh)
    # going back to the defaults must not return the alpha1=0.3 values
    assert ensemble.hydrodynamic_radius(return_mean=True) == pytest.approx(default_rh)


def test_kr_rh_matches_kirkwood_riseman_definition():
    n_residues = 30
    ensemble = random_walk_ensemble(n_residues=n_residues)
    maps = ensemble.distance_maps()

    # 1/Rh = (1/N^2) sum_{i != j} 1/r_ij, written over the full (redundant) map
    off_diagonal = ~np.eye(n_residues, dtype=bool)
    expected = np.array([n_residues**2 / np.sum(1.0 / m[off_diagonal]) for m in maps])

    np.testing.assert_allclose(ensemble.hydrodynamic_radius(mode="kr"), expected, rtol=1e-12)


def test_kr_rh_is_n_over_n_minus_one_times_inverse_mean_inverse_distance():
    n_residues, spacing_A = 20, 3.81
    separations = np.abs(np.arange(n_residues)[:, None] - np.arange(n_residues)[None, :])
    ensemble = Ensemble((spacing_A * separations)[None], "ACDEFGHIKLMNPQRSTVWY")

    # the old (pre-fix) value 1/<1/r_ij> was too small by (N-1)/N
    n_pairs = n_residues * (n_residues - 1) / 2
    inverse_distance_sum = sum((n_residues - s) / (spacing_A * s) for s in range(1, n_residues))
    old_value = n_pairs / inverse_distance_sum
    expected = n_residues**2 / (2 * inverse_distance_sum)

    assert expected == pytest.approx(old_value * n_residues / (n_residues - 1))
    np.testing.assert_allclose(ensemble.hydrodynamic_radius(mode="kr"), [expected])


def test_nygaard_rh_raises_for_very_short_chains():
    ensemble = random_walk_ensemble(n_residues=3)
    with pytest.raises(ValueError, match="mode='kr'"):
        ensemble.hydrodynamic_radius(mode="nygaard")

    # Kirkwood-Riseman still works and is positive
    assert np.all(ensemble.hydrodynamic_radius(mode="kr") > 0)


def test_rg_and_rh_return_copies_of_the_cache():
    ensemble = random_walk_ensemble()
    rg = ensemble.radius_of_gyration()
    rg_reference = rg.copy()
    rg[:] = -1.0
    np.testing.assert_array_equal(ensemble.radius_of_gyration(), rg_reference)

    for mode in ("kr", "nygaard"):
        rh = ensemble.hydrodynamic_radius(mode=mode)
        rh_reference = rh.copy()
        rh[:] = -1.0
        np.testing.assert_array_equal(ensemble.hydrodynamic_radius(mode=mode), rh_reference)


# ---------------------------------------------------------------------------
# Ensemble: indexing and validation
# ---------------------------------------------------------------------------


def test_local_rg_of_whole_chain_equals_global_rg():
    ensemble = random_walk_ensemble()
    np.testing.assert_allclose(
        ensemble.local_radius_of_gyration(0, ensemble.sequence_length),
        ensemble.radius_of_gyration(),
    )


@pytest.mark.parametrize(("start", "end"), [(10, 25), (10, 10), (12, 10), (-1, 5)])
def test_local_rg_rejects_invalid_regions(start, end):
    ensemble = random_walk_ensemble(n_residues=20)
    with pytest.raises(ValueError, match="end is exclusive"):
        ensemble.local_radius_of_gyration(start, end)


@pytest.mark.parametrize("j", [-1, 20])
def test_rij_validates_j(j):
    ensemble = random_walk_ensemble(n_residues=20)
    with pytest.raises(ValueError, match="Invalid residue index j"):
        ensemble.rij(0, j)


def test_constructor_rejects_upper_triangular_maps():
    maps = random_walk_ensemble().distance_maps()
    with pytest.raises(ValueError, match="must be symmetric"):
        Ensemble(np.triu(maps, 1), "G" * maps.shape[1])


def test_constructor_rejects_nonzero_diagonal():
    maps = random_walk_ensemble().distance_maps().copy()
    maps[3, 4, 4] = 1.0
    with pytest.raises(ValueError, match="zero diagonal"):
        Ensemble(maps, "G" * maps.shape[1])


def test_constructor_leaves_non_finite_maps_to_check_for_errors():
    maps = random_walk_ensemble().distance_maps().copy()
    maps[2, 0, 5] = maps[2, 5, 0] = np.nan
    ensemble = Ensemble(maps, "G" * maps.shape[1])
    assert ensemble.check_for_errors(verbose=False) == [2]


def test_invalid_sequence_error_lists_valid_residues():
    maps = random_walk_ensemble(n_residues=3).distance_maps()
    with pytest.raises(ValueError) as error:
        Ensemble(maps, "GXG")
    assert "{configs.VALID_AA}" not in str(error.value)
    assert str(configs.VALID_AA) in str(error.value)


# ---------------------------------------------------------------------------
# Ensemble: check_for_errors keeps the attached trajectory
# ---------------------------------------------------------------------------


def ensemble_with_bad_map_frame(bad_frame: int = 5):
    """
    Build an ensemble with structures where one distance map is corrupted.

    Parameters
    ----------
    bad_frame : int, optional
        Index of the frame whose end-to-end distance is set to an impossible
        500 A in the distance map (but not in the coordinates). Default 5.

    Returns
    -------
    tuple
        (Ensemble, coordinates in nm of the attached trajectory).
    """
    coordinates_A = random_walk_coordinates(12, 20)
    maps = distance_maps_from_coordinates(coordinates_A)
    maps[bad_frame, 0, -1] = maps[bad_frame, -1, 0] = 500.0
    sequence = "GS" * 10
    ensemble = Ensemble(maps, sequence, ssprot_ensemble=ssprotein_from_coordinates(sequence, coordinates_A))
    return ensemble, ensemble.trajectory.traj.xyz.copy()


def test_check_for_errors_slices_trajectory_by_default():
    ensemble, original_xyz_nm = ensemble_with_bad_map_frame(bad_frame=5)
    bad_frames = ensemble.check_for_errors(remove_errors=True, verbose=False)

    keep = [i for i in range(len(original_xyz_nm)) if i != 5]
    assert bad_frames == [5]
    assert ensemble.has_structures
    assert len(ensemble) == ensemble.trajectory.n_frames == 11
    np.testing.assert_allclose(ensemble.trajectory.traj.xyz, original_xyz_nm[keep])


def test_check_for_errors_rebuilds_trajectory_only_when_asked(monkeypatch):
    ensemble, _ = ensemble_with_bad_map_frame()
    rebuild_calls = []
    monkeypatch.setattr(
        ensemble,
        "build_ensemble_trajectory",
        lambda **kwargs: rebuild_calls.append(kwargs),
    )
    ensemble.check_for_errors(remove_errors=True, verbose=False, rebuild_trajectory=True)
    assert rebuild_calls == [{"force_recompute": True}]
    assert len(ensemble) == 11


def test_check_for_errors_refuses_mismatched_trajectory():
    coordinates_A = random_walk_coordinates(12, 20)
    maps = distance_maps_from_coordinates(coordinates_A)
    maps[5, 0, -1] = maps[5, -1, 0] = 500.0
    sequence = "GS" * 10
    short_trajectory = ssprotein_from_coordinates(sequence, coordinates_A[:10])
    ensemble = Ensemble(maps, sequence, ssprot_ensemble=short_trajectory)

    with pytest.raises(ValueError, match="10 frames but there are 12 distance maps"):
        ensemble.check_for_errors(remove_errors=True, verbose=False)
    # nothing was modified
    assert len(ensemble) == 12
    assert ensemble.trajectory.n_frames == 10


# ---------------------------------------------------------------------------
# Ensemble.reweight_bme: caching and failures
# ---------------------------------------------------------------------------


def rg_ensemble_and_values():
    """
    Build a random-walk ensemble and its per-frame Rg as a BME input column.

    Returns
    -------
    tuple
        (Ensemble with 40 frames, Rg values in Angstroms, shape (40, 1)).
    """
    ensemble = random_walk_ensemble(n_frames=40, n_residues=20)
    return ensemble, ensemble.radius_of_gyration()[:, None]


def test_reweight_bme_reuses_cache_only_for_identical_inputs():
    np.random.seed(BME_LAMBDA_SEED)
    ensemble, calculated = rg_ensemble_and_values()
    mean_rg = float(calculated.mean())
    observable = [ExperimentalObservable(mean_rg - 0.5, 0.5)]

    first = ensemble.reweight_bme(observable, calculated, theta=1.0, verbose=False)
    assert ensemble.reweight_bme(observable, calculated, theta=1.0, verbose=False) is first

    new_theta = ensemble.reweight_bme(observable, calculated, theta=10.0, verbose=False)
    assert new_theta is not first and new_theta.theta == 10.0

    new_observable = [ExperimentalObservable(mean_rg + 0.5, 0.5)]
    shifted = ensemble.reweight_bme(new_observable, calculated, theta=10.0, verbose=False)
    assert shifted.observables[0].value == pytest.approx(mean_rg + 0.5)

    # an in-place edit of calculated_values must not be answered from the cache
    calculated *= 1.1
    edited = ensemble.reweight_bme(new_observable, calculated, theta=10.0, verbose=False)
    assert edited is not shifted


def test_reweight_bme_does_not_cache_failed_fit(monkeypatch):
    np.random.seed(BME_LAMBDA_SEED)
    ensemble, calculated = rg_ensemble_and_values()
    observable = [ExperimentalObservable(float(calculated.mean()), 0.5)]
    ensemble.reweight_bme(observable, calculated, theta=1.0, verbose=False)
    assert ensemble.has_bme_weights

    original_run = BME._run_single_theta_optimization
    monkeypatch.setattr(
        BME,
        "_run_single_theta_optimization",
        lambda self, **kwargs: make_failed(original_run(self, **kwargs)),
    )
    with pytest.warns(RuntimeWarning, match="BME optimization failed"):
        failed = ensemble.reweight_bme(observable, calculated, theta=2.0, verbose=False)

    assert not failed.success
    # the earlier result (from different inputs) is dropped, not silently kept
    assert ensemble.bme_result is None
    assert not ensemble.has_bme_weights
    with pytest.raises(ValueError, match="no successful BME reweighting"):
        ensemble.radius_of_gyration(return_mean=True, use_bme_weights=True)


def test_reweight_bme_accepts_explicit_theta_grid(bme_problem):
    np.random.seed(BME_LAMBDA_SEED)
    observables, calculated = bme_problem
    ensemble = random_walk_ensemble(n_frames=len(calculated), n_residues=10)
    theta_grid = [0.05, 0.3, 1.0, 4.0, 9.0]

    result = ensemble.reweight_bme(observables, calculated, theta_range=theta_grid, verbose=False)
    scan = result.metadata["theta_scan_result"]
    np.testing.assert_allclose(scan.theta_values, theta_grid)
    assert result.success


# ---------------------------------------------------------------------------
# BME theta scan
# ---------------------------------------------------------------------------


def test_build_theta_grid_tuple_is_a_range_and_list_is_a_grid():
    np.testing.assert_allclose(build_theta_grid((0.01, 10.0), 4), [0.01, 0.1, 1.0, 10.0])
    np.testing.assert_allclose(build_theta_grid((1.0, 4.0), 4, log_scale=False), [1, 2, 3, 4])
    np.testing.assert_allclose(build_theta_grid([0.5, 0.1, 3.0], 99), [0.5, 0.1, 3.0])
    np.testing.assert_allclose(build_theta_grid(np.array([0.2, 0.4]), 99), [0.2, 0.4])


@pytest.mark.parametrize(
    "theta_range",
    [(0.1, 1.0, 10.0), (0.0, 1.0), (-1.0, 1.0), [1.0], [0.1, 0.0, 1.0], [0.1, np.inf], np.ones((2, 2))],
)
def test_build_theta_grid_rejects_invalid_ranges(theta_range):
    with pytest.raises(ValueError):
        build_theta_grid(theta_range, 5)


def test_theta_scan_excludes_failed_fits_from_knee(bme_problem, monkeypatch):
    observables, calculated = bme_problem
    theta_values = build_theta_grid((0.01, 10.0), 10)

    np.random.seed(BME_LAMBDA_SEED)
    all_success = theta_scan(observables, calculated, theta_range=list(theta_values))
    assert all(res.success for res in all_success.results)

    # fail the fit at the largest theta: before the fix its NaN chi-squared made
    # the knee finders return index 0 regardless of the curve
    failing_theta = theta_values[-1]
    original_fit = BME.fit

    def fit_failing_at_largest_theta(self, *args, theta=None, **kwargs):
        result = original_fit(self, *args, theta=theta, **kwargs)
        if theta is not None and np.isclose(theta, failing_theta):
            return make_failed(result)
        return result

    monkeypatch.setattr(BME, "fit", fit_failing_at_largest_theta)
    np.random.seed(BME_LAMBDA_SEED)
    with pytest.warns(RuntimeWarning, match="1 of 10 BME fits"):
        scan = theta_scan(observables, calculated, theta_range=list(theta_values))

    assert not scan.results[-1].success
    assert np.isnan(scan.kl_divergence_values[-1])
    assert scan.results[scan.optimal_idx].success

    expected_idx, _ = find_optimal_theta(scan.chi_squared_values[:-1], scan.kl_divergence_values[:-1])
    assert scan.optimal_idx == expected_idx
    assert scan.optimal_idx != 0


def test_theta_scan_raises_when_every_fit_fails(bme_problem, monkeypatch):
    observables, calculated = bme_problem
    original_fit = BME.fit
    monkeypatch.setattr(BME, "fit", lambda self, *args, **kwargs: make_failed(original_fit(self, *args, **kwargs)))
    with pytest.raises(RuntimeError, match="All 4 BME fits"):
        theta_scan(observables, calculated, n_points=4)


def test_find_optimal_theta_rejects_non_finite_values():
    chi_squared = np.array([5.0, 2.0, 1.0, np.nan])
    kl_divergence = np.array([0.0, 0.1, 0.5, np.nan])
    with pytest.raises(ValueError, match="must be finite"):
        find_optimal_theta(chi_squared, kl_divergence)


def test_kl_divergence_is_nan_for_failed_fit_and_inf_for_zero_phi(bme_problem):
    observables, calculated = bme_problem
    np.random.seed(BME_LAMBDA_SEED)
    result = BME(observables, calculated).fit(theta=1.0, verbose=False)

    assert result.kl_divergence == pytest.approx(-np.log(result.phi))
    assert np.isnan(make_failed(result).kl_divergence)
    assert dataclasses.replace(result, phi=0.0).kl_divergence == np.inf


def test_single_fit_does_not_pass_deprecated_disp_option(bme_problem):
    observables, calculated = bme_problem
    np.random.seed(BME_LAMBDA_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = BME(observables, calculated).fit(theta=1.0, verbose=True)
    assert result.success


def test_print_summary_lists_n_show_theta_values(bme_problem, capsys):
    observables, calculated = bme_problem
    np.random.seed(BME_LAMBDA_SEED)
    scan = theta_scan(observables, calculated, n_points=6)
    capsys.readouterr()

    scan.print_summary(n_show=3)
    rows = [line for line in capsys.readouterr().out.splitlines() if "✓" in line or "✗" in line]
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# starling_viz
# ---------------------------------------------------------------------------


def test_starling_viz_imports_without_sparrow(monkeypatch):
    # a None entry makes any "import sparrow..." raise ImportError
    monkeypatch.setitem(sys.modules, "sparrow", None)
    monkeypatch.delitem(sys.modules, "starling.frontend.starling_viz", raising=False)
    module = importlib.import_module("starling.frontend.starling_viz")
    assert callable(module.plot_protein_arcs)


@pytest.mark.parametrize("links", [[[0, 5, 0.4]], [[0, 5, 0.4], [2, 9, 0.4]]])
def test_plot_protein_arcs_draws_and_saves_when_contact_scores_are_equal(links, tmp_path):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from starling.frontend.starling_viz import plot_protein_arcs

    filename = tmp_path / "arcs.png"
    returned = plot_protein_arcs("GSGSGSGSGSGS", links, contacts=True, filename=str(filename))
    try:
        assert returned is None
        assert filename.exists()
        assert len(plt.gcf().axes[0].lines) == len(links)
    finally:
        plt.close("all")
