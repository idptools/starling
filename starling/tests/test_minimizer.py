"""
Tests for the Mpipi-GG minimizer (starling.minimizer).

These build small synthetic chains rather than running STARLING, so they are
fast and never need model weights. They check that the parameter set matches
Mpipi-GG, that the batched PyTorch force field matches the plain NumPy
reference implementation and its own autograd gradient, and that relaxation
repairs local geometry while leaving global dimensions alone.
"""

from typing import Any

import numpy as np
import pytest
import torch

from starling.minimizer import (
    DistanceRestraints,
    MpipiGG,
    relax_conformations,
    relax_ensemble,
)
from starling.minimizer.fire import fire_minimize
from starling.minimizer.langevin import (
    BOLTZMANN_KJ_PER_MOL_K,
    DEFAULT_TEMPERATURE_K,
    langevin_thermalize,
)
from starling.minimizer.parameters import (
    BOND_FORCE_CONSTANT,
    BOND_LENGTH,
    COULOMB_CONSTANT,
    KCAL_TO_KJ,
    MPIPI_GG_CHARGES,
    MPIPI_GG_RESIDUES,
    build_sequence_parameters,
    debye_length,
    load_pair_table,
)
from starling.minimizer.potentials import (
    debye_huckel,
    harmonic_bond,
    mpipi_gg_energy,
    wang_frenkel,
)

# a mix of charged, aromatic, aliphatic and proline residues
SEQ = "MKDEYGSPLRRAEWFKDVHQNTSGIC"


def _random_chain(n, seed, bond=BOND_LENGTH, min_distance=6.5):
    """
    Build a random-walk chain with ideal bonds and no close non-bonded pairs.

    Each new bead is placed one bond length from the previous one in a random
    direction, and rejected if it lands within min_distance of any earlier
    bead other than its bonded neighbour. The default of 6.5 A keeps nearly
    every non-bonded pair outside its Wang-Frenkel sigma, so the chain is
    close to something Mpipi-GG is happy with.
    """
    rng = np.random.default_rng(seed)
    xyz = [np.zeros(3)]
    attempts = 0
    while len(xyz) < n:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("could not grow a self-avoiding chain")
        step = rng.normal(size=3)
        candidate = xyz[-1] + bond * step / np.linalg.norm(step)
        if len(xyz) >= 2:
            if np.min(np.linalg.norm(np.array(xyz[:-1]) - candidate, axis=1)) < min_distance:
                continue
        xyz.append(candidate)
    return np.array(xyz)


def _distance_maps(xyz):
    """(n_frames, n, n) distance maps for (n_frames, n, 3) coordinates."""
    return np.linalg.norm(xyz[:, :, None, :] - xyz[:, None, :, :], axis=-1)


def _noisy_ensemble(n_frames=4, noise=0.7, seed=0):
    """
    Ideal chains plus Gaussian noise on every bead.

    The noise mimics what MDS does to STARLING structures: global shape is
    kept, but bond lengths spread out and some beads overlap.
    """
    rng = np.random.default_rng(seed)
    truth = np.stack([_random_chain(len(SEQ), seed + i) for i in range(n_frames)])
    noisy = truth + rng.normal(scale=noise, size=truth.shape)
    return truth, noisy


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------


def test_pair_table_is_symmetric_and_complete():
    table = load_pair_table()
    assert table.residues == MPIPI_GG_RESIDUES
    for name in ("sigma", "epsilon", "mu", "nu"):
        values = getattr(table, name)
        assert values.shape == (20, 20)
        assert np.all(np.isfinite(values))
        assert np.array_equal(values, values.T)
    assert np.all(table.sigma > 0)
    assert np.all(table.epsilon > 0)

    # Mpipi-GG uses mu = 2 and nu = 1 for every amino acid pair
    assert np.all(table.mu == 2.0)
    assert np.all(table.nu == 1.0)


def test_epsilon_is_converted_to_kj_per_mol():
    # M-W is not touched by any of the GG modifications, so it should match
    # Supplementary Table 11 of Joseph et al. (1.233991 kJ/mol, 0.675573 nm)
    table = load_pair_table()
    m, w = MPIPI_GG_RESIDUES.index("M"), MPIPI_GG_RESIDUES.index("W")
    assert table.epsilon[m, w] == pytest.approx(1.233991, abs=1e-5)
    assert table.sigma[m, w] == pytest.approx(6.75573, abs=1e-4)


def test_pair_table_matches_finches():
    finches = pytest.importorskip("finches")
    from finches.forcefields.mpipi import Mpipi_model

    model = Mpipi_model(version="Mpipi_GGv1")
    table = load_pair_table()
    for i, a in enumerate(MPIPI_GG_RESIDUES):
        for j, b in enumerate(MPIPI_GG_RESIDUES):
            assert table.sigma[i, j] == model.SIGMA_ALL[a][b]
            # FINCHES stores epsilon in kcal/mol
            assert table.epsilon[i, j] == pytest.approx(model.EPSILON_ALL[a][b] * KCAL_TO_KJ, rel=1e-12)
            assert table.mu[i, j] == model.MU_ALL[a][b]
            assert table.nu[i, j] == model.NU_ALL[a][b]
        assert MPIPI_GG_CHARGES[a] == model.CHARGE_ALL[a]
    assert finches is not None


def test_charges():
    assert MPIPI_GG_CHARGES["K"] == MPIPI_GG_CHARGES["R"] == 0.75
    assert MPIPI_GG_CHARGES["D"] == MPIPI_GG_CHARGES["E"] == -0.75
    assert MPIPI_GG_CHARGES["H"] == 0.375
    charged = {"K", "R", "D", "E", "H"}
    assert all(MPIPI_GG_CHARGES[r] == 0.0 for r in MPIPI_GG_RESIDUES if r not in charged)


def test_debye_length():
    assert debye_length(150) == pytest.approx(7.9008, abs=1e-3)
    assert debye_length(0) == np.inf
    with pytest.raises(ValueError):
        debye_length(-1)


def test_coulomb_constant_matches_lammps():
    # LAMMPS real units: 332.06371 kcal mol^-1 A e^-2
    assert COULOMB_CONSTANT == pytest.approx(332.06371 * KCAL_TO_KJ, rel=1e-6)


def test_build_sequence_parameters_rejects_invalid_residues():
    with pytest.raises(ValueError):
        build_sequence_parameters("MKXE")
    with pytest.raises(ValueError):
        build_sequence_parameters("")


# ------------------------------------------------------------------------------
# Reference potentials
# ------------------------------------------------------------------------------


def test_wang_frenkel_shape():
    sigma, epsilon = 6.0, 1.5
    assert wang_frenkel(sigma, sigma, epsilon) == pytest.approx(0.0, abs=1e-12)
    assert wang_frenkel(3 * sigma, sigma, epsilon) == pytest.approx(0.0, abs=1e-12)
    assert wang_frenkel(3.5 * sigma, sigma, epsilon) == 0.0

    r = np.linspace(sigma, 3 * sigma, 200001)
    assert wang_frenkel(r, sigma, epsilon).min() == pytest.approx(-epsilon, rel=1e-6)


def test_harmonic_bond_and_debye_huckel():
    assert harmonic_bond(BOND_LENGTH) == 0.0
    assert harmonic_bond(BOND_LENGTH + 1.0) == pytest.approx(40.15)

    # opposite charges attract, like charges repel, and nothing past the cutoff
    assert debye_huckel(5.0, 0.75, -0.75, 7.9) < 0
    assert debye_huckel(5.0, 0.75, 0.75, 7.9) > 0
    assert debye_huckel(36.0, 0.75, 0.75, 7.9) == 0.0


# ------------------------------------------------------------------------------
# PyTorch force field
# ------------------------------------------------------------------------------


def test_torch_energy_matches_numpy_reference():
    _, noisy = _noisy_ensemble(n_frames=3)
    ff = MpipiGG(SEQ, device="cpu")
    energies = ff.energy(torch.as_tensor(noisy))

    for k in range(noisy.shape[0]):
        ref = mpipi_gg_energy(noisy[k], ff.parameters, ff.debye_length)
        for term in ("bond", "wang_frenkel", "debye_huckel", "total"):
            assert float(energies[term][k]) == pytest.approx(ref[term], rel=1e-10)


def test_forces_match_autograd():
    truth, noisy = _noisy_ensemble(n_frames=3)
    ff = MpipiGG(SEQ, device="cpu")
    restraints = DistanceRestraints(_distance_maps(truth), sigma=ff.parameters.sigma, tolerance=0.2)
    index = torch.arange(noisy.shape[0])

    x = torch.as_tensor(noisy).requires_grad_(True)
    ff.energy(x, restraints, index)["total"].sum().backward()
    assert x.grad is not None

    forces = ff.evaluate(torch.as_tensor(noisy), restraints, index).forces
    assert torch.allclose(forces, -x.grad, rtol=1e-8, atol=1e-8)


def test_bonded_neighbours_are_excluded_from_nonbonded_terms():
    # a K-E pair at the bond length would be deep inside the WF wall and
    # strongly attractive under Debye-Huckel if it were not excluded
    ff = MpipiGG("KE", device="cpu")
    x = torch.tensor([[[0.0, 0.0, 0.0], [BOND_LENGTH, 0.0, 0.0]]], dtype=torch.float64)
    energies = ff.energy(x)
    assert float(energies["wang_frenkel"][0]) == 0.0
    assert float(energies["debye_huckel"][0]) == 0.0
    assert float(energies["total"][0]) == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(ff.evaluate(x).forces, torch.zeros_like(x))


def test_forcefield_rejects_bad_shapes():
    ff = MpipiGG(SEQ, device="cpu")
    with pytest.raises(ValueError):
        ff.evaluate(torch.zeros(1, len(SEQ) - 1, 3, dtype=torch.float64))


# ------------------------------------------------------------------------------
# Restraints
# ------------------------------------------------------------------------------


def test_restraints_are_flat_bottomed_and_skip_local_pairs():
    n = 10
    chain = _random_chain(n, seed=3)[None]
    reference = _distance_maps(chain)
    restraints = DistanceRestraints(reference, min_separation=4, tolerance=0.5, force_constant=10.0)
    index = torch.zeros(1, dtype=torch.long)

    # inside the tolerance there is no energy and no force
    within = torch.as_tensor(reference + 0.4)
    dudr, energy = restraints.pair_terms(within, index, compute_energy=True)
    assert energy is not None and float(energy[0]) == 0.0
    assert torch.count_nonzero(dudr) == 0

    # beyond it, only pairs at least min_separation apart are penalised
    beyond = torch.as_tensor(reference + 1.5)
    dudr, energy = restraints.pair_terms(beyond, index, compute_energy=True)
    separation = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    assert torch.all(dudr[0][torch.as_tensor(separation < 4)] == 0)
    assert torch.all(dudr[0][torch.as_tensor(separation >= 4)] > 0)

    k_pair = restraints.pair_force_constant
    expected = 0.5 * k_pair * 1.0**2 * restraints.n_restrained_pairs
    assert energy is not None and float(energy[0]) == pytest.approx(expected)


def test_restraint_reference_floor():
    reference = np.full((1, 6, 6), 2.0)
    sigma = np.full((6, 6), 5.0)
    restraints = DistanceRestraints(reference, sigma=sigma, min_separation=2, tolerance=0.0, reference_floor=1.0)
    index = torch.zeros(1, dtype=torch.long)

    # a pair sitting exactly at sigma is at the (floored) reference
    at_sigma = torch.full((1, 6, 6), 5.0, dtype=torch.float64)
    dudr, _ = restraints.pair_terms(at_sigma, index)
    assert torch.count_nonzero(dudr) == 0


# ------------------------------------------------------------------------------
# Minimization and relaxation
# ------------------------------------------------------------------------------


def test_fire_restores_bond_lengths():
    # a straight chain with every bond compressed to 3 A
    n = 12
    x0 = torch.zeros(1, n, 3, dtype=torch.float64)
    x0[0, :, 0] = 3.0 * torch.arange(n, dtype=torch.float64)
    ff = MpipiGG("G" * n, device="cpu")

    result = fire_minimize(x0, lambda x, index: ff.evaluate(x).forces, force_tolerance=0.01)
    assert bool(result.converged[0])

    bonds = torch.linalg.vector_norm(torch.diff(result.coordinates[0], dim=0), dim=-1)
    assert torch.allclose(bonds, torch.full_like(bonds, BOND_LENGTH), atol=0.01)


def test_relaxation_repairs_local_geometry_and_keeps_global_shape():
    truth, noisy = _noisy_ensemble(n_frames=4)
    # minimization only: thermalization deliberately spreads bonds back out
    result = relax_conformations(
        noisy,
        SEQ,
        reference_distances=_distance_maps(truth),
        device="cpu",
        progress_bar=False,
        thermalization_steps=0,
    )

    assert result.reference == "distance_map"
    assert result.converged.all()
    assert np.all(result.max_force < 1.0)

    # local geometry is repaired...
    assert np.all(result.before.max_bond_deviation > 0.5)
    assert np.all(result.after.max_bond_deviation < 0.25)
    assert np.all(result.after.n_clashes == 0)
    assert np.all(result.energy_after["total"] < result.energy_before["total"])

    # ...while global dimensions stay at those of the reference structures
    # (the noise itself shifts Rg slightly, so compare against the reference)
    centred = truth - truth.mean(axis=1, keepdims=True)
    rg_truth = np.sqrt((centred**2).sum(axis=-1).mean(axis=-1))
    assert np.all(np.abs(result.after.radius_of_gyration / rg_truth - 1) < 0.01)
    assert np.all(result.after.long_range_rmsd < result.before.long_range_rmsd)


def test_relaxation_without_a_reference_map():
    _, noisy = _noisy_ensemble(n_frames=2, seed=10)
    result = relax_conformations(noisy, SEQ, device="cpu", progress_bar=False, thermalization_steps=0)

    assert result.reference == "coordinates"
    assert np.all(result.before.long_range_rmsd == 0.0)
    assert np.all(result.after.max_bond_deviation < 0.25)
    assert np.all(result.after.long_range_rmsd < 1.0)


def test_relaxation_is_independent_of_batching():
    # minimization only: thermalization noise depends on how frames are batched
    _, noisy = _noisy_ensemble(n_frames=5, seed=20)
    kwargs: dict[str, Any] = dict(device="cpu", progress_bar=False, thermalization_steps=0)
    together = relax_conformations(noisy, SEQ, **kwargs)
    apart = relax_conformations(noisy, SEQ, batch_size=2, **kwargs)
    assert np.allclose(together.coordinates, apart.coordinates, atol=1e-8)
    assert np.array_equal(together.n_steps, apart.n_steps)


def test_relax_result_summary_and_trajectory():
    _, noisy = _noisy_ensemble(n_frames=2, seed=30)
    result = relax_conformations(noisy, SEQ, device="cpu", progress_bar=False)

    summary = result.summary()
    assert "radius of gyration" in summary
    assert "2/2" in summary

    traj = result.to_trajectory()
    assert traj.n_frames == 2
    assert traj.n_atoms == len(SEQ)
    # MDTraj works in nm
    assert np.allclose(traj.xyz * 10.0, result.coordinates, atol=1e-4)


@pytest.mark.parametrize(
    "recorded, override, expected",
    [(300, None, 300), (300, 50, 50), (300, 0, 0), (0, None, 0), (None, None, 150)],
)
def test_relax_ensemble_returns_a_new_ensemble(recorded, override, expected):
    from soursop.sstrajectory import SSTrajectory

    from starling.structure.coordinates import create_ca_topology_from_coords
    from starling.structure.ensemble import Ensemble

    truth, noisy = _noisy_ensemble(n_frames=3, seed=40)
    maps = _distance_maps(truth)
    protein = SSTrajectory(TRJ=create_ca_topology_from_coords(SEQ, noisy / 10.0)).proteinTrajectoryList[0]
    ensemble = Ensemble(maps, SEQ, ssprot_ensemble=protein, ionic_strength=recorded)

    relaxed, result = relax_ensemble(ensemble, ionic_strength=override, device="cpu", progress_bar=False, seed=0)

    assert relaxed is not ensemble
    assert relaxed.has_structures
    assert np.array_equal(relaxed.distance_maps(), maps)
    assert np.allclose(relaxed.trajectory.traj.xyz * 10.0, result.coordinates, atol=1e-4)
    assert result.settings["ionic_strength"] == expected
    assert relaxed.ionic_strength == expected
    assert ensemble.ionic_strength == recorded
    # the original ensemble keeps its unrelaxed structures
    assert np.allclose(ensemble.trajectory.traj.xyz * 10.0, noisy, atol=1e-4)
    assert result.reference == "distance_map"


def test_relax_conformations_rejects_bad_input():
    _, noisy = _noisy_ensemble(n_frames=2, seed=50)
    with pytest.raises(ValueError):
        relax_conformations(noisy[:, :-1], SEQ, device="cpu", progress_bar=False)

    bad = noisy.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        relax_conformations(bad, SEQ, device="cpu", progress_bar=False)

    with pytest.raises(ValueError):
        relax_conformations(
            noisy,
            SEQ,
            reference_distances=np.zeros((1, len(SEQ), len(SEQ))),
            device="cpu",
            progress_bar=False,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_matches_cpu():
    truth, noisy = _noisy_ensemble(n_frames=3, seed=60)
    maps = _distance_maps(truth)
    # minimization only: CPU and MPS draw different thermalization noise
    kwargs: dict[str, Any] = dict(reference_distances=maps, progress_bar=False, thermalization_steps=0)
    cpu = relax_conformations(noisy, SEQ, device="cpu", **kwargs)
    mps = relax_conformations(noisy, SEQ, device="mps", **kwargs)

    assert mps.converged.all()
    assert np.allclose(mps.after.radius_of_gyration, cpu.after.radius_of_gyration, atol=0.05)
    assert np.allclose(mps.after.mean_bond_length, cpu.after.mean_bond_length, atol=0.01)


# ------------------------------------------------------------------------------
# Thermalization
# ------------------------------------------------------------------------------


def _bond_lengths(xyz):
    """Consecutive-bead distances (A) for conformations of shape (n_frames, n, 3) in A."""
    return np.linalg.norm(np.diff(np.asarray(xyz), axis=1), axis=-1)


def _thermal_bond_std(temperature_K=DEFAULT_TEMPERATURE_K):
    """Equipartition spread sqrt(kT / k) of a harmonic Mpipi-GG bond, in A."""
    return np.sqrt(BOLTZMANN_KJ_PER_MOL_K * temperature_K / BOND_FORCE_CONSTANT)


# thermal fluctuations lengthen the mean bond beyond the 3.81 A minimum
# (Mpipi-GG simulations of 200 natural IDRs at 300 K average 3.854 A)
THERMAL_MEAN_BOND_RANGE = (3.80, 3.90)


def _assert_thermal_bonds(bonds):
    """Bonds (A) look like a 300 K Mpipi-GG chain: right mean, thermal spread."""
    low, high = THERMAL_MEAN_BOND_RANGE
    assert low < bonds.mean() < high
    assert bonds.std() == pytest.approx(_thermal_bond_std(), rel=0.2)


def test_langevin_samples_the_harmonic_boltzmann_distribution():
    # independent 3D springs: each coordinate's variance must be kT / k
    generator = torch.Generator().manual_seed(0)
    x0 = torch.zeros(20000, 1, 3, dtype=torch.float64)
    x = langevin_thermalize(
        x0,
        lambda x, index: -BOND_FORCE_CONSTANT * x,
        n_steps=500,
        generator=generator,
    )
    assert x.std().item() == pytest.approx(_thermal_bond_std(), rel=0.03)


def test_langevin_rejects_bad_settings_and_handles_zero_steps():
    x0 = torch.zeros(2, 4, 3, dtype=torch.float64)
    force = lambda x, index: -x  # noqa: E731

    unchanged = langevin_thermalize(x0, force, n_steps=0)
    assert torch.equal(unchanged, x0) and unchanged is not x0

    with pytest.raises(ValueError):
        langevin_thermalize(x0, force, n_steps=-1)
    with pytest.raises(ValueError):
        langevin_thermalize(x0, force, n_steps=10, temperature_K=0.0)
    with pytest.raises(ValueError):
        langevin_thermalize(x0[0], force, n_steps=10)


def test_thermalization_restores_thermal_bond_spread():
    truth, noisy = _noisy_ensemble(n_frames=20, seed=70)
    kwargs: dict[str, Any] = dict(reference_distances=_distance_maps(truth), device="cpu", progress_bar=False)
    minimized = relax_conformations(noisy, SEQ, thermalization_steps=0, **kwargs)
    thermalized = relax_conformations(noisy, SEQ, seed=0, **kwargs)

    minimized_bonds = _bond_lengths(minimized.coordinates)
    thermal_bonds = _bond_lengths(thermalized.coordinates)

    # minimization leaves bonds far narrower than a 300 K Mpipi-GG chain...
    assert minimized_bonds.std() < 0.25 * _thermal_bond_std()
    # ...and thermalization gives them the equipartition spread back
    _assert_thermal_bonds(thermal_bonds)

    # the restraints still hold the global shape of the reference structures
    centred = truth - truth.mean(axis=1, keepdims=True)
    rg_truth = np.sqrt((centred**2).sum(axis=-1).mean(axis=-1))
    assert np.all(np.abs(thermalized.after.radius_of_gyration / rg_truth - 1) < 0.03)
    assert thermalized.settings["thermalization_steps"] == 250


def test_thermalization_is_reproducible_with_a_seed():
    _, noisy = _noisy_ensemble(n_frames=2, seed=80)
    kwargs: dict[str, Any] = dict(device="cpu", progress_bar=False, thermalization_steps=50)
    first = relax_conformations(noisy, SEQ, seed=1, **kwargs)
    again = relax_conformations(noisy, SEQ, seed=1, **kwargs)
    other = relax_conformations(noisy, SEQ, seed=2, **kwargs)

    assert np.array_equal(first.coordinates, again.coordinates)
    assert not np.allclose(first.coordinates, other.coordinates)

    with pytest.raises(ValueError):
        relax_conformations(noisy, SEQ, device="cpu", thermalization_steps=-1)


# ------------------------------------------------------------------------------
# Integration with generate() and the CLI
# ------------------------------------------------------------------------------


def test_generate_rejects_non_bool_relax():
    from starling.frontend.ensemble_generation import generate

    with pytest.raises(ValueError, match="relax must be True or False"):
        generate(SEQ, relax="yes", output_directory=".")


def test_generate_relaxes_by_default():
    import inspect

    from starling.frontend.ensemble_generation import generate

    assert inspect.signature(generate).parameters["relax"].default is True


@pytest.mark.parametrize("sampler", ["ddpm", "ddim", "dpmpp", "plms"])
@pytest.mark.parametrize("salt", [0, 300])
@pytest.mark.parametrize("return_data", [False, True])
def test_generate_records_and_conditions_on_ionic_strength(monkeypatch, tmp_path, sampler, salt, return_data):
    from types import SimpleNamespace

    from starling.inference import generation
    from starling.data.schedulers import cosine_beta_schedule
    from starling import generate, load_ensemble

    sequence = "ACDE"
    map_ = np.array(
        [
            [
                [0.0, 3.8, 7.6, 11.4],
                [3.8, 0.0, 3.8, 7.6],
                [7.6, 3.8, 0.0, 3.8],
                [11.4, 7.6, 3.8, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    conditions = []

    def sequence2labels(tokens, mask, ionic_strength):
        conditions.append(ionic_strength.clone())
        return tokens

    betas = cosine_beta_schedule(1000)
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    previous = torch.cat([torch.ones(1), alpha_bar[:-1]])
    diffusion = SimpleNamespace(
        device=torch.device("cpu"),
        num_timesteps=1000,
        betas=betas,
        alphas_cumprod=alpha_bar,
        alphas_cumprod_prev=previous,
        sqrt_recip_alphas=(1 - betas).rsqrt(),
        sqrt_one_minus_alphas_cumprod=(1 - alpha_bar).sqrt(),
        posterior_variance=betas * (1 - previous) / (1 - alpha_bar),
        latent_space_scaling_factor=1.0,
        sequence2labels=sequence2labels,
        model=lambda x, t, context, mask: torch.zeros_like(x),
    )
    decoder = SimpleNamespace(decode=lambda x: torch.from_numpy(map_)[:, None].repeat(len(x), 1, 1, 1))
    monkeypatch.setattr(generation.model_manager, "get_models", lambda **kwargs: (decoder, diffusion))

    ensembles = generate(
        {"test": sequence},
        conformations=1,
        device=torch.device("cpu"),
        steps=12,
        sampler=sampler,
        return_structures=False,
        batch_size=1,
        output_directory=str(tmp_path),
        return_data=return_data,
        verbose=False,
        show_progress_bar=False,
        show_per_step_progress_bar=False,
        pdb_trajectory=False,
        ionic_strength=salt,
    )

    if return_data:
        assert ensembles["test"].ionic_strength == salt
    else:
        assert ensembles is None
    saved = load_ensemble(str(tmp_path / "test.starling"))
    assert saved.ionic_strength == salt
    assert conditions, "The real sampler must condition the network"
    for condition in conditions:
        torch.testing.assert_close(condition, torch.tensor([[salt]]))
    np.testing.assert_array_equal(saved.distance_maps(), map_)


def _run_cli(*cli_args):
    """Run the starling CLI in a subprocess and return the CompletedProcess."""
    import subprocess
    import sys

    argv = ["starling", *cli_args]
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.argv={argv!r}; from starling.scripts.starling_main_cli import main; main()",
        ],
        capture_output=True,
        text=True,
    )


def test_cli_exposes_relax_flag():
    result = _run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "--relax" in result.stdout


def test_cli_relax_requires_return_structures(tmp_path):
    # this fails in the argument checks, before any model is loaded
    result = _run_cli(SEQ, "--relax", "-o", str(tmp_path))
    assert result.returncode == 1
    assert "requires -r/--return_structures" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_relax_coordinates_works_in_nanometres():
    from starling.inference.generation import _relax_coordinates

    truth, noisy = _noisy_ensemble(n_frames=2, seed=70)
    relaxed = _relax_coordinates(
        SEQ,
        noisy / 10.0,
        _distance_maps(truth),
        ionic_strength=150,
        device="cpu",
        batch_size=100,
        show_progress_bar=False,
    )

    assert relaxed.shape == noisy.shape
    # thermalization spreads individual bonds, so check the mean: a units slip
    # would put it out by a factor of 10
    bonds = np.linalg.norm(np.diff(relaxed, axis=1), axis=-1)
    assert bonds.mean() == pytest.approx(BOND_LENGTH / 10.0, abs=0.02)


def test_error_filter_screens_the_relaxed_structures(monkeypatch):
    """
    With relax on, the 3D error screen must see the relaxed structures.

    In the first round, the stand-in for relaxation stretches every second
    conformer far past the physical bound, so if (and only if) the screen runs
    after relaxation those conformers are discarded and topped up.
    """
    from starling.inference import generation

    seq = "MKTAYIAKQRQ"
    n = len(seq)
    straight = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) * 3.8
    sampled = []
    relax_calls = []

    def fake_sample(sampler, sequence, conformations, *args, **kwargs):
        sampled.append(conformations)
        return torch.from_numpy(np.stack([straight] * conformations))

    def fake_relax(sequence, coordinates, distance_maps, *args, **kwargs):
        relax_calls.append(len(coordinates))
        out = np.array(coordinates, dtype=np.float64)
        if len(relax_calls) == 1:
            out[::2] *= 5.0
        return out

    monkeypatch.setattr(generation, "_sample_distance_maps", fake_sample)
    monkeypatch.setattr(generation, "_relax_coordinates", fake_relax)

    def run(relax):
        return generation._generate_error_filtered_conformers(
            sampler=None,
            sequence=seq,
            conformations=4,
            batch_size=100,
            show_per_step_progress_bar=False,
            constraint=None,
            return_structures=True,
            device="cpu",
            show_progress_bar=False,
            relax=relax,
        )

    maps, coords, discarded = run(relax=True)
    assert len(maps) == len(coords) == 4
    assert discarded == 2
    assert relax_calls == [4, 2], "every reconstructed conformer is relaxed"
    assert sampled == [4, 2]

    relax_calls.clear()
    maps, coords, discarded = run(relax=False)
    assert relax_calls == []
    assert discarded == 0


@pytest.mark.slow
def test_generate_relaxes_structures_end_to_end():
    from starling import generate

    kwargs = dict(
        conformations=8,
        return_structures=True,
        return_single_ensemble=True,
        show_progress_bar=False,
        show_per_step_progress_bar=False,
    )
    relaxed = generate(SEQ, **kwargs)

    # MDTraj stores coordinates in nm
    bonds = _bond_lengths(relaxed.trajectory.traj.xyz * 10.0)

    # relaxed bonds sit at the Mpipi-GG length with the thermal spread of a
    # 300 K simulation. Raw weighted SMACOF (~3.73 +/- 0.08 A for this
    # sequence) and minimization alone (+/- ~0.03 A) both fail the spread check
    _assert_thermal_bonds(bonds)


@pytest.mark.slow
def test_relaxation_repairs_unweighted_smacof_structures():
    """
    Relaxation still repairs structures from unweighted SMACOF.

    Until October 2026 STARLING reconstructed 3D structures with unweighted
    SMACOF, which compresses bonds (~3.2 A) and occasionally breaks the chain.
    The minimizer was built to repair those structures, so we keep checking
    that it can, using the unweighted stress distance_matrix_to_3d_structure_torch_mds
    gives when weights is None. Note this starts from classical MDS rather
    than the random starts the old reconstruction used, which is the only
    starting point the current code offers.
    """
    from starling import generate
    from starling.structure.coordinates import distance_matrix_to_3d_structure_torch_mds

    distance_maps = generate(
        SEQ,
        conformations=16,
        return_structures=False,
        return_single_ensemble=True,
        show_progress_bar=False,
        show_per_step_progress_bar=False,
    ).distance_maps()

    raw, _ = distance_matrix_to_3d_structure_torch_mds(distance_maps, device="cpu", progress_bar=False, weights=None)
    result = relax_conformations(
        raw,
        SEQ,
        reference_distances=distance_maps,
        device="cpu",
        progress_bar=False,
    )

    # control: unweighted stress compresses bonds (measured 3.35-3.44 A for
    # this sequence), otherwise the checks below would show nothing
    assert _bond_lengths(raw).mean() < BOND_LENGTH - 0.25

    # relaxation restores the bonds to the Mpipi-GG length with thermal spread...
    relaxed_bonds = _bond_lengths(result.coordinates)
    _assert_thermal_bonds(relaxed_bonds)

    # ...without changing global dimensions much. Thermal motion moves Rg a
    # little: on 4000 natural-IDR conformations mean |dRg| was 0.3%, max 3.3%
    rg_change = np.abs(result.after.radius_of_gyration / result.before.radius_of_gyration - 1)
    assert rg_change.mean() < 0.01
    assert np.all(rg_change < 0.05)
