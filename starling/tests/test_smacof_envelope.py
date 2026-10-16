"""
Quality envelope for SMACOF reconstruction across lengths and ensemble dimensions.

SMACOF's settings (classical-MDS starting point, iteration budget, stopping
tolerance and map-error weights) were chosen empirically, and are known to work
for highly compact through highly expanded conformers. These tests pin that
down. They sample STARLING ensembles for sequences spanning the full range of
lengths and of length-normalized ensemble dimensions, reconstruct them exactly
as generate() does before each relaxation mode ("standard" and "fast"), and
check that each case still fits its distance maps as well as it did when the
envelope was measured (2026-10-10). Any change to the SMACOF settings must keep
passing them; a change that makes any case worse needs a deliberate decision
and a re-measured envelope.

Each case is held to its own reference values rather than to one global limit.
Fit errors span a factor of 20 across the range (expanded chains fit far better
than compact ones), so a global limit set by the worst case would let the
expanded end degrade several-fold unnoticed.

The fast mode is checked where it differs from the standard one, before
relaxation: its classical-MDS starting point must match the full
eigendecomposition's, and its shorter SMACOF run must stay within its own
envelope. After relaxation the two modes are compared directly, but that check
is coarse. Relaxation refits every structure to its distance map, so even a
random SMACOF start, or a single SMACOF iteration, relaxed to within 6% of the
standard structures' fit.
"""

import numpy as np
import pytest
import torch

from starling import configs
from starling.frontend.ensemble_generation import generate
from starling.inference import generation
from starling.inference.generation import (
    FAST_RELAX_SMACOF_ITERATIONS,
    RELAX_FAST,
    RELAX_MODES,
    RELAX_STANDARD,
    _reconstruct_coordinates,
    _smacof_settings,
)
from starling.structure.coordinates import (
    DEFAULT_SMACOF_ITERATIONS,
    FULL_EIGENDECOMPOSITION,
    SMACOF_FLOAT32_MATMUL_PRECISION,
    SUBSPACE_ITERATION,
    _classical_mds,
    distance_matrix_to_3d_structure_torch_mds,
    generate_3d_coordinates_from_distances,
    smacof_converged,
)
from starling.structure.weighted_stress import map_error_weights
from starling.utilities import check_device, float32_matmul_precision

# Sequence lengths tested, from short to the longest STARLING accepts
LENGTHS = (20, 100, 250, configs.MAX_SEQUENCE_LENGTH)

# Repeat motifs whose STARLING ensembles span compact to expanded. Measured
# length-normalized Rg (Rg / sqrt(N), Angstroms) at 380 residues: WG 1.09,
# SYGQ 1.26, GS 2.81, poly-E 5.90.
MOTIFS = ("WG", "SYGQ", "GS", "E")
MOST_COMPACT_MOTIF = "WG"
MOST_EXPANDED_MOTIF = "E"

# Conformations sampled per (length, motif)
CONFORMATIONS_PER_CASE = 12

# Maps are sampled on the CPU with a fixed seed. CPU and CUDA draw different
# random numbers for the same seed, and the CPU stream is the same on every
# machine, so every machine checks the same maps. SMACOF itself runs on the
# default device, which is what is under test.
SAMPLING_SEED = 0
SAMPLING_DEVICE = "cpu"

# Reference quality of each case, averaged over its conformations, measured
# with the standard settings of 2026-10-10:
#   relative RMS distance error, sqrt(sum (d - D)^2 / sum D^2) over all pairs
#   relative weighted stress, the same with the map-error weights SMACOF fits
#   mean |bonded distance - map distance|, in Angstroms (before relaxation)
# SMACOF on CPU and on CUDA gave the same averages to within 0.04%.
REFERENCE_QUALITY = {
    ("WG", 20): (0.0416, 0.0408, 0.113),
    ("SYGQ", 20): (0.0372, 0.0369, 0.100),
    ("GS", 20): (0.0379, 0.0376, 0.093),
    ("E", 20): (0.0174, 0.0180, 0.067),
    ("WG", 100): (0.0472, 0.0498, 0.398),
    ("SYGQ", 100): (0.0281, 0.0305, 0.349),
    ("GS", 100): (0.0148, 0.0167, 0.312),
    ("E", 100): (0.0054, 0.0061, 0.206),
    ("WG", 250): (0.0707, 0.0725, 0.547),
    ("SYGQ", 250): (0.0522, 0.0542, 0.537),
    ("GS", 250): (0.0181, 0.0195, 0.449),
    ("E", 250): (0.0065, 0.0070, 0.351),
    ("WG", 380): (0.0995, 0.1009, 0.566),
    ("SYGQ", 380): (0.0749, 0.0766, 0.560),
    ("GS", 380): (0.0217, 0.0228, 0.475),
    ("E", 380): (0.0099, 0.0102, 0.399),
}

# Reference quality of the SMACOF structures that relax="fast" relaxes
# (subspace-iteration start, FAST_RELAX_SMACOF_ITERATIONS iterations),
# measured the same way on 2026-10-10. They fit the maps exactly as well as the
# standard structures; only their bonds are rougher before relaxation (bond
# error 4-28% higher), which relaxation repairs. CPU and CUDA agreed to within
# 0.2%.
REFERENCE_QUALITY_FAST = {
    ("WG", 20): (0.0421, 0.0417, 0.129),
    ("SYGQ", 20): (0.0376, 0.0376, 0.114),
    ("GS", 20): (0.0384, 0.0384, 0.107),
    ("E", 20): (0.0179, 0.0189, 0.086),
    ("WG", 100): (0.0471, 0.0498, 0.411),
    ("SYGQ", 100): (0.0280, 0.0306, 0.365),
    ("GS", 100): (0.0148, 0.0168, 0.325),
    ("E", 100): (0.0054, 0.0062, 0.225),
    ("WG", 250): (0.0707, 0.0725, 0.562),
    ("SYGQ", 250): (0.0522, 0.0542, 0.555),
    ("GS", 250): (0.0181, 0.0196, 0.464),
    ("E", 250): (0.0065, 0.0070, 0.365),
    ("WG", 380): (0.0995, 0.1009, 0.590),
    ("SYGQ", 380): (0.0749, 0.0766, 0.585),
    ("GS", 380): (0.0217, 0.0229, 0.493),
    ("E", 380): (0.0099, 0.0103, 0.418),
}

REFERENCE_QUALITY_BY_RELAX_MODE = {RELAX_STANDARD: REFERENCE_QUALITY, RELAX_FAST: REFERENCE_QUALITY_FAST}

# How much worse than its reference a case may get. Far above the 0.04%
# hardware spread, and below the degradations this must catch: cutting SMACOF
# to 20 iterations worsened bond errors by 30-150%, and a random starting point
# worsened the fit of long expanded chains by 14-52%.
RELATIVE_MARGIN = 0.10

# Per-conformation backstops, so a single bad conformation cannot hide in an
# average (worst measured: RMS error 0.14, Rg error 0.008, bond error 0.62 A)
MAX_RELATIVE_RMS_DISTANCE_ERROR = 0.20
MAX_RELATIVE_RG_ERROR = 0.02
MAX_MEAN_BOND_ERROR_A = 0.8

# The tests only mean something if the sequences really span compact to
# expanded. From 100 residues up, the measured length-normalized Rg was at
# most 1.37 A for WG and at least 4.73 A for poly-E.
MIN_LENGTH_FOR_COVERAGE_CHECK = 100
MAX_COMPACT_NORMALIZED_RG_A = 1.6
MIN_EXPANDED_NORMALIZED_RG_A = 4.0

# The fast mode's classical-MDS starting structures must match the full
# eigendecomposition's: largest change in any inter-residue distance, relative
# to the largest map distance. Measured at most 1.5e-4 on STARLING maps (float32
# rounding on CUDA) and 9.5e-4 on noisy random-walk chains whose 3rd to 9th
# eigenvalues are close. An unconverged start (4 subspace iterations rather
# than 20) was 2.5e-2 off on STARLING maps.
MAX_RELATIVE_START_DIFFERENCE = 2e-3

# Seed for the thermalization when relaxing with either mode
RELAXATION_SEED = 0

# How far the relaxed ensembles of the two modes may differ, per case. Over 16
# cases and 5 relaxation seeds the largest differences were, at 20 residues
# (and from 100 residues up): fit +3.7% (+0.9%), mean Rg 1.2% (0.2%), mean
# bond length 0.04 A (0.01 A), mean closest non-bonded distance 0.18 A
# (0.12 A). The fit is held to RELATIVE_MARGIN like the envelope above.
MAX_RELAXED_RG_DIFFERENCE = 0.03
MAX_RELAXED_BOND_DIFFERENCE_A = 0.1
MAX_RELAXED_CONTACT_DIFFERENCE_A = 0.4


def radius_of_gyration_from_maps(distance_maps: np.ndarray) -> np.ndarray:
    """
    Radius of gyration of each conformation, from its distance map.

    Uses Rg^2 = sum_ij r_ij^2 / (2 N^2), which holds for equal-mass beads.

    Parameters
    ----------
    distance_maps : np.ndarray
        Distance maps in Angstroms, shape (n_conformations, N, N).

    Returns
    -------
    np.ndarray
        Rg in Angstroms, shape (n_conformations,).
    """
    n_residues = distance_maps.shape[-1]
    return np.sqrt((distance_maps**2).sum(axis=(1, 2)) / (2 * n_residues**2))


def radius_of_gyration_from_coordinates(coordinates_a: np.ndarray) -> np.ndarray:
    """
    Radius of gyration of each conformation, from its coordinates.

    Parameters
    ----------
    coordinates_a : np.ndarray
        Coordinates in Angstroms, shape (n_conformations, N, 3).

    Returns
    -------
    np.ndarray
        Rg in Angstroms, shape (n_conformations,).
    """
    centred = coordinates_a - coordinates_a.mean(axis=1, keepdims=True)
    return np.sqrt((centred**2).sum(axis=-1).mean(axis=1))


def repeat_sequence(motif: str, n_residues: int) -> str:
    """
    A sequence of n_residues made by repeating a motif.

    Parameters
    ----------
    motif : str
        Amino-acid motif to repeat.
    n_residues : int
        Sequence length; the last repeat is cut short if needed.

    Returns
    -------
    str
        The repeat sequence.
    """
    return (motif * n_residues)[:n_residues]


def pairwise_distances(coordinates_a: np.ndarray) -> np.ndarray:
    """
    All inter-residue distances of each conformation.

    Parameters
    ----------
    coordinates_a : np.ndarray
        Coordinates in Angstroms, shape (n_conformations, N, 3).

    Returns
    -------
    np.ndarray
        Distances in Angstroms, shape (n_conformations, N, N).
    """
    return np.linalg.norm(coordinates_a[:, :, None] - coordinates_a[:, None], axis=-1)


def mean_bond_length_a(coordinates_a: np.ndarray) -> float:
    """
    Mean distance between sequence neighbours over a whole ensemble.

    Parameters
    ----------
    coordinates_a : np.ndarray
        Coordinates in Angstroms, shape (n_conformations, N, 3).

    Returns
    -------
    float
        Mean bond length in Angstroms.
    """
    return float(np.linalg.norm(np.diff(coordinates_a, axis=1), axis=-1).mean())


def closest_nonbonded_distance_a(coordinates_a: np.ndarray) -> np.ndarray:
    """
    Distance between the closest pair of non-bonded residues (|i - j| >= 2) in each conformation.

    Parameters
    ----------
    coordinates_a : np.ndarray
        Coordinates in Angstroms, shape (n_conformations, N, 3), with N >= 3.

    Returns
    -------
    np.ndarray
        Closest non-bonded distance in Angstroms, shape (n_conformations,).
    """
    n_residues = coordinates_a.shape[1]
    separation = np.abs(np.subtract.outer(np.arange(n_residues), np.arange(n_residues)))
    distances = pairwise_distances(coordinates_a)
    return np.where(separation >= 2, distances, np.inf).min(axis=(1, 2))


def reconstruction_quality(distance_maps: np.ndarray, coordinates_a: np.ndarray) -> dict[str, np.ndarray]:
    """
    Measure how well each reconstructed structure fits its distance map.

    Parameters
    ----------
    distance_maps : np.ndarray
        Distance maps in Angstroms, shape (n_conformations, N, N).
    coordinates_a : np.ndarray
        Reconstructed coordinates in Angstroms, shape (n_conformations, N, 3).

    Returns
    -------
    dict of str to np.ndarray
        Per-conformation arrays, each of shape (n_conformations,):
        "rms" (relative RMS distance error over all pairs), "weighted_stress"
        (the same with SMACOF's map-error weights), "rg" (relative Rg error)
        and "bond" (mean bonded-distance error in Angstroms).
    """
    n_residues = distance_maps.shape[-1]
    distances = pairwise_distances(coordinates_a)
    error = distances - distance_maps

    upper_i, upper_j = np.triu_indices(n_residues, k=1)
    rms = np.sqrt(
        (error[:, upper_i, upper_j] ** 2).sum(axis=1) / (distance_maps[:, upper_i, upper_j] ** 2).sum(axis=1)
    )

    weights = map_error_weights(n_residues).numpy().astype(np.float64)
    weighted_stress = np.sqrt((weights * error**2).sum(axis=(1, 2)) / (weights * distance_maps**2).sum(axis=(1, 2)))

    rg_map = radius_of_gyration_from_maps(distance_maps)
    rg = np.abs(radius_of_gyration_from_coordinates(coordinates_a) - rg_map) / rg_map

    bond = np.abs(np.diagonal(error, 1, 1, 2)).mean(axis=1)
    return {"rms": rms, "weighted_stress": weighted_stress, "rg": rg, "bond": bond}


@pytest.fixture(scope="module")
def sample_maps():
    """
    Sample each (length, motif) case once, for every test in this module.

    Returns
    -------
    callable
        sample_maps(n_residues, motif) -> distance maps in Angstroms (float64),
        shape (CONFORMATIONS_PER_CASE, n_residues, n_residues).
    """
    cache: dict[tuple[int, str], np.ndarray] = {}

    def sample(n_residues: int, motif: str) -> np.ndarray:
        key = (n_residues, motif)
        if key not in cache:
            ensemble = generate(
                repeat_sequence(motif, n_residues),
                conformations=CONFORMATIONS_PER_CASE,
                seed=SAMPLING_SEED,
                device=SAMPLING_DEVICE,
                return_structures=False,
                return_single_ensemble=True,
                show_progress_bar=False,
                show_per_step_progress_bar=False,
            )
            cache[key] = ensemble.distance_maps().astype(np.float64)
        return cache[key]

    return sample


@pytest.fixture(scope="module")
def reconstruct(sample_maps):
    """
    Reconstruct each case with the SMACOF settings generate() uses before a relaxation mode, without relaxing.

    Returns
    -------
    callable
        reconstruct(n_residues, motif, relax_mode) -> (distance_maps,
        coordinates_a), the sampled maps and their SMACOF structures, both in
        Angstroms.
    """
    smacof_device = check_device(None)

    def smacof_structures(n_residues: int, motif: str, relax_mode: str) -> tuple[np.ndarray, np.ndarray]:
        distance_maps = sample_maps(n_residues, motif)
        initialization, n_iter = _smacof_settings(relax_mode)
        coordinates_a, _ = distance_matrix_to_3d_structure_torch_mds(
            distance_maps,
            batch_size=configs.DEFAULT_BATCH_SIZE,
            n_iter=n_iter,
            device=smacof_device,
            progress_bar=False,
            weights=map_error_weights(n_residues, device=smacof_device),
            initialization=initialization,
        )
        return distance_maps, coordinates_a.astype(np.float64)

    return smacof_structures


@pytest.fixture(scope="module")
def relax(sample_maps):
    """
    Reconstruct and relax each case exactly as generate() does in a relaxation mode.

    Returns
    -------
    callable
        relax(n_residues, motif, relax_mode) -> (distance_maps, coordinates_a),
        the sampled maps and their relaxed structures, both in Angstroms.
        Both modes use the same relaxation seed.
    """
    device = check_device(None)

    def relaxed_structures(n_residues: int, motif: str, relax_mode: str) -> tuple[np.ndarray, np.ndarray]:
        distance_maps = sample_maps(n_residues, motif)
        coordinates_nm = _reconstruct_coordinates(
            repeat_sequence(motif, n_residues),
            distance_maps,
            configs.DEFAULT_IONIC_STRENGTH,
            device,
            configs.DEFAULT_BATCH_SIZE,
            show_progress_bar=False,
            relax=relax_mode,
            verbose=False,
            relaxation_seed=RELAXATION_SEED,
        )
        return distance_maps, coordinates_nm.astype(np.float64) * configs.CONVERT_ANGSTROM_TO_NM

    return relaxed_structures


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.parametrize("relax_mode", RELAX_MODES)
@pytest.mark.parametrize("n_residues", LENGTHS)
@pytest.mark.parametrize("motif", MOTIFS)
def test_reconstruction_stays_within_its_envelope(reconstruct, n_residues, motif, relax_mode):
    """Before relaxation, each case fits its maps as well as its reference in each mode, and no conformation is far off."""
    distance_maps, coordinates_a = reconstruct(n_residues, motif, relax_mode)
    assert np.isfinite(coordinates_a).all()
    quality = reconstruction_quality(distance_maps, coordinates_a)

    reference_rms, reference_weighted_stress, reference_bond = REFERENCE_QUALITY_BY_RELAX_MODE[relax_mode][
        (motif, n_residues)
    ]
    assert quality["rms"].mean() <= reference_rms * (1 + RELATIVE_MARGIN)
    assert quality["weighted_stress"].mean() <= reference_weighted_stress * (1 + RELATIVE_MARGIN)
    assert quality["bond"].mean() <= reference_bond * (1 + RELATIVE_MARGIN)

    assert quality["rms"].max() < MAX_RELATIVE_RMS_DISTANCE_ERROR
    assert quality["rg"].max() < MAX_RELATIVE_RG_ERROR
    assert quality["bond"].max() < MAX_MEAN_BOND_ERROR_A


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.parametrize("n_residues", [length for length in LENGTHS if length >= MIN_LENGTH_FOR_COVERAGE_CHECK])
def test_sequences_span_compact_to_expanded(sample_maps, n_residues):
    """Guard the guard: the motifs must still give highly compact and highly expanded ensembles."""
    compact_maps = sample_maps(n_residues, MOST_COMPACT_MOTIF)
    expanded_maps = sample_maps(n_residues, MOST_EXPANDED_MOTIF)
    compact_rg = radius_of_gyration_from_maps(compact_maps).mean() / np.sqrt(n_residues)
    expanded_rg = radius_of_gyration_from_maps(expanded_maps).mean() / np.sqrt(n_residues)
    assert compact_rg < MAX_COMPACT_NORMALIZED_RG_A
    assert expanded_rg > MIN_EXPANDED_NORMALIZED_RG_A


def relative_start_difference(distance_maps: np.ndarray, device: torch.device | str) -> np.ndarray:
    """
    How far the fast classical-MDS start is from the full eigendecomposition's, per conformation.

    Parameters
    ----------
    distance_maps : np.ndarray
        Distance maps in Angstroms, shape (n_conformations, N, N).
    device : torch.device or str
        Device to compute both starts on.

    Returns
    -------
    np.ndarray
        Largest change in any inter-residue distance between the two starting
        structures, relative to the largest distance in the map, shape
        (n_conformations,).
    """
    maps = torch.as_tensor(distance_maps, dtype=torch.float32, device=device)
    # the precision SMACOF runs both starts at inside generate()
    with float32_matmul_precision(SMACOF_FLOAT32_MATMUL_PRECISION):
        full_start = _classical_mds(maps, initialization=FULL_EIGENDECOMPOSITION)
        fast_start = _classical_mds(maps, initialization=SUBSPACE_ITERATION)
    difference = np.abs(
        pairwise_distances(full_start.cpu().numpy().astype(np.float64))
        - pairwise_distances(fast_start.cpu().numpy().astype(np.float64))
    )
    largest_distance = np.maximum(np.asarray(distance_maps).max(axis=(1, 2)), np.finfo(np.float32).tiny)
    return difference.max(axis=(1, 2)) / largest_distance


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.parametrize("n_residues", LENGTHS)
@pytest.mark.parametrize("motif", MOTIFS)
def test_fast_start_matches_the_full_eigendecomposition(sample_maps, n_residues, motif):
    """The fast mode starts SMACOF from the same structures as the standard one, from compact to expanded."""
    difference = relative_start_difference(sample_maps(n_residues, motif), check_device(None))
    assert difference.max() < MAX_RELATIVE_START_DIFFERENCE


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.parametrize("n_residues", LENGTHS)
@pytest.mark.parametrize("motif", MOTIFS)
def test_fast_relaxation_matches_standard_relaxation(relax, n_residues, motif):
    """
    Relaxed fast ensembles match relaxed standard ones in fit, Rg, bonds and closest contacts.

    This is a coarse check (see the module docstring); the fast mode is
    checked closely before relaxation, by the start and envelope tests above.
    """
    distance_maps, standard_a = relax(n_residues, motif, RELAX_STANDARD)
    _, fast_a = relax(n_residues, motif, RELAX_FAST)
    assert np.isfinite(fast_a).all()

    standard_quality = reconstruction_quality(distance_maps, standard_a)
    fast_quality = reconstruction_quality(distance_maps, fast_a)
    for metric in ("rms", "weighted_stress"):
        assert fast_quality[metric].mean() <= standard_quality[metric].mean() * (1 + RELATIVE_MARGIN)

    standard_rg = radius_of_gyration_from_coordinates(standard_a).mean()
    fast_rg = radius_of_gyration_from_coordinates(fast_a).mean()
    assert abs(fast_rg / standard_rg - 1) < MAX_RELAXED_RG_DIFFERENCE

    assert abs(mean_bond_length_a(fast_a) - mean_bond_length_a(standard_a)) < MAX_RELAXED_BOND_DIFFERENCE_A

    contact_difference = closest_nonbonded_distance_a(fast_a).mean() - closest_nonbonded_distance_a(standard_a).mean()
    assert abs(contact_difference) < MAX_RELAXED_CONTACT_DIFFERENCE_A


# ---------------------------------------------------------------------------
# Stopping-rule reporting and relaxation-mode wiring (fast; no model weights)
# ---------------------------------------------------------------------------

# Ideal C-alpha virtual bond length used to build test chains, in Angstroms
BOND_LENGTH_A = 3.8


def random_chain_maps(n_frames: int, n_residues: int, seed: int, noise_a: float) -> np.ndarray:
    """
    Distance maps of random-walk chains, optionally made non-Euclidean by noise.

    Parameters
    ----------
    n_frames : int
        Number of maps.
    n_residues : int
        Beads per chain.
    seed : int
        Seed for numpy's generator.
    noise_a : float
        Standard deviation of symmetric noise added to every distance, in
        Angstroms. 0 gives exactly embeddable maps.

    Returns
    -------
    np.ndarray
        float32 maps in Angstroms, shape (n_frames, n_residues, n_residues).
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_frames, n_residues, 3))
    steps *= BOND_LENGTH_A / np.linalg.norm(steps, axis=-1, keepdims=True)
    maps = pairwise_distances(np.cumsum(steps, axis=1))
    noise = rng.normal(scale=noise_a, size=maps.shape)
    maps = np.abs(maps + (noise + noise.transpose(0, 2, 1)) / 2)
    for frame in maps:
        np.fill_diagonal(frame, 0.0)
    return maps.astype(np.float32)


def test_smacof_converged_separates_settled_maps_from_capped_ones():
    """Exactly embeddable maps settle well before the cap; noisy maps given two iterations cannot."""
    exact_maps = random_chain_maps(n_frames=3, n_residues=12, seed=0, noise_a=0.0)
    _, exact_stress = distance_matrix_to_3d_structure_torch_mds(exact_maps, device="cpu", progress_bar=False)
    assert smacof_converged(exact_stress).all()

    noisy_maps = random_chain_maps(n_frames=3, n_residues=40, seed=1, noise_a=2.0)
    _, capped_stress = distance_matrix_to_3d_structure_torch_mds(
        noisy_maps, n_iter=2, device="cpu", progress_bar=False
    )
    assert not smacof_converged(capped_stress).any()


def test_unrelaxed_reconstruction_matches_the_public_path_and_reports_the_cap(capsys):
    """generate()'s unrelaxed path gives exactly the public function's structures, and says how many hit the cap."""
    maps = random_chain_maps(n_frames=5, n_residues=30, seed=2, noise_a=1.0)
    expected_nm = generate_3d_coordinates_from_distances("cpu", 4, maps, progress_bar=False)
    coordinates_nm = _reconstruct_coordinates(
        "A" * 30, maps, 150, "cpu", 4, show_progress_bar=False, relax=False, verbose=True
    )
    np.testing.assert_array_equal(coordinates_nm, expected_nm)

    # the same reconstruction through the public function, to count the cap hits independently
    _, stress = distance_matrix_to_3d_structure_torch_mds(
        maps, batch_size=4, device="cpu", progress_bar=False, weights=map_error_weights(30, device="cpu")
    )
    n_capped = int((~smacof_converged(stress)).sum())
    assert f"SMACOF: {n_capped} of 5 conformation(s) ran all 300 iterations" in capsys.readouterr().out


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("n_residues", [1, 2, 3, 8, 9, 40, 200])
@pytest.mark.parametrize("noise_a", [0.0, 2.0])
def test_fast_start_matches_the_full_start_at_any_length(device, n_residues, noise_a):
    """
    Subspace iteration gives the full eigendecomposition's start, including for
    chains shorter than its block of vectors or than three residues.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if n_residues == 1:
        maps = np.zeros((3, 1, 1), dtype=np.float32)
    else:
        maps = random_chain_maps(n_frames=6, n_residues=n_residues, seed=n_residues, noise_a=noise_a)
    assert relative_start_difference(maps, device).max() < MAX_RELATIVE_START_DIFFERENCE


def test_fast_start_is_the_same_every_time():
    """The fast start draws its random block from a fixed seed, so it repeats exactly."""
    maps = torch.as_tensor(random_chain_maps(n_frames=4, n_residues=50, seed=3, noise_a=1.0))
    first = _classical_mds(maps, initialization=SUBSPACE_ITERATION)
    torch.manual_seed(12345)
    again = _classical_mds(maps, initialization=SUBSPACE_ITERATION)
    torch.testing.assert_close(first, again, rtol=0, atol=0)


def test_reconstruction_rejects_unknown_initializations():
    maps = random_chain_maps(n_frames=2, n_residues=10, seed=4, noise_a=0.0)
    with pytest.raises(ValueError, match="initialization must be one of"):
        distance_matrix_to_3d_structure_torch_mds(maps, device="cpu", progress_bar=False, initialization="lanczos")


def test_each_relaxation_mode_has_its_smacof_settings():
    """Unrelaxed structures keep the standard settings, since nothing afterwards repairs a shorter run."""
    assert _smacof_settings(False) == (FULL_EIGENDECOMPOSITION, DEFAULT_SMACOF_ITERATIONS)
    assert _smacof_settings(RELAX_STANDARD) == (FULL_EIGENDECOMPOSITION, DEFAULT_SMACOF_ITERATIONS)
    assert _smacof_settings(RELAX_FAST) == (SUBSPACE_ITERATION, FAST_RELAX_SMACOF_ITERATIONS)
    # the backend takes only normalized modes; generate() turns True into "standard"
    for unknown in (True, None, "quick", "Fast"):
        with pytest.raises(ValueError, match="relax must be one of"):
            _smacof_settings(unknown)


@pytest.mark.parametrize(
    "relax_mode,expected_initialization,expected_iterations,expect_relaxation",
    [
        (False, FULL_EIGENDECOMPOSITION, DEFAULT_SMACOF_ITERATIONS, False),
        (RELAX_STANDARD, FULL_EIGENDECOMPOSITION, DEFAULT_SMACOF_ITERATIONS, True),
        (RELAX_FAST, SUBSPACE_ITERATION, FAST_RELAX_SMACOF_ITERATIONS, True),
    ],
)
def test_reconstruct_coordinates_follows_the_relaxation_mode(
    monkeypatch, relax_mode, expected_initialization, expected_iterations, expect_relaxation
):
    """Each mode runs SMACOF with its own settings, and both relaxing modes relax the same way."""
    maps = random_chain_maps(n_frames=3, n_residues=12, seed=5, noise_a=0.5)
    smacof_calls = []
    relaxation_calls = []
    real_mds_coordinates = generation._mds_coordinates

    def recording_mds_coordinates(*args, **kwargs):
        smacof_calls.append(kwargs)
        return real_mds_coordinates(*args, **kwargs)

    def recording_relax_coordinates(sequence, coordinates, distance_maps, *args, **kwargs):
        relaxation_calls.append(kwargs)
        return coordinates.cpu().numpy() / configs.CONVERT_ANGSTROM_TO_NM

    monkeypatch.setattr(generation, "_mds_coordinates", recording_mds_coordinates)
    monkeypatch.setattr(generation, "_relax_coordinates", recording_relax_coordinates)
    coordinates_nm = _reconstruct_coordinates(
        "A" * 12, maps, 150, "cpu", 4, show_progress_bar=False, relax=relax_mode, verbose=False, relaxation_seed=9
    )

    assert coordinates_nm.shape == (3, 12, 3)
    assert len(smacof_calls) == 1
    assert smacof_calls[0]["initialization"] == expected_initialization
    assert smacof_calls[0]["n_iter"] == expected_iterations
    if expect_relaxation:
        assert len(relaxation_calls) == 1
        assert relaxation_calls[0]["seed"] == 9
    else:
        assert relaxation_calls == []
