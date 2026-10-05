"""
Do STARLING's faster samplers bias real predictions relative to DDPM?

DDPM, which steps through all 1000 timesteps, is the sampler the network was
trained for, so with the released checkpoints it is the reference. DDIM, PLMS
and DPM++ approximate the same reverse process in far fewer steps, and any
systematic difference from DDPM is bias they introduce. Each is compared
with a DDPM ensemble of the same sequence on:

- global dimensions: mean radius of gyration, the mean over residue pairs of
  the relative difference in <d_ij>, and the spread of all distances
  (geometric mean over pairs of the std(d_ij) ratio);
- local geometry: the mean and the spread of i,i+1 and i,i+2 distances.

Measured on alpha-synuclein with 400 conformations per sampler (October
2026, MPS), against the 95% range of DDPM compared with itself (200 vs 200
conformations):

    metric                 DDPM noise      DDIM-30  PLMS-30  DPM++-12
    mean Rg (%)            -3.5 to +3.2    -0.1     +2.0     +0.9
    <d_ij> (%)             -2.8 to +2.5    -0.4     +1.4     +0.6
    spread of all d_ij     0.96 to 1.04    0.98     1.02     1.04
    i,i+1 mean (%)         -0.03 to +0.03  -0.09    +0.04    +0.07
    i,i+1 spread           0.97 to 1.03    0.93     1.03     1.07
    i,i+2 spread           0.98 to 1.02    0.94     1.03     1.05

A 51-residue natural IDR (IDR_10013, A0A1P8BB09_ARATH) gave the same
pattern: i,i+1 / i,i+2 spread ratios 0.92 / 0.95 (DDIM-30), 1.04 / 1.04
(PLMS-30) and 1.07 / 1.06 (DPM++-12), with every global metric inside the
DDPM noise. PLMS-30 came out slightly expanded in both sequences (mean Rg
+2.0% and +3.4%), each within the noise on its own.

Global dimensions show no bias for any sampler. Locally, DDIM-30 (the
default) narrows the fluctuations of neighbouring distances by ~6-8% and
DPM++-12 widens them by ~5-7%; PLMS-30 is closest to DDPM. These are the
same discretization effects the exact-predictor tests measure precisely
(test_exact_denoiser_bias.py). With 200 conformations per sampler, sampling
noise alone moves a local spread ratio by ~2.5% (95%), so this module only
fails a sampler whose local spread is off by more than 12%, and leaves the
smaller, known effects to the deterministic exact-predictor tests.

Generating the DDPM reference takes ~10 minutes on MPS and much less on a
GPU, so these tests are marked slow as well as requiring weights.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.requires_weights]

# alpha-synuclein (UniProt P37840), a well-characterized 140-residue IDP
SEQUENCE: Final[str] = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
)

N_CONFORMATIONS: Final[int] = 200
SEED: Final[int] = 1

# the faster samplers at the settings STARLING uses (DDIM-30 is the default)
FAST_SAMPLERS: Final[tuple[tuple[str, int], ...]] = (("ddim", 30), ("plms", 30), ("dpmpp", 12))

# Tolerances, about 4x the 95% DDPM-vs-DDPM noise at 200 conformations for
# global dimensions; local means and spreads as described in the docstring
RG_TOLERANCE_PERCENT: Final[float] = 6.0
MEAN_DISTANCE_TOLERANCE_PERCENT: Final[float] = 5.0
GLOBAL_SPREAD_TOLERANCE: Final[float] = 0.08
LOCAL_MEAN_TOLERANCE_PERCENT: Final[float] = 2.0
LOCAL_SPREAD_TOLERANCE: Final[float] = 0.12

# sequence separations treated as local geometry
LOCAL_SEPARATIONS: Final[tuple[int, ...]] = (1, 2)


def _sample_maps(sampler: str, steps: int) -> np.ndarray:
    """
    Generate distance maps for SEQUENCE with one sampler.

    Parameters
    ----------
    sampler : str
        STARLING sampler name.
    steps : int
        Sampler steps.

    Returns
    -------
    np.ndarray
        Distance maps in Angstroms, shape (N_CONFORMATIONS, n, n), float64.
    """
    from starling import generate

    torch.manual_seed(SEED)
    ensemble = generate(
        SEQUENCE,
        conformations=N_CONFORMATIONS,
        sampler=sampler,
        steps=steps,
        return_structures=False,
        return_single_ensemble=True,
        show_progress_bar=False,
        show_per_step_progress_bar=False,
    )
    return ensemble.distance_maps().astype(np.float64)


def _radius_of_gyration_A(maps: np.ndarray) -> np.ndarray:
    """Radius of gyration of each map, sqrt(sum_ij d_ij^2 / (2 n^2)), in Angstroms."""
    n = maps.shape[-1]
    return np.sqrt((maps**2).sum(axis=(1, 2)) / (2 * n**2))


def _pair_distances(maps: np.ndarray, separation: int | None = None) -> np.ndarray:
    """Distances of every pair i < j (or only |i - j| == separation), shape (n_maps, n_pairs)."""
    i, j = np.triu_indices(maps.shape[-1], k=1)
    if separation is not None:
        keep = (j - i) == separation
        i, j = i[keep], j[keep]
    return maps[:, i, j]


@pytest.fixture(scope="module")
def ddpm_maps() -> np.ndarray:
    """The DDPM-1000 reference ensemble, generated once for the module."""
    return _sample_maps("ddpm", 1000)


@pytest.fixture(scope="module", params=FAST_SAMPLERS, ids=lambda param: f"{param[0]}-{param[1]}")
def sampler_maps(request) -> np.ndarray:
    """One faster sampler's ensemble, generated once per sampler."""
    sampler, steps = request.param
    return _sample_maps(sampler, steps)


def test_global_dimensions_match_ddpm(ddpm_maps, sampler_maps):
    rg_difference = 100 * (_radius_of_gyration_A(sampler_maps).mean() / _radius_of_gyration_A(ddpm_maps).mean() - 1)
    assert abs(rg_difference) < RG_TOLERANCE_PERCENT, f"mean Rg differs by {rg_difference:+.2f}%"

    distances, reference = _pair_distances(sampler_maps), _pair_distances(ddpm_maps)
    mean_difference = 100 * np.mean(distances.mean(axis=0) / reference.mean(axis=0) - 1)
    assert abs(mean_difference) < MEAN_DISTANCE_TOLERANCE_PERCENT, f"<d_ij> differs by {mean_difference:+.2f}%"

    spread_ratio = np.exp(np.mean(np.log(distances.std(axis=0) / reference.std(axis=0))))
    assert abs(spread_ratio - 1) < GLOBAL_SPREAD_TOLERANCE, f"distance spread ratio {spread_ratio:.3f}"


@pytest.mark.parametrize("separation", LOCAL_SEPARATIONS)
def test_local_distances_match_ddpm(ddpm_maps, sampler_maps, separation):
    distances = _pair_distances(sampler_maps, separation)
    reference = _pair_distances(ddpm_maps, separation)

    mean_difference = 100 * (distances.mean() / reference.mean() - 1)
    assert abs(mean_difference) < LOCAL_MEAN_TOLERANCE_PERCENT, f"mean differs by {mean_difference:+.2f}%"

    spread_ratio = distances.std(axis=0).mean() / reference.std(axis=0).mean()
    assert abs(spread_ratio - 1) < LOCAL_SPREAD_TOLERANCE, f"spread ratio {spread_ratio:.3f}"
