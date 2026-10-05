"""
Does any of STARLING's samplers bias the distribution it samples?

Each sampler is driven by the exact noise predictor for a known latent
distribution (see exact_denoiser.py), so the network cannot contribute any
error: whatever bias remains is the sampler's own. Every sampler, at the
settings STARLING uses, has to reproduce

- the mean (to within 2% of the distribution's standard deviation),
- the spread (standard deviation within +/- 5%),
- and for a two-mode mixture, the mode weights (within 0.01), positions
  (within 0.2 mode standard deviations) and widths (within +/- 5%)

of three distributions: a unit-variance Gaussian, a narrow Gaussian (std
0.2, where coarse time steps have the most effect), and an asymmetric
mixture of two well-separated modes.

Two sampler settings currently fail and are marked as strict expected
failures, so the bias is recorded explicitly and the tests flag any change
in it (a strict xfail that starts passing fails the suite):

- DDIM-30, STARLING's default, narrows the narrow Gaussian by ~9%
  (std ratio 0.908). It is first-order in time, and 30 steps is coarse.
- DPM++-12 widens the modes of the mixture by ~13% and ~5% (std ratios
  1.13 and 1.05) and pushes them slightly apart.

Both are discretization errors: test_bias_vanishes_with_more_steps shows
them shrinking towards zero as the step count rises. DDPM (1000 steps, the
sampler the model was trained for) and PLMS-30 pass everything.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest

from starling.tests.sampler_tests.exact_denoiser import GaussianMixture, draw_samples

# 250 latents of 24x24 = 144,000 independent samples per run, which pins a
# spread ratio to ~0.2% and a mean to ~0.3% of the std (one standard error)
N_CONFORMATIONS: Final[int] = 250

# what counts as unbiased; see the module docstring
MEAN_TOLERANCE: Final[float] = 0.02  # in units of the distribution's std
SPREAD_TOLERANCE: Final[float] = 0.05  # relative error in std
MODE_WEIGHT_TOLERANCE: Final[float] = 0.01  # absolute
MODE_POSITION_TOLERANCE: Final[float] = 0.2  # in units of the mode's std

UNIT_GAUSSIAN: Final = GaussianMixture.gaussian(mean=0.3, std=1.0)
NARROW_GAUSSIAN: Final = GaussianMixture.gaussian(mean=0.3, std=0.2)
TWO_MODES: Final = GaussianMixture(weights=(0.3, 0.7), means=(1.5, -1.0), stds=(0.25, 0.25))

# halfway between the two modes
MODE_BOUNDARY: Final[float] = 0.25

# every sampler at the settings STARLING uses (DDIM-30 is the default)
DEFAULT_SAMPLERS: Final[tuple[tuple[str, int], ...]] = (
    ("ddpm", 1000),
    ("ddim", 30),
    ("plms", 30),
    ("dpmpp", 12),
)


# known sampler biases, recorded as strict expected failures (see module docstring)
DDIM_NARROWS: Final = pytest.mark.xfail(
    strict=True,
    reason="DDIM-30 under-disperses narrow distributions: std ratio 0.908 with an exact predictor (first-order steps)",
)
DPMPP_WIDENS_MODES: Final = pytest.mark.xfail(
    strict=True,
    reason="DPM++-12 over-disperses sharp modes: mode std ratios 1.13 / 1.05 with an exact predictor",
)


def _speed_marks(sampler: str) -> tuple[pytest.MarkDecorator, ...]:
    # DDPM evaluates all 1000 timesteps, so each of its cases takes ~10 s
    return (pytest.mark.slow,) if sampler == "ddpm" else ()


def _sampler_ids(sampler: str, steps: int) -> str:
    return f"{sampler}-{steps}"


def _assert_mean_and_spread(samples: np.ndarray, distribution: GaussianMixture) -> None:
    mean_error = (samples.mean() - distribution.mean) / distribution.std
    spread_ratio = samples.std() / distribution.std
    assert abs(mean_error) < MEAN_TOLERANCE, f"mean off by {mean_error:+.4f} std"
    assert abs(spread_ratio - 1.0) < SPREAD_TOLERANCE, f"std ratio {spread_ratio:.3f}"


@pytest.mark.parametrize(
    ("sampler", "steps", "distribution"),
    [
        pytest.param(
            sampler,
            steps,
            distribution,
            id=f"{_sampler_ids(sampler, steps)}-{label}",
            marks=(*marks, *_speed_marks(sampler)),
        )
        for sampler, steps in DEFAULT_SAMPLERS
        for label, distribution, marks in (
            ("unit", UNIT_GAUSSIAN, ()),
            ("narrow", NARROW_GAUSSIAN, (DDIM_NARROWS,) if sampler == "ddim" else ()),
        )
    ],
)
def test_gaussian_mean_and_spread_are_unbiased(sampler, steps, distribution):
    samples = draw_samples(sampler, steps, distribution, N_CONFORMATIONS)
    _assert_mean_and_spread(samples, distribution)


@pytest.mark.parametrize(
    ("sampler", "steps"),
    [
        pytest.param(
            sampler,
            steps,
            id=_sampler_ids(sampler, steps),
            marks=((DPMPP_WIDENS_MODES,) if sampler == "dpmpp" else ()) + _speed_marks(sampler),
        )
        for sampler, steps in DEFAULT_SAMPLERS
    ],
)
def test_two_mode_weights_positions_and_widths_are_unbiased(sampler, steps):
    samples = draw_samples(sampler, steps, TWO_MODES, N_CONFORMATIONS)
    upper = samples > MODE_BOUNDARY

    # the minority mode keeps its weight...
    assert abs(upper.mean() - TWO_MODES.weights[0]) < MODE_WEIGHT_TOLERANCE, f"weight {upper.mean():.4f}"

    # ...and each mode keeps its position and width
    for in_mode, mean, std in (
        (upper, TWO_MODES.means[0], TWO_MODES.stds[0]),
        (~upper, TWO_MODES.means[1], TWO_MODES.stds[1]),
    ):
        mode = samples[in_mode]
        position_error = (mode.mean() - mean) / std
        width_ratio = mode.std() / std
        assert abs(position_error) < MODE_POSITION_TOLERANCE, f"mode at {mean} shifted by {position_error:+.3f} std"
        assert abs(width_ratio - 1.0) < SPREAD_TOLERANCE, f"mode at {mean} has std ratio {width_ratio:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("sampler", "steps_list", "distribution", "final_tolerance"),
    [
        # DDIM's under-dispersion of the narrow Gaussian: 0.908 -> 0.971 -> 0.992
        pytest.param("ddim", (30, 100, 300), NARROW_GAUSSIAN, 0.02, id="ddim-narrow"),
        # DPM++'s widening of the minority mode: 1.13 -> 1.07 -> 1.02
        pytest.param("dpmpp", (12, 20, 50), TWO_MODES, 0.03, id="dpmpp-two-modes"),
    ],
)
def test_bias_vanishes_with_more_steps(sampler, steps_list, distribution, final_tolerance):
    """The known biases are discretization error: they shrink towards zero with more steps."""
    errors = []
    for steps in steps_list:
        samples = draw_samples(sampler, steps, distribution, N_CONFORMATIONS)
        if distribution is TWO_MODES:
            samples = samples[samples > MODE_BOUNDARY]
            reference_std = TWO_MODES.stds[0]
        else:
            reference_std = distribution.std
        errors.append(abs(samples.std() / reference_std - 1.0))

    assert all(later < earlier for earlier, later in zip(errors, errors[1:])), f"errors {np.round(errors, 3)}"
    assert errors[-1] < final_tolerance, f"still {errors[-1]:.3f} off at {steps_list[-1]} steps"


@pytest.mark.parametrize(
    ("sampler", "steps"),
    [
        pytest.param(sampler, steps, id=_sampler_ids(sampler, steps), marks=_speed_marks(sampler))
        for sampler, steps in DEFAULT_SAMPLERS
    ],
)
def test_latent_scaling_is_undone_exactly(sampler, steps):
    """Every sampler divides by the latent scaling factor once, and only once."""
    unscaled = draw_samples(sampler, steps, UNIT_GAUSSIAN, 20, seed=3, scaling_factor=1.0)
    scaled = draw_samples(sampler, steps, UNIT_GAUSSIAN, 20, seed=3, scaling_factor=2.0)
    np.testing.assert_allclose(scaled * 2.0, unscaled, rtol=1e-6, atol=1e-6)
