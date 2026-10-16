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

DDIM with very few steps (1 to 10) also has to reproduce the mean. Its
timestep grid used to start far below the noise level of the initial latents
(timestep 1, 501, 801 or 901), which missed the mean by up to 3 standard
deviations; it is now stretched to start in pure noise. With 10 to 50 steps
that must not narrow the spread by more than 5% beyond what the old grid did.
(With 2 to 5 steps the old grids' spread was partly the initial noise passed
through as if it were signal, and the stretched grids narrow it further.)
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest

from starling.samplers import sampler_utilities
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

# For few-step DDIM. Even timestep 992, where every grid now starts, holds a
# little signal (sqrt(alpha_bar) = 0.011), which leaves the mean of this
# distribution ~1% of its std short; sampling error is ~0.3%. Before the grid
# was stretched, 1, 2, 5 and 10 steps missed it by 3.3, 0.31, 0.21 and 0.13 std.
FEW_STEP_GAUSSIAN: Final = GaussianMixture.gaussian(mean=1.0, std=0.3)
FEW_STEP_MEAN_TOLERANCE: Final[float] = 0.03  # in units of the distribution's std

# How much further from 1 the std ratio of few-step DDIM may be on the
# stretched grid than on the legacy grid (measured: at most 0.03 further, for
# 10 steps on the narrow Gaussian: 0.717 vs 0.748)
FEW_STEP_SPREAD_TOLERANCE: Final[float] = 0.05

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


@pytest.mark.filterwarnings("ignore:DDIM with .* steps narrows the ensemble")
@pytest.mark.parametrize("steps", [1, 2, 5, 10])
def test_few_step_ddim_reproduces_the_mean(steps):
    """Few-step DDIM starts in pure noise, so it gets the mean right (it still narrows the spread)."""
    samples = draw_samples("ddim", steps, FEW_STEP_GAUSSIAN, N_CONFORMATIONS)
    mean_error = (samples.mean() - FEW_STEP_GAUSSIAN.mean) / FEW_STEP_GAUSSIAN.std
    assert abs(mean_error) < FEW_STEP_MEAN_TOLERANCE, f"mean off by {mean_error:+.4f} std"


@pytest.mark.parametrize("distribution", [FEW_STEP_GAUSSIAN, NARROW_GAUSSIAN], ids=["mean-1", "narrow"])
@pytest.mark.parametrize("steps", [10, 20, 25, 50])
def test_few_step_ddim_spread_is_no_worse_than_on_the_legacy_grid(monkeypatch, steps, distribution):
    """Stretching the grid to start in pure noise must not cost few-step DDIM more than 5% of spread."""
    stretched = draw_samples("ddim", steps, distribution, N_CONFORMATIONS)

    # the legacy grid is what ddim_timesteps() builds when no grid is stretched
    monkeypatch.setattr(sampler_utilities, "MAX_START_SIGNAL_SCALE", float("inf"))
    legacy = draw_samples("ddim", steps, distribution, N_CONFORMATIONS)

    stretched_error = abs(stretched.std() / distribution.std - 1.0)
    legacy_error = abs(legacy.std() / distribution.std - 1.0)
    assert stretched_error <= legacy_error + FEW_STEP_SPREAD_TOLERANCE, (
        f"std ratio off by {stretched_error:.3f} (legacy grid {legacy_error:.3f})"
    )


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
