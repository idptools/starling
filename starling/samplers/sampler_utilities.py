"""
Stateless helpers shared by STARLING's samplers.

These are the pieces of sampling that more than one sampler needs but that
do not depend on a sampler's state: input validation, the timestep grid that
DDIM and PLMS both step through, and the DDIM update, which PLMS also uses
(it only changes which noise prediction is fed into it).

References
----------
[1] Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit
    models. ICLR 2021. arXiv:2010.02502.
[2] Lin, S., Liu, B., Li, J., & Yang, X. (2024). Common diffusion noise
    schedules and sample steps are flawed. WACV 2024. arXiv:2305.08891.
"""

from __future__ import annotations

from typing import Final, TypeGuard

import numpy as np
import torch

# Turning a noisy latent into an estimate of the clean latent divides by
# sqrt(alpha_bar_t), so any error in the predicted noise is amplified
# 1/sqrt(alpha_bar_t)-fold. DDIM and PLMS never start from a timestep where
# that amplification would exceed 100x.
MINIMUM_SIGNAL_SCALE: Final[float] = 0.01

# Sampling starts from pure noise, N(0, I), so the grid's first timestep must
# be one where the latents hold almost no signal. A grid that starts where
# sqrt(alpha_bar) is above this is stretched up to the noisiest safe timestep
# (see ddim_timesteps()). 0.02 leaves the 30-step grid (which starts at
# sqrt(alpha_bar) ~ 0.012 on the cosine schedule) untouched.
MAX_START_SIGNAL_SCALE: Final[float] = 0.02

# Ways of spacing the DDIM/PLMS timestep grid
UNIFORM_DISCRETIZATION: Final[str] = "uniform"
QUADRATIC_DISCRETIZATION: Final[str] = "quad"

# The quadratic grid spans the first 80% of the training timesteps, as in the
# reference DDIM implementation (https://github.com/ermongroup/ddim)
QUADRATIC_GRID_FRACTION: Final[float] = 0.8

# The DDIM/PLMS grid starts one timestep above zero, and its final step lands
# on timestep 0
GRID_OFFSET: Final[int] = 1
FINAL_TIMESTEP: Final[int] = 0


def is_integer(value: object) -> TypeGuard[int | np.integer]:
    """
    Check whether a value is a Python or NumPy integer (and not a bool).

    Parameters
    ----------
    value : object
        Value to check.

    Returns
    -------
    bool
        True if value is an int or np.integer other than True/False.
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def check_step_count(n_steps: object, num_timesteps: int) -> None:
    """
    Check that a requested number of sampler steps is usable.

    Parameters
    ----------
    n_steps : object
        Requested number of steps.
    num_timesteps : int
        Number of timesteps the diffusion model was trained with, which is the
        most steps any sampler can take.

    Raises
    ------
    ValueError
        If n_steps is not an integer between 1 and num_timesteps.
    """
    if not is_integer(n_steps) or not 1 <= n_steps <= num_timesteps:
        raise ValueError(f"n_steps must be an integer between 1 and {num_timesteps}")


def check_conformation_count(num_conformations: object) -> None:
    """
    Check that a requested number of conformations is a positive integer.

    Parameters
    ----------
    num_conformations : object
        Requested number of conformations.

    Raises
    ------
    ValueError
        If num_conformations is not a positive integer.
    """
    if not is_integer(num_conformations) or num_conformations < 1:
        raise ValueError("num_conformations must be a positive integer")


def ddim_timesteps(
    alphas_cumprod: torch.Tensor,
    n_steps: int,
    discretization: str = UNIFORM_DISCRETIZATION,
) -> tuple[list[int], list[int]]:
    """
    Build the timestep grid that DDIM and PLMS step through.

    The uniform grid takes every (num_timesteps // n_steps)-th timestep
    starting from 1, so for 1000 timesteps and 30 steps it is 1, 34, ..., 991:
    31 steps, not 30. We keep this legacy behaviour so results match earlier
    versions of STARLING. The quadratic grid spaces n_steps timesteps
    quadratically over the first 80% of the schedule and can repeat timesteps
    at its low end.

    If the grid's noisiest timestep still has sqrt(alpha_bar) above
    MAX_START_SIGNAL_SCALE (every quadratic grid, and uniform grids with few
    steps, e.g. 10 steps would start at timestep 901), it is replaced by a
    grid with the same number of steps and the same shape (even or quadratic
    spacing) that is stretched to run from timestep 1 up to the noisiest safe
    timestep (see stretched_timestep_grid()). For the uniform grid this is
    the "linspace" spacing of Lin et al. [2].

    Any timestep where sqrt(alpha_bar) < MINIMUM_SIGNAL_SCALE is dropped and
    replaced by a single step at the noisiest timestep that is still safe.

    Parameters
    ----------
    alphas_cumprod : torch.Tensor
        Cumulative signal fraction alpha_bar_t for every training timestep,
        shape (num_timesteps,).
    n_steps : int
        Requested number of steps.
    discretization : str, optional
        'uniform' (default) or 'quad'.

    Returns
    -------
    tuple of (list of int, list of int)
        The timestep each step starts from, noisiest first, and the timestep
        each step lands on. The last step lands on timestep 0.

    Raises
    ------
    ValueError
        If the discretization is unknown or no timestep has enough signal.
    """
    num_timesteps = len(alphas_cumprod)

    if discretization == UNIFORM_DISCRETIZATION:
        stride = num_timesteps // n_steps
        ascending = np.arange(0, num_timesteps - 1, stride) + GRID_OFFSET
    elif discretization == QUADRATIC_DISCRETIZATION:
        grid_end = np.sqrt(num_timesteps * QUADRATIC_GRID_FRACTION)
        ascending = (np.linspace(0, grid_end, n_steps) ** 2).astype(int) + GRID_OFFSET
    else:
        raise ValueError(
            f"unknown discretization {discretization!r}; use {UNIFORM_DISCRETIZATION!r} or {QUADRATIC_DISCRETIZATION!r}"
        )

    # the noisiest timestep with enough signal (the schedule is monotonic, so
    # that is the last one that passes)
    usable = torch.nonzero(alphas_cumprod.sqrt() >= MINIMUM_SIGNAL_SCALE).flatten()
    if not len(usable):
        raise ValueError(f"DDIM requires timesteps with sqrt(alpha) >= {MINIMUM_SIGNAL_SCALE}")
    noisiest_safe_timestep = int(usable[-1])

    # A latent at signal level a = sqrt(alpha_bar) has mean a * mu, but the
    # initial N(0, I) latents have mean 0. Starting the grid where a is not ~0
    # treats pure noise as such a latent, and the sampler carries the missing
    # a * mu through to its samples. For Gaussian data (mean mu, std s) the
    # deterministic sampler maps a shift d in the starting latents to about
    # s * d in the samples, so the ensemble mean is biased by about a * mu * s,
    # i.e. about a times the data's std for latents of order one (10 uniform
    # steps would start at timestep 901, where a = 0.15, and miss the mean of
    # N(1, 0.3^2) by 0.13 std). With 1 step the grid would be just timestep 1,
    # so the "samples" would be the initial noise. Such grids are stretched to
    # start at the noisiest safe timestep (Lin et al. [2]). We stretch rather
    # than shift them: shifting the whole grid up would also move its fine,
    # low-noise end up (the quadratic grid's would start at timestep 192), and
    # the last step would then jump from there to timestep 0 in one go, which
    # turns the latents into the posterior mean at that noise level and loses
    # spread (std ratio ~0.7 for N(1, 0.3^2) with 30 quadratic steps).
    if alphas_cumprod[int(ascending[-1])].sqrt() > MAX_START_SIGNAL_SCALE:
        ascending = stretched_timestep_grid(len(ascending), noisiest_safe_timestep, discretization)

    # cap the grid at the noisiest safe timestep
    if np.any(ascending > noisiest_safe_timestep):
        safe = ascending[ascending < noisiest_safe_timestep]
        ascending = np.append(safe, noisiest_safe_timestep)

    timesteps = [int(timestep) for timestep in ascending[::-1]]
    next_timesteps = [*timesteps[1:], FINAL_TIMESTEP]
    return timesteps, next_timesteps


def stretched_timestep_grid(n_points: int, noisiest_timestep: int, discretization: str) -> np.ndarray:
    """
    Build a DDIM/PLMS timestep grid that spans every usable timestep.

    The grid runs from timestep GRID_OFFSET (1) up to noisiest_timestep with
    n_points timesteps, spaced evenly for the uniform grid or quadratically
    for the quadratic grid (so the quadratic grid covers the fraction
    (noisiest_timestep - 1) / num_timesteps of the schedule instead of
    QUADRATIC_GRID_FRACTION). It is built from its noisy end, so a grid of
    one point is just noisiest_timestep.

    Parameters
    ----------
    n_points : int
        Number of timesteps in the grid (one per step).
    noisiest_timestep : int
        The grid's noisiest timestep.
    discretization : str
        'uniform' or 'quad'.

    Returns
    -------
    np.ndarray
        The timesteps in ascending order, as integers. The quadratic grid can
        repeat timesteps at its low end, like the legacy quadratic grid.
    """
    assert n_points >= 1, "a grid needs at least one timestep"
    assert noisiest_timestep >= GRID_OFFSET, "the noisiest timestep must not be below the grid's start"

    if discretization == UNIFORM_DISCRETIZATION:
        descending = np.linspace(noisiest_timestep, GRID_OFFSET, n_points)
    else:
        # the quadratic grid spaces sqrt(timestep - GRID_OFFSET) evenly
        descending = np.linspace(np.sqrt(noisiest_timestep - GRID_OFFSET), 0, n_points) ** 2 + GRID_OFFSET

    return np.rint(descending[::-1]).astype(int)


def ddim_update(
    latents: torch.Tensor,
    predicted_noise: torch.Tensor,
    alpha_bar: torch.Tensor,
    next_alpha_bar: torch.Tensor,
    noise_scale: torch.Tensor | float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Take one DDIM step from noise level alpha_bar to next_alpha_bar.

    This is Eq. 12 of Song et al. [1]: estimate the clean latent from the
    predicted noise, then re-noise it to the next level along the same noise
    direction, adding fresh noise of size noise_scale (sigma_t in [1]). With
    noise_scale = 0 the step is deterministic.

    Parameters
    ----------
    latents : torch.Tensor
        Noisy latents at the current level, shape (batch, 1, 24, 24).
    predicted_noise : torch.Tensor
        Noise predicted for these latents, same shape.
    alpha_bar : torch.Tensor
        Scalar cumulative signal fraction of the current level.
    next_alpha_bar : torch.Tensor
        Scalar cumulative signal fraction of the level to step to.
    noise_scale : torch.Tensor or float, optional
        Standard deviation of the fresh noise added (Eq. 16 of [1]). Default
        0, which draws no random numbers. A Python float avoids a GPU-to-CPU
        synchronization when deciding whether to add noise.
    generator : torch.Generator or None, optional
        Generator the fresh noise is drawn from. Default None, which draws
        from torch's global generator.

    Returns
    -------
    torch.Tensor
        Latents at the next noise level, same shape as latents.
    """
    clean_estimate = (latents - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
    direction_to_latents = (1.0 - next_alpha_bar - noise_scale**2).sqrt() * predicted_noise
    next_latents = next_alpha_bar.sqrt() * clean_estimate + direction_to_latents

    if noise_scale > 0:
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
        next_latents = next_latents + noise_scale * noise

    return next_latents
