"""
A batched FIRE energy minimizer.

FIRE (the Fast Inertial Relaxation Engine) minimizes an energy by running
damped molecular dynamics: velocities are steered towards the force, the
time step grows while the system keeps moving downhill, and the system is
stopped dead (and the time step cut) the moment it starts moving uphill. It
only needs forces, copes well with the very steep repulsive walls that
overlapping beads produce, and is the standard choice for relaxing
particle systems.

This implementation follows FIRE 2.0 (Guenole et al. 2020): semi-implicit
Euler integration, velocity mixing after the velocity update, an initial
delay before the time step can shrink, and a half-step backtrack whenever
the power F.v turns negative. Every conformation in the batch has its own
time step, mixing parameter and convergence state, so conformations are
minimized independently even though they are integrated together. Once a
conformation has converged it is dropped from the batch, so the force
function only ever sees conformations that are still moving.

To stop the enormous forces from badly overlapping beads throwing a
conformation apart, each step is capped so that no bead moves more than
max_step Angstroms.

References
----------
Bitzek, E., Koskinen, P., Gahler, F., Moseler, M., & Gumbsch, P. (2006).
Structural relaxation made simple. Physical Review Letters, 97(17), 170201.

Guenole, J., Noehring, W. G., Vaid, A., Houlle, F., Xie, Z., Prakash, A., &
Bitzek, E. (2020). Assessment and optimization of the fast inertial
relaxation engine (FIRE) for energy minimization in atomistic simulations
and its implementation in LAMMPS. Computational Materials Science, 175,
109584.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Integral

import torch
from tqdm.auto import tqdm

# signature of the function FIRE minimizes: given coordinates (batch, n, 3)
# and the index of each conformation in the full set, return forces
ForceFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FIREParameters:
    """
    Settings for the FIRE minimizer.

    Time is in arbitrary units (every bead has unit mass), so dt values are
    only meaningful relative to one another and to the stiffness of the
    energy surface. The defaults are tuned for Mpipi-GG in kJ/mol and
    Angstroms, where the stiffest routine term is the 80.3 kJ mol^-1 A^-2
    bond.

    Attributes
    ----------
    dt_start : float
        Initial time step. Default 0.01.

    dt_max : float
        Largest time step allowed. Default 0.1.

    dt_min : float
        Smallest time step allowed. Default 1e-4.

    n_delay : int
        Number of consecutive downhill steps required before the time step
        may grow, and the number of initial steps during which it may not
        shrink. Default 5.

    f_inc : float
        Factor the time step grows by. Default 1.1.

    f_dec : float
        Factor the time step shrinks by after an uphill step. Default 0.5.

    alpha_start : float
        Initial velocity-mixing parameter. Default 0.1.

    f_alpha : float
        Factor alpha shrinks by on every downhill step past the delay.
        Default 0.99.

    max_step : float
        Largest distance, in Angstroms, any bead may move in one step.
        Default 0.1.
    """

    dt_start: float = 0.01
    dt_max: float = 0.1
    dt_min: float = 1e-4
    n_delay: int = 5
    f_inc: float = 1.1
    f_dec: float = 0.5
    alpha_start: float = 0.1
    f_alpha: float = 0.99
    max_step: float = 0.1


@dataclass
class FIREResult:
    """
    Outcome of a FIRE minimization.

    Attributes
    ----------
    coordinates : torch.Tensor
        Final coordinates, shape (batch, n, 3).

    converged : torch.Tensor
        Whether each conformation reached the force tolerance, shape (batch,),
        dtype bool.

    n_steps : torch.Tensor
        Number of steps each conformation took, shape (batch,), dtype long.

    max_force : torch.Tensor
        Largest per-bead force magnitude in the final coordinates, in the
        force function's units, shape (batch,).
    """

    coordinates: torch.Tensor
    converged: torch.Tensor
    n_steps: torch.Tensor
    max_force: torch.Tensor


def _vector_norm(values: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """Compute norms without squaring large float32 clash forces directly."""
    if values.dtype != torch.float32:
        return torch.linalg.vector_norm(values, dim=dim)
    scale = values.abs().amax(dim=dim, keepdim=True).clamp_min(1e-30)
    return (torch.linalg.vector_norm(values / scale, dim=dim, keepdim=True) * scale).squeeze(dim)


def _max_bead_force(forces: torch.Tensor) -> torch.Tensor:
    """Largest per-bead force magnitude in each conformation, shape (batch,)."""
    per_bead: torch.Tensor = _vector_norm(forces, dim=-1)
    return per_bead.amax(dim=-1)


def fire_minimize(
    coordinates: torch.Tensor,
    force_function: ForceFunction,
    force_tolerance: float = 1.0,
    max_steps: int = 5000,
    parameters: FIREParameters | None = None,
    progress_bar: bool = False,
) -> FIREResult:
    """
    Minimize a batch of conformations with FIRE.

    Parameters
    ----------
    coordinates : torch.Tensor
        Starting coordinates, shape (batch, n, 3). Not modified.

    force_function : callable
        force_function(x, index) returns the forces on the conformations x
        (shape (b, n, 3)), where index (shape (b,), dtype long) gives the
        position of each of those conformations in the full batch. As
        conformations converge they are dropped, so b shrinks over time.

    force_tolerance : float, optional
        A conformation has converged once no bead feels a force larger than
        this. Default 1.0 (kJ/mol/Angstrom for Mpipi-GG).

    max_steps : int, optional
        Maximum number of steps for any conformation. Default 5000.

    parameters : FIREParameters, optional
        FIRE settings. If None (default), FIREParameters() is used.

    progress_bar : bool, optional
        If True, show a progress bar of converged conformations. Default
        False.

    Returns
    -------
    FIREResult
        Final coordinates, convergence flags, step counts and final maximum
        forces.

    Raises
    ------
    ValueError
        If the coordinates are not (batch, n, 3), or the tolerance or step
        limit are not positive.
    """
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"coordinates must have shape (batch, n, 3), got {tuple(coordinates.shape)}")
    if not math.isfinite(force_tolerance) or force_tolerance <= 0:
        raise ValueError(f"force_tolerance must be positive, got {force_tolerance}")
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 0:
        raise ValueError(f"max_steps must be a nonnegative integer, got {max_steps}")

    p = parameters if parameters is not None else FIREParameters()

    device, dtype = coordinates.device, coordinates.dtype
    n_frames = coordinates.shape[0]

    # full-batch outputs, filled in as conformations converge (or run out of steps)
    x_out = coordinates.clone()
    converged_out = torch.zeros(n_frames, dtype=torch.bool, device=device)
    steps_out = torch.zeros(n_frames, dtype=torch.long, device=device)
    fmax_out = torch.zeros(n_frames, dtype=dtype, device=device)

    # working state for the conformations still being minimized
    index = torch.arange(n_frames, device=device)
    x = coordinates.clone()
    v = torch.zeros_like(x)
    forces = force_function(x, index)
    fmax = _max_bead_force(forces)
    dt = torch.full((n_frames,), p.dt_start, dtype=dtype, device=device)
    alpha = torch.full((n_frames,), p.alpha_start, dtype=dtype, device=device)
    n_positive = torch.zeros(n_frames, dtype=torch.long, device=device)

    # conformations whose forces stop being finite are retired immediately
    failed = ~torch.isfinite(fmax)

    bar = tqdm(total=n_frames, disable=not progress_bar, desc="Relaxing")

    step = 0
    while True:
        # retire conformations that have converged, failed or run out of steps
        converged = (fmax < force_tolerance) & ~failed
        done = converged | failed
        if step >= max_steps:
            done = torch.ones_like(done)

        if done.any():
            finished = index[done]
            x_out[finished] = x[done]
            converged_out[finished] = converged[done]
            steps_out[finished] = step
            fmax_out[finished] = fmax[done]
            if progress_bar:
                bar.update(int(done.sum()))

            keep = ~done
            if not keep.any():
                break

            index, x, v, forces, fmax, failed = (
                index[keep],
                x[keep],
                v[keep],
                forces[keep],
                fmax[keep],
                failed[keep],
            )
            dt, alpha, n_positive = dt[keep], alpha[keep], n_positive[keep]

        # FIRE 2.0 time step and mixing control, per conformation
        if dtype == torch.float32:
            # Only the sign matters; rescale before multiplication to avoid overflow.
            f_scale = forces.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
            v_scale = v.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
            power = ((forces / f_scale) * (v / v_scale)).sum(dim=(1, 2))
        else:
            power = (forces * v).sum(dim=(1, 2))
        downhill = power > 0

        n_positive = torch.where(downhill, n_positive + 1, torch.zeros_like(n_positive))
        grow = downhill & (n_positive > p.n_delay)
        dt = torch.where(grow, torch.clamp(dt * p.f_inc, max=p.dt_max), dt)
        alpha = torch.where(grow, alpha * p.f_alpha, alpha)

        uphill = ~downhill
        # no time step cut during the initial delay (FIRE 2.0). Tensor masks
        # handle an empty uphill set without synchronizing with the host.
        shrink = uphill & (step >= p.n_delay)
        dt = torch.where(shrink, torch.clamp(dt * p.f_dec, min=p.dt_min), dt)
        alpha = torch.where(uphill, torch.full_like(alpha, p.alpha_start), alpha)

        # step back half a step and stop dead
        uphill_3d = uphill[:, None, None]
        x = torch.where(uphill_3d, x - 0.5 * dt[:, None, None] * v, x)
        v = torch.where(uphill_3d, torch.zeros_like(v), v)

        # semi-implicit Euler velocity update, then mix towards the force
        dt_3d = dt[:, None, None]
        v = v + dt_3d * forces

        v_norm = _vector_norm(v, dim=(1, 2))
        f_norm = _vector_norm(forces, dim=(1, 2))
        ratio = torch.where(f_norm > 0, v_norm / f_norm, torch.zeros_like(f_norm))
        a_3d = alpha[:, None, None]
        v = (1.0 - a_3d) * v + a_3d * ratio[:, None, None] * forces

        # cap the step so no bead moves more than max_step; the velocity is
        # scaled with it so that it reflects how far the beads actually moved
        dx = dt_3d * v
        largest = _vector_norm(dx, dim=-1).amax(dim=-1)
        scale = torch.clamp(p.max_step / largest.clamp_min(1e-30), max=1.0)
        scale_3d = scale[:, None, None]
        x_previous = x
        x = x + scale_3d * dx
        v = scale_3d * v

        forces = force_function(x, index)
        fmax = _max_bead_force(forces)

        # a conformation whose forces blow up goes back to its last good
        # coordinates and is retired as not converged on the next pass,
        # rather than poisoning the output
        bad = ~torch.isfinite(fmax)
        x = torch.where(bad[:, None, None], x_previous, x)
        failed = failed | bad

        step += 1

    bar.close()

    return FIREResult(
        coordinates=x_out,
        converged=converged_out,
        n_steps=steps_out,
        max_force=fmax_out,
    )
