"""
Batched Langevin thermalization of minimized conformations.

Minimization leaves every conformation at the bottom of its local energy
well, which is a zero-temperature structure. For Mpipi-GG that means bonds
sit at 3.81 A with almost no spread, whereas at 300 K a harmonic bond with
k = 80.3 kJ mol^-1 A^-2 fluctuates with a standard deviation of
sqrt(kT / k) = 0.18 A, which is exactly what Mpipi-GG simulations show.
Running a short stretch of Langevin dynamics after minimization, under the
same force field and restraints, puts those thermal fluctuations back: each
conformation becomes a sample from the local Boltzmann distribution around
its STARLING structure rather than the minimum of it.

We integrate with BAOAB (Leimkuhler & Matthews 2013): half kick (B), half
drift (A), an exact Ornstein-Uhlenbeck velocity refresh (O), half drift (A),
half kick (B). Of the common Langevin splittings it has the smallest error in
the configurational distribution, which matters here because we only keep
positions.

Every bead gets the same mass. The Boltzmann distribution over positions does
not depend on the masses, so this only changes how quickly that distribution
is reached, not what is sampled.

Units are Angstroms, picoseconds, kJ/mol and g/mol throughout.

References
----------
Leimkuhler, B., & Matthews, C. (2013). Rational construction of stochastic
numerical methods for molecular sampling. Applied Mathematics Research
eXpress, 2013(1), 34-56.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Final

import torch

from starling.minimizer.fire import ForceFunction

# Boltzmann constant in kJ mol^-1 K^-1 (CODATA 2018: k_B * N_A)
BOLTZMANN_KJ_PER_MOL_K: Final[float] = 0.008314462618

# a force in kJ/mol/A on a mass in g/mol gives an acceleration in A/ps^2 of
# 100 times their ratio: (1e3 J/mol / 1e-10 m) / (1e-3 kg/mol) = 1e16 m/s^2,
# and 1 m/s^2 = 1e-14 A/ps^2
ACCELERATION_PER_FORCE_OVER_MASS: Final[float] = 100.0

# Mpipi-GG simulations (and STARLING's training data) were run at 300 K
DEFAULT_TEMPERATURE_K: Final[float] = 300.0

# mean residue mass of a protein, in g/mol; see the module docstring for why
# a single value is fine
DEFAULT_BEAD_MASS: Final[float] = 110.0

# 20 fs, the Mpipi-GG simulation timestep. The stiffest routine motion is the
# bond vibration (period ~0.74 ps at this mass), so this is ~37 steps a period
DEFAULT_TIMESTEP_PS: Final[float] = 0.02

# strong coupling to the heat bath: bond vibrations (angular frequency
# ~8.5 ps^-1) are close to critically damped, so local degrees of freedom
# thermalize within a few ps while large-scale motions stay slow
DEFAULT_FRICTION_PER_PS: Final[float] = 10.0


def langevin_thermalize(
    coordinates: torch.Tensor,
    force_function: ForceFunction,
    n_steps: int,
    temperature_K: float = DEFAULT_TEMPERATURE_K,
    timestep_ps: float = DEFAULT_TIMESTEP_PS,
    friction_per_ps: float = DEFAULT_FRICTION_PER_PS,
    bead_mass: float = DEFAULT_BEAD_MASS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Run BAOAB Langevin dynamics on a batch of conformations.

    Velocities start from the Maxwell-Boltzmann distribution at
    temperature_K, so the system starts at the right kinetic temperature and
    only the potential energy needs to equilibrate.

    Parameters
    ----------
    coordinates : torch.Tensor
        Starting coordinates in Angstroms, shape (batch, n, 3). Not modified.

    force_function : callable
        force_function(x, index) returns forces in kJ/mol/A on the
        conformations x (shape (batch, n, 3)); index (shape (batch,), dtype
        long) is always the full batch here.

    n_steps : int
        Number of Langevin steps. Zero returns a copy of the input.

    temperature_K : float, optional
        Temperature in Kelvin. Default 300.

    timestep_ps : float, optional
        Integration timestep in picoseconds. Default 0.02.

    friction_per_ps : float, optional
        Langevin friction coefficient in ps^-1. Default 10.

    bead_mass : float, optional
        Mass of every bead in g/mol. Default 110.

    generator : torch.Generator, optional
        Random number generator on the coordinates' device, for reproducible
        noise. If None (default), torch's global generator is used.

    Returns
    -------
    torch.Tensor
        Coordinates after n_steps, shape (batch, n, 3).

    Raises
    ------
    ValueError
        If the coordinates are not (batch, n, 3), or a setting is out of
        range.
    RuntimeError
        If any force becomes non-finite, which means the timestep is too
        large for the forces involved. Checked before returning; no invalid
        trajectory is returned.
    """
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"coordinates must have shape (batch, n, 3), got {tuple(coordinates.shape)}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
        raise ValueError(f"n_steps must be a nonnegative integer, got {n_steps}")
    for name, value in (
        ("temperature_K", temperature_K),
        ("timestep_ps", timestep_ps),
        ("friction_per_ps", friction_per_ps),
        ("bead_mass", bead_mass),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    x = coordinates.clone()
    if n_steps == 0:
        return x

    index = torch.arange(x.shape[0], device=x.device)
    kT = BOLTZMANN_KJ_PER_MOL_K * temperature_K

    # thermal velocity scale sqrt(kT / m) in A/ps (kJ/g is 100 A^2/ps^2)
    thermal_speed = math.sqrt(ACCELERATION_PER_FORCE_OVER_MASS * kT / bead_mass)
    acceleration_scale = ACCELERATION_PER_FORCE_OVER_MASS / bead_mass

    # exact Ornstein-Uhlenbeck update over one timestep
    velocity_decay = math.exp(-friction_per_ps * timestep_ps)
    noise_scale = thermal_speed * math.sqrt(1.0 - velocity_decay**2)

    def noise() -> torch.Tensor:
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    v = thermal_speed * noise()
    acceleration = acceleration_scale * force_function(x, index)
    finite = torch.isfinite(acceleration).all()
    half_step = 0.5 * timestep_ps

    for _ in range(n_steps):
        v = v + half_step * acceleration  # B
        x = x + half_step * v  # A
        v = velocity_decay * v + noise_scale * noise()  # O
        x = x + half_step * v  # A

        forces = force_function(x, index)
        # Retain failures on device, including transient ones. Inspect once
        # at the output boundary rather than synchronizing every timestep.
        finite = finite & torch.isfinite(forces).all()
        acceleration = acceleration_scale * forces
        v = v + half_step * acceleration  # B

    if not finite:
        raise RuntimeError(
            "non-finite forces during Langevin thermalization; the timestep "
            f"({timestep_ps} ps) is too large for these structures"
        )
    return x
