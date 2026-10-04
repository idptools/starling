"""
Relax STARLING conformations with the Mpipi-GG force field.

STARLING predicts distance maps, and 3D structures are then reconstructed from
those maps by multidimensional scaling (MDS). MDS reproduces the long-range
structure of each map well, but short distances come out comparatively
poorly. Unweighted MDS compresses bonds to ~3.1-3.2 A (against 3.81 A in
Mpipi-GG) with a wide spread, overlaps beads, and occasionally breaks the
chain; the weighted reconstruction used since October 2026 is much better but
still leaves bonds short (~3.6 A) and some beads too close. The distance maps
themselves do not have these problems.

relax_conformations() fixes this by minimizing the Mpipi-GG energy of each
conformation while holding its global shape in place with flat-bottomed
distance restraints on every pair of residues far apart in sequence (see
starling.minimizer.restraints). The restraints hold the structure to the
STARLING distance map where one is available, or to the input coordinates
otherwise. Local geometry is left entirely to the force field. A short
Langevin run at 300 K then puts back the thermal fluctuations that
minimization removes (see starling.minimizer.langevin), so bond, angle and
dihedral distributions match a 300 K Mpipi-GG simulation rather than its
zero-temperature minimum.

relax_ensemble() does the same for a STARLING Ensemble and hands back a new
Ensemble carrying the relaxed structures.

Units are Angstroms, kJ/mol and kJ/mol/Angstrom throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import TYPE_CHECKING, Any, Final
import warnings

import numpy as np
import numpy.typing as npt
import torch
from tqdm.auto import tqdm

from starling import configs
from starling.minimizer.fire import FIREParameters, ForceFunction, fire_minimize
from starling.minimizer.forcefield import ENERGY_TERMS, MpipiGG, pairwise_distances
from starling.minimizer.langevin import DEFAULT_TEMPERATURE_K, langevin_thermalize
from starling.minimizer.parameters import BOND_LENGTH
from starling.minimizer.restraints import (
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_MIN_SEPARATION,
    DEFAULT_REFERENCE_FLOOR,
    DEFAULT_TOLERANCE,
    DistanceRestraints,
)

if TYPE_CHECKING:
    import mdtraj as md

    from starling.structure.ensemble import Ensemble


# a conformation is relaxed once no bead feels more than this force
# (kJ/mol/Angstrom; kT is ~2.5 kJ/mol at 300 K)
DEFAULT_FORCE_TOLERANCE: Final[float] = 1.0
DEFAULT_MAX_STEPS: Final[int] = 5000

# Langevin steps (0.02 ps each) run after minimization to restore thermal
# fluctuations. On 20 natural IDRs (200 conformations each) 250 steps already
# gives bonds the Mpipi-GG simulation width (0.0175 vs 0.0177 nm) and brings
# the angle and dihedral distributions 5x and 3x closer to the simulations;
# 1000 or 2500 steps changed none of this and cost 4-10x more
DEFAULT_THERMALIZATION_STEPS: Final[int] = 250

# conformations minimized together; memory scales as batch_size * n^2
DEFAULT_BATCH_SIZE: Final[int] = 100

# two non-bonded beads closer than this fraction of their sigma count as a
# clash. Mpipi-GG salt bridges sit at ~0.7 sigma, so this is set below that
DEFAULT_CLASH_FRACTION: Final[float] = 0.6

# the per-conformation quantities reported by GeometryDiagnostics
_DIAGNOSTIC_FIELDS: Final[tuple[str, ...]] = (
    "mean_bond_length",
    "max_bond_deviation",
    "n_clashes",
    "min_contact_ratio",
    "radius_of_gyration",
    "end_to_end_distance",
    "long_range_rmsd",
)


@dataclass(frozen=True)
class GeometryDiagnostics:
    """
    Per-conformation geometry checks, before or after relaxation.

    Every attribute is an array with one entry per conformation.

    Attributes
    ----------
    mean_bond_length : np.ndarray
        Mean distance between sequence neighbours, in Angstroms.

    max_bond_deviation : np.ndarray
        Largest |bond length - 3.81 A| in the conformation, in Angstroms.

    n_clashes : np.ndarray
        Number of non-bonded pairs (|i - j| >= 2) closer than clash_fraction
        times their Wang-Frenkel sigma.

    min_contact_ratio : np.ndarray
        Smallest r_ij / sigma_ij over all non-bonded pairs.

    radius_of_gyration : np.ndarray
        Radius of gyration in Angstroms.

    end_to_end_distance : np.ndarray
        Distance between the first and last residue in Angstroms.

    long_range_rmsd : np.ndarray
        Root-mean-square deviation, in Angstroms, of the restrained
        (sequence-distant) distances from their reference values (the
        STARLING distance map or the input coordinates).
    """

    mean_bond_length: npt.NDArray[np.float64]
    max_bond_deviation: npt.NDArray[np.float64]
    n_clashes: npt.NDArray[np.int64]
    min_contact_ratio: npt.NDArray[np.float64]
    radius_of_gyration: npt.NDArray[np.float64]
    end_to_end_distance: npt.NDArray[np.float64]
    long_range_rmsd: npt.NDArray[np.float64]


@dataclass
class RelaxationResult:
    """
    Relaxed conformations and a record of what relaxation changed.

    Attributes
    ----------
    sequence : str
        The amino acid sequence.

    coordinates : np.ndarray
        Relaxed (minimized, then thermalized unless thermalization_steps
        was 0) coordinates in Angstroms, shape (n_conformations, n, 3).

    initial_coordinates : np.ndarray
        The coordinates that were passed in, in Angstroms.

    converged : np.ndarray
        Whether each conformation reached the force tolerance (bool).

    n_steps : np.ndarray
        Number of minimization steps each conformation took.

    max_force : np.ndarray
        Largest force on any bead at the end of minimization (before any
        thermalization), in kJ/mol/Angstrom.

    energy_before : dict of str to np.ndarray
        Energy of each term (see starling.minimizer.forcefield.ENERGY_TERMS)
        for every conformation before relaxation, in kJ/mol. The restraint
        term is measured against the floored reference, so it can be
        non-zero even when the reference is the input coordinates.

    energy_after : dict of str to np.ndarray
        As energy_before, after relaxation.

    before : GeometryDiagnostics
        Geometry checks on the input coordinates.

    after : GeometryDiagnostics
        Geometry checks on the relaxed coordinates.

    reference : str
        What the restraints held the structures to: 'distance_map' or
        'coordinates'.

    settings : dict
        The settings the relaxation ran with.
    """

    sequence: str
    coordinates: npt.NDArray[np.float64]
    initial_coordinates: npt.NDArray[np.float64]
    converged: npt.NDArray[np.bool_]
    n_steps: npt.NDArray[np.int64]
    max_force: npt.NDArray[np.float64]
    energy_before: dict[str, npt.NDArray[np.float64]]
    energy_after: dict[str, npt.NDArray[np.float64]]
    before: GeometryDiagnostics
    after: GeometryDiagnostics
    reference: str
    settings: dict[str, float | int | str] = field(default_factory=dict)

    # .........................................................................
    #
    def __len__(self) -> int:
        """Number of conformations."""
        return int(self.coordinates.shape[0])

    # .........................................................................
    #
    def summary(self) -> str:
        """
        Human-readable before/after summary of the relaxation.

        Returns
        -------
        str
            A short multi-line report of ensemble-averaged diagnostics.
        """
        b, a = self.before, self.after

        def rel_change(x0: np.ndarray, x1: np.ndarray) -> float:
            return float(np.mean(np.abs(x1 - x0) / x0) * 100.0)

        lines = [
            f"Relaxed {len(self)} conformations of a {len(self.sequence)}-residue "
            f"sequence (restrained to the {self.reference.replace('_', ' ')})",
            f"  converged                  : {int(self.converged.sum())}/{len(self)} "
            f"(median {int(np.median(self.n_steps))} steps)",
            f"  mean bond length (A)       : {b.mean_bond_length.mean():6.2f} -> "
            f"{a.mean_bond_length.mean():6.2f}   (Mpipi-GG {BOND_LENGTH})",
            f"  max |bond - 3.81| (A)      : {b.max_bond_deviation.mean():6.2f} -> "
            f"{a.max_bond_deviation.mean():6.2f}   (mean over conformations)",
            f"  clashes per conformation   : {b.n_clashes.mean():6.1f} -> {a.n_clashes.mean():6.1f}",
            f"  min r/sigma                : {b.min_contact_ratio.mean():6.2f} -> "
            f"{a.min_contact_ratio.mean():6.2f}   (mean over conformations)",
            f"  radius of gyration (A)     : {b.radius_of_gyration.mean():6.2f} -> "
            f"{a.radius_of_gyration.mean():6.2f}   "
            f"(mean per-conformation change "
            f"{rel_change(b.radius_of_gyration, a.radius_of_gyration):.1f}%)",
            f"  end-to-end distance (A)    : {b.end_to_end_distance.mean():6.2f} -> "
            f"{a.end_to_end_distance.mean():6.2f}   "
            f"(mean per-conformation change "
            f"{rel_change(b.end_to_end_distance, a.end_to_end_distance):.1f}%)",
            f"  long-range RMSD to ref (A) : {b.long_range_rmsd.mean():6.2f} -> {a.long_range_rmsd.mean():6.2f}",
        ]
        return "\n".join(lines)

    # .........................................................................
    #
    def to_trajectory(self) -> md.Trajectory:
        """
        Relaxed conformations as an MDTraj trajectory of CA beads.

        Returns
        -------
        mdtraj.Trajectory
            Trajectory with one CA atom per residue (coordinates in nm).
        """
        from starling.structure.coordinates import create_ca_topology_from_coords

        return create_ca_topology_from_coords(self.sequence, self.coordinates / configs.CONVERT_ANGSTROM_TO_NM)


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------


def _as_coordinate_array(
    coordinates: npt.ArrayLike | torch.Tensor, n_residues: int
) -> npt.NDArray[np.float64] | torch.Tensor:
    """Validate coordinates without moving device-resident tensors to the CPU."""
    xyz = coordinates.detach() if isinstance(coordinates, torch.Tensor) else np.asarray(coordinates, dtype=np.float64)
    original_shape = tuple(xyz.shape)
    if xyz.ndim == 2:
        xyz = xyz[np.newaxis]

    if xyz.ndim != 3 or xyz.shape[1:] != (n_residues, 3):
        raise ValueError(
            f"coordinates must have shape (n_conformations, {n_residues}, 3) or "
            f"({n_residues}, 3) to match the sequence, got {original_shape}"
        )
    finite = torch.isfinite(xyz).all() if isinstance(xyz, torch.Tensor) else np.all(np.isfinite(xyz))
    if not finite:
        raise ValueError("coordinates contain NaN or infinite values")

    return xyz


def _as_reference_array(
    reference_distances: npt.ArrayLike | torch.Tensor, n_frames: int, n_residues: int
) -> npt.NDArray[np.float64] | torch.Tensor:
    """Validate reference maps without moving device-resident tensors to the CPU."""
    ref = (
        reference_distances.detach()
        if isinstance(reference_distances, torch.Tensor)
        else np.asarray(reference_distances, dtype=np.float64)
    )
    original_shape = tuple(ref.shape)
    if ref.ndim == 2:
        ref = ref[np.newaxis]

    if ref.shape != (n_frames, n_residues, n_residues):
        raise ValueError(
            f"reference_distances must have shape ({n_frames}, {n_residues}, "
            f"{n_residues}) to match the coordinates, got {original_shape}"
        )
    finite = torch.isfinite(ref).all() if isinstance(ref, torch.Tensor) else np.all(np.isfinite(ref))
    if not finite:
        raise ValueError("reference_distances contain NaN or infinite values")

    return ref


def _geometry_diagnostics(
    coordinates: torch.Tensor,
    reference: torch.Tensor,
    sigma: torch.Tensor,
    min_separation: int,
    clash_fraction: float,
) -> dict[str, torch.Tensor]:
    """
    Compute GeometryDiagnostics quantities for a batch on its own device.

    Parameters
    ----------
    coordinates : torch.Tensor
        Coordinates in Angstroms, shape (batch, n, 3).

    reference : torch.Tensor
        Unfloored reference distances in Angstroms, shape (batch, n, n).

    sigma : torch.Tensor
        Wang-Frenkel sigma for every residue pair, shape (n, n).

    min_separation : int
        Sequence separation from which pairs count as long-range.

    clash_fraction : float
        Clash threshold in units of sigma.

    Returns
    -------
    dict of str to torch.Tensor
        One (batch,) tensor per name in _DIAGNOSTIC_FIELDS.
    """
    n = coordinates.shape[1]
    device = coordinates.device

    r = pairwise_distances(coordinates)
    bonds = r.diagonal(offset=1, dim1=1, dim2=2)

    sequence_index = torch.arange(n, device=device)
    separation = (sequence_index[:, None] - sequence_index[None, :]).abs()
    upper_nonbonded = separation >= 2
    upper_nonbonded = upper_nonbonded & (sequence_index[:, None] < sequence_index)
    long_range = (separation >= min_separation) & (sequence_index[:, None] < sequence_index)

    ratio = r / sigma
    inf = torch.full_like(ratio, float("inf"))
    contact_ratio = torch.where(upper_nonbonded, ratio, inf)

    centred = coordinates - coordinates.mean(dim=1, keepdim=True)
    rg = torch.sqrt((centred**2).sum(dim=-1).mean(dim=-1))

    n_long = int(long_range.sum())
    if n_long > 0:
        sq_dev = torch.where(long_range, (r - reference) ** 2, torch.zeros_like(r))
        lr_rmsd = torch.sqrt(sq_dev.sum(dim=(1, 2)) / n_long)
    else:
        lr_rmsd = torch.zeros(coordinates.shape[0], dtype=r.dtype, device=device)

    if n > 1:
        mean_bond = bonds.mean(dim=-1)
        max_dev = (bonds - BOND_LENGTH).abs().amax(dim=-1)
    else:
        mean_bond = torch.full_like(rg, float("nan"))
        max_dev = torch.zeros_like(rg)

    return {
        "mean_bond_length": mean_bond,
        "max_bond_deviation": max_dev,
        "n_clashes": (contact_ratio < clash_fraction).sum(dim=(1, 2)),
        "min_contact_ratio": contact_ratio.flatten(1).amin(dim=-1),
        "radius_of_gyration": rg,
        "end_to_end_distance": r[:, 0, -1],
        "long_range_rmsd": lr_rmsd,
    }


def _to_diagnostics(chunks: list[dict[str, torch.Tensor]]) -> GeometryDiagnostics:
    """Stitch per-batch diagnostic tensors into one GeometryDiagnostics."""
    values: dict[str, np.ndarray] = {}
    for name in _DIAGNOSTIC_FIELDS:
        joined = torch.cat([chunk[name] for chunk in chunks]).cpu().numpy()
        values[name] = joined.astype(np.int64) if name == "n_clashes" else joined.astype(np.float64)
    return GeometryDiagnostics(**values)


# ------------------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------------------


def _try_fused_forces(
    forcefield: MpipiGG, restraints: DistanceRestraints, coordinates: torch.Tensor, index: torch.Tensor
) -> ForceFunction | None:
    """Warm the CUDA kernel before noise is drawn; otherwise use eager forces."""
    try:
        from starling.minimizer.fused_forces import fused_forces

        force = fused_forces(forcefield, restraints)
        force(coordinates, index)
        return force
    except Exception as error:
        warnings.warn(
            f"CUDA force fusion unavailable; using eager float32 relaxation: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def relax_conformations(
    coordinates: npt.ArrayLike | torch.Tensor,
    sequence: str,
    reference_distances: npt.ArrayLike | torch.Tensor | None = None,
    ionic_strength: float = configs.DEFAULT_IONIC_STRENGTH,
    min_separation: int = DEFAULT_MIN_SEPARATION,
    tolerance: float = DEFAULT_TOLERANCE,
    restraint_force_constant: float = DEFAULT_FORCE_CONSTANT,
    reference_floor: float = DEFAULT_REFERENCE_FLOOR,
    force_tolerance: float = DEFAULT_FORCE_TOLERANCE,
    max_steps: int = DEFAULT_MAX_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device | None = None,
    fire_parameters: FIREParameters | None = None,
    clash_fraction: float = DEFAULT_CLASH_FRACTION,
    progress_bar: bool = True,
    thermalization_steps: int = DEFAULT_THERMALIZATION_STEPS,
    temperature_K: float = DEFAULT_TEMPERATURE_K,
    seed: int | None = None,
    compile_forces: bool = False,
) -> RelaxationResult:
    """
    Relax conformations with Mpipi-GG while holding their global shape.

    Each conformation is energy-minimized under the Mpipi-GG force field
    (harmonic bonds, Wang-Frenkel and Debye-Huckel; see
    starling.minimizer.forcefield.MpipiGG) plus flat-bottomed harmonic
    restraints on every pair of residues at least min_separation apart in
    sequence (see starling.minimizer.restraints.DistanceRestraints). The
    restraints keep long-range distances within tolerance of their reference
    values, so global dimensions barely move, while bond lengths, local
    contacts and steric clashes relax to what the force field wants.

    The minimized structures are then thermalized with a short run of
    Langevin dynamics at temperature_K under the same energy (see
    starling.minimizer.langevin). Minimization alone gives zero-temperature
    structures whose bonds barely vary; thermalization restores the thermal
    fluctuations a 300 K Mpipi-GG simulation has. Pass
    thermalization_steps=0 to return the minimized structures.

    Minimization and thermalization use float32 forces, accumulation, and
    state on every device. CUDA FIRE and thermalization share one reusable Triton kernel
    when available, otherwise eager float32 forces. Seeded runs remain
    reproducible but differ from the earlier float64 thermalization.

    Parameters
    ----------
    coordinates : array-like or torch.Tensor
        Conformations in Angstroms, shape (n_conformations, n, 3) or (n, 3).
        Note that MDTraj and SOURSOP store coordinates in nm, so multiply
        those by 10 first.

    sequence : str
        Amino acid sequence (20 standard amino acids only).

    reference_distances : array-like or torch.Tensor, optional
        Distance maps in Angstroms to restrain each conformation to, shape
        (n_conformations, n, n), typically the STARLING distance maps the
        conformations were built from. If None (default), each conformation
        is restrained to its own starting distances.

    ionic_strength : float, optional
        Ionic strength in mM, which sets the Debye length. Default is
        STARLING's default ionic strength (150 mM).

    min_separation : int, optional
        Residue pairs with |i - j| >= min_separation are restrained; closer
        pairs are left to the force field. Default 4.

    tolerance : float, optional
        Half-width of the restraints' flat bottom in Angstroms. Default 0.5.

    restraint_force_constant : float, optional
        Overall restraint stiffness in kJ mol^-1 A^-2, shared out across each
        bead's restraint partners (see starling.minimizer.restraints).
        Default 20.0. Stiffer restraints keep long-range distances closer to
        the reference but stop bonds from fully relaxing: at 1000 the MDS
        bond compression is barely corrected.

    reference_floor : float, optional
        Reference distances are raised to at least this multiple of the pair's
        Wang-Frenkel sigma, so restraints never hold a clash in place.
        Default 1.0.

    force_tolerance : float, optional
        A conformation is converged once no bead feels a force above this, in
        kJ/mol/Angstrom. Default 1.0.

    max_steps : int, optional
        Maximum number of minimization steps per conformation. Default 5000.

    batch_size : int, optional
        Number of conformations minimized together. Memory use scales as
        batch_size * n^2, so lower this for long sequences on small GPUs.
        Default 100.

    device : str or torch.device, optional
        Device to run on ('cpu', 'cuda', 'cuda:N' or 'mps'). If None
        (default), the fastest available device is used.

    fire_parameters : FIREParameters, optional
        Settings for the FIRE minimizer. If None (default), the defaults in
        starling.minimizer.fire.FIREParameters are used.

    clash_fraction : float, optional
        Two non-bonded beads closer than this fraction of their sigma are
        reported as a clash in the diagnostics. Only affects reporting.
        Default 0.6.

    progress_bar : bool, optional
        If True (default), show a progress bar.

    thermalization_steps : int, optional
        Number of Langevin steps (0.02 ps each) run after minimization.
        Default DEFAULT_THERMALIZATION_STEPS; 0 skips thermalization.

    temperature_K : float, optional
        Thermalization temperature in Kelvin. Default 300, the temperature
        of the Mpipi-GG simulations STARLING was trained on.

    seed : int, optional
        Seed for the thermalization noise. If None (default), torch's global
        generator is used, so torch.manual_seed() upstream still makes runs
        reproducible.

    compile_forces : bool, optional
        Fuse the complete thermalization force calculation with torch.compile
        on CUDA instead of the reusable kernel. Default False. Adds compilation startup time, amortized over
        repeated batches; FIRE remains eager. Compilation failures warn and
        fall back to eager thermalization before consuming random draws.

    Returns
    -------
    RelaxationResult
        Relaxed coordinates plus before/after energies and geometry
        diagnostics. Call .summary() for a readable overview.

    Raises
    ------
    ValueError
        If the coordinates or reference maps do not match the sequence, or
        contain non-finite values, or a setting is out of range.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for name, value in (("max_steps", max_steps), ("thermalization_steps", thermalization_steps)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer, got {value}")
    for name, value in (("force_tolerance", force_tolerance), ("temperature_K", temperature_K)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}")

    forcefield = MpipiGG(sequence, ionic_strength=ionic_strength, device=device, dtype=torch.float32)
    n = forcefield.n_residues

    xyz = _as_coordinate_array(coordinates, n)
    n_frames = xyz.shape[0]

    ref_array = None if reference_distances is None else _as_reference_array(reference_distances, n_frames, n)

    device_t, dtype = forcefield.device, forcefield.dtype
    if not isinstance(compile_forces, bool):
        raise TypeError("compile_forces must be a bool")
    if compile_forces and device_t.type != "cuda":
        raise ValueError("compile_forces requires a CUDA device")
    compilation_available = compile_forces
    compiled_batches = 0
    fusion_available = device_t.type == "cuda" and not compile_forces
    fused_batches = 0
    fused_fire_batches = 0
    sigma = torch.as_tensor(forcefield.parameters.sigma, dtype=dtype, device=device_t)

    relaxed = np.empty(xyz.shape, dtype=np.float64)
    converged = np.zeros(n_frames, dtype=bool)
    n_steps = np.zeros(n_frames, dtype=np.int64)
    max_force = np.zeros(n_frames, dtype=np.float64)
    energy_chunks: dict[str, list[np.ndarray]] = {
        f"{stage}_{term}": [] for stage in ("before", "after") for term in ENERGY_TERMS
    }
    before_chunks: list[dict[str, torch.Tensor]] = []
    after_chunks: list[dict[str, torch.Tensor]] = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_t)
        generator.manual_seed(seed)

    bar = tqdm(total=n_frames, disable=not progress_bar, desc="Relaxing")

    for start in range(0, n_frames, batch_size):
        stop = min(start + batch_size, n_frames)
        x0 = torch.as_tensor(xyz[start:stop], dtype=dtype, device=device_t)

        if ref_array is None:
            reference = pairwise_distances(x0)
        else:
            reference = torch.as_tensor(ref_array[start:stop], dtype=dtype, device=device_t)

        restraints = DistanceRestraints(
            reference,
            sigma=sigma,
            min_separation=min_separation,
            tolerance=tolerance,
            force_constant=restraint_force_constant,
            reference_floor=reference_floor,
            device=device_t,
            dtype=dtype,
        )
        batch_index = torch.arange(stop - start, device=device_t)

        def force_function(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return forcefield.evaluate(x, restraints, index).forces

        e_before = forcefield.energy(x0, restraints, batch_index)
        before_chunks.append(_geometry_diagnostics(x0, reference, sigma, min_separation, clash_fraction))

        fire_force_function: ForceFunction = force_function
        if fusion_available:
            fused_force = _try_fused_forces(forcefield, restraints, x0, batch_index)
            if fused_force is None:
                fusion_available = False
            else:
                fire_force_function = fused_force
                fused_fire_batches += 1
        fire = fire_minimize(
            x0,
            fire_force_function,
            force_tolerance=force_tolerance,
            max_steps=max_steps,
            parameters=fire_parameters,
        )

        thermal_force_function: ForceFunction = fire_force_function
        thermal_coordinates = fire.coordinates
        if fire_force_function is not force_function and thermalization_steps > 0:
            fused_batches += 1
        if compilation_available and thermalization_steps > 0:
            try:
                compiled_force = torch.compile(thermal_force_function, fullgraph=True, dynamic=True)
                # Compile before the stochastic loop, so fallback cannot repeat
                # or discard random draws. FIRE's changing batch stays eager.
                compiled_force(thermal_coordinates, batch_index)
                thermal_force_function = compiled_force
                compiled_batches += 1
            except Exception as error:
                warnings.warn(
                    f"Whole-force compilation failed; using eager thermalization: {type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                compilation_available = False

        # put thermal fluctuations back into the zero-temperature minima
        x_final = langevin_thermalize(
            thermal_coordinates,
            thermal_force_function,
            thermalization_steps,
            temperature_K=temperature_K,
            generator=generator,
        )

        e_after = forcefield.energy(x_final, restraints, batch_index)
        after_chunks.append(_geometry_diagnostics(x_final, reference, sigma, min_separation, clash_fraction))

        relaxed[start:stop] = x_final.cpu().numpy()
        converged[start:stop] = fire.converged.cpu().numpy()
        n_steps[start:stop] = fire.n_steps.cpu().numpy()
        max_force[start:stop] = fire.max_force.cpu().numpy()
        for term in ENERGY_TERMS:
            energy_chunks[f"before_{term}"].append(e_before[term].cpu().numpy())
            energy_chunks[f"after_{term}"].append(e_after[term].cpu().numpy())

        bar.update(stop - start)

    bar.close()

    return RelaxationResult(
        sequence=forcefield.sequence,
        coordinates=relaxed,
        initial_coordinates=(xyz.cpu().numpy().astype(np.float64) if isinstance(xyz, torch.Tensor) else xyz),
        converged=converged,
        n_steps=n_steps,
        max_force=max_force,
        energy_before={t: np.concatenate(energy_chunks[f"before_{t}"]).astype(np.float64) for t in ENERGY_TERMS},
        energy_after={t: np.concatenate(energy_chunks[f"after_{t}"]).astype(np.float64) for t in ENERGY_TERMS},
        before=_to_diagnostics(before_chunks),
        after=_to_diagnostics(after_chunks),
        reference="coordinates" if ref_array is None else "distance_map",
        settings={
            "ionic_strength": float(ionic_strength),
            "debye_length": forcefield.debye_length,
            "min_separation": int(min_separation),
            "tolerance": float(tolerance),
            "restraint_force_constant": float(restraint_force_constant),
            "reference_floor": float(reference_floor),
            "force_tolerance": float(force_tolerance),
            "max_steps": int(max_steps),
            "thermalization_steps": int(thermalization_steps),
            "temperature_K": float(temperature_K),
            "device": str(device_t),
            "compile_forces": compile_forces,
            "compiled_thermalization_batches": compiled_batches,
            "fused_thermalization_batches": fused_batches,
            "fused_minimization_batches": fused_fire_batches,
            "thermalization_dtype": "float32" if thermalization_steps > 0 else "not_run",
            "minimization_dtype": str(dtype).removeprefix("torch."),
        },
    )


def relax_ensemble(
    ensemble: Ensemble,
    use_distance_maps: bool = True,
    ionic_strength: float | None = None,
    progress_bar: bool = True,
    **kwargs: Any,
) -> tuple[Ensemble, RelaxationResult]:
    """
    Relax the 3D structures of a STARLING ensemble with Mpipi-GG.

    Takes the ensemble's reconstructed structures (building them by MDS first
    if they do not exist yet), relaxes them with relax_conformations(), and
    returns a new Ensemble that carries the same distance maps and the
    relaxed structures. The original ensemble is left untouched.

    Generated ensembles record their ionic strength. Legacy or manually created
    ensembles may not have this metadata; STARLING's default is used for those.

    Parameters
    ----------
    ensemble : starling.structure.ensemble.Ensemble
        The ensemble to relax.

    use_distance_maps : bool, optional
        If True (default), restrain each structure to the STARLING distance
        map it was built from. If False, restrain each structure to its own
        starting distances.

    ionic_strength : float, optional
        Ionic strength in mM. Overrides the ensemble metadata; if omitted,
        uses the recorded value or STARLING's default (150 mM).

    progress_bar : bool, optional
        If True (default), show progress bars.

    **kwargs
        Any other keyword argument of relax_conformations() (for example
        tolerance, min_separation, device or batch_size).

    Returns
    -------
    tuple
        [0] Ensemble: a new ensemble with the relaxed structures attached.
        [1] RelaxationResult: the relaxed coordinates and diagnostics.
    """
    from soursop.sstrajectory import SSTrajectory

    from starling.structure.ensemble import Ensemble

    if ionic_strength is None:
        ionic_strength = (
            ensemble.ionic_strength if ensemble.ionic_strength is not None else configs.DEFAULT_IONIC_STRENGTH
        )

    # SOURSOP/MDTraj coordinates are in nm
    trajectory = ensemble.build_ensemble_trajectory(progress_bar=progress_bar)
    coordinates = trajectory.traj.xyz.astype(np.float64) * configs.CONVERT_ANGSTROM_TO_NM
    distance_maps = ensemble.distance_maps()

    result = relax_conformations(
        coordinates,
        ensemble.sequence,
        reference_distances=distance_maps if use_distance_maps else None,
        ionic_strength=ionic_strength,
        progress_bar=progress_bar,
        **kwargs,  # type: ignore[arg-type]
    )

    relaxed_protein = SSTrajectory(TRJ=result.to_trajectory()).proteinTrajectoryList[0]
    relaxed_ensemble = Ensemble(
        distance_maps,
        ensemble.sequence,
        ssprot_ensemble=relaxed_protein,
        ionic_strength=ionic_strength,
    )

    return relaxed_ensemble, result
