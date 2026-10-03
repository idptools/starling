"""
Distance restraints that hold the global shape of a conformation in place.

The idea is to let the force field repair local geometry (bond lengths,
short-range contacts, overlapping beads) while keeping the long-range
structure of each conformation where STARLING put it. We do this with flat-
bottomed harmonic restraints on every pair of residues separated by at least
min_separation positions in sequence:

    U_ij = (k_pair / 2) max(0, |r_ij - r0_ij| - tolerance)^2

where r0_ij is the reference distance. Pairs closer together in sequence are
left entirely to the force field. Inside the tolerance a restraint exerts no
force at all, so small adjustments are free and only real departures from the
reference are resisted.

Because a bead takes part in roughly n restraints, the stiffness per pair is
normalised by the mean number of restraint partners per bead:

    k_pair = force_constant / mean(partners per bead)

so force_constant sets the overall stiffness resisting displacement of a bead
independent of chain length, rather than growing with n.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import torch

# default restraint settings; see DistanceRestraints for what each one does
DEFAULT_MIN_SEPARATION: Final[int] = 4
DEFAULT_TOLERANCE: Final[float] = 0.5
DEFAULT_FORCE_CONSTANT: Final[float] = 20.0
DEFAULT_REFERENCE_FLOOR: Final[float] = 1.0


class DistanceRestraints:
    """
    Flat-bottomed harmonic restraints on sequence-distant residue pairs.

    Holds one (n, n) reference distance map per conformation. When the
    force field is evaluated on a batch, frame_index says which reference
    each conformation in the batch is held to, which lets the minimizer
    drop converged conformations from the batch without copying references
    around.

    Attributes
    ----------
    min_separation : int
        Pairs with |i - j| >= min_separation are restrained.

    tolerance : float
        Half-width of the flat bottom in Angstroms.

    force_constant : float
        Overall restraint stiffness in kJ mol^-1 A^-2 (see module docstring).

    pair_force_constant : float
        Force constant applied to each restrained pair in kJ mol^-1 A^-2.

    n_restrained_pairs : int
        Number of unique residue pairs that are restrained.
    """

    def __init__(
        self,
        reference_distances: npt.ArrayLike | torch.Tensor,
        sigma: npt.ArrayLike | torch.Tensor | None = None,
        min_separation: int = DEFAULT_MIN_SEPARATION,
        tolerance: float = DEFAULT_TOLERANCE,
        force_constant: float = DEFAULT_FORCE_CONSTANT,
        reference_floor: float = DEFAULT_REFERENCE_FLOOR,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Set up restraints for a set of conformations.

        Parameters
        ----------
        reference_distances : array-like or torch.Tensor
            Reference distance maps in Angstroms, shape (n_frames, n, n) or
            (n, n) for a single conformation. These are symmetrised here, so
            upper- or lower-triangular maps are not accepted; pass full maps.

        sigma : array-like or torch.Tensor, optional
            Wang-Frenkel sigma for every residue pair in Angstroms, shape
            (n, n). If given, every reference distance is raised to at least
            reference_floor * sigma_ij, so a restraint never tries to hold two
            beads in a steric clash that the force field is trying to resolve.
            Default None (no floor).

        min_separation : int, optional
            Minimum sequence separation |i - j| of a restrained pair. Default
            4, so bonds, i/i+2 and i/i+3 pairs are left to the force field.

        tolerance : float, optional
            Half-width of the flat bottom in Angstroms: a restrained distance
            can move this far from its reference without any penalty. Default
            0.5.

        force_constant : float, optional
            Overall restraint stiffness in kJ mol^-1 A^-2, which is divided
            among each bead's restraint partners (see module docstring).
            Default 20.0.

        reference_floor : float, optional
            Minimum reference distance in units of sigma_ij. Only used if
            sigma is given. Default 1.0.

        device : str or torch.device, optional
            Device to hold the references on. Default 'cpu'.

        dtype : torch.dtype, optional
            Floating point type. Default torch.float64.

        Raises
        ------
        ValueError
            If the reference maps are not square, sigma does not match them,
            or any setting is out of range.
        """
        if min_separation < 1:
            raise ValueError(f"min_separation must be >= 1, got {min_separation}")
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        if force_constant < 0:
            raise ValueError(f"force_constant must be >= 0, got {force_constant}")
        if reference_floor < 0:
            raise ValueError(f"reference_floor must be >= 0, got {reference_floor}")

        reference = torch.as_tensor(reference_distances, dtype=dtype, device=device)
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if reference.ndim != 3 or reference.shape[1] != reference.shape[2]:
            raise ValueError(
                "reference_distances must have shape (n_frames, n, n) or (n, n), "
                f"got {tuple(reference.shape)}"
            )

        n = reference.shape[1]
        reference = 0.5 * (reference + reference.transpose(1, 2))

        if sigma is not None:
            sigma_t = torch.as_tensor(sigma, dtype=dtype, device=device)
            if sigma_t.shape != (n, n):
                raise ValueError(
                    f"sigma must have shape ({n}, {n}) to match the reference maps, "
                    f"got {tuple(sigma_t.shape)}"
                )
            reference = torch.maximum(reference, reference_floor * sigma_t)

        separation = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        mask = separation >= min_separation
        partners_per_bead = mask.sum(axis=1)
        mean_partners = float(partners_per_bead.mean()) if n > 0 else 0.0

        self.min_separation: int = int(min_separation)
        self.tolerance: float = float(tolerance)
        self.force_constant: float = float(force_constant)
        self.pair_force_constant: float = (
            self.force_constant / mean_partners if mean_partners > 0 else 0.0
        )
        self.n_restrained_pairs: int = int(mask.sum()) // 2

        self._reference = reference
        self._mask = torch.as_tensor(mask, device=device)

        # cache of the reference maps for the most recent frame_index, so an
        # unchanged batch does not re-gather its references every step
        self._cached_index: torch.Tensor | None = None
        self._cached_reference: torch.Tensor | None = None

    # .........................................................................
    #
    @property
    def n_frames(self) -> int:
        """int: Number of reference distance maps held."""
        return int(self._reference.shape[0])

    # .........................................................................
    #
    def _reference_for(self, frame_index: torch.Tensor) -> torch.Tensor:
        """Reference maps for the given frames, reusing the last gather if possible."""
        cached = self._cached_index
        if (
            cached is None
            or cached.shape != frame_index.shape
            or not torch.equal(cached, frame_index)
        ):
            self._cached_index = frame_index.clone()
            self._cached_reference = self._reference.index_select(0, frame_index)

        assert self._cached_reference is not None
        return self._cached_reference

    # .........................................................................
    #
    def pair_terms(
        self,
        distances: torch.Tensor,
        frame_index: torch.Tensor,
        compute_energy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Restraint dU/dr (and optionally energy) for every pair.

        Parameters
        ----------
        distances : torch.Tensor
            Current distances in Angstroms, shape (batch, n, n).

        frame_index : torch.Tensor
            Index of the reference map for each conformation in the batch,
            shape (batch,), dtype long.

        compute_energy : bool, optional
            If True, also return the per-conformation energy. Default False.

        Returns
        -------
        tuple
            [0] torch.Tensor: dU/dr in kJ/mol/Angstrom, shape (batch, n, n).
            [1] torch.Tensor or None: energy in kJ/mol, shape (batch,).
        """
        deviation = distances - self._reference_for(frame_index)
        excess = torch.clamp_min(deviation.abs() - self.tolerance, 0.0)
        excess = torch.where(self._mask, excess, torch.zeros_like(excess))

        dudr = self.pair_force_constant * excess * torch.sign(deviation)

        energy = None
        if compute_energy:
            # every pair appears twice in the full matrix
            energy = 0.25 * self.pair_force_constant * (excess**2).sum(dim=(1, 2))

        return dudr, energy
