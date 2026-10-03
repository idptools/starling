"""
Batched PyTorch implementation of the Mpipi-GG force field.

MpipiGG binds the force field to one sequence and a set of solution
conditions, and then evaluates forces (and optionally energies) for a batch
of conformations of that sequence at once, on CPU, CUDA or MPS.

Every term in the energy (bonds, Wang-Frenkel, Debye-Huckel, and any
restraints) depends only on inter-residue distances, so for each term we
compute dU/dr on the full (batch, n, n) distance matrix and turn the summed
derivatives into forces with a single batched matrix product:

    F_i = sum_j g_ij (x_i - x_j),  where g_ij = -(dU_ij / dr_ij) / r_ij

which avoids ever building a (batch, n, n, 3) displacement tensor. Energies
are only computed on request, because the minimizer only needs forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from starling import configs, utilities
from starling.minimizer.parameters import (
    BOND_FORCE_CONSTANT,
    BOND_LENGTH,
    COULOMB_CONSTANT,
    DEFAULT_COULOMB_CUTOFF,
    DEFAULT_DIELECTRIC,
    WANG_FRENKEL_CUTOFF_RATIO,
    SequenceParameters,
    build_sequence_parameters,
    debye_length as compute_debye_length,
)
from starling.structure.coordinates import get_tensor_dtype

if TYPE_CHECKING:
    from starling.minimizer.restraints import DistanceRestraints

# Distances are clamped to at least this value (Angstroms) before any
# potential is evaluated. Beads this close only occur in badly broken input
# structures; the clamp keeps the steep Wang-Frenkel wall finite in float32.
MIN_DISTANCE: Final[float] = 0.1

# names of the energy terms, in the order they are reported
ENERGY_TERMS: Final[tuple[str, ...]] = (
    "bond",
    "wang_frenkel",
    "debye_huckel",
    "restraint",
    "total",
)


@dataclass
class ForceFieldEvaluation:
    """
    Forces (and optionally energies) for a batch of conformations.

    Attributes
    ----------
    forces : torch.Tensor
        Force on every bead in kJ/mol/Angstrom, shape (batch, n, 3).

    energies : dict of str to torch.Tensor, or None
        Per-conformation energy in kJ/mol for each term in ENERGY_TERMS, each
        of shape (batch,). None unless energies were requested.
    """

    forces: torch.Tensor
    energies: dict[str, torch.Tensor] | None


def pairwise_distances(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Compute every inter-bead distance for a batch of conformations.

    This uses the direct (difference-based) distance calculation rather than
    the matrix-multiplication shortcut, which loses precision for beads that
    are close together relative to their distance from the origin.

    Parameters
    ----------
    coordinates : torch.Tensor
        Bead positions in Angstroms, shape (batch, n, 3).

    Returns
    -------
    torch.Tensor
        Distances in Angstroms, shape (batch, n, n).
    """
    return torch.cdist(
        coordinates, coordinates, compute_mode="donot_use_mm_for_euclid_dist"
    )


def forces_from_pair_derivatives(
    coordinates: torch.Tensor, distances: torch.Tensor, dudr: torch.Tensor
) -> torch.Tensor:
    """
    Turn pairwise energy derivatives into per-bead forces.

    Parameters
    ----------
    coordinates : torch.Tensor
        Bead positions in Angstroms, shape (batch, n, 3). These should be
        centred on the origin to keep the matrix product well conditioned.

    distances : torch.Tensor
        Distances in Angstroms, shape (batch, n, n), with no zero entries
        (including on the diagonal).

    dudr : torch.Tensor
        Symmetric derivative of the pair energy with respect to distance, in
        kJ/mol/Angstrom, shape (batch, n, n), zero on the diagonal.

    Returns
    -------
    torch.Tensor
        Forces in kJ/mol/Angstrom, shape (batch, n, 3).
    """
    g = -dudr / distances
    return g.sum(dim=-1, keepdim=True) * coordinates - torch.bmm(g, coordinates)


class MpipiGG:
    """
    The Mpipi-GG force field for one sequence.

    Holds the per-pair parameters for the sequence on the chosen device and
    evaluates forces and energies for batches of conformations. The energy
    is the sum of

    * harmonic bonds between sequence neighbours (k = 80.3 kJ mol^-1 A^-2,
      r0 = 3.81 A);
    * the Wang-Frenkel potential between every pair of residues two or more
      positions apart, truncated at 3 sigma;
    * the Debye-Huckel screened Coulomb potential between every pair of
      charged residues two or more positions apart, truncated at the Coulomb
      cutoff.

    Directly bonded neighbours are excluded from both non-bonded terms, as in
    Mpipi; without this exclusion the Wang-Frenkel wall (sigma is roughly 5-8
    A) would fight the 3.81 A bond.

    Attributes
    ----------
    sequence : str
        The amino acid sequence.

    parameters : SequenceParameters
        The Mpipi-GG pair parameters for the sequence (NumPy arrays).

    debye_length : float
        Debye screening length in Angstroms.

    dielectric : float
        Relative dielectric constant.

    coulomb_cutoff : float
        Debye-Huckel cutoff in Angstroms.

    device : torch.device
        Device every tensor lives on.

    dtype : torch.dtype
        Floating point type used for every calculation.
    """

    def __init__(
        self,
        sequence: str,
        ionic_strength: float = configs.DEFAULT_IONIC_STRENGTH,
        dielectric: float = DEFAULT_DIELECTRIC,
        debye_length: float | None = None,
        coulomb_cutoff: float = DEFAULT_COULOMB_CUTOFF,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Build the force field for a sequence.

        Parameters
        ----------
        sequence : str
            Amino acid sequence made up of the 20 standard amino acids.

        ionic_strength : float, optional
            Ionic strength in mM, used to set the Debye length. Default is
            STARLING's default ionic strength (150 mM). Ignored if
            debye_length is given.

        dielectric : float, optional
            Relative dielectric constant. Default 80.

        debye_length : float, optional
            Debye screening length in Angstroms. If None (default) this is
            computed from ionic_strength (see
            starling.minimizer.parameters.debye_length()).

        coulomb_cutoff : float, optional
            Debye-Huckel cutoff in Angstroms. Default 35, as in Mpipi.

        device : str or torch.device, optional
            Device to run on ('cpu', 'cuda', 'cuda:N' or 'mps'). If None
            (default), the fastest available device is used.

        dtype : torch.dtype, optional
            Floating point type. If None (default), float64 is used everywhere
            except MPS, which only supports float32.

        Raises
        ------
        ValueError
            If the sequence contains a residue Mpipi-GG has no parameters for,
            or if the dielectric, Debye length or cutoff are not positive.
        """
        if dielectric <= 0:
            raise ValueError(f"dielectric must be positive, got {dielectric}")
        if coulomb_cutoff <= 0:
            raise ValueError(f"coulomb_cutoff must be positive, got {coulomb_cutoff}")

        if debye_length is None:
            debye_length = compute_debye_length(ionic_strength)
        elif debye_length <= 0:
            raise ValueError(f"debye_length must be positive, got {debye_length}")

        self.parameters: SequenceParameters = build_sequence_parameters(sequence)
        self.sequence: str = self.parameters.sequence
        self.debye_length: float = float(debye_length)
        self.dielectric: float = float(dielectric)
        self.coulomb_cutoff: float = float(coulomb_cutoff)

        self.device: torch.device = utilities.check_device(device)
        self.dtype: torch.dtype = (
            dtype if dtype is not None else get_tensor_dtype(self.device)
        )

        self._build_tensors()

    # .........................................................................
    #
    @property
    def n_residues(self) -> int:
        """int: Number of residues (beads) in the sequence."""
        return len(self.sequence)

    # .........................................................................
    #
    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        """Move a NumPy array onto the force field's device and dtype."""
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    # .........................................................................
    #
    def _build_tensors(self) -> None:
        """
        Precompute every per-pair quantity the force and energy loops need.

        Where every pair shares the same Wang-Frenkel exponent (true for all
        20 amino acids in Mpipi-GG, where mu = 2 and nu = 1) we store it as a
        plain float, which lets torch.pow take its faster scalar path.
        """
        p = self.parameters
        n = self.n_residues

        separation = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        nonbonded = separation >= 2

        # Wang-Frenkel: mask, sigma, cutoff, epsilon * alpha and the exponents.
        # (R / r)^(2 mu) is (R / sigma)^(2 mu) (sigma / r)^(2 mu), so we store
        # the constant (R / sigma)^(2 mu) and only take one power per step
        self._wf_mask = torch.as_tensor(nonbonded, device=self.device)
        self._sigma = self._tensor(p.sigma)
        self._wf_cutoff = self._tensor(WANG_FRENKEL_CUTOFF_RATIO * p.sigma)
        self._eps_alpha = self._tensor(p.epsilon * p.alpha)

        self._two_mu: float | torch.Tensor
        self._two_nu: float | torch.Tensor
        self._cutoff_power: float | torch.Tensor
        if np.all(p.mu == p.mu.flat[0]) and np.all(p.nu == p.nu.flat[0]):
            mu, nu = float(p.mu.flat[0]), float(p.nu.flat[0])
            self._two_mu = 2.0 * mu
            self._two_nu = 2.0 * nu
            self._cutoff_power = WANG_FRENKEL_CUTOFF_RATIO ** (2.0 * mu)
        else:
            self._two_mu = self._tensor(2.0 * p.mu)
            self._two_nu = self._tensor(2.0 * p.nu)
            self._cutoff_power = self._tensor(WANG_FRENKEL_CUTOFF_RATIO ** (2.0 * p.mu))

        # Debye-Huckel: q_i q_j e^2 N_A / (4 pi eps0 eps_r), only for charged
        # non-bonded pairs. Sequences with fewer than two charges skip it
        qq = COULOMB_CONSTANT * np.outer(p.charges, p.charges) / self.dielectric
        dh_mask = nonbonded & (qq != 0.0)
        self._has_electrostatics = bool(dh_mask.any())
        self._dh_mask = torch.as_tensor(dh_mask, device=self.device)
        self._qq = self._tensor(np.where(dh_mask, qq, 0.0))
        self._kappa = 0.0 if np.isinf(self.debye_length) else 1.0 / self.debye_length

    # .........................................................................
    #
    def _wang_frenkel(
        self, r: torch.Tensor, compute_energy: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Wang-Frenkel dU/dr (and optionally energy) for every pair.

        Parameters
        ----------
        r : torch.Tensor
            Distances in Angstroms, shape (batch, n, n), already clamped to
            MIN_DISTANCE.

        compute_energy : bool
            If True, also return the per-conformation energy.

        Returns
        -------
        tuple
            [0] torch.Tensor: dU/dr in kJ/mol/Angstrom, shape (batch, n, n).
            [1] torch.Tensor or None: energy in kJ/mol, shape (batch,).
        """
        within = self._wf_mask & (r < self._wf_cutoff)

        # a = (sigma / r)^(2 mu) and b = (R / r)^(2 mu)
        a = torch.pow(self._sigma / r, self._two_mu)
        b = self._cutoff_power * a
        b_minus_1 = b - 1.0

        # dphi/dr = -(2 mu eps alpha / r) (b - 1)^(2 nu - 1) [a (b - 1) + 2 nu (a - 1) b]
        dudr = (
            -(self._two_mu * self._eps_alpha / r)
            * torch.pow(b_minus_1, self._two_nu - 1.0)
            * (a * b_minus_1 + self._two_nu * (a - 1.0) * b)
        )
        dudr = torch.where(within, dudr, torch.zeros_like(dudr))

        energy = None
        if compute_energy:
            phi = self._eps_alpha * (a - 1.0) * torch.pow(b_minus_1, self._two_nu)
            phi = torch.where(within, phi, torch.zeros_like(phi))
            # every pair appears twice in the full matrix
            energy = 0.5 * phi.sum(dim=(1, 2))

        return dudr, energy

    # .........................................................................
    #
    def _debye_huckel(
        self, r: torch.Tensor, compute_energy: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Debye-Huckel dU/dr (and optionally energy) for every pair.

        Parameters
        ----------
        r : torch.Tensor
            Distances in Angstroms, shape (batch, n, n), already clamped to
            MIN_DISTANCE.

        compute_energy : bool
            If True, also return the per-conformation energy.

        Returns
        -------
        tuple
            [0] torch.Tensor: dU/dr in kJ/mol/Angstrom, shape (batch, n, n).
            [1] torch.Tensor or None: energy in kJ/mol, shape (batch,).
        """
        within = self._dh_mask & (r <= self.coulomb_cutoff)

        u = self._qq * torch.exp(-self._kappa * r) / r
        u = torch.where(within, u, torch.zeros_like(u))

        # d/dr [exp(-kappa r) / r] = -(kappa + 1 / r) exp(-kappa r) / r
        dudr = -u * (self._kappa + 1.0 / r)

        energy = 0.5 * u.sum(dim=(1, 2)) if compute_energy else None

        return dudr, energy

    # .........................................................................
    #
    def evaluate(
        self,
        coordinates: torch.Tensor,
        restraints: DistanceRestraints | None = None,
        frame_index: torch.Tensor | None = None,
        compute_energy: bool = False,
    ) -> ForceFieldEvaluation:
        """
        Forces (and optionally energies) for a batch of conformations.

        Parameters
        ----------
        coordinates : torch.Tensor
            Bead positions in Angstroms, shape (batch, n, 3), on this force
            field's device.

        restraints : DistanceRestraints, optional
            Restraints to add to the Mpipi-GG energy. If None (default) only
            the force field is evaluated.

        frame_index : torch.Tensor, optional
            For each conformation in the batch, the index of the restraint
            reference it should be held to (see DistanceRestraints). Required
            if restraints are given.

        compute_energy : bool, optional
            If True, also compute the energy of every term. Default False.

        Returns
        -------
        ForceFieldEvaluation
            Forces in kJ/mol/Angstrom and, if requested, energies in kJ/mol.

        Raises
        ------
        ValueError
            If the coordinates do not have shape (batch, n, 3), or restraints
            are given without a frame_index.
        """
        if coordinates.ndim != 3 or coordinates.shape[1:] != (self.n_residues, 3):
            raise ValueError(
                f"coordinates must have shape (batch, {self.n_residues}, 3), got "
                f"{tuple(coordinates.shape)}"
            )
        if restraints is not None and frame_index is None:
            raise ValueError("frame_index is required when restraints are given")

        # forces are translation invariant; centring keeps the batched matrix
        # product in forces_from_pair_derivatives() well conditioned
        x = coordinates - coordinates.mean(dim=1, keepdim=True)
        r = pairwise_distances(x).clamp_min(MIN_DISTANCE)

        dudr, e_wf = self._wang_frenkel(r, compute_energy)

        e_dh: torch.Tensor | None = None
        if self._has_electrostatics:
            dh_dudr, e_dh = self._debye_huckel(r, compute_energy)
            dudr = dudr + dh_dudr

        e_restraint: torch.Tensor | None = None
        if restraints is not None:
            assert frame_index is not None
            restraint_dudr, e_restraint = restraints.pair_terms(
                r, frame_index, compute_energy
            )
            dudr = dudr + restraint_dudr

        # harmonic bonds live on the first off-diagonals; both non-bonded terms
        # are masked to zero there, so we can add the bond derivative in place
        bond_r = r.diagonal(offset=1, dim1=1, dim2=2)
        bond_dudr = BOND_FORCE_CONSTANT * (bond_r - BOND_LENGTH)
        dudr.diagonal(offset=1, dim1=1, dim2=2).add_(bond_dudr)
        dudr.diagonal(offset=-1, dim1=1, dim2=2).add_(bond_dudr)

        forces = forces_from_pair_derivatives(x, r, dudr)

        energies = None
        if compute_energy:
            zeros = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)
            e_bond = (0.5 * BOND_FORCE_CONSTANT * (bond_r - BOND_LENGTH) ** 2).sum(-1)
            energies = {
                "bond": e_bond,
                "wang_frenkel": e_wf if e_wf is not None else zeros,
                "debye_huckel": e_dh if e_dh is not None else zeros,
                "restraint": e_restraint if e_restraint is not None else zeros,
            }
            energies["total"] = (
                energies["bond"]
                + energies["wang_frenkel"]
                + energies["debye_huckel"]
                + energies["restraint"]
            )

        return ForceFieldEvaluation(forces=forces, energies=energies)

    # .........................................................................
    #
    def energy(
        self,
        coordinates: torch.Tensor,
        restraints: DistanceRestraints | None = None,
        frame_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Energies of a batch of conformations, split by term.

        Parameters
        ----------
        coordinates : torch.Tensor
            Bead positions in Angstroms, shape (batch, n, 3).

        restraints : DistanceRestraints, optional
            Restraints to include. Default None.

        frame_index : torch.Tensor, optional
            Restraint reference index for each conformation (see evaluate()).

        Returns
        -------
        dict of str to torch.Tensor
            Energy in kJ/mol for each term in ENERGY_TERMS, each of shape
            (batch,).
        """
        result = self.evaluate(
            coordinates, restraints, frame_index, compute_energy=True
        )
        assert result.energies is not None
        return result.energies
