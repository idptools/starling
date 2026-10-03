"""
Reference NumPy implementations of the Mpipi-GG potentials.

These are the plain, readable definitions of the three terms that make up the
Mpipi-GG energy. The batched PyTorch implementation in
starling.minimizer.forcefield is what the minimizer actually uses; these
functions exist so that implementation can be checked against something
simple, and so individual potentials can be evaluated and plotted directly.

Units are Angstroms for distances and kJ/mol for energies throughout.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from starling.minimizer.parameters import (
    BOND_FORCE_CONSTANT,
    BOND_LENGTH,
    COULOMB_CONSTANT,
    DEFAULT_COULOMB_CUTOFF,
    DEFAULT_DIELECTRIC,
    WANG_FRENKEL_CUTOFF_RATIO,
    SequenceParameters,
    wang_frenkel_alpha,
)


def harmonic_bond(
    r: npt.ArrayLike,
    force_constant: float = BOND_FORCE_CONSTANT,
    bond_length: float = BOND_LENGTH,
) -> npt.NDArray[np.float64]:
    """
    Harmonic bond energy, E = (k / 2) (r - r0)^2.

    Parameters
    ----------
    r : array-like
        Bond length(s) in Angstroms.

    force_constant : float, optional
        Spring constant k in kJ mol^-1 A^-2. Default 80.3 (8.03 J mol^-1 pm^-2).

    bond_length : float, optional
        Equilibrium bond length r0 in Angstroms. Default 3.81.

    Returns
    -------
    np.ndarray
        Energy in kJ/mol for each bond length.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    return np.asarray(0.5 * force_constant * (r_arr - bond_length) ** 2)


def wang_frenkel(
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    epsilon: npt.ArrayLike,
    mu: npt.ArrayLike = 2.0,
    nu: npt.ArrayLike = 1.0,
    cutoff_ratio: float = WANG_FRENKEL_CUTOFF_RATIO,
) -> npt.NDArray[np.float64]:
    """
    Wang-Frenkel pair potential (Joseph et al., equation 4).

        phi(r) = epsilon alpha [(sigma / r)^(2 mu) - 1] [(R / r)^(2 mu) - 1]^(2 nu)

    for r <= R = cutoff_ratio * sigma, and zero beyond R. Alpha is chosen so
    the minimum of the well is exactly -epsilon.

    Parameters
    ----------
    r : array-like
        Distance(s) between the two beads in Angstroms.

    sigma : array-like
        Size parameter(s) in Angstroms, broadcastable against r.

    epsilon : array-like
        Well depth(s) in kJ/mol, broadcastable against r.

    mu : array-like, optional
        Exponent(s) mu. Default 2.

    nu : array-like, optional
        Exponent(s) nu. Default 1.

    cutoff_ratio : float, optional
        Cutoff R in units of sigma. Default 3.0.

    Returns
    -------
    np.ndarray
        Energy in kJ/mol at each distance.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    epsilon_arr = np.asarray(epsilon, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)

    cutoff = cutoff_ratio * sigma_arr
    alpha = wang_frenkel_alpha(mu_arr, nu_arr, cutoff_ratio=cutoff_ratio)

    repulsive = np.power(sigma_arr / r_arr, 2.0 * mu_arr) - 1.0
    envelope = np.power(np.power(cutoff / r_arr, 2.0 * mu_arr) - 1.0, 2.0 * nu_arr)
    energy = epsilon_arr * alpha * repulsive * envelope

    # the closed form does not vanish past the cutoff, so truncate explicitly
    return np.asarray(np.where(r_arr <= cutoff, energy, 0.0))


def debye_huckel(
    r: npt.ArrayLike,
    q1: npt.ArrayLike,
    q2: npt.ArrayLike,
    debye_length: float,
    dielectric: float = DEFAULT_DIELECTRIC,
    cutoff: float = DEFAULT_COULOMB_CUTOFF,
) -> npt.NDArray[np.float64]:
    """
    Debye-Huckel screened Coulomb potential (Joseph et al., equation 3).

        E(r) = q1 q2 exp(-r / lambda_D) / (4 pi epsilon_0 epsilon_r r)

    truncated (without shifting) at the cutoff, as in Mpipi.

    Parameters
    ----------
    r : array-like
        Distance(s) between the two beads in Angstroms.

    q1 : array-like
        Charge(s) of the first bead in units of the elementary charge.

    q2 : array-like
        Charge(s) of the second bead in units of the elementary charge.

    debye_length : float
        Debye screening length in Angstroms (np.inf turns screening off).

    dielectric : float, optional
        Relative dielectric constant. Default 80.

    cutoff : float, optional
        Cutoff in Angstroms. Default 35.

    Returns
    -------
    np.ndarray
        Energy in kJ/mol at each distance.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    q1_arr = np.asarray(q1, dtype=np.float64)
    q2_arr = np.asarray(q2, dtype=np.float64)

    energy = (
        COULOMB_CONSTANT
        * q1_arr
        * q2_arr
        * np.exp(-r_arr / debye_length)
        / (dielectric * r_arr)
    )
    return np.asarray(np.where(r_arr <= cutoff, energy, 0.0))


def mpipi_gg_energy(
    coordinates: npt.ArrayLike,
    parameters: SequenceParameters,
    debye_length: float,
    dielectric: float = DEFAULT_DIELECTRIC,
    coulomb_cutoff: float = DEFAULT_COULOMB_CUTOFF,
) -> dict[str, float]:
    """
    Total Mpipi-GG energy of a single conformation, split by term.

    Non-bonded terms (Wang-Frenkel and Debye-Huckel) are evaluated for every
    pair of residues except directly bonded neighbours, which interact only
    through the harmonic bond. This is a straightforward O(n^2) loop-free
    evaluation intended for testing and for spot checks, not for speed.

    Parameters
    ----------
    coordinates : array-like
        Bead positions in Angstroms, shape (n, 3).

    parameters : SequenceParameters
        Mpipi-GG parameters for the sequence (see
        starling.minimizer.parameters.build_sequence_parameters()).

    debye_length : float
        Debye screening length in Angstroms.

    dielectric : float, optional
        Relative dielectric constant. Default 80.

    coulomb_cutoff : float, optional
        Debye-Huckel cutoff in Angstroms. Default 35.

    Returns
    -------
    dict
        Energies in kJ/mol under the keys 'bond', 'wang_frenkel',
        'debye_huckel' and 'total'.

    Raises
    ------
    ValueError
        If the coordinates are not (n, 3) with n matching the sequence.
    """
    xyz = np.asarray(coordinates, dtype=np.float64)
    n = len(parameters.sequence)
    if xyz.shape != (n, 3):
        raise ValueError(
            f"coordinates must have shape ({n}, 3) for this sequence, got {xyz.shape}"
        )

    bonds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    e_bond = float(harmonic_bond(bonds).sum())

    # every pair separated by two or more positions in sequence
    i, j = np.triu_indices(n, k=2)
    r = np.linalg.norm(xyz[i] - xyz[j], axis=1)

    e_wf = float(
        wang_frenkel(
            r,
            parameters.sigma[i, j],
            parameters.epsilon[i, j],
            parameters.mu[i, j],
            parameters.nu[i, j],
        ).sum()
    )

    e_dh = float(
        debye_huckel(
            r,
            parameters.charges[i],
            parameters.charges[j],
            debye_length,
            dielectric=dielectric,
            cutoff=coulomb_cutoff,
        ).sum()
    )

    return {
        "bond": e_bond,
        "wang_frenkel": e_wf,
        "debye_huckel": e_dh,
        "total": e_bond + e_wf + e_dh,
    }
