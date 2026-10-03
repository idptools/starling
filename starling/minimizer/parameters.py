"""
Mpipi-GG parameters and physical constants.

This module holds everything needed to define the Mpipi-GG force field for a
protein sequence: the Wang-Frenkel pair parameters for the 20 amino acids, the
per-residue charges, the bonded parameters, and the constants for the
Debye-Huckel screened electrostatics. It does not evaluate any energies itself;
see starling.minimizer.potentials for plain NumPy reference implementations of
the potentials and starling.minimizer.forcefield for the batched PyTorch
implementation used by the minimizer.

The Wang-Frenkel parameters are read from data/mpipi_gg_pairs.tsv, which is a
plain-text copy of the Mpipi-GG (GGv1) parameter set used in FINCHES. Epsilon is
stored in that file in kcal/mol (the native LAMMPS real-unit values) and
converted to kJ/mol when loaded, so every energy in this package is in kJ/mol,
every distance is in Angstroms, and every force is in kJ/mol/Angstrom.

Note that the FINCHES parameter pickles are also in kcal/mol, even though the
FINCHES docstrings describe them as kJ/mol. For example, the M-W epsilon is
0.294931 in the pickle and 1.233991 kJ/mol in Supplementary Table 11 of
Joseph et al. (a ratio of 4.184).

References
----------
Joseph, J. A., Reinhardt, A., Aguirre, A., Chew, P. Y., Russell, K. O.,
Espinosa, J. R., Garaizar, A., & Collepardo-Guevara, R. (2021). Physics-driven
coarse-grained model for biomolecular phase separation with near-quantitative
accuracy. Nature Computational Science, 1(11), 732-743.

Lotthammer, J. M., Ginell, G. M., Griffith, D., Emenecker, R. J., & Holehouse,
A. S. (2024). Direct prediction of intrinsically disordered protein
conformational properties from sequence. Nature Methods, 21(3), 465-476.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Final

import numpy as np
import numpy.typing as npt

# ------------------------------------------------------------------------------
# Physical constants and bonded parameters
# ------------------------------------------------------------------------------

# the 20 amino acids Mpipi-GG covers, in the order used for every per-residue
# lookup in this module (this matches starling.configs.VALID_AA)
MPIPI_GG_RESIDUES: Final[str] = "ACDEFGHIKLMNPQRSTVWY"

# thermochemical calorie; converts the kcal/mol epsilons in the data file to kJ/mol
KCAL_TO_KJ: Final[float] = 4.184

# harmonic bond: E = (k / 2) (r - r0)^2 with k = 8.03 J mol^-1 pm^-2, which is
# 80.3 kJ mol^-1 A^-2, and r0 = 381 pm (Joseph et al., equation 2)
BOND_FORCE_CONSTANT: Final[float] = 80.3
BOND_LENGTH: Final[float] = 3.81

# the Wang-Frenkel potential goes to zero at R_ij = 3 sigma_ij
WANG_FRENKEL_CUTOFF_RATIO: Final[float] = 3.0

# e^2 N_A / (4 pi epsilon_0) in kJ mol^-1 A, from the exact SI values of the
# elementary charge and Avogadro's number and CODATA 2018 epsilon_0. This is
# 1389.35 kJ mol^-1 A, the same as the LAMMPS real-unit Coulomb constant
# (332.06371 kcal mol^-1 A)
_ELEMENTARY_CHARGE: Final[float] = 1.602176634e-19
_AVOGADRO: Final[float] = 6.02214076e23
_VACUUM_PERMITTIVITY: Final[float] = 8.8541878128e-12
COULOMB_CONSTANT: Final[float] = (
    (_ELEMENTARY_CHARGE**2 * _AVOGADRO / (4.0 * np.pi * _VACUUM_PERMITTIVITY))
    * 1e10
    / 1000.0
)

# relative dielectric constant of water used by Mpipi
DEFAULT_DIELECTRIC: Final[float] = 80.0

# Mpipi truncates the Debye-Huckel potential at 3.5 nm
DEFAULT_COULOMB_CUTOFF: Final[float] = 35.0

# Debye length in A is DEBYE_LENGTH_PREFACTOR / sqrt(ionic strength in M). This
# is the room-temperature value for a monovalent salt in water used by FINCHES,
# which gives 7.90 A at 150 mM (the Mpipi paper uses 7.95 A)
DEBYE_LENGTH_PREFACTOR: Final[float] = 3.06

# Mpipi-GG charges in units of the elementary charge. Charged residues carry
# +/- 0.75 e and histidine carries half that
MPIPI_GG_CHARGES: Final[dict[str, float]] = {
    "A": 0.0,
    "C": 0.0,
    "D": -0.75,
    "E": -0.75,
    "F": 0.0,
    "G": 0.0,
    "H": 0.375,
    "I": 0.0,
    "K": 0.75,
    "L": 0.0,
    "M": 0.0,
    "N": 0.0,
    "P": 0.0,
    "Q": 0.0,
    "R": 0.75,
    "S": 0.0,
    "T": 0.0,
    "V": 0.0,
    "W": 0.0,
    "Y": 0.0,
}

_PAIR_TABLE_FILENAME: Final[str] = "mpipi_gg_pairs.tsv"


# ------------------------------------------------------------------------------
# Parameter containers
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MpipiGGPairTable:
    """
    Wang-Frenkel parameters for every pair of the 20 amino acids.

    Every array is (20, 20) and symmetric, with rows and columns in the order
    of MPIPI_GG_RESIDUES.

    Attributes
    ----------
    residues : str
        The residue order used for the rows and columns.

    sigma : np.ndarray
        Size parameter in Angstroms; the distance at which the Wang-Frenkel
        potential crosses zero.

    epsilon : np.ndarray
        Well depth in kJ/mol.

    mu : np.ndarray
        Wang-Frenkel exponent mu (dimensionless).

    nu : np.ndarray
        Wang-Frenkel exponent nu (dimensionless).
    """

    residues: str
    sigma: npt.NDArray[np.float64]
    epsilon: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    nu: npt.NDArray[np.float64]


@dataclass(frozen=True)
class SequenceParameters:
    """
    Mpipi-GG parameters for every pair of residues in one sequence.

    Built by build_sequence_parameters(). Pair arrays are (n, n) and
    symmetric, where n is the length of the sequence; the diagonal holds the
    self-pair values but is never used.

    Attributes
    ----------
    sequence : str
        The amino acid sequence the parameters were built for.

    sigma : np.ndarray
        Wang-Frenkel size parameter for each residue pair, in Angstroms.

    epsilon : np.ndarray
        Wang-Frenkel well depth for each residue pair, in kJ/mol.

    mu : np.ndarray
        Wang-Frenkel exponent mu for each residue pair.

    nu : np.ndarray
        Wang-Frenkel exponent nu for each residue pair.

    alpha : np.ndarray
        Wang-Frenkel prefactor alpha for each residue pair (see
        wang_frenkel_alpha()).

    charges : np.ndarray
        Charge on each residue in units of the elementary charge, shape (n,).
    """

    sequence: str
    sigma: npt.NDArray[np.float64]
    epsilon: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    nu: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    charges: npt.NDArray[np.float64]


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_pair_table() -> MpipiGGPairTable:
    """
    Read the Mpipi-GG Wang-Frenkel pair parameters from the packaged data file.

    The file lists each unordered residue pair once, so we fill in both
    triangles of every matrix here. Epsilon is converted from kcal/mol (as
    stored) to kJ/mol. The result is cached, so the file is only read once per
    session.

    Returns
    -------
    MpipiGGPairTable
        The (20, 20) sigma, epsilon, mu and nu matrices.

    Raises
    ------
    ValueError
        If the file is missing a residue pair, lists a pair twice, or names a
        residue outside the 20 amino acids.
    """
    index = {residue: i for i, residue in enumerate(MPIPI_GG_RESIDUES)}
    n_types = len(MPIPI_GG_RESIDUES)

    tables = {
        name: np.full((n_types, n_types), np.nan)
        for name in ("sigma", "epsilon", "mu", "nu")
    }

    data_file = resources.files("starling.minimizer").joinpath(
        f"data/{_PAIR_TABLE_FILENAME}"
    )
    with data_file.open("r") as fh:
        rows = csv.DictReader(
            (line for line in fh if not line.startswith("#")), delimiter="\t"
        )
        for row in rows:
            r1, r2 = row["residue_1"], row["residue_2"]
            if r1 not in index or r2 not in index:
                raise ValueError(
                    f"Unrecognized residue pair [{r1}-{r2}] in {_PAIR_TABLE_FILENAME}"
                )

            i, j = index[r1], index[r2]
            if not np.isnan(tables["sigma"][i, j]):
                raise ValueError(
                    f"Residue pair [{r1}-{r2}] appears more than once in "
                    f"{_PAIR_TABLE_FILENAME}"
                )

            values = {
                "sigma": float(row["sigma"]),
                "epsilon": float(row["epsilon"]) * KCAL_TO_KJ,
                "mu": float(row["mu"]),
                "nu": float(row["nu"]),
            }
            for name, value in values.items():
                tables[name][i, j] = value
                tables[name][j, i] = value

    if np.isnan(tables["sigma"]).any():
        missing = [
            f"{MPIPI_GG_RESIDUES[i]}-{MPIPI_GG_RESIDUES[j]}"
            for i, j in zip(*np.nonzero(np.isnan(tables["sigma"])))
            if i <= j
        ]
        raise ValueError(
            f"{_PAIR_TABLE_FILENAME} is missing the residue pairs: {missing}"
        )

    return MpipiGGPairTable(
        residues=MPIPI_GG_RESIDUES,
        sigma=tables["sigma"],
        epsilon=tables["epsilon"],
        mu=tables["mu"],
        nu=tables["nu"],
    )


def wang_frenkel_alpha(
    mu: npt.ArrayLike,
    nu: npt.ArrayLike,
    cutoff_ratio: float = WANG_FRENKEL_CUTOFF_RATIO,
) -> npt.NDArray[np.float64]:
    """
    Compute the Wang-Frenkel prefactor alpha.

    Alpha scales the potential so that its minimum is exactly -epsilon
    (Joseph et al., equation 5):

        alpha = 2 nu (R / sigma)^(2 mu) [(2 nu + 1) / (2 nu ((R / sigma)^(2 mu) - 1))]^(2 nu + 1)

    Parameters
    ----------
    mu : array-like
        Wang-Frenkel exponent(s) mu.

    nu : array-like
        Wang-Frenkel exponent(s) nu.

    cutoff_ratio : float, optional
        R / sigma, the cutoff in units of sigma. Default 3.0, as in Mpipi.

    Returns
    -------
    np.ndarray
        Alpha for each (mu, nu), broadcast to a common shape.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)

    cutoff_term = np.power(cutoff_ratio, 2.0 * mu_arr)
    bracket = (2.0 * nu_arr + 1.0) / (2.0 * nu_arr * (cutoff_term - 1.0))

    return np.asarray(
        2.0 * nu_arr * cutoff_term * np.power(bracket, 2.0 * nu_arr + 1.0),
        dtype=np.float64,
    )


def debye_length(ionic_strength: float) -> float:
    """
    Debye screening length for a monovalent salt in water at room temperature.

    Uses the same convention as FINCHES: DEBYE_LENGTH_PREFACTOR / sqrt(I),
    with I in molar. This gives 7.90 Angstroms at 150 mM.

    Parameters
    ----------
    ionic_strength : float
        Ionic strength in mM (the unit STARLING uses throughout). A value of
        zero means no screening, in which case the Debye length is infinite.

    Returns
    -------
    float
        Debye length in Angstroms (np.inf if ionic_strength is zero).

    Raises
    ------
    ValueError
        If ionic_strength is negative.
    """
    if ionic_strength < 0:
        raise ValueError(f"Ionic strength must be >= 0 mM, got {ionic_strength}")

    if ionic_strength == 0:
        return float(np.inf)

    return float(DEBYE_LENGTH_PREFACTOR / np.sqrt(ionic_strength / 1000.0))


def validate_sequence(sequence: str) -> str:
    """
    Check that a sequence only contains residues Mpipi-GG has parameters for.

    Parameters
    ----------
    sequence : str
        Amino acid sequence. Case is ignored.

    Returns
    -------
    str
        The sequence in upper case.

    Raises
    ------
    ValueError
        If the sequence is empty or contains anything other than the 20
        standard amino acids.
    """
    if not isinstance(sequence, str) or len(sequence) == 0:
        raise ValueError("sequence must be a non-empty string")

    sequence = sequence.upper()
    invalid = sorted(set(sequence) - set(MPIPI_GG_RESIDUES))
    if invalid:
        raise ValueError(
            f"sequence contains residues Mpipi-GG has no parameters for: {invalid}. "
            f"Valid residues are {MPIPI_GG_RESIDUES}"
        )

    return sequence


def build_sequence_parameters(sequence: str) -> SequenceParameters:
    """
    Expand the Mpipi-GG parameter tables onto every residue pair of a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence made up of the 20 standard amino acids.

    Returns
    -------
    SequenceParameters
        (n, n) pair parameter matrices and (n,) charges for the sequence.

    Raises
    ------
    ValueError
        If the sequence contains a residue Mpipi-GG has no parameters for.
    """
    sequence = validate_sequence(sequence)
    table = load_pair_table()

    type_index = np.array([MPIPI_GG_RESIDUES.index(r) for r in sequence])
    pair = np.ix_(type_index, type_index)

    mu = table.mu[pair]
    nu = table.nu[pair]

    return SequenceParameters(
        sequence=sequence,
        sigma=table.sigma[pair],
        epsilon=table.epsilon[pair],
        mu=mu,
        nu=nu,
        alpha=wang_frenkel_alpha(mu, nu),
        charges=np.array([MPIPI_GG_CHARGES[r] for r in sequence], dtype=np.float64),
    )
