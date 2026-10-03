"""
Mpipi-GG relaxation of STARLING conformations.

STARLING's 3D structures are reconstructed from predicted distance maps by
multidimensional scaling, which gets the global shape right but leaves local
geometry (bond lengths, close contacts) physically unrealistic. This package
relaxes those structures under the Mpipi-GG force field while restraining
long-range distances, so local geometry is repaired and global dimensions are
left essentially unchanged.

The main entry points are

* relax_ensemble(): relax the structures of a STARLING Ensemble.
* relax_conformations(): relax raw coordinates (in Angstroms) for a sequence.
* MpipiGG: the batched PyTorch Mpipi-GG force field itself.
"""

from starling.minimizer.fire import FIREParameters, fire_minimize
from starling.minimizer.forcefield import MpipiGG
from starling.minimizer.relax import (
    GeometryDiagnostics,
    RelaxationResult,
    relax_conformations,
    relax_ensemble,
)
from starling.minimizer.restraints import DistanceRestraints

__all__ = [
    "DistanceRestraints",
    "FIREParameters",
    "GeometryDiagnostics",
    "MpipiGG",
    "RelaxationResult",
    "fire_minimize",
    "relax_conformations",
    "relax_ensemble",
]
