"""
Time STARLING generating 3D structures with the fast MDS reconstruction and relaxation.

This is generate(return_structures=True, relax="fast"): distance maps, the
faster MDS reconstruction, then the same Mpipi-GG minimization and
thermalization as the standard mode.

Usage
-----
python benchmark_mds_fast.py [--sequence SEQ] [--conformations N] [--repeats N] [--device DEVICE]

Run with --help for every option.
"""

from benchmark_common import run

if __name__ == "__main__":
    run(["fast"], __doc__.strip().splitlines()[0])
