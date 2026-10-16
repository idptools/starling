"""
Time STARLING generating 3D structures with the standard MDS reconstruction and relaxation.

This is what generate(return_structures=True) does by default: distance maps,
the standard MDS reconstruction, then Mpipi-GG minimization and
thermalization (relax="standard").

Usage
-----
python benchmark_mds_standard.py [--sequence SEQ] [--conformations N] [--repeats N] [--device DEVICE]

Run with --help for every option.
"""

from benchmark_common import run

if __name__ == "__main__":
    run(["standard"], __doc__.strip().splitlines()[0])
