"""
Time STARLING generating distance maps only (no 3D structures).

Usage
-----
python benchmark_distance_maps.py [--sequence SEQ] [--conformations N] [--repeats N] [--device DEVICE]

Run with --help for every option.
"""

from benchmark_common import run

if __name__ == "__main__":
    run(["maps"], __doc__.strip().splitlines()[0])
