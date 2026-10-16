"""
Time STARLING in all three modes and print a comparison table.

Runs, in order: distance maps only, maps plus standard MDS structures
(relaxed), and maps plus fast MDS structures (relaxed). The models load once,
during the first mode's warm-up.

Usage
-----
python run_all.py [--sequence SEQ] [--conformations N] [--repeats N] [--device DEVICE]

Run with --help for every option.
"""

from benchmark_common import MODES, print_summary, run

if __name__ == "__main__":
    results = run(list(MODES), __doc__.strip().splitlines()[0])
    print_summary(results)
