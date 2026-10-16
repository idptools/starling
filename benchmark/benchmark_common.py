"""
Shared code for the STARLING throughput benchmarks.

Every benchmark script in this directory times starling.generate() for one
sequence in one or more output modes and reports conformers per minute:

maps
    Distance maps only (return_structures=False).
standard
    Distance maps plus 3D structures from the standard MDS reconstruction,
    relaxed (minimized and thermalized) with Mpipi-GG
    (return_structures=True, relax="standard"; generate()'s default when
    structures are requested).
fast
    Distance maps plus 3D structures from the fast MDS reconstruction,
    relaxed the same way (return_structures=True, relax="fast").

For each mode we first make one untimed warm-up call, which loads the models
(only the first mode run in a session pays for this) and sets up the
device's kernels, and then time --repeats calls of --conformations
conformations each. Times are wall-clock seconds from the call to its
return, so they include everything a user waits for: sampling, decoding,
and (for the structure modes) reconstruction and relaxation. Every other
generate() setting (sampler, steps, batch size, ionic strength) is
STARLING's default unless given on the command line, so the numbers describe
what a user gets out of the box.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any, Final

import torch

import starling
from starling import configs
from starling.utilities import check_device

# alpha-synuclein (UniProt P37840), 140 residues: a widely studied IDR of
# typical length, and the sequence starling-benchmark uses
ALPHA_SYNUCLEIN: Final[str] = (
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
)

# generate() keyword arguments for each mode, and how each is described
MODES: Final[dict[str, dict[str, Any]]] = {
    "maps": {"return_structures": False},
    "standard": {"return_structures": True, "relax": "standard"},
    "fast": {"return_structures": True, "relax": "fast"},
}
MODE_LABELS: Final[dict[str, str]] = {
    "maps": "Distance maps only",
    "standard": "Maps + standard MDS, relaxed",
    "fast": "Maps + fast MDS, relaxed",
}

DEFAULT_REPEATS: Final[int] = 3
DEFAULT_SEED: Final[int] = 0
SECONDS_PER_MINUTE: Final[float] = 60.0


@dataclass(frozen=True)
class ModeResult:
    """
    Timings for one mode.

    Attributes
    ----------
    mode : str
        A key of MODES.
    warm_up_seconds : float
        Wall-clock time of the untimed warm-up call (includes loading the
        models if this was the first mode run in the session).
    run_seconds : list of float
        Wall-clock time of each timed call.
    conformations : int
        Conformations generated per timed call.
    """

    mode: str
    warm_up_seconds: float
    run_seconds: list[float]
    conformations: int

    @property
    def conformers_per_minute(self) -> list[float]:
        """list of float: Throughput of each timed call."""
        return [self.conformations / seconds * SECONDS_PER_MINUTE for seconds in self.run_seconds]

    @property
    def median_conformers_per_minute(self) -> float:
        """float: Median throughput over the timed calls."""
        return statistics.median(self.conformers_per_minute)


def build_parser(description: str) -> argparse.ArgumentParser:
    """
    Command-line options shared by every benchmark script.

    Parameters
    ----------
    description : str
        Description shown by --help.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the shared options.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--sequence",
        default=ALPHA_SYNUCLEIN,
        help="amino acid sequence to generate ensembles for (default: alpha-synuclein, 140 residues)",
    )
    parser.add_argument(
        "--conformations",
        type=int,
        default=configs.DEFAULT_NUMBER_CONFS,
        help=f"conformations per timed call (default: {configs.DEFAULT_NUMBER_CONFS}, STARLING's default)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"timed calls per mode; the median is reported (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--device", default=None, help="'cpu', 'mps', 'cuda' or 'cuda:N' (default: the fastest available)"
    )
    parser.add_argument("--sampler", default=configs.DEFAULT_SAMPLER, help=f"default: {configs.DEFAULT_SAMPLER}")
    parser.add_argument(
        "--steps", type=int, default=configs.DEFAULT_STEPS, help=f"denoising steps (default: {configs.DEFAULT_STEPS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=configs.DEFAULT_BATCH_SIZE,
        help=f"conformations sampled at once (default: {configs.DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"default: {DEFAULT_SEED}")
    return parser


def check_arguments(args: argparse.Namespace) -> None:
    """
    Reject settings that would make the benchmark meaningless.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.

    Raises
    ------
    ValueError
        If the number of conformations, repeats, steps or the batch size is
        not positive.
    """
    for name in ("conformations", "repeats", "steps", "batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1, got {getattr(args, name)}")


def hardware_description(device: torch.device) -> str:
    """
    One line naming the device the benchmark runs on.

    Parameters
    ----------
    device : torch.device
        The device.

    Returns
    -------
    str
        E.g. 'cuda:0 (NVIDIA RTX A4000)', 'mps (arm, macOS 15.5)' or
        'cpu (x86_64, 16 cores)'.
    """
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return f"mps ({platform.machine()}, macOS {platform.mac_ver()[0]})"
    return f"cpu ({platform.machine()}, {os.cpu_count()} cores)"


def print_header(args: argparse.Namespace, device: torch.device) -> None:
    """
    Print the software, hardware and settings the numbers belong to.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.
    device : torch.device
        Device the benchmark runs on.
    """
    print(f"STARLING {starling.__version__}, PyTorch {torch.__version__}, Python {platform.python_version()}")
    print(f"Device: {hardware_description(device)}")
    print(
        f"Sequence: {len(args.sequence)} residues; {args.conformations} conformations per call, "
        f"timed calls per mode: {args.repeats}; {args.sampler} with {args.steps} steps, batch size {args.batch_size}"
    )
    print()


def time_generate(args: argparse.Namespace, device: torch.device, mode: str, conformations: int, seed: int) -> float:
    """
    Wall-clock seconds for one generate() call.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options (sequence, sampler, steps, batch size).
    device : torch.device
        Device to run on.
    mode : str
        A key of MODES.
    conformations : int
        Conformations to generate.
    seed : int
        Seed for sampling and relaxation.

    Returns
    -------
    float
        Seconds from the call to its return.
    """
    # generate() prints which sampler it uses on every call; the header
    # already says, so that line is dropped (warnings go to stderr and still show)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        starling.generate(
            args.sequence,
            conformations=conformations,
            device=str(device),
            sampler=args.sampler,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=seed,
            return_single_ensemble=True,
            show_progress_bar=False,
            show_per_step_progress_bar=False,
            **MODES[mode],
        )
    # generate() returns NumPy arrays, so the GPU has finished; synchronizing
    # anyway keeps the timing honest if that ever changes
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def benchmark_mode(args: argparse.Namespace, device: torch.device, mode: str) -> ModeResult:
    """
    Warm up, then time --repeats calls of one mode, printing each as it finishes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.
    device : torch.device
        Device to run on.
    mode : str
        A key of MODES.

    Returns
    -------
    ModeResult
        The warm-up and timed-call results.
    """
    print(f"{MODE_LABELS[mode]}")

    # one batch is enough to load the models and set up every kernel the
    # timed calls use
    warm_up_conformations = min(args.batch_size, args.conformations)
    warm_up_seconds = time_generate(args, device, mode, warm_up_conformations, args.seed)
    print(f"  warm-up: {warm_up_seconds:.1f} s for {warm_up_conformations} conformations (not counted)")

    run_seconds: list[float] = []
    for repeat in range(args.repeats):
        seconds = time_generate(args, device, mode, args.conformations, args.seed + repeat + 1)
        run_seconds.append(seconds)
        rate = args.conformations / seconds * SECONDS_PER_MINUTE
        print(f"  call {repeat + 1}: {seconds:.1f} s, {rate:,.0f} conformers/min")

    result = ModeResult(mode, warm_up_seconds, run_seconds, args.conformations)
    print(f"  median: {result.median_conformers_per_minute:,.0f} conformers/min")
    print()
    return result


def print_summary(results: list[ModeResult]) -> None:
    """
    Print the median throughput of every mode in one table.

    Parameters
    ----------
    results : list of ModeResult
        One result per mode, in the order they ran.
    """
    print(f"{'Mode':32s} {'conformers/min (median)':>24s}")
    for result in results:
        print(f"{MODE_LABELS[result.mode]:32s} {result.median_conformers_per_minute:>24,.0f}")


def run(modes: list[str], description: str) -> list[ModeResult]:
    """
    Parse the command line and benchmark each mode in turn.

    Parameters
    ----------
    modes : list of str
        Keys of MODES to benchmark, in order.
    description : str
        Description shown by --help.

    Returns
    -------
    list of ModeResult
        One result per mode.
    """
    args = build_parser(description).parse_args()
    check_arguments(args)
    device = check_device(args.device)

    print_header(args, device)
    return [benchmark_mode(args, device, mode) for mode in modes]
