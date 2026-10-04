import argparse
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np

import starling
from starling import configs
from starling._version import __version__
from starling.frontend.ensemble_generation import generate
from starling.utilities import check_device


def print_starling():
    print("\n-------------------------------------------------------")
    print(f"     STARLING (version {__version__})           ")
    print("-------------------------------------------------------")


def _describe_weights(ref):
    """
    Describe where a weights reference will actually be loaded from.

    Parameters
    ----------
    ref : str
        Local path or URL as stored in ``configs``.

    Returns
    -------
    str
        The local file that will be used, or a note that the file is missing
        and would be downloaded from ``ref``.
    """
    if not ref.startswith("http"):
        return ref
    for candidate in configs.candidate_weights_paths(os.path.basename(ref)):
        if os.path.isfile(candidate):
            return candidate
    return f"NOT FOUND LOCALLY (will be downloaded from {ref})"


def print_info():
    # local imports if needed
    print_starling()
    print("Using models at the following locations:")
    print(f"  VAE  model weights: {_describe_weights(configs.DEFAULT_ENCODER_WEIGHTS_PATH)}")
    print(f"  DDPM model weights: {_describe_weights(configs.DEFAULT_DDPM_WEIGHTS_PATH)}")
    print(f"  Offline mode       : {'ON (STARLING_OFFLINE set)' if configs.is_offline() else 'off'}")
    print("  Local weight directories searched (in order):")
    for d in (configs.DEFAULT_MODEL_DIR, configs.torch_hub_checkpoint_dir()):
        print(f"    - {d}")
    print("-------------------------------------------------------")
    print("CONFIG INFO:")
    print("  Default # of confs :", configs.DEFAULT_NUMBER_CONFS)
    print("  Default batch size :", configs.DEFAULT_BATCH_SIZE)
    print("  Default steps      :", configs.DEFAULT_STEPS)
    print(f"  Default device     : {check_device(None)}")
    print("-------------------------------------------------------")
    print("Need help - please raise an issue on GitHub: https://github.com/idptools/starling/issues")

    print("\n")


def main():
    # Initialize the argument parser
    parser = ArgumentParser(description="Generate distance maps using STARLING.")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "user_input",
        type=str,
        help="Input sequences in various formats (file, string, list, or dict)",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--conformations",
        type=int,
        default=configs.DEFAULT_NUMBER_CONFS,
        help=f"Number of conformations to generate (default: {configs.DEFAULT_NUMBER_CONFS})",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use for predictions (default: None, auto-detected)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=configs.DEFAULT_STEPS,
        help=f"Requested sampler steps; DDPM uses the full training schedule (default: {configs.DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--sampler",
        choices=("ddim", "ddpm", "dpmpp", "plms"),
        default=configs.DEFAULT_SAMPLER,
        help=f"Sampling method (default: {configs.DEFAULT_SAMPLER})",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=configs.DEFAULT_BATCH_SIZE,
        help=f"Batch size to use for sampling (default: {configs.DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default=".",
        help="Directory to save output (default: '.')",
    )
    parser.add_argument(
        "--outname",
        type=str,
        default=None,
        help="If provided and a single sequence is provided, defines the prefix ahead of .pdb/.xtc/.npy extensions (default: None)",
    )
    parser.add_argument(
        "-r",
        "--return_structures",
        action="store_true",
        default=False,
        help="Return the 3D structures (default: False)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output (default: False)",
    )
    parser.add_argument(
        "--ionic_strength",
        dest="ionic_strength",
        type=int,
        default=configs.DEFAULT_IONIC_STRENGTH,
        help=f"Ionic strength (in mM) for the prediction. Default: {configs.DEFAULT_IONIC_STRENGTH} mM.",
    )
    parser.add_argument(
        "--disable_progress_bar",
        dest="progress_bar",
        action="store_false",
        default=True,
        help="Disable progress bar during generation (default: False)",
    )
    parser.add_argument(
        "--remove-errors",
        dest="remove_errors",
        action="store_true",
        default=False,
        help=(
            "Discard conformations with physically impossible inter-residue "
            "distances and regenerate replacements, so the number of conformations "
            "written still matches -c/--conformations. Filtering happens before "
            "anything is written to disk, so the .xtc/.pdb and .starling outputs "
            "contain only clean conformations. With -r/--return_structures the "
            "reconstructed 3D conformations are screened as well as the distance "
            "maps (default: False)"
        ),
    )
    parser.add_argument(
        "--relax",
        dest="relax",
        action="store_true",
        default=False,
        help=(
            "Relax the reconstructed 3D conformations with the Mpipi-GG force "
            "field, restrained to their STARLING distance maps, then thermalize "
            "them briefly at 300 K. This fixes the compressed bonds and close "
            "contacts left by MDS reconstruction, giving local geometry close "
            "to Mpipi-GG simulations while keeping global dimensions. Requires "
            "-r/--return_structures. "
            "With --remove-errors, relaxation runs before the 3D error screen "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--compile-forces",
        action="store_true",
        help="Fuse CUDA thermalization forces for sustained throughput; adds compilation startup time. Requires --relax -r.",
    )

    # will need to update this default...
    parser.add_argument(
        "--info",
        action="store_true",
        default=False,
        help="Print STARLING information only",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="Print STARLING version only",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # first. informational args overide anything else
    if args.info:
        print_info()
        sys.exit(0)
    elif args.version:
        print(__version__)
        sys.exit(0)
    elif args.user_input is None:
        print(parser.format_usage())  # Print the command signature
        print("ERROR: STARLING requires user_input or --version", file=sys.stderr)
        sys.exit(1)
    else:
        pass

    ### sanity checks
    # check if the output directory exists
    if not os.path.exists(args.output_directory):
        print(
            f"ERROR: Output directory {args.output_directory} does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.compile_forces and (not args.relax or not args.return_structures):
        parser.error("--compile-forces requires --relax and -r/--return_structures")

    # relaxation acts on 3D structures, so without -r it would silently do nothing
    if args.relax and not args.return_structures:
        print(
            "ERROR: --relax relaxes the 3D structures, so it requires -r/--return_structures.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.verbose:
        print_starling()

    # Call the generate function with parsed arguments
    generate(
        user_input=args.user_input,
        conformations=args.conformations,
        ionic_strength=args.ionic_strength,
        device=args.device,
        steps=args.steps,
        sampler=args.sampler,
        return_structures=args.return_structures,
        batch_size=args.batch_size,
        output_directory=args.output_directory,
        output_name=args.outname,
        return_data=False,
        verbose=args.verbose,
        show_progress_bar=args.progress_bar,
        remove_errors=args.remove_errors,
        relax=args.relax,
        compile_forces=args.compile_forces,
    )


#
# BENCHMARKING CODE BELOW
#

# default sequence is alpha-synuclein
ALPHA_SYN = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"


def measure_runtime(
    user_input,
    steps=configs.DEFAULT_STEPS,
    batch_size=configs.DEFAULT_BATCH_SIZE,
    device=None,
    cooltime=600,
    single_run=0,
    compile=False,
):
    """
    Measure runtime of Starling conformational generation for a given sequence.

    Parameters
    ----------
    user_input : str
        Protein sequence to process.

    steps : int
        Number of steps to run the DDPM model.

    batch_size : int
        Batch size for processing.

    device : str
        Device to run on (e.g., 'cpu', 'cuda', 'mps').

    cooltime : int
        Time to cool down the hardware after each run (default: 600 seconds).

    single_run : int
        If set means we just do a single run with the specified number of conformations,
        otherwise we will do 10 predictions linearly spaced beteen 10 and 1000 confomers.

    """

    if single_run != 0:
        conformations_list = [int(single_run)]
    else:
        conformations_list = np.linspace(10, 1000, num=10, dtype=int)

    if device is None:
        device_name = str(check_device(device))
    else:
        device_name = device

    # initialize matrices to store runtime and radius of gyration
    runtime_matrix = np.zeros((len(conformations_list), 2))
    rg_matrix = np.zeros((len(conformations_list), 2))

    # iterate over conformations and steps
    for i, conf in enumerate(conformations_list):
        start_time = time.perf_counter()
        data = generate(
            user_input,
            conformations=conf,
            device=device,
            batch_size=batch_size,
            steps=steps,
            verbose=True,
            return_structures=False,
            return_single_ensemble=True,
        )
        end_time = time.perf_counter()

        runtime_matrix[i, 0] = conf
        runtime_matrix[i, 1] = end_time - start_time

        rg_matrix[i, 0] = conf
        rg_matrix[i, 1] = data.radius_of_gyration(return_mean=True)

        if compile:
            compile_string = "compile_on"
        else:
            compile_string = "compile_off"

        np.savetxt(
            f"rg_matrix_{steps}_steps_{device_name}_{batch_size}_batchsize_{compile_string}.csv",
            rg_matrix,
            delimiter=", ",
        )
        np.savetxt(
            f"runtime_matrix_{steps}_steps_{device_name}_{batch_size}_batchsize_{compile_string}.csv",
            runtime_matrix,
            delimiter=", ",
        )

        # have a snoozle to cool down the hardware (could use -80 alternatively? TBD).
        if i != len(conformations_list) - 1:
            time.sleep(cooltime)


def starling_benchmark():
    parser = argparse.ArgumentParser(
        description="Benchmark STARLING performance based on various tweakable parameters."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=configs.DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {configs.DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=configs.DEFAULT_STEPS,
        help=f"Number of steps to run the DDPM model (default: {configs.DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=ALPHA_SYN,
        help="Protein sequence to process. Default is alpha-synuclein.",
    )
    parser.add_argument(
        "--cooltime",
        type=int,
        default=20,
        help="Time to cool down the hardware after each run (default: 20 seconds).",
    )
    parser.add_argument(
        "--single-run",
        type=int,
        default=0,
        help="By default, starling-benchmark runs a series of 10 predictions for 10-1000 confomers. If you want to run a single prediction, specify the number of conformations here.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable compilation (default: False). NB: As of 2025-09-20, compilation is only supported on CUDA devices.",
    )

    args = parser.parse_args()

    if args.compile:
        print("Compiling models...")
        starling.set_compilation_options(enabled=True)
        print("Compilation complete.")

    measure_runtime(
        args.sequence,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        cooltime=args.cooltime,
        single_run=args.single_run,
        compile=args.compile,
    )
