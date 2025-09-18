import argparse
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import psutil

from starling import configs
from starling._version import __version__
from starling.frontend.ensemble_generation import generate
from starling.utilities import check_device, check_file_exists


def print_starling():
    print("\n-------------------------------------------------------")
    print(f"     STARLING (version {__version__})           ")
    print("-------------------------------------------------------")


def print_info():
    # local imports if needed
    print_starling()
    print("Using models at the following locations:")
    print(f"  VAE  model weights: {configs.DEFAULT_ENCODER_WEIGHTS_PATH}")
    print(f"  DDPM model weights: {configs.DEFAULT_DDPM_WEIGHTS_PATH}")
    print("-------------------------------------------------------")
    print("CONFIG INFO:")
    print("  Default # of confs :", configs.DEFAULT_NUMBER_CONFS)
    print("  Default batch size :", configs.DEFAULT_BATCH_SIZE)
    print("  Default steps      :", configs.DEFAULT_STEPS)
    print("  Default # of CPUs  :", configs.DEFAULT_CPU_COUNT_MDS)
    print("  Default # MDS jobs :", configs.DEFAULT_MDS_NUM_INIT)
    print(f"  Default device     : {check_device(None)}")
    print("-------------------------------------------------------")
    print(
        "Need help - please raise an issue on GitHub: https://github.com/idptools/starling/issues"
    )

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
        help=f"Number of steps to run the DDPM model (default: {configs.DEFAULT_STEPS})",
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
        "--num-cpus",
        dest="num_cpus",
        type=int,
        default=configs.DEFAULT_CPU_COUNT_MDS,
        help=f"Sets the max number of CPUs to use. Default: {configs.DEFAULT_CPU_COUNT_MDS}.",
    )
    parser.add_argument(
        "--num-mds-init",
        dest="num_mds_init",
        type=int,
        default=configs.DEFAULT_MDS_NUM_INIT,
        help=f"Sets the number of MDS jobs to be run in parallel. More may give better reconstruction but requires 1:1 with #CPUs to avoid performance penalty. Default: {configs.DEFAULT_MDS_NUM_INIT}.",
    )
    parser.add_argument(
        "--salt",
        dest="salt",
        type=int,
        default=configs.DEFAULT_SALT,
        help=f"Salt (in mM) for the prediction. Default: {configs.DEFAULT_SALT} mM.",
    )
    parser.add_argument(
        "--disable_progress_bar",
        dest="progress_bar",
        action="store_false",
        default=True,
        help="Disable progress bar during generation (default: False)",
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

    if args.verbose:
        print_starling()

    print("Salt concentration: ", args.salt, " mM")
    # Call the generate function with parsed arguments
    generate(
        user_input=args.user_input,
        conformations=args.conformations,
        salt=args.salt,
        device=args.device,
        steps=args.steps,
        return_structures=args.return_structures,
        batch_size=args.batch_size,
        num_cpus_mds=args.num_cpus,
        num_mds_init=args.num_mds_init,
        output_directory=args.output_directory,
        output_name=args.outname,
        return_data=False,
        verbose=args.verbose,
        show_progress_bar=args.progress_bar,
    )


#
# BENCHMARKING CODE BELOW
#

# default sequence is alpha-synuclein
ALPHA_SYN = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"


def measure_runtime(
    user_input, steps=25, batch_size=100, device=None, cooltime=600, single_run=0
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

        runtime_matrix[i,0] = conf
        runtime_matrix[i,1] = end_time - start_time

        rg_matrix[i, 0] = conf
        rg_matrix[i, 1] = data.radius_of_gyration(return_mean=True)

        np.savetxt(
            f"rg_matrix_{steps}_steps_{device_name}_{batch_size}_batchsize.csv",
            rg_matrix,
            delimiter=", ",
        )
        np.savetxt(
            f"runtime_matrix_{steps}_steps_{device_name}_{batch_size}_batchsize.csv",
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
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of steps to run the DDPM model (default: 25)",
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

    args = parser.parse_args()

    measure_runtime(
        args.sequence,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        cooltime=args.cooltime,
        single_run=args.single_run,
    )
