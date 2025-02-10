import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
from soursop.sstrajectory import SSTrajectory

from starling.structure import ensemble
from starling.utilities import parse_output_path, symmetrize_distance_maps


## USER-FACING FUNCTIONS
def starling2xtc():
    # Initialize the argument parser
    parser = ArgumentParser(
        description="Convert .starling ensembles to a PDB topology file and XTC trajectory"
    )

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "input_file", type=str, help="Input starling file", default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Directory and/or filename to save output (default: '.')",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    outname = parse_output_path(args)

    # build an ensemble from the starling file
    E = ensemble.load_ensemble(args.input_file)

    E.save_trajectory(outname)


def starling2pdb():
    # Initialize the argument parser
    parser = ArgumentParser(description="Convert .starling files to a PDB trajectory")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "input_file", type=str, help="Input starling file", default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Directory and/or filename to save output (default: '.')",
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    outname = parse_output_path(args)

    # build an ensemble from the starling file
    E = ensemble.load_ensemble(args.input_file)

    # save the ensemble as a pdb trajectory
    E.save_trajectory(outname, pdb_trajectory=True)


def starling2numpy():
    # Initialize the argument parser
    parser = ArgumentParser(description="Convert .starling ensembles to a numpy array")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "input_file", type=str, help="Input starling file", default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Directory and/or filename to save output (default: '.')",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    outname = parse_output_path(args)

    # build an ensemble from the starling file
    E = ensemble.load_ensemble(args.input_file)

    dm = E.distance_maps()
    np.save(outname, dm)


def starling2sequence():
    # Initialize the argument parser
    parser = ArgumentParser(description="Convert .starling ensembles to a numpy array")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "input_file", type=str, help="Input starling file", default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Directory and/or filename to save output (default: '.')",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # build an ensemble from the starling file
    E = ensemble.load_ensemble(args.input_file)

    print(E.sequence)


def starling2info():
    # Initialize the argument parser
    parser = ArgumentParser(description="Convert .starling ensembles to a numpy array")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "input_file", type=str, help="Input starling file", default=None
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # build an ensemble from the starling file
    E = ensemble.load_ensemble(args.input_file)

    print("-------------------------------")
    print("STARLING Generated ensemble")
    print("-------------------------------")
    print(f"Number of conformations     : {E.number_of_conformations}")
    print(f"Generate with STARLING      : {E._Ensemble__metadata['VERSION']}")
    print(f"Generate on ....            : {E._Ensemble__metadata['DATE']}")
    print(
        f"DDPM_WEIGHTS_PATH           : {E._Ensemble__metadata['DEFAULT_DDPM_WEIGHTS_PATH']}"
    )
    print(
        f"ENCODER_WEIGHTS_PATH        : {E._Ensemble__metadata['DEFAULT_ENCODER_WEIGHTS_PATH']}"
    )
    print(f"Average radius of gyration  : {E.radius_of_gyration(return_mean=True)}")
    print(f"Average end-to-end distance : {E.end_to_end_distance(return_mean=True)}")
    print(f"Sequence: {E.sequence}")
    print("-------------------------------")


def starling2starling():
    # Initialize the argument parser
    parser = ArgumentParser(
        description="Convert .starling ensembles to a different .starling ensemble! Right now this is for error correction."
    )

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "input_file", type=str, help="Input starling file", default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Directory and/or filename to save output (default: '.')",
    )
    parser.add_argument(
        "--error-check",
        action="store_true",
        default=False,
        help="If set then the we do error checking to scan the ensemble for any issues.",
    )
    parser.add_argument(
        "--remove-errors",
        action="store_true",
        default=False,
        help="If set then the we output a new .starling file with any error-identified confomers removed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set then the fixed file just overwrites the input file (including path information)...",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    outname = parse_output_path(args)

    # build an ensemble from the starling file
    E = ensemble.load_ensemble(args.input_file)

    if args.error_check:
        nframes = len(E)
        E.check_for_errors(remove_errors=args.remove_errors, verbose=True)
        if args.remove_errors and len(E) != nframes:
            if args.overwrite:
                outname = args.input_file
            else:
                outname = "fixed_" + outname
            E.save(outname, verbose=False)


def numpy2starling():
    # Initialize the argument parser
    parser = ArgumentParser(
        description="Convert nd.numpy array (generated by STARLING) files to a .starling file. Note this is ONLY to support backwards compatiblity for ensembles generated previously"
    )

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument(
        "input_file", type=str, help="Input starling file", default=None
    )
    parser.add_argument(
        "-s",
        "--sequence",
        type=str,
        default=None,
        help="Amino acid sequence of the protein; this is a required argument",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Directory and/or filename to save output (default: '.')",
    )
    parser.add_argument("-x", "--xtc", type=str, default=None, help="XTC trajectory")
    parser.add_argument("-p", "--pdb", type=str, default=None, help="PDB trajectory")
    parser.add_argument(
        "--build-structures",
        action="store_true",
        default=False,
        help="If set then the .starling file will include a reconstructred ensemble unless an XTC ensemble is passed in which case this is ignored.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.pdb is None and args.sequence is None:
        print("Error: EITHER sequence is a required or a PDB file is required")
        sys.exit(1)

    if args.xtc is not None and args.pdb is None:
        print("Error: PDB file is required when XTC file is provided")
        sys.exit(1)

    outname = parse_output_path(args)

    # build an ensemble from the starling file
    try:
        distance_maps = np.load(args.input_file)
    except Exception as e:
        print("Error loading numpy array: ", e)
        sys.exit(1)

    # if a pdb trajectory is provided, we can use it to get the sequence
    pdb_only = False
    try:
        if args.pdb and args.xtc is None:
            ssprot = SSTrajectory(args.pdb, args.pdb).proteinTrajectoryList[0]
            pdb_only = True
        elif args.pdb and args.xtc:
            ssprot = SSTrajectory(args.xtc, args.pdb).proteinTrajectoryList[0]
        else:
            ssprot = None
    except Exception as e:
        print("Error loading structural information: ", e)
        sys.exit(1)

    # if now sequence parsed get it from the parsed pdb trajectory
    if args.sequence is None:
        args.sequence = ssprot.get_amino_acid_sequence(oneletter=True)

    # overide this so we don't load a single conformer structure as the ensemble
    if pdb_only:
        ssprot = None

    # check sequence and distance map match in length
    seq_len = distance_maps.shape[1]
    if len(args.sequence) != seq_len:
        print("Error: sequence length does not match the distance map")
        sys.exit(1)

    # build
    E = ensemble.Ensemble(distance_maps, args.sequence, ssprot_ensemble=ssprot)

    if args.build_structures:
        E.build_ensemble_trajectory()

    # not ideal but we assign date to the file creation/touch time
    stat = os.stat(args.input_file)
    creation_time = stat.st_ctime
    DATE = time.ctime(creation_time)

    # finally we over-write the metadata
    E._Ensemble__metadata["DEFAULT_ENCODER_WEIGHTS_PATH"] = "UNKNOWN (from numpy array)"
    E._Ensemble__metadata["DEFAULT_DDPM_WEIGHTS_PATH"] = "UNKNOWN (from numpy array)"
    E._Ensemble__metadata["VERSION"] = "UNKNOWN (from numpy array)"
    E._Ensemble__metadata["DATE"] = DATE

    E.save(outname)


def xtc2starling():
    from pathlib import Path

    # Initialize the argument parser
    parser = ArgumentParser(description="Convert .xtc files to .starling ensembles")

    # Add command-line arguments corresponding to the parameters of the generate function
    parser.add_argument("--xtc", type=str, help="Input xtc file", default=None)
    parser.add_argument("--pdb", type=str, help="PDB topology file", default=None)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Directory and/or filename to save output (default: '.')",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    outname = Path(args.xtc).parts[-1].replace(".xtc", ".starling")
    ss_traj = SSTrajectory(args.xtc, args.pdb)
    ss_protein = ss_traj.proteinTrajectoryList[0]
    sequence = ss_protein.get_amino_acid_sequence(oneletter=True)
    distance_maps, _ = ss_protein.get_distance_map(return_instantaneous_maps=True)
    sym_distance_maps = symmetrize_distance_maps(distance_maps)

    # build an ensemble from the starling file
    ens = ensemble.Ensemble(sym_distance_maps, sequence)

    ens.save(outname)
