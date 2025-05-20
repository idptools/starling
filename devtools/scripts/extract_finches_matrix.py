import argparse
import glob
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from finches.epsilon_calculation import InteractionMatrixConstructor
from finches.forcefields.calvados import calvados_model
from finches.forcefields.mpipi import Mpipi_model
from tqdm import tqdm

from starling.data.tokenizer import StarlingTokenizer


def load_hdf5(filename):
    """Load data from an HDF5 file."""
    data = {}
    with h5py.File(filename, "r") as f:
        data["seq"] = f["seq"][()]

    return data


def get_interaction_matrix(sequence, IMC):
    """Calculate the interaction matrix from distance maps."""
    interaction_matrix = IMC.calculate_pairwise_homotypic_matrix(sequence)

    # Zero out the diagonal
    np.fill_diagonal(interaction_matrix, 0)

    return interaction_matrix


def row_normalize(matrix):
    row_norms = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)

    return matrix / row_norms


def frob_norm(matrix):
    frob_norm_value = np.linalg.norm(matrix, ord="fro")  # scalar
    return matrix / frob_norm_value


def append_to_h5(filename, interaction_matrix):
    interaction_matrix_row_norm = row_normalize(interaction_matrix)
    interaction_matrix_frob_norm = frob_norm(interaction_matrix)
    with h5py.File(filename, "a") as f:
        # Add a new dataset
        if "finches_row_norm" in f:
            del f["finches_row_norm"]  # Delete if it exists to overwrite cleanly
        if "finches" in f:
            del f["finches"]
        if "finches_frob_norm" in f:
            del f["finches_frob_norm"]
        f.create_dataset("finches", data=interaction_matrix)
        f.create_dataset("finches_row_norm", data=interaction_matrix_row_norm)
        f.create_dataset("finches_frob_norm", data=interaction_matrix_frob_norm)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract interaction matrices from .h5 files."
    )
    parser.add_argument(
        "--salt",
        type=float,
        help="Salt concentration in M",
    )
    parser.add_argument(
        "--forcefield",
        type=str,
        choices=["mpipi", "calvados"],
        help="Forcefield to use (mpipi or calvados)",
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="/work/bnovak/projects/sequence2ensemble/lammps_data/300mM_data/mPIPIgg_300mM/*encoded.h5",
        help="Glob pattern for input files",
    )
    return parser.parse_args()


def main():
    """Extract interaction matrices from .h5 files."""
    args = parse_arguments()

    # Initialize the appropriate forcefield model based on arguments
    if args.forcefield == "mpipi":
        ff_params = Mpipi_model(version="Mpipi_GGv1", salt=args.salt)
    else:  # calvados
        ff_params = calvados_model(version="CALVADOS2", salt=args.salt)

    print(f"Using {args.forcefield} forcefield with salt concentration {args.salt} M")

    IMC = InteractionMatrixConstructor(parameters=ff_params)

    files = glob.glob(args.input_pattern)
    print(f"Found {len(files)} files to process")

    tokenizer = StarlingTokenizer()

    # Process each file
    for file_name in tqdm(files):
        try:
            data = load_hdf5(file_name)
            sequence = tokenizer.decode(data["seq"])
            interaction_matrix = get_interaction_matrix(sequence, IMC)

            # Save the interaction matrix to the same file
            append_to_h5(file_name, interaction_matrix)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    main()
