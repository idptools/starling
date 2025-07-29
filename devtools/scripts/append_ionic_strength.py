import argparse
import glob

import h5py
import hdf5plugin
from tqdm import tqdm


def append_ionic_strength(filename, ionic_strength):
    """Add ionic strength value to an HDF5 file."""
    with h5py.File(filename, "a") as f:
        # Delete existing ionic_strength if it exists
        if "ionic_strength" in f:
            del f["ionic_strength"]
        # Create the dataset with the scalar value
        f.create_dataset("ionic_strength", data=ionic_strength)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Append ionic strength value to HDF5 files."
    )
    parser.add_argument(
        "--salt",
        type=float,
        required=True,
        help="Ionic strength value in mM to append to the files",
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="/work/bnovak/projects/sequence2ensemble/lammps_data/300mM_data/mPIPIgg_300mM/*encoded.h5",
        help="Glob pattern for input files",
    )
    return parser.parse_args()


def main():
    """Append ionic strength to .h5 files."""
    args = parse_arguments()

    print(f"Will append ionic strength of {args.salt} mM to matching files")

    # Find all files matching the pattern
    files = glob.glob(args.input_pattern)
    print(f"Found {len(files)} files to process")

    # Process each file
    for file_name in tqdm(files):
        try:
            # Add the ionic strength to the file
            append_ionic_strength(file_name, args.salt)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    main()
