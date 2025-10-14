import glob
import multiprocessing as mp
from functools import partial
from pathlib import Path

import h5py
import hdf5plugin
import pandas as pd
from tqdm import tqdm

from starling.data.tokenizer import StarlingTokenizer
from starling.structure.ensemble import Ensemble


def load_hdf5(filename):
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
    return data


def get_global_dimensions(sequence, distance_maps):
    ensemble = Ensemble(distance_maps, sequence)

    return ensemble.radius_of_gyration().mean(), ensemble.end_to_end_distance().mean()


def process_file(file_name, tokenizer):
    """Process a single file and return its results."""
    try:
        data = load_hdf5(file_name)
        sequence = tokenizer.decode(data["seq"])
        distance_maps = data["dm"][:, : len(sequence), : len(sequence)]
        rg, re = get_global_dimensions(sequence, distance_maps)

        return {
            "name": file_name.name,
            "rg": rg,
            "re": re,
            "sequence_length": len(sequence),
            "status": "success",
        }
    except Exception as e:
        # Handle any errors that might occur during processing
        return {"name": file_name.name, "status": f"error: {str(e)}"}


def main():
    tokenizer = StarlingTokenizer()

    path = Path(
        "/work/bnovak/projects/sequence2ensemble/lammps_data/20mM_data/mPIPIgg_20mM"
    )
    files = [f for f in path.glob("*.h5") if "encoded" not in f.name]

    # Determine the number of processes to use
    num_processes = min(mp.cpu_count(), len(files))
    print(f"Using {num_processes} processes to process {len(files)} files")

    # Create a partial function with the tokenizer already included
    process_fn = partial(process_file, tokenizer=tokenizer)

    # Process files in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm(pool.imap(process_fn, files), total=len(files)))

    # Filter successful results
    successful_results = [r for r in results if r.get("status") == "success"]
    error_results = [r for r in results if r.get("status") != "success"]

    if error_results:
        print(f"Warning: {len(error_results)} files failed to process")

    # Create DataFrame from successful results
    results_df = pd.DataFrame(successful_results)

    # Drop the status column as it's no longer needed
    if "status" in results_df.columns:
        results_df = results_df.drop(columns=["status"])

    # Save results to CSV
    output_path = path.parent / "global_dimensions_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
