import multiprocessing as mp
from argparse import ArgumentParser
from collections import OrderedDict

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from finches.forcefields.mpipi import Mpipi_model, harmonic
from tabulate import tabulate
from tqdm import tqdm

from starling.models.vqvae import VQVAE


def track_codebook_usage(encoder_outputs, codebook_vectors):
    """
    Tracks the utilization of codebook entries during training.

    Args:
        encoder_outputs (torch.Tensor): The output of the encoder before quantization.
                                        Shape: (batch_size, num_latents, latent_dim)
        codebook_vectors (torch.Tensor): The learnable codebook vectors.
                                         Shape: (num_codebook_entries, latent_dim)

    Returns:
        usage_counts (np.ndarray): Count of how many times each codebook entry was selected.
    """
    # Compute distances to all codebook entries

    # Shape: (batch_size, num_latents, num_codebook_entries)
    distances = torch.cdist(encoder_outputs.view(-1, 1), codebook_vectors)

    # Find the closest codebook entry for each encoder output
    # Shape: (batch_size, num_latents)
    closest_codebook_indices = torch.argmin(distances, dim=-1)

    # Count how many times each codebook entry is used
    num_codebook_entries = codebook_vectors.shape[0]
    usage_counts = (
        torch.bincount(
            closest_codebook_indices.view(-1), minlength=num_codebook_entries
        )
        .cpu()
        .numpy()
    )

    return usage_counts


def int_to_seq(int_seq):
    """
    Convert an integer sequence to a string sequence.
    """
    aa_to_int = {
        "0": 0,
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
    }
    reversed_dict = {v: k for k, v in aa_to_int.items()}
    seq = ""
    for i in int_seq:
        if i != 0:
            seq += str(reversed_dict[i])
    return seq


def symmetrize(dm):
    """
    Symmetrize a distance map.
    """
    dm = np.array([np.triu(m, k=1) + np.triu(m, k=1).T for m in dm])

    return dm


def load_hdf5_compressed(file_path, keys_to_load=None):
    """
    Loads data from an HDF5 file.

    Parameters:
        - file_path (str): Path to the HDF5 file.
        - keys_to_load (list): List of keys to load. If None, loads all keys.
    Returns:
        - dict: Dictionary containing loaded data.
    """
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        keys = keys_to_load if keys_to_load else f.keys()
        for key in keys:
            if key == "dm":
                data_dict[key] = f[key][...]
            else:
                data_dict[key] = f[key][...]
    return data_dict


def get_codebook_usage(model, distance_maps):
    encoder_outputs = model.encode_to_prequant(distance_maps)
    codebook_vectors = model.quantize.embedding.weight
    usage_counts = track_codebook_usage(encoder_outputs, codebook_vectors)
    total_entries = len(usage_counts)
    used_entries = np.count_nonzero(usage_counts)
    utilization_percentage = (used_entries / total_entries) * 100
    return utilization_percentage


def get_errors(recon_dm, dm, mask):
    recon = F.mse_loss(recon_dm, dm, reduction="none")

    recon = recon * mask
    all_mse = recon.sum(axis=(1, 2)) / mask.sum(axis=(1, 2))

    all_mse = np.array([i.item() for i in all_mse])

    recon_bonds = [i.diagonal(offset=1) for i in recon]
    mask_bonds = [i.diagonal(offset=1) for i in mask]

    bonds_mse = np.array(
        [(i.sum() / j.sum()).item() for i, j in zip(recon_bonds, mask_bonds)]
    )

    return all_mse, bonds_mse


def read_input_file(file_path):
    """
    Read the input file and return a list of paths to the HDF5 files.
    """
    paths = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            paths.append(line)
    return paths


def prepare_data(data):
    """
    Prepare the data for inference.
    """
    dm = torch.from_numpy(data)
    dm = dm.unsqueeze(1)

    return dm


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vae", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--outfile", type=str, default="summary_stats_vae.csv")
    args = parser.parse_args()

    # Get the number of cores
    num_cores = mp.cpu_count()
    # Create a pool of workers
    pool = mp.Pool(num_cores)

    # Load the VQVAE model
    vae = VQVAE.load_from_checkpoint(args.vae, map_location=args.device)

    # Read the input file
    paths = read_input_file(args.input)

    # Start a dataframe
    results = OrderedDict()

    for path in tqdm(paths):
        sequence_stats = OrderedDict()
        data = load_hdf5_compressed(path, keys_to_load=["dm", "seq"])

        # Get the data
        data["seq"] = int_to_seq(data["seq"])
        ground_truth_dm = prepare_data(data["dm"])

        num_batches = ground_truth_dm.shape[0] // args.batch
        remaining_samples = ground_truth_dm.shape[0] % args.batch

        recon_dm = []

        for batch in range(num_batches):
            recon_dm.append(
                get_codebook_usage(
                    vae,
                    ground_truth_dm[batch * args.batch : (batch + 1) * args.batch].to(
                        args.device
                    ),
                )
            )

        if remaining_samples > 0:
            recon_dm.append(
                get_codebook_usage(
                    vae,
                    ground_truth_dm[
                        (batch + 1) * args.batch : (batch + 1) * args.batch
                        + remaining_samples
                    ].to(args.device),
                )
            )

        sequence_stats["usage"] = round(np.mean(recon_dm), 4)

        mask = data["dm"] != 0
        sequence_stats["Sequence Length"] = mask[0].diagonal(offset=1).sum() + 1

        results[path] = sequence_stats

    results_df = pd.DataFrame(results).T

    # Calculate the mean and max of each column
    mean_values = results_df.mean().round(4)
    max_values = results_df.max()

    results_df.loc["Overall Mean"] = mean_values
    results_df.loc["Overall Max"] = max_values

    results_df.to_csv(args.outfile, index=True)

    summary = results_df.tail(2).reset_index(drop=False)

    formatted_summary = tabulate(
        summary, headers="keys", tablefmt="pipe", floatfmt=".4f", showindex=False
    )

    print(formatted_summary)


if __name__ == "__main__":
    main()
