import os
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sparrow
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tabulate import tabulate
from tqdm import tqdm

from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNetConditional
from starling.models.vae import VAE
from starling.samplers.ddim_sampler import DDIMSampler


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


def symmetrize_distance_map(dist_map):
    # Ensure the distance map is 2D

    dist_map = dist_map.squeeze(0) if dist_map.dim() == 3 else dist_map

    # Create a copy of the distance map to modify
    sym_dist_map = dist_map.clone()

    # Replace the lower triangle with the upper triangle values
    mask_upper_triangle = torch.triu(torch.ones_like(dist_map), diagonal=1).bool()
    mask_lower_triangle = ~mask_upper_triangle

    # Set lower triangle values to be the same as the upper triangle
    sym_dist_map[mask_lower_triangle] = dist_map.T[mask_lower_triangle]

    # Set diagonal values to zero
    sym_dist_map.fill_diagonal_(0)

    return sym_dist_map.detach().cpu().numpy().squeeze()


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


def reconstruct(model, distance_maps):
    recon_dm, _ = model(distance_maps)
    recon_dm = recon_dm.detach().cpu().numpy().squeeze()

    return recon_dm


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
    paths = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            paths[line[0]] = line[1]
    return paths


def prepare_data(data):
    """
    Prepare the data for inference.
    """
    dm = torch.from_numpy(data)
    dm = dm.unsqueeze(1)

    return dm


def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def hellingers_summary(
    gt_dms, recon_dms, save_path="hellinger_heatmap.png", plot_histogram=False
):
    """
    Compute Hellinger distances between corresponding residue pairs in ground truth and reconstructed distance maps.

    Parameters
    ----------
    gt_dms : np.ndarray
        Ground truth distance maps of shape (num_gt_conformers, padded_length, padded_length)
        where padded_length is the padded number of residues in the sequence
    recon_dms : np.ndarray
        Reconstructed distance maps of shape (num_recon_conformers, actual_length, actual_length)
        where actual_length is the actual number of residues in the sequence

    Returns
    -------
    np.ndarray
        Array of shape (actual_length, actual_length) containing Hellinger distances for each residue pair
    """
    actual_length = recon_dms.shape[1]

    # Trim ground truth maps to match the size of reconstructed maps
    gt_dms_trimmed = gt_dms[:, :actual_length, :actual_length]

    assert gt_dms_trimmed.shape[1:] == recon_dms.shape[1:], (
        "Inconsistent shapes between trimmed ground truth and reconstructed distance maps"
    )

    colors = ["#4c72b0", "#dd8452"]

    n_bins = 25
    # Initialize the result array
    hellinger_distances = np.zeros((actual_length, actual_length))
    gt_histograms = []
    recon_histograms = []
    for i in range(actual_length):
        for j in range(actual_length):
            global_min = min(gt_dms_trimmed[:, i, j].min(), recon_dms[:, i, j].min())

            global_max = max(gt_dms_trimmed[:, i, j].max(), recon_dms[:, i, j].max())

            bin_width = (global_max - global_min) / n_bins

            gt_values = gt_dms_trimmed[:, i, j]
            recon_values = recon_dms[:, i, j]

            hist_gt, _ = np.histogram(
                gt_values,
                bins=n_bins,
                range=(global_min, global_max),
                density=True,
            )

            hist_gt *= bin_width
            gt_histograms.append(hist_gt)
            hist_recon, _ = np.histogram(
                recon_values,
                bins=n_bins,
                range=(global_min, global_max),
                density=True,
            )
            hist_recon *= bin_width
            recon_histograms.append(hist_recon)
            os.makedirs("plots", exist_ok=True)
            if plot_histogram:
                # Plot histogram for ground truth values (gt_values)
                plt.hist(
                    gt_values,
                    bins=n_bins,
                    range=(global_min, global_max),
                    density=True,
                    alpha=0.6,  # Slightly increase the opacity
                    color=colors[0],  # Use custom color
                    edgecolor="black",  # Add edge color for visibility
                    label="Ground Truth",  # Label for legend
                    linewidth=1.2,  # Line width for the edges
                )

                # Plot histogram for reconstructed values (recon_values)
                plt.hist(
                    recon_values,
                    bins=n_bins,
                    range=(global_min, global_max),
                    density=True,
                    alpha=0.6,
                    color=colors[1],
                    edgecolor="black",
                    label="Reconstruction",
                    linewidth=1.2,
                )

                # Add labels, title, and legend
                plt.xlabel("Value", fontsize=14)
                plt.ylabel("Density", fontsize=14)
                plt.title(f"Residue Histogram for {i}-{j}", fontsize=16)
                plt.legend(fontsize=12)

                # Adjust ticks and add gridlines for better readability
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

                # Save the plot with high DPI and tight layout
                plt.savefig(
                    f"plots/residue_{i}_{j}_histogram.png", dpi=300, bbox_inches="tight"
                )

                # Close the figure to free memory
                plt.close()

            # Compute Hellinger distance
            hellinger_distances[i, j] = hellinger_distance(hist_gt, hist_recon)

    # Create and save the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(hellinger_distances, cmap="viridis", annot=False, cbar=True)
    plt.title(
        "Hellinger Distances between Ground Truth and Reconstructed Distance Maps"
    )
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return hellinger_distances


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vae", type=str, required=True)
    parser.add_argument("--ddpm", type=str, required=True)
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--conformations", type=int, default=100)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outfile", type=str, default="summary_stats_ddpm.csv")
    args = parser.parse_args()

    # Load the VAE model
    vae = VAE.load_from_checkpoint(args.vae, map_location=args.device)

    # Load the UNet model
    UNet_model = UNetConditional(
        in_channels=1,
        out_channels=1,
        base=64,
        norm="group",
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=384,
    )

    # Load the DDPM model
    diffusion = DiffusionModel.load_from_checkpoint(
        args.ddpm,
        model=UNet_model,
        encoder_model=vae,
        map_location=args.device,
    )

    # Construct a sampler
    if args.ddim:
        sampler = DDIMSampler(ddpm_model=diffusion, n_steps=args.steps)
    else:
        sampler = diffusion

    # Read the input file
    paths = read_input_file(args.input)

    # Start a dataframe
    results = OrderedDict()

    for path in tqdm(paths):
        sequence_stats = OrderedDict()

        if Path(paths[path]).suffix == ".h5":
            data = load_hdf5_compressed(paths[path], keys_to_load=["dm", "seq"])
            mpipi_dm = data["dm"]
            sequence = int_to_seq(data["seq"])
        else:
            mpipi_dm = []
            sequence = paths[path]

        num_batches = args.conformations // args.batch
        remaining_samples = args.conformations % args.batch

        starling_dm = []

        for batch in range(num_batches):
            distance_map = sampler.sample(args.batch, labels=sequence)
            starling_dm.append(
                [
                    symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                    for dm in distance_map
                ]
            )

        if remaining_samples > 0:
            distance_map = sampler.sample(remaining_samples, labels=sequence)
            starling_dm.append(
                [
                    symmetrize_distance_map(dm[:, : len(sequence), : len(sequence)])
                    for dm in distance_map
                ]
            )
        starling_dm = np.concatenate(starling_dm, axis=0)

        hellinger_distances = hellingers_summary(
            mpipi_dm, starling_dm, save_path=f"plots/hellinger_heatmap_{path}.png"
        )

        upper_triangle_bond_offset = np.triu(hellinger_distances, k=1).mean()

        albatross_prediction = sparrow.Protein(sequence).predictor.end_to_end_distance()
        starling_prediction = starling_dm.mean(axis=0)[0, -1]

        if len(mpipi_dm) > 0:
            mpipi_prediction = round(mpipi_dm.mean(axis=0)[0, len(sequence) - 1], 4)
            mean_mpipi = mpipi_dm[:, : len(sequence), : len(sequence)].mean(axis=0)
            mean_starling = starling_dm.mean(axis=0)
            dm_difference = abs(mean_starling - mean_mpipi)

            dm_difference_mean = round(dm_difference.mean(), 4)
            dm_difference_max = round(dm_difference.max(), 4)
        else:
            mpipi_prediction = None
            dm_difference_mean = None
            dm_difference_max = None

        sequence_stats["STARLING"] = round(starling_prediction, 4)
        sequence_stats["ALBATROSS"] = round(albatross_prediction, 4)
        sequence_stats["Mpipi"] = mpipi_prediction
        sequence_stats["Average_dm_abe"] = dm_difference_mean
        sequence_stats["Max_dm_abe"] = dm_difference_max
        sequence_stats["Mean_hellinger_distance"] = round(upper_triangle_bond_offset, 4)

        results[path] = sequence_stats

    results_df = pd.DataFrame(results).T

    correlation = {
        "STARLING": round(
            pearsonr(
                results_df["STARLING"],
                results_df["ALBATROSS"]
                if mpipi_prediction is None
                else results_df["Mpipi"],
            )[0],
            4,
        )
    }

    if len(mpipi_dm) > 0:
        correlation["ALBATROSS"] = round(
            pearsonr(
                results_df["ALBATROSS"],
                results_df["Mpipi"],
            )[0],
            4,
        )

    results_df.loc["Correlation"] = correlation

    results_df.to_csv(args.outfile, index=True)

    summary = results_df.tail(1).reset_index(drop=False)

    formatted_summary = tabulate(
        summary, headers="keys", tablefmt="pipe", floatfmt=".4f", showindex=False
    )

    print(formatted_summary)


if __name__ == "__main__":
    main()
