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

from starling.models.vae import VAE


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


def finches_potential_energy(data):
    mpipi = Mpipi_model()
    interaction_energy = 0
    dm = data[0]
    seq = data[1]
    bonds = np.diagonal(dm, offset=1)
    harmonic_energy = harmonic(bonds).sum()
    sequence = list(seq)
    for num, residue in enumerate(sequence):
        for next_residue in range(2, len(sequence[num:])):
            residue_interaction = mpipi.compute_full_Mpipi(
                residue,
                sequence[num + next_residue],
                dm[num, num + next_residue],
            )
            interaction_energy += residue_interaction
    return interaction_energy, harmonic_energy, interaction_energy + harmonic_energy


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

    # Load the VAE model
    vae = VAE.load_from_checkpoint(args.vae, map_location=args.device)

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
                reconstruct(
                    vae,
                    ground_truth_dm[batch * args.batch : (batch + 1) * args.batch].to(
                        args.device
                    ),
                )
            )

        if remaining_samples > 0:
            recon_dm.append(
                reconstruct(
                    vae,
                    ground_truth_dm[
                        (batch + 1) * args.batch : (batch + 1) * args.batch
                        + remaining_samples
                    ].to(args.device),
                )
            )

        recon_dm = np.concatenate(recon_dm, axis=0)

        mask = data["dm"] != 0
        mask = mask ^ np.tril(mask)
        all_mse, bonds_mse = get_errors(
            torch.from_numpy(recon_dm), torch.from_numpy(data["dm"]), mask
        )

        ## Calculate potential energy
        # recon_energy_data = zip(
        #    recon_dm, [data["seq"] for _ in range(recon_dm.shape[0])]
        # )
        # ground_truth_energy_data = zip(
        #    data["dm"], [data["seq"] for _ in range(recon_dm.shape[0])]
        # )

        ## Run the calculations in parallel
        # recon_results = pool.map(finches_potential_energy, recon_energy_data)
        # ground_truth_results = pool.map(
        #    finches_potential_energy, ground_truth_energy_data
        # )

        # recon_interaction_energy, recon_harmonic_energy, _ = zip(*recon_results)
        # ground_truth_interaction_energy, ground_truth_harmonic_energy, _ = zip(
        #    *ground_truth_results
        # )

        sequence_stats["mse"] = round(all_mse.mean(), 4)
        sequence_stats["std_mse"] = round(all_mse.std(), 4)
        sequence_stats["max_mse"] = round(all_mse.max(), 4)

        sequence_stats["bond_mse"] = round(bonds_mse.mean(), 4)
        sequence_stats["std_bond_mse"] = round(bonds_mse.std(), 4)
        sequence_stats["max_bond_mse"] = round(bonds_mse.max(), 4)

        ## Positive difference means the model is generating distance maps that are less stable
        # difference = list(recon_interaction_energy) - np.array(
        #    ground_truth_interaction_energy
        # )

        # sequence_stats["Potential_energy_abe"] = round(difference.mean(), 4)
        # sequence_stats["Max_potential_energy_abe"] = round(difference.max(), 4)

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
