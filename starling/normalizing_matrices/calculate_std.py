import glob
import multiprocessing as mp
from pathlib import Path

import numpy as np
from IPython import embed
from tqdm import tqdm

shape = (768, 768)


def MaxPad(original_array):
    # Pad the distance map to a desired shape, here we are using
    # (768, 768) because largest sequences are 750 residues long
    # and 768 can be divided by 2 a bunch of times leading to nice
    # behavior during conv2d and conv2transpose down- and up-sampling
    pad_height = max(0, shape[0] - original_array.shape[0])
    pad_width = max(0, shape[1] - original_array.shape[1])
    return np.pad(
        original_array,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )


mean_matrix = np.load("mean_matrix.npy")


def calculate_running_mean(dir):
    files_to_load = glob.glob(f"{dir}*gz")
    distances = np.array([np.loadtxt(i) for i in files_to_load])
    shapes = distances.shape
    deviations = abs((distances - mean_matrix[: shapes[1], : shapes[2]])) ** 2
    deviations = np.sum(deviations, axis=0)
    deviations = MaxPad(deviations)
    return deviations


dirs = sorted(
    glob.glob("/home/bnovak/projects/VAE_training/random_walk_pretraining/*/")
)

# calculate_running_mean(dirs[0])

deviations_per_sequence = []

with mp.Pool(processes=32) as pool:
    results = list(tqdm(pool.imap(calculate_running_mean, dirs), total=len(dirs)))
    for deviation in results:
        deviations_per_sequence.append(deviation)

np.save("deviations_per_sequence.npy", deviations_per_sequence)

non_zero_counts = np.count_nonzero(deviations_per_sequence, axis=0) * 1000
non_zero_counts[non_zero_counts == 0] = 1

total_deviations = np.sum(deviations_per_sequence, axis=0)

print(non_zero_counts[0])

np.save(
    "standard_deviation_per_residue.npy", np.sqrt(total_deviations / non_zero_counts)
)
