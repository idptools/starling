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


def calculate_running_mean(dir):
    files_to_load = glob.glob(f"{dir}*gz")
    distances = np.array([np.loadtxt(i) for i in files_to_load])
    means = np.mean(distances, axis=0)
    means = MaxPad(means)
    return means


dirs = sorted(
    glob.glob("/home/bnovak/projects/VAE_training/random_walk_pretraining/*/")
)


mean_per_sequence = []

with mp.Pool(processes=32) as pool:
    results = list(tqdm(pool.imap(calculate_running_mean, dirs), total=len(dirs)))
    for mean in results:
        mean_per_sequence.append(mean)

np.save("mean_per_seq_length.npy", mean_per_sequence)

non_zero_counts = np.count_nonzero(mean_per_sequence, axis=0)
non_zero_counts[non_zero_counts == 0] = 1

total_mean_sum = np.sum(mean_per_sequence, axis=0)

print(non_zero_counts[0])

np.save("mean_matrix.npy", total_mean_sum / non_zero_counts)
