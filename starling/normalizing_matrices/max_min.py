import glob

import numpy as np
from IPython import embed
from tqdm import tqdm

dirs = sorted(
    glob.glob("/home/bnovak/projects/VAE_training/random_walk_pretraining/*/")
)

mean_matrix = np.load("mean_matrix.npy")
std_matrix = np.load("standard_deviation_per_residue.npy")


def standard(original_array):
    height, width = original_array[0].shape
    standardized_data = (original_array - mean_matrix[:height, :width]) / (
        std_matrix[:height, :width] + 1e-5
    )

    return standardized_data


def MaxPad(original_array):
    # Pad the distance map to a desired shape, here we are using
    # (768, 768) because largest sequences are 750 residues long
    # and 768 can be divided by 2 a bunch of times leading to nice
    # behavior during conv2d and conv2transpose down- and up-sampling
    pad_height = max(0, 768 - original_array.shape[0])
    pad_width = max(0, 768 - original_array.shape[1])
    return np.pad(
        original_array,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )


max_matrix = np.zeros((768, 768))
min_matrix = np.ones((768, 768)) * 1000

for dir in tqdm(dirs):
    files_to_load = glob.glob(f"{dir}*gz")
    distances = np.array([np.loadtxt(i) for i in files_to_load])
    distances_standard = standard(distances)
    max_distances_standard = MaxPad(np.max(distances_standard, axis=0))
    min_distances_standard = MaxPad(np.min(distances_standard, axis=0))
    # embed()
    max_matrix = np.maximum(max_distances_standard, max_matrix)
    min_matrix = np.minimum(min_distances_standard, min_matrix)

embed()
