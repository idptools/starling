from typing import List

import h5py
import numpy as np


def one_hot_encode(sequence):
    """
    One-hot encodes a sequence.
    """
    # Define the mapping of each amino acid to a unique integer
    aa_to_int = {
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

    # One-hot encode the sequence
    one_hot_sequence = np.zeros((len(sequence), len(aa_to_int) + 1), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in aa_to_int:
            one_hot_sequence[i, aa_to_int[aa]] = 1
        else:
            one_hot_sequence[i, aa_to_int["X"]] = 1
    return one_hot_sequence


def MaxPad(original_array: np.array, shape: tuple) -> np.array:
    """
    A function that takes in a distance map and pads it to a desired shape

    Parameters
    ----------
    original_array : np.array
        A distance map

    Returns
    -------
    np.array
        A distance map padded to a desired shape
    """
    # Pad the distance map to a desired shape
    pad_height = max(0, shape[0] - original_array.shape[0])
    pad_width = max(0, shape[1] - original_array.shape[1])
    return np.pad(
        original_array,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )


def load_hdf5_compressed(file_path, frame, keys_to_load=None):
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
                data_dict[key] = f[key][frame]
            else:
                data_dict[key] = f[key][...]
    return data_dict


def read_tsv_file(tsv_file: str) -> List:
    """
    A function that reads the paths to distance maps from a txt file

    Parameters
    ----------
    txt_file : str
        A path to a tsv file containing the paths to distance maps as a first column
        and index of a distance map to load as a second column

    Returns
    -------
    List
        A list of paths to distance maps
    """
    paths = []
    with open(tsv_file, "r") as file:
        for line in file:
            line = line.strip()
            line = line.split("\t")[0:2]
            paths.append(line)
    return paths


def symmetrize(matrix):
    """
    Symmetrizes a matrix.
    """
    if np.array_equal(matrix, matrix.T):
        return matrix
    else:
        # Extract upper triangle excluding diagonal
        upper_triangle = np.triu(matrix, k=1)
        # Symmetrize upper triangle by mirroring
        sym_matrix = upper_triangle + upper_triangle.T
        # Add diagonal elements (to handle odd-sized matrices)
        sym_matrix += np.diag(np.diag(matrix))
        return sym_matrix
