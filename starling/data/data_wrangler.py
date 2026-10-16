import h5py
import hdf5plugin
import numpy as np
import pandas as pd

# Character used to right-pad shorter sequences in one_hot_encode; it is the padding
# token (index 0) of the amino-acid alphabet below
PADDING_TOKEN = "0"


def one_hot_encode(sequences):
    """
    One-hot encodes a sequence.

    Sequences shorter than the longest one are right-padded with the padding
    token "0", so a batch of different-length sequences is encoded exactly as if
    each had been padded with "0" by hand.
    """
    # Define the mapping of each amino acid to a unique integer
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

    num_sequences = len(sequences)
    sequence_length = max(len(sequence) for sequence in sequences)
    num_classes = len(aa_to_int)

    # Convert sequences to a padded numpy array of integers; np.array on the ragged
    # per-sequence lists would raise for sequences of different lengths
    int_sequences = np.full(
        (num_sequences, sequence_length), aa_to_int[PADDING_TOKEN], dtype=np.int64
    )
    for i, sequence in enumerate(sequences):
        int_sequences[i, : len(sequence)] = [aa_to_int[aa] for aa in sequence]

    # Create an array of zeros with shape (num_sequences, sequence_length, num_classes)
    one_hot_encoded_seq = np.zeros(
        (num_sequences, sequence_length, num_classes), dtype=np.float32
    )

    # Use advanced indexing to set the appropriate elements to 1
    for i, sequence in enumerate(int_sequences):
        one_hot_encoded_seq[i, np.arange(len(sequence)), sequence] = 1

    return one_hot_encoded_seq


def MaxPad(original_array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    A function that takes in a distance map and pads it to a desired shape

    Parameters
    ----------
    original_array : np.array
        A distance map
    shape : tuple[int, int]
        The (height, width) to zero-pad the distance map to. Must be at least as
        large as the distance map in both dimensions.

    Returns
    -------
    np.array
        A distance map padded to a desired shape

    Raises
    ------
    ValueError
        If the distance map is larger than shape in either dimension, since
        padding cannot shrink it.
    """
    if original_array.shape[0] > shape[0] or original_array.shape[1] > shape[1]:
        raise ValueError(
            f"Cannot pad a distance map of shape {original_array.shape} to the "
            f"smaller shape {tuple(shape)}"
        )

    # Pad the distance map to a desired shape
    pad_height = shape[0] - original_array.shape[0]
    pad_width = shape[1] - original_array.shape[1]
    return np.pad(
        original_array,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )


def load_hdf5_compressed(file_path, frame=None, keys_to_load=None):
    """
    Loads data from an HDF5 file, optionally for a specific frame.

    Parameters:
        - file_path (str): Path to the HDF5 file.
        - frame (int, optional): Specific frame to load from 'dm' or 'latents'. If None, loads full datasets.
        - keys_to_load (list, optional): Specific dataset keys to load. If None, loads all available keys.

    Returns:
        - dict: {key: np.ndarray or np.ndarray slice}

    Raises:
        - IndexError: If frame is out of range for 'dm' or 'latents'.
    """
    special_keys = {"dm", "latents"}
    data = {}

    with h5py.File(file_path, "r") as f:
        keys = keys_to_load if keys_to_load is not None else list(f.keys())
        for key in keys:
            dataset = f[key]
            if frame is None or key not in special_keys:
                data[key] = dataset[()]
                continue

            # Fail loudly rather than silently returning every frame when the
            # requested frame does not exist
            num_frames = dataset.shape[0]
            if not -num_frames <= frame < num_frames:
                raise IndexError(
                    f"frame {frame} is out of range for '{key}' in {file_path}, "
                    f"which has {num_frames} frames"
                )
            data[key] = dataset[frame]
    return data


def read_tsv_file(tsv_file: str) -> pd.DataFrame:
    """
    A function that reads the paths to distance maps from a tsv file

    Parameters
    ----------
    tsv_file : str
        A path to a tsv file containing the paths to distance maps as a first column
        and index of a distance map to load as a second column

    Returns
    -------
    pandas.DataFrame
        Rows containing paths to distance maps and their indices.
    """
    df = pd.read_csv(tsv_file, sep="\t", header=None, usecols=range(2))
    return df


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
