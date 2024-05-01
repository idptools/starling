import os
from typing import List

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from finches.frontend.mpipi_frontend import Mpipi_frontend
from IPython import embed


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(
        self, txt_file: str, target_shape: int, pretraining: bool = False
    ) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch

        Parameters
        ----------
        txt_file : str
            A path to a text file containing the paths to distance maps
        target_shape : int
            The desired shape of the distance maps. The distance map will be
            padded
        """
        self.data_path = self.read_paths(txt_file)
        self.target_shape = (target_shape, target_shape)
        self.pretraining = pretraining

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        # Get a single data sample
        try:
            sample = np.loadtxt(self.data_path[index], dtype=np.float32)
        except Exception as e:
            data_path, frame = self.data_path[index].split()
            data = self.load_hdf5_compressed(
                data_path, keys_to_load=["dm", "seq"], frame=int(frame)
            )
            sample = self.symmetrize(data["dm"]).astype(np.float32)
            sequence = data["seq"][()].decode()

        if not self.pretraining:
            encoder_condition = self.get_interaction_matrix(sequence)
            encoder_condition = self.symmetrize(encoder_condition)
            decoder_condition = self.one_hot_encode(sequence)
            encoder_condition = self.MaxPad(
                encoder_condition, shape=(self.target_shape)
            )
            decoder_condition = self.MaxPad(
                decoder_condition, shape=(self.target_shape[0], 20)
            )

            encoder_condition = torch.from_numpy(
                encoder_condition.astype(np.float32)
            ).unsqueeze(0)
            decoder_condition = torch.from_numpy(decoder_condition.astype(np.float32))
        else:
            encoder_condition = {}
            decoder_condition = {}

        # Resize the input distance map with padding
        sample = self.MaxPad(sample, shape=(self.target_shape))

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        return {
            "data": sample,
            "encoder_condition": encoder_condition,
            "decoder_condition": decoder_condition,
        }

    def read_paths(self, txt_file: str) -> List:
        """
        A function that reads the paths to distance maps from a txt file

        Parameters
        ----------
        txt_file : str
            A path to the text file containing the paths to distance maps

        Returns
        -------
        List
            A list of paths to distance maps
        """
        paths = []
        with open(txt_file, "r") as file:
            for line in file:
                path = line.strip()
                paths.append(path)
        return paths

    def MaxPad(self, original_array: np.array, shape: tuple) -> np.array:
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

    def get_interaction_matrix(self, sequence):
        mf = Mpipi_frontend()
        interaction_matrix = mf.intermolecular_idr_matrix(
            sequence, sequence, window_size=1
        )
        return interaction_matrix[0][0]

    def one_hot_encode(self, sequence):
        """
        One-hot encodes a sequence.
        """
        # Define the mapping of each amino acid to a unique integer
        aa_to_int = {
            "A": 0,
            "C": 1,
            "D": 2,
            "E": 3,
            "F": 4,
            "G": 5,
            "H": 6,
            "I": 7,
            "K": 8,
            "L": 9,
            "M": 10,
            "N": 11,
            "P": 12,
            "Q": 13,
            "R": 14,
            "S": 15,
            "T": 16,
            "V": 17,
            "W": 18,
            "Y": 19,
        }

        # One-hot encode the sequence
        one_hot_sequence = np.zeros((len(sequence), len(aa_to_int)), dtype=np.float32)
        for i, aa in enumerate(sequence):
            if aa in aa_to_int:
                one_hot_sequence[i, aa_to_int[aa]] = 1
            else:
                one_hot_sequence[i, aa_to_int["X"]] = 1
        return one_hot_sequence

    def load_hdf5_compressed(self, file_path, frame, keys_to_load=None):
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

    def symmetrize(self, matrix):
        """
        Symmetrizes a matrix.
        """
        if np.array_equal(matrix, matrix.T):
            return matrix
        else:
            return matrix + matrix.T - np.diag(np.diag(matrix))


# Step 2: Create a data module
class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=None,
        target_shape=None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.num_workers = int(os.cpu_count() / 4)

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(
                self.train_data,
                target_shape=self.target_shape,
            )
            self.val_dataset = MatrixDataset(
                self.val_data,
                target_shape=self.target_shape,
            )
        if stage == "test":
            self.test_dataset = MatrixDataset(
                self.test_data,
                target_shape=self.target_shape,
            )
        if stage == "predict":
            self.predict_dataset = MatrixDataset(
                self.predict_data,
                target_shape=self.target_shape,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
