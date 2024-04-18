import os
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file: str, target_shape: int) -> None:
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

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        # Get a single data sample
        sample = np.loadtxt(self.data_path[index], dtype=np.float32)

        # Resize the input distance map with padding
        sample = self.MaxPad(sample)

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        return {"data": sample}

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

    def MaxPad(self, original_array: np.array) -> np.array:
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
        pad_height = max(0, self.target_shape[0] - original_array.shape[0])
        pad_width = max(0, self.target_shape[1] - original_array.shape[1])
        return np.pad(
            original_array,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )

    def normalize_by_normalization_matrix(self, original_array):
        # Divide by some normalization matrix
        shape = np.shape(original_array)
        normalized_matrix = (
            original_array / self.normalization_matrix[: shape[0], : shape[1]]
        )
        return normalized_matrix

    def generate_normalization_matrix(self, shape, bond_length=3.81):
        # This normalization matrix is setup so that the constant distances
        # (i.e., bond lengths) stay constant by each element within the matrix
        # being an accumulated count of residues times the bond_length
        norm_matrix = np.zeros(shape, dtype=int)
        for i in range(shape[0]):
            norm_matrix[i, i : shape[1]] = np.arange(shape[1] - i)

        norm_matrix = norm_matrix * bond_length
        norm_matrix = norm_matrix + norm_matrix.T
        norm_matrix[norm_matrix == 0] = 1

        return norm_matrix

    def generate_afrc_distance_map(self, target_shape):
        # Normalize by the average distance map for an ideal homopolymer
        from afrc import AnalyticalFRC

        seq = "G" * target_shape[0]
        protein = AnalyticalFRC(seq)
        distance_map = protein.get_distance_map()
        distance_map = distance_map + distance_map.T
        distance_map[distance_map == 0] = 1

        return distance_map


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
