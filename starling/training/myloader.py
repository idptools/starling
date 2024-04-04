import os

import numpy as np
import pkg_resources
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data import load_norm_matrices


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, normalize, target_shape):
        self.data_path = self.read_paths(txt_file)
        self.normalize = normalize
        self.target_shape = (target_shape, target_shape)

        self.normalization_tactics = {
            "length": self.normalize_by_length,
            "bond_length": self.normalize_by_normalization_matrix,
            "afrc": self.normalize_by_normalization_matrix,
            "log": self.normalize_by_log10,
            "normalize_and_scale": self.normalize_and_scale,
            "mean": self.normalize_by_mean_gSAW,
        }

        if self.normalize is not None:
            if self.normalize == "bond_length":
                self.normalization_matrix = self.generate_normalization_matrix(
                    self.target_shape
                )
            elif self.normalize == "afrc":
                self.normalization_matrix = self.generate_afrc_distance_map(
                    self.target_shape
                )
            elif self.normalize == "normalize_and_scale":
                directory_path = pkg_resources.resource_filename("starling", "data/")
                self.mean_dataset = np.load(directory_path + "mean_384_dataset.npy")
                self.std_dataset = np.load(directory_path + "std_384_dataset.npy")
                self.max_standard = np.load(
                    directory_path + "max_standard_384_dataset.npy"
                )
                self.min_standard = np.load(
                    directory_path + "min_standard_384_dataset.npy"
                )
            elif self.normalize == "mean":
                directory_path = pkg_resources.resource_filename("starling", "data/")
                self.mean_dataset = np.load(directory_path + "mean_384_dataset.npy")

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        # Get a single data sample
        sample = np.loadtxt(self.data_path[index], dtype=np.float32)

        # Normalize your distance map according to user input
        if self.normalize is not None:
            sample = self.normalization_tactics[self.normalize](sample)

        # Resize the input distance map with padding
        sample = self.MaxPad(sample)

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        return {"input": sample}

    def read_paths(self, txt_file):
        paths = []
        with open(txt_file, "r") as file:
            for line in file:
                path = line.strip()
                paths.append(path)
        return paths

    def MaxPad(self, original_array):
        # Pad the distance map to a desired shape
        pad_height = max(0, self.target_shape[0] - original_array.shape[0])
        pad_width = max(0, self.target_shape[1] - original_array.shape[1])
        return np.pad(
            original_array,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )

    def normalize_by_length(self, original_array):
        # Normalize the distances by square root of N where N is num_residues
        # Does not seem to work super well because the distances that should
        # stay constant start to depend on N (i.e., bond length)
        normalized_distance_map = original_array / np.sqrt(len(original_array))
        return normalized_distance_map

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

    def normalize_by_log10(self, original_array):
        np.fill_diagonal(original_array, 1)
        normalized_matrix = np.log10(original_array)
        return normalized_matrix

    def reciprocal_log(self, data):
        data_recip = 1 / (np.log(data))
        data_recip[np.isinf(data_recip)] = 0
        return data_recip

    def calculate_min_max_reference(self, array):
        self.max_array = np.maximum(array, self.max_array)
        array[array == 0] = np.inf
        self.min_array = np.minimum(array, self.min_array)

    def min_max_scale(self, data):
        denominator = self.max_array - self.min_array
        denominator[denominator == 0] = 1
        normalized_data = (data - self.min_array) / denominator
        return normalized_data

    def reciprocal_log_normalization(self, original_array):
        reciprocal_log_data = self.reciprocal_log(original_array)
        self.calculate_min_max_reference(reciprocal_log_data)
        normalized_data = self.min_max_scale(reciprocal_log_data)
        return normalized_data

    def normalize_and_scale(self, data):
        self.std_dataset[self.std_dataset.round(3) == 0] = 1
        # Zero mean the data and standardize std
        data_standard = (data - self.mean_dataset) / self.std_dataset

        denominator = self.max_standard - self.min_standard
        denominator[denominator.round(3) == 0] = 1

        data_normal = (data_standard - self.min_standard) / denominator
        np.fill_diagonal(data_normal, 0)
        return data_normal.astype(np.float32)

    def normalize_by_mean_gSAW(self, data):
        self.mean_dataset[self.mean_dataset == 0] = 1
        data_norm = data / self.mean_dataset
        np.fill_diagonal(data_norm, 0)
        return data_norm.astype(np.float32)


# Step 2: Create a data module
class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=None,
        normalize=None,
        target_shape=None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.normalize = normalize
        self.target_shape = target_shape
        self.num_workers = int(os.cpu_count() / 4)

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(
                self.train_data,
                normalize=self.normalize,
                target_shape=self.target_shape,
            )
            self.val_dataset = MatrixDataset(
                self.val_data,
                normalize=self.normalize,
                target_shape=self.target_shape,
            )
        if stage == "test":
            self.test_dataset = MatrixDataset(
                self.test_data,
                normalize=self.normalize,
                target_shape=self.target_shape,
            )
        if stage == "predict":
            self.predict_dataset = MatrixDataset(
                self.predict_data,
                normalize=self.normalize,
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
