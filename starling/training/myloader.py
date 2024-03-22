import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data import load_norm_matrices


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, args):
        self.args = args
        self.data_path = self.read_paths(txt_file)

        # Hard coded in so that the dimensions during encoding and decoding match
        # self.target_shape = (768, 768)
        self.target_shape = (192, 192)

        self.normalization_tactics = {
            "length": self.normalize_by_length,
            "bond_length": self.normalize_by_normalization_matrix,
            "afrc": self.normalize_by_normalization_matrix,
            "log": self.normalize_by_log10,
            "normalize_and_scale": self.normalize_and_scale,
        }

        self.resizing_tactics = {"pad": self.MaxPad, "interpolate": self.interpolate}

        if args.normalize == "bond_length":
            self.normalization_matrix = self.generate_normalization_matrix(
                self.target_shape
            )
        elif args.normalize == "afrc":
            self.normalization_matrix = self.generate_afrc_distance_map(
                self.target_shape
            )
        elif args.normalize == "normalize_and_scale":
            (
                self.mean_matrix,
                self.std_matrix,
                self.max_standard,
                self.min_standard,
                self.max_expected_distances,
            ) = load_norm_matrices.load_matrices()

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        # Get a single data sample
        sample = np.loadtxt(self.data_path[index], dtype=np.float32)

        # Normalize your distance map according to user input
        if self.args.normalize is not None:
            sample = self.normalization_tactics[self.args.normalize](sample)

        # Resize the input distance map with padding or resizing
        tactic = "interpolate" if self.args.interpolate else "pad"
        sample = self.resizing_tactics[tactic](sample)

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        # embed()

        return {"input": sample}

    def read_paths(self, txt_file):
        paths = []
        with open(txt_file, "r") as file:
            for line in file:
                path = line.strip()
                paths.append(path)
        return paths

    def MaxPad(self, original_array):
        # Pad the distance map to a desired shape, here we are using
        # (768, 768) because largest sequences are 750 residues long
        # and 768 can be divided by 2 a bunch of times leading to nice
        # behavior during conv2d and conv2transpose down- and up-sampling
        pad_height = max(0, self.target_shape[0] - original_array.shape[0])
        pad_width = max(0, self.target_shape[1] - original_array.shape[1])
        return np.pad(
            original_array,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )

    def interpolate(self, distance_map):
        # This BICUBIC method was tested to perform the best
        # on distance maps (smallest error, max error ~0.07A)
        method = "bicubic"

        # Set up the distance map shapes to be right for transform
        # needed [B, C, H, W]
        distance_map_upsample = torch.from_numpy(distance_map).unsqueeze(0).unsqueeze(0)

        # Transform the data
        distance_map_upsample = torch.nn.functional.interpolate(
            distance_map_upsample, size=self.target_shape, mode=method
        )
        # Convert back to an array with shape (H, W)
        distance_map_upsample = distance_map_upsample.squeeze().numpy()
        np.fill_diagonal(distance_map_upsample, 0)
        distance_map_upsample = distance_map_upsample.clip(
            min=0, max=(np.max(distance_map))
        )

        return distance_map_upsample

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
        # embed()

        return distance_map

    def normalize_by_log10(self, original_array):
        #! We should change this by filling the diagonal with 1 which will after log10 equal to
        normalized_matrix = np.log10(original_array + 0.005)
        return normalized_matrix

    def normalize_and_scale(self, original_array):
        height, width = original_array.shape
        standardized_data = (
            original_array - self.mean_matrix[:height, :width]
        ) / self.std_matrix[:height, :width]

        denominator = (
            self.max_standard[:height, :width] - self.min_standard[:height, :width]
        )
        denominator[denominator == 0] = 1

        scaled_data = (
            2 * (standardized_data - self.min_standard[:height, :width]) / (denominator)
            - 1
        )
        # scaled_data = original_array / (
        #     self.max_expected_distances[:height, :width] + 1e-5
        # )

        return scaled_data


# Step 2: Create a data module
class LightningModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size, args):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.args = args

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.train_data, self.args)
        self.val_dataset = MyDataset(self.val_data, self.args)
        self.test_dataset = MyDataset(self.test_data, self.args)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size
        )
