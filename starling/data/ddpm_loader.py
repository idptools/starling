import hdf5plugin
import numpy as np
import pytorch_lightning as pl
import torch

from starling.data.data_wrangler import (
    load_hdf5_compressed,
    read_tsv_file,
)


def sequence_to_indices(sequence, aa_to_int):
    """
    Converts a single sequence to integer indices based on a mapping.

    Parameters:
    - sequence (str): A single sequence string.
    - aa_to_int (dict): A dictionary mapping amino acids to integers.

    Returns:
    - torch.Tensor: Tensor of integer indices.
    """
    int_sequence = [aa_to_int[aa] for aa in sequence]
    return torch.tensor(int_sequence, dtype=torch.int64)


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file: str, labels: str) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch
        tsv_file : str
            A path to a tsv file containing the paths to distance maps as a first column
            and index of a distance map to load as a second column
        labels : str
            Which labels to use for the dataset, learnable or fixed (finches interaction matrix).
        """
        self.data = read_tsv_file(tsv_file)
        self.labels = labels
        self.aa_to_int = {
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, frame = self.data.iloc[index]
        data = load_hdf5_compressed(
            data_path, keys_to_load=["dm", "seq"], frame=int(frame)
        )
        distance_map = data["dm"].astype(np.float16)
        distance_map = torch.from_numpy(distance_map).unsqueeze(0)

        sequence = data["seq"].astype(np.int32)
        sequence = torch.from_numpy(sequence)

        return distance_map, sequence


# Step 2: Create a data module
class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=None,
        labels=None,
        num_workers=None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.labels = labels
        # self.num_workers = int(os.cpu_count() / 4)
        self.num_workers = num_workers

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(
                self.train_data,
                labels=self.labels,
            )
            self.val_dataset = MatrixDataset(
                self.val_data,
                labels=self.labels,
            )
        if stage == "test":
            self.test_dataset = MatrixDataset(
                self.test_data,
                labels=self.labels,
            )
        if stage == "predict":
            self.predict_dataset = MatrixDataset(
                self.predict_data,
                labels=self.labels,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=5,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=5,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
