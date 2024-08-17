import os

import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data.data_wrangler import (
    MaxPad,
    load_hdf5_compressed,
    one_hot_encode,
    read_tsv_file,
    symmetrize,
)


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file: str, target_shape: int, labels: str) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch
        tsv_file : str
            A path to a tsv file containing the paths to distance maps as a first column
            and index of a distance map to load as a second column
        target_shape : int
            The desired shape of the distance maps. The distance map will be
            padded
        labels : str
        """
        self.data = read_tsv_file(tsv_file)
        self.target_shape = (target_shape, target_shape)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, frame = self.data.iloc[index]

        data = load_hdf5_compressed(
            data_path, keys_to_load=["dm", "seq"], frame=int(frame)
        )

        sample = symmetrize(data["dm"]).astype(np.float32)

        # Resize the input distance map with padding
        sample = MaxPad(sample, shape=(self.target_shape))

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        #if self.labels == "finches":
        #    sequence = self.get_interaction_matrix(data["seq"][()].decode())
        #    sequence = MaxPad(sequence, shape=(self.target_shape))
        #    sequence = torch.from_numpy(sequence).to(torch.float32)

        if self.labels == "learned-embeddings":
            sequence = (
                torch.argmax(
                    torch.from_numpy(
                        one_hot_encode(data["seq"][()].decode().ljust(384, "0"))
                    ),
                    dim=-1,
                )
                .to(torch.int64)
                .squeeze()
            )

        return sample, sequence

    #def get_interaction_matrix(self, sequence):
    #    mf = Mpipi_frontend()
    #    interaction_matrix = mf.intermolecular_idr_matrix(
    #        sequence, sequence, window_size=1
    #    )
    #    return interaction_matrix[0][0]


# Step 2: Create a data module
class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=None,
        target_shape=None,
        labels=None,
        num_workers=None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.labels = labels
        self.num_workers = num_workers

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(
                self.train_data,
                target_shape=self.target_shape,
                labels=self.labels,
            )
            self.val_dataset = MatrixDataset(
                self.val_data,
                target_shape=self.target_shape,
                labels=self.labels,
            )
        if stage == "test":
            self.test_dataset = MatrixDataset(
                self.test_data,
                target_shape=self.target_shape,
                labels=self.labels,
            )
        if stage == "predict":
            self.predict_dataset = MatrixDataset(
                self.predict_data,
                target_shape=self.target_shape,
                labels=self.labels,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
