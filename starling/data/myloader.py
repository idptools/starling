import os

import numpy as np
import pytorch_lightning as pl
import torch
from finches.epsilon_calculation import Interaction_Matrix_Constructor
from finches.forcefields.mPiPi import mPiPi_model
from finches.frontend.mpipi_frontend import Mpipi_frontend
from IPython import embed

from starling.data.data_wrangler import (
    MaxPad,
    load_hdf5_compressed,
    one_hot_encode,
    read_tsv_file,
    symmetrize,
)


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file: str, target_shape: int, finches_conditioning) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch
        tsv_file : str
            A path to a tsv file containing the paths to distance maps as a first column
            and index of a distance map to load as a second column
        target_shape : int
            The desired shape of the distance maps. The distance map will be
            padded
        """
        self.data = read_tsv_file(tsv_file)
        self.target_shape = (target_shape, target_shape)
        self.finches_conditioning = finches_conditioning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # data_path, frame = self.data[index]
        data_path, frame = self.data.iloc[index]
        sample, sequence = load_hdf5_compressed(
            data_path, keys_to_load=["dm", "seq"], frame=int(frame)
        )
        sample = symmetrize(sample).astype(np.float32)

        # Resize the input distance map with padding
        sample = MaxPad(sample, shape=(self.target_shape))

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        if self.finches_conditioning:
            # Should I make the diagonal 0? - this would be as positional encodings for the model
            labels = self.get_interaction_matrix(sequence)
            labels = MaxPad(labels, shape=(self.target_shape))
            labels = torch.from_numpy(labels).to(torch.float32)
        else:
            labels = None

        # sequence = torch.argmax(
        #     torch.from_numpy(one_hot_encode(sequence.ljust(384, "0"))), dim=-1
        # ).to(torch.int64)

        # Memory leak discussion
        # https://github.com/pytorch/pytorch/issues/13246

        # return {
        #     "data": sample,
        #     "sequence": sequence,
        #     "length": torch.tensor([len(sequence) - 1]),
        #     "labels": labels,
        # }

        return sample, sequence, labels

    def get_interaction_matrix(self, sequence):
        mf = Mpipi_frontend()
        interaction_matrix = mf.intermolecular_idr_matrix(
            sequence, sequence, window_size=1
        )
        return interaction_matrix[0][0]

    def epsilon_vector(self, sequence):
        # Initialize a finches.forcefields.Mpipi.Mpipi_model object
        Mpipi_GGv1_params = mPiPi_model(version="mPiPi_GGv1")

        # initialize an InteractionMatrixConstructor
        IMC = Interaction_Matrix_Constructor(parameters=Mpipi_GGv1_params)

        attractive, repulsive = IMC.calculate_epsilon_vectors(sequence, sequence)
        # Pad with zeros up to length 384
        attractive = np.pad(attractive, (0, 384 - len(attractive)), "constant")
        repulsive = np.pad(repulsive, (0, 384 - len(repulsive)), "constant")

        attractive = np.array(attractive, dtype=np.float32)
        repulsive = np.array(repulsive, dtype=np.float32)

        # Concatenate the padded arrays
        epsilon_vector = np.concatenate((attractive, repulsive))

        return epsilon_vector


# Step 2: Create a data module
class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=None,
        target_shape=None,
        finches_conditioning=False,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.target_shape = target_shape
        # self.num_workers = int(os.cpu_count() / 4)
        self.num_workers = 16
        self.finches_conditioning = finches_conditioning

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(
                self.train_data,
                target_shape=self.target_shape,
                finches_conditioning=self.finches_conditioning,
            )
            self.val_dataset = MatrixDataset(
                self.val_data,
                target_shape=self.target_shape,
                finches_conditioning=self.finches_conditioning,
            )
        if stage == "test":
            self.test_dataset = MatrixDataset(
                self.test_data,
                target_shape=self.target_shape,
                finches_conditioning=self.finches_conditioning,
            )
        if stage == "predict":
            self.predict_dataset = MatrixDataset(
                self.predict_data,
                target_shape=self.target_shape,
                finches_conditioning=self.finches_conditioning,
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
