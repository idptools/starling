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
    read_tsv_file,
    symmetrize,
)


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(
        self, tsv_file: str, target_shape: int, pretraining: bool = False
    ) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch

        Parameters
        ----------
        tsv_file : str
            A path to a tsv file containing the paths to distance maps as a first column
            and index of a distance map to load as a second column
        target_shape : int
            The desired shape of the distance maps. The distance map will be
            padded
        """
        self.data = read_tsv_file(tsv_file)
        self.target_shape = (target_shape, target_shape)
        self.pretraining = pretraining

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get a single data sample
        try:
            sample = np.loadtxt(self.data[index], dtype=np.float32)
        except Exception as e:
            data_path, frame = self.data[index]
            data = load_hdf5_compressed(
                data_path, keys_to_load=["dm", "seq"], frame=int(frame)
            )
            sample = symmetrize(data["dm"]).astype(np.float32)
            sequence = data["seq"][()].decode()

        if not self.pretraining:
            encoder_condition = self.get_interaction_matrix(sequence)
            encoder_condition = symmetrize(encoder_condition)
            encoder_condition = MaxPad(encoder_condition, shape=(self.target_shape))

            encoder_condition = torch.from_numpy(
                encoder_condition.astype(np.float32)
            ).unsqueeze(0)
        else:
            encoder_condition = {}
            sequence = {}

        # Resize the input distance map with padding
        sample = MaxPad(sample, shape=(self.target_shape))

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        return {
            "data": sample,
            "encoder_condition": encoder_condition,
            "decoder_condition": sequence,
            # "decoder_condition": torch.from_numpy(self.epsilon_vector(sequence)),
            "length": torch.tensor([len(sequence) - 1]),
        }

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
        pretraining=False,
    ):
        super().__init__()
        self.train_data = train_data
        self.pretraining = pretraining
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
                pretraining=self.pretraining,
            )
            self.val_dataset = MatrixDataset(
                self.val_data,
                target_shape=self.target_shape,
                pretraining=self.pretraining,
            )
        if stage == "test":
            self.test_dataset = MatrixDataset(
                self.test_data,
                target_shape=self.target_shape,
                pretraining=self.pretraining,
            )
        if stage == "predict":
            self.predict_dataset = MatrixDataset(
                self.predict_data,
                target_shape=self.target_shape,
                pretraining=self.pretraining,
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
