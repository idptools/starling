import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data.data_wrangler import (
    MaxPad,
    load_hdf5_compressed,
    read_tsv_file,
    symmetrize,
)


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file: str, target_shape: int) -> None:
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, frame = self.data.iloc[index]

        sample = load_hdf5_compressed(data_path, keys_to_load=["dm"], frame=int(frame))

        sample = symmetrize(sample["dm"]).astype(np.float32)

        # Resize the input distance map with padding
        sample = MaxPad(sample, shape=(self.target_shape))

        # Add a channel dimension using unsqueeze
        sample = torch.from_numpy(sample).unsqueeze(0)

        return sample


# Step 2: Create a data module
class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=None,
        target_shape=None,
        num_workers=None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.num_workers = num_workers  # Set this to 16

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
