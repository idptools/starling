import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data.data_wrangler import load_hdf5_compressed, read_tsv_file


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file: str) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch
        tsv_file : str
            A path to a tsv file containing the paths to distance maps as a first column
            and index of a distance map to load as a second column
        """
        self.data = read_tsv_file(tsv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, frame = self.data.iloc[index]

        distance_map = load_hdf5_compressed(
            data_path, keys_to_load=["dm"], frame=int(frame)
        )["dm"]

        # Add a channel dimension using unsqueeze
        distance_map = torch.from_numpy(distance_map).unsqueeze(0)

        return distance_map


# Step 2: Create a data module
class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=None,
        num_workers=None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers  # Set this to 16

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(
                self.train_data,
            )
            self.val_dataset = MatrixDataset(
                self.val_data,
            )
        if stage == "test":
            self.test_dataset = MatrixDataset(
                self.test_data,
            )
        if stage == "predict":
            self.predict_dataset = MatrixDataset(
                self.predict_data,
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
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=5,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=5,
            pin_memory=True,
        )
