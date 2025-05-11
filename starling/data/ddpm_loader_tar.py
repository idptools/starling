import glob
import io
import os

import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from omegaconf import OmegaConf
from tqdm import tqdm


def npy_decoder(key, data):
    return np.load(io.BytesIO(data), allow_pickle=False)


class DDPMDataloader(pl.LightningDataModule):
    def __init__(
        self,
        config,
        effective_batch_size=None,
    ):
        super().__init__()

        self.config = config

        self.dataset = self.config.dataset
        self.early_batch_size = self.config.early_batch_size
        self.train_shuffle_buffer = 4000
        self.val_shuffle_buffer = 4000
        self.loader_shuffle_buffer = 4000
        self.batch_size = self.config.batch_size

        # Calculate batch counts
        self.effective_batch_size = effective_batch_size
        self.n_train_batches = 20_000_000 // self.effective_batch_size
        self.n_val_batches = 4_000_000 // self.effective_batch_size

    def setup(self, stage=None):
        # Create the WebDataset instances for training and validation
        train_dir = os.path.join(self.dataset, "train")
        val_dir = os.path.join(self.dataset, "validation")

        train_tar_files = sorted(glob.glob(f"{train_dir}/*.tar"))
        val_tar_files = sorted(glob.glob(f"{val_dir}/*.tar"))

        # Check if we found any files
        if not train_tar_files:
            raise FileNotFoundError(f"No train tar files found in {train_dir}")
        if not val_tar_files:
            raise FileNotFoundError(f"No validation tar files found in {val_dir}")

        self.train_dataset = (
            wds.WebDataset(
                train_tar_files,
                nodesplitter=wds.split_by_node,  # Distributes tar files across nodes
                resampled=True,  # Enables infinite iteration through the dataset
                shardshuffle=True,
            )
            .shuffle(4000)  # Creates and constantly maintains a buffer of 4000 samples
            .decode(npy_decoder)
            .to_tuple("sequence.npy", "distance_map.npy")
            .batched(self.early_batch_size, self.early_collate)
        )
        self.val_dataset = (
            wds.WebDataset(
                val_tar_files,
                nodesplitter=wds.split_by_node,
                shardshuffle=True,
            )
            .shuffle(4000)
            .decode(npy_decoder)
            .to_tuple("sequence.npy", "distance_map.npy")
            .batched(self.early_batch_size, self.early_collate)
        )

    def train_dataloader(self):
        wds_loader = (
            wds.WebLoader(
                self.train_dataset,
                batch_size=None,
                num_workers=self.config.num_workers,
                pin_memory=True,
                prefetch_factor=2,
            )
            .unbatched()
            .shuffle(4000)
            .with_epoch(self.n_train_batches)  # sets "epoch" size in iterable dataset
            .batched(self.batch_size)
            .with_length(self.n_train_batches)  # defines __len__ for the dataset
        )
        return wds_loader

    def val_dataloader(self):
        wds_loader = (
            wds.WebLoader(
                self.val_dataset,
                batch_size=None,
                num_workers=self.config.num_workers,
                pin_memory=True,
                prefetch_factor=2,
            )
            .unbatched()
            .shuffle(4000)
            .with_epoch(self.n_val_batches)
            .batched(self.batch_size, partial=False)
            .with_length(self.n_val_batches)
        )
        return wds_loader

    def early_collate(self, sample):
        """Optimized collation that uses numpy operations for speed"""
        if not sample:
            return [], []

        # Use numpy's stack for efficient memory usage
        sequences = np.stack([data[0] for data in sample], axis=0, dtype=np.int32)
        distance_maps = np.stack([data[1] for data in sample], axis=0)

        distance_maps = distance_maps[:, np.newaxis, :, :]
        distance_maps = torch.from_numpy(distance_maps).to(torch.bfloat16)

        return distance_maps, sequences


def check_shapes(input_ids, attention_mask, labels):
    assert input_ids.shape[0] == attention_mask.shape[0] == labels.shape[0], (
        "Batch size mismatch"
    )
    assert input_ids.shape[1] == attention_mask.shape[1] == labels.shape[1], (
        "Sequence length mismatch"
    )


if __name__ == "__main__":
    config_path = os.path.join("..", "configs", "dataloader", "dataloader.yaml")
    cfg = OmegaConf.load(config_path)

    effective_batch_size = 1 * 4 * 16

    data_module = DDPMDataloader(cfg, effective_batch_size)

    # Initialize the datasets
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in tqdm(train_loader):
        # import pdb

        # pdb.set_trace()
        pass
