import glob
import io
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import webdataset as wds
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def npy_decoder(key, data):
    return np.load(io.BytesIO(data), allow_pickle=False)


class DDPMDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        config,
        effective_batch_size=None,
    ):
        super().__init__()
        self.config = config
        self.dataset_dir = self.config.dataset
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.prefetch_factor = getattr(self.config, "prefetch_factor", 2)
        self.shuffle_buffer = getattr(self.config, "shuffle_buffer", 10000)

        self.ionic_strength = getattr(self.config, "ionic_strength", 150)

        # Calculate number of batches (can be replaced with metadata file reading)
        train_size = getattr(self.config, "train_size", 1_000_000)
        val_size = getattr(self.config, "val_size", 100_000)
        self.effective_batch_size = effective_batch_size or self.batch_size
        self.n_train_batches = int(train_size) // int(self.effective_batch_size)
        self.n_val_batches = int(val_size) // int(self.effective_batch_size)

    def setup(self, stage=None):
        """Create dataset objects for training and validation"""
        # Find tar files
        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "validation")

        # Include .tar.zst, .tar.gz
        train_tar_files = sorted(glob.glob(f"{train_dir}/*.tar*"))
        val_tar_files = sorted(glob.glob(f"{val_dir}/*.tar*"))

        # Error checking
        if not train_tar_files:
            raise FileNotFoundError(f"No train tar files found in {train_dir}")
        if not val_tar_files:
            raise FileNotFoundError(f"No validation tar files found in {val_dir}")

        # Create datasets - simpler pipeline with clear stages
        self.train_dataset = self._create_dataset(
            tar_files=train_tar_files, is_training=True
        )

        self.val_dataset = self._create_dataset(
            tar_files=val_tar_files, is_training=False
        )

    def _create_dataset(
        self,
        tar_files: List[str],
        is_training: bool = False,
    ):
        """Create a WebDataset with appropriate processing"""
        # Configure the options based on training or validation
        dataset = wds.WebDataset(
            tar_files,
            nodesplitter=wds.split_by_node,
            resampled=is_training,  # Only enable infinite iteration for training
            shardshuffle=True,
        )

        # Start the processing pipeline
        pipeline = dataset.shuffle(self.shuffle_buffer).decode(self._npz_decoder)

        pipeline = pipeline.map(self._apply_filter_map)

        # Complete the processing pipeline
        return pipeline.map(self._process_sample).batched(
            self.batch_size, partial=not is_training
        )

    def _apply_filter_map(self, sample):
        """Map function that applies filtering by returning None for filtered samples"""
        if self._filter_sample(sample):
            return sample
        else:
            return None

    def _filter_sample(self, sample):
        """Filter samples based on custom training criteria"""

        # Example filtering based on distance map properties
        ionic_strength = sample["ionic_strength_mm.npz"]

        if ionic_strength == self.ionic_strength:
            return True
        else:
            return False

    def _npz_decoder(self, key, data):
        """Decoder for NPZ files with error handling"""
        try:
            npz_data = np.load(io.BytesIO(data), allow_pickle=False)
            return npz_data["array"]
        except Exception as e:
            print(f"Error decoding {key}: {e}")
            return None

    def _process_sample(self, sample):
        """Process a single sample"""
        # Extract the distance map and sequence
        latents = sample["latent.npz"]
        sequence = sample["sequence.npz"]

        # Add channel dimension if needed
        if len(latents.shape) == 2:
            latents = latents[np.newaxis, :, :]

        if len(sequence.shape) == 1:
            sequence = sequence[np.newaxis, :]

        # Return a tuple with all data components
        return (latents, sequence)

    def _collate_fn(self, batch):
        """Collate function that handles empty batches"""
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        # Stack latent vectors
        latents = np.stack([item for item in batch[0]])

        # Stack sequences
        sequences = np.stack([item[0] for item in batch[1]])

        # Get the maximum sequence length
        max_seq_length = (sequences != 0).sum(axis=1).max()

        # Remove extraneous padding elements
        sequences = sequences[:, :max_seq_length]

        attention_mask = sequences != 0

        # Convert to torch tensors (on CPU to avoid slowdown)
        latents = torch.from_numpy(latents)
        sequences = torch.from_numpy(sequences).to(torch.int32)
        attention_mask = torch.from_numpy(attention_mask)

        # Return as dictionary for clearer access in training loop
        return {
            "latent": latents,
            "sequence": sequences,
            "attention_mask": attention_mask,
        }

    def train_dataloader(self):
        return (
            wds.WebLoader(
                self.train_dataset,
                batch_size=None,  # Already batched
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
                collate_fn=self._collate_fn,
            )
            .with_epoch(self.n_train_batches)
            .with_length(self.n_train_batches)
        )

    def val_dataloader(self):
        return (
            wds.WebLoader(
                self.val_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
                collate_fn=self._collate_fn,
            )
            .with_epoch(self.n_val_batches)
            .with_length(self.n_val_batches)
        )


if __name__ == "__main__":
    config_path = os.path.join("..", "configs", "dataloader", "dataloader.yaml")
    cfg = OmegaConf.load(config_path)

    effective_batch_size = 1 * 4 * 16

    data_module = DDPMDataLoader(cfg.tar, effective_batch_size)

    # Initialize the datasets
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in tqdm(train_loader):
        # import pdb

        # pdb.set_trace()
        pass
