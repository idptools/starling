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


class VAEdataloader(pl.LightningDataModule):
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
        self.apply_filter = getattr(self.config, "apply_filter", True)
        if self.apply_filter:
            self.filter_data = pd.read_csv(
                os.path.join(self.dataset_dir, "acceptance_probs.csv")
            )
            self.bin_starts = self.filter_data["bin_start"].values
            self.bin_ends = self.filter_data["bin_end"].values
            self.accept_probs = self.filter_data["accept_prob"].values

            self.accept_prob_table = np.zeros(384 + 1)
            for start, end, prob in zip(
                self.bin_starts, self.bin_ends, self.accept_probs
            ):
                self.accept_prob_table[start + 1 : end + 1] = prob

        # Calculate number of batches (can be replaced with metadata file reading)
        train_size = getattr(self.config, "train_size", 1_000_000)
        val_size = getattr(self.config, "val_size", 100_000)
        self.effective_batch_size = effective_batch_size or self.batch_size
        self.n_train_batches = int(train_size) // int(self.effective_batch_size)
        self.n_val_batches = int(val_size) // int(self.effective_batch_size)

    def __get_accept_prob(self, seq_length):
        return self.accept_prob_table[seq_length]

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
            tar_files=train_tar_files, is_training=True, apply_filter=self.apply_filter
        )

        self.val_dataset = self._create_dataset(
            tar_files=val_tar_files, is_training=False, apply_filter=False
        )

    def _create_dataset(
        self,
        tar_files: List[str],
        is_training: bool = False,
        apply_filter: bool = False,
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

        # Apply filter only if requested (for training)
        if apply_filter:
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
        # Your filtering criteria here
        if sample is None or "distance_map.npz" not in sample:
            return False

        # Example filtering based on distance map properties
        sequence_length = (sample["sequence.npz"] != 0).sum()

        if np.random.random() < self.__get_accept_prob(sequence_length):
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
        if sample is None or "distance_map.npz" not in sample:
            return None  # Skip bad samples

        # Extract the distance map
        distance_map = sample["distance_map.npz"]

        # Add channel dimension if needed
        if len(distance_map.shape) == 2:
            distance_map = distance_map[np.newaxis, :, :]

        return (distance_map,)

    def _collate_fn(self, batch):
        """Collate function that handles empty batches"""
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        # Stack distance maps
        distance_maps = np.stack([item for item in batch[0]])

        # Convert to torch tensor and desired data type (on CPU to avoid slowdown)
        distance_maps = torch.from_numpy(distance_maps)

        # BFloat16 conversion can happen on GPU during training
        return distance_maps

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
    from collections import Counter

    import matplotlib.pyplot as plt

    config_path = os.path.join("..", "configs", "dataloader", "vae_dataloader.yaml")
    cfg = OmegaConf.load(config_path)

    effective_batch_size = 1 * 4 * 16

    data_module = VAEdataloader(cfg.tar, effective_batch_size)

    # Initialize the datasets
    data_module.setup()
    train_loader = data_module.train_dataloader()

    size_counter = Counter()
    total = 0

    for i, batch in enumerate(tqdm(train_loader)):
        sizes = (batch[:, 0, 0, :] != 0).sum(dim=-1) + 1
        for s in sizes.tolist():
            size_counter[s] += 1
            total += 1

        # if (i + 1) % 100 == 0:
        # print(f"\nFinal size distribution after {i + 1} batches:")
        # for size, count in sorted(size_counter.items()):
        #     print(f"Size {size}: {count}/{total} ({count / total:.4f})")

        if (i + 1) % 100 == 0:
            # Plot and save bar plot
            plt.figure(figsize=(10, 5))
            sorted_items = sorted(size_counter.items())
            x = [size for size, _ in sorted_items]
            y = [count / total for _, count in sorted_items]
            plt.bar(x, y, color="skyblue")
            plt.xlabel("Size")
            plt.ylabel("Fraction of Total")
            plt.title(f"Size Distribution after {i + 1} Batches")
            plt.tight_layout()
            plt.savefig("size_distribution.png")
            plt.close()

    # for batch in tqdm(train_loader):
    #     sizes = (batch[:, 0, 0, :] != 0).sum(dim=-1) + 1
    #     import pdb

    #     pdb.set_trace()  # Debugging breakpoint
    #     pass
