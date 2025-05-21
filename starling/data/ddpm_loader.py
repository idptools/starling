from collections import namedtuple

import hdf5plugin
import numpy as np
import pytorch_lightning as pl
import torch

from starling.data.data_wrangler import load_hdf5_compressed, read_tsv_file


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


# Define a named tuple at the module level
BatchOutput = namedtuple(
    "BatchOutput", ["latent_distance_maps", "sequences", "masks", "ionic_strength"]
)


def collate_batch_with_padding(batch):
    # Separate the distance maps, sequences, and ionic strengths
    distance_maps, sequences, ionic_strengths = zip(*batch)

    # Find the maximum sequence length in this batch
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)

    # Pre-allocate tensors for the entire batch
    padded_sequences = torch.zeros(batch_size, max_len, dtype=sequences[0].dtype)
    sequence_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Fill pre-allocated tensors
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_sequences[i, :seq_len] = seq
        sequence_masks[i, :seq_len] = 1

    # Stack distance maps
    distance_maps = torch.stack(distance_maps, dim=0)

    # Ionic strengths
    ionic_strengths = torch.stack(ionic_strengths).unsqueeze(1)

    # Return a named tuple for better type safety and readability
    return BatchOutput(
        latent_distance_maps=distance_maps,
        sequences=padded_sequences,
        masks=sequence_masks,
        ionic_strength=ionic_strengths,
    )


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file: str) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch
        tsv_file : str
            A path to a tsv file containing the paths to distance maps as a first column
            and index of a distance map to load as a second column
        labels : str
            Which labels to use for the dataset, learnable or fixed (finches interaction matrix).
        """
        self.data = read_tsv_file(tsv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, frame = self.data.iloc[index]
        data = load_hdf5_compressed(
            data_path,
            keys_to_load=["latents", "seq", "ionic_strength"],
            frame=int(frame),
        )
        distance_map = data["latents"]
        distance_map = torch.from_numpy(distance_map).unsqueeze(0)

        sequence = data["seq"].astype(np.int32)
        remove_padded = sequence != 0
        sequence = sequence[remove_padded]
        sequence = torch.from_numpy(sequence)

        ionic_strength = torch.tensor(data["ionic_strength"], dtype=torch.float32)

        return distance_map, sequence, ionic_strength


# Step 2: Create a data module
class DDPMDataloader(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_data = config.h5.train
        self.val_data = config.h5.validation
        self.test_data = config.h5.test
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(self.train_data)
            self.val_dataset = MatrixDataset(self.val_data)
        if stage == "test":
            self.test_dataset = MatrixDataset(self.test_data)
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
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_batch_with_padding,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=collate_batch_with_padding,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_batch_with_padding,
        )
