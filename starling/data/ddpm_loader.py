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


def collate_batch_with_padding(batch):
    # Separate the distance maps and sequences
    distance_maps, sequences, interaction_matrices = zip(*batch)

    # Find the maximum sequence length in this batch
    max_len = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)

    # Pre-allocate tensors for the entire batch
    padded_sequences = torch.zeros(batch_size, max_len, dtype=sequences[0].dtype)
    sequence_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padded_IMC = torch.zeros(batch_size, max_len, 384, dtype=torch.float32)

    # Fill pre-allocated tensors
    for i, (seq, matrix) in enumerate(zip(sequences, interaction_matrices)):
        seq_len = seq.size(0)
        padded_sequences[i, :seq_len] = seq
        sequence_masks[i, :seq_len] = 1

        # Convert matrix to tensor once outside the indexing operation
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32)
        padded_IMC[i, :seq_len, :seq_len] = matrix_tensor

    # Stack distance maps
    distance_maps = torch.stack(distance_maps, dim=0)

    return distance_maps, padded_sequences, padded_IMC, sequence_masks


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file: str, sequence_context) -> None:
        """
        A class that creates a dataset of distance maps compatible with PyTorch
        tsv_file : str
            A path to a tsv file containing the paths to distance maps as a first column
            and index of a distance map to load as a second column
        labels : str
            Which labels to use for the dataset, learnable or fixed (finches interaction matrix).
        """
        self.data = read_tsv_file(tsv_file)
        self.sequence_context = sequence_context

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index: Index of the item to retrieve

        Returns:
            tuple: (distance_map, sequence, interaction_matrix)
                - distance_map: Tensor of shape [1, height, width]
                - sequence: Tensor of integer indices representing amino acids
                - interaction_matrix: Normalized interaction matrix
        """
        # 1. Extract data path and frame number from dataframe
        data_path, frame = self.data.iloc[index]

        data = load_hdf5_compressed(
            data_path,
            keys_to_load=["latents", "seq", self.sequence_context],
            frame=int(frame),
        )

        # Process data more efficiently
        distance_map = torch.from_numpy(data["latents"]).unsqueeze(0)

        # Process sequence in one step
        sequence = data["seq"]
        valid_indices = sequence != 0
        sequence = torch.from_numpy(sequence[valid_indices].astype(np.int32))

        interaction_matrix = data[self.sequence_context]

        return distance_map, sequence, interaction_matrix


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
        self.sequence_context = config.sequence_context

    def prepare_data(self):
        # Implement any data download or preprocessing here
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MatrixDataset(self.train_data, self.sequence_context)
            self.val_dataset = MatrixDataset(self.val_data, self.sequence_context)
        if stage == "test":
            self.test_dataset = MatrixDataset(self.test_data, self.sequence_context)
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
