import hdf5plugin
import numpy as np
import pytorch_lightning as pl
import torch
from finches.epsilon_calculation import InteractionMatrixConstructor
from finches.forcefields.mpipi import Mpipi_model

from starling.data.data_wrangler import load_hdf5_compressed, read_tsv_file
from starling.data.tokenizer import StarlingTokenizer


def row_norm(A):
    """
    Normalize the rows of a matrix A.

    Parameters:
    - A (torch.Tensor): Input matrix of shape [batch_size, channels, height, width].

    Returns:
    - torch.Tensor: Row-normalized matrix.
    """
    norm = torch.norm(A, p=2, dim=1, keepdim=True)
    return A / norm


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
    """
    Custom collate function that pads sequences to the maximum length within the batch.

    Parameters:
    - batch: List of tuples (distance_map, sequence)

    Returns:
    - distance_maps: Tensor of shape [batch_size, channels, height, width]
    - padded_sequences: Tensor of shape [batch_size, max_seq_length]
    - sequence_masks: Tensor of shape [batch_size, max_seq_length], 1 for real, 0 for padding
    """
    # Separate the distance maps and sequences
    distance_maps, sequences, interaction_matrices = zip(*batch)

    # Find the maximum sequence length in this batch
    max_len = max(seq.size(0) for seq in sequences)

    # Create padded sequences and masks
    padded_sequences = []
    sequence_masks = []

    # Pad each sequence to the maximum length
    for seq, matrix in zip(sequences, interaction_matrices):
        seq_len = seq.size(0)
        # Create padding
        padding = torch.zeros(max_len - seq_len, dtype=seq.dtype)
        # Pad sequence
        padded_seq = torch.cat([seq, padding], dim=0)
        padded_sequences.append(padded_seq)

        # Create mask (1 for real tokens, 0 for padding)
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:seq_len] = 1
        sequence_masks.append(mask)

    # Stack into tensors
    distance_maps = torch.stack(distance_maps, dim=0)
    padded_sequences = torch.stack(padded_sequences, dim=0)
    sequence_masks = torch.stack(sequence_masks, dim=0)

    return distance_maps, padded_sequences, sequence_masks.bool()


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

        self.tokenizer = StarlingTokenizer()

        Mpipi_GGv1_params_20 = Mpipi_model(version="Mpipi_GGv1", salt=20)
        Mpipi_GGv1_params_150 = Mpipi_model(version="Mpipi_GGv1", salt=150)
        Mpipi_GGv1_params_300 = Mpipi_model(version="Mpipi_GGv1", salt=300)

        self.mpipi_20 = InteractionMatrixConstructor(Mpipi_GGv1_params_20)
        self.mpipi_150 = InteractionMatrixConstructor(Mpipi_GGv1_params_150)
        self.mpipi_300 = InteractionMatrixConstructor(Mpipi_GGv1_params_300)

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

        # 2. Load compressed HDF5 data
        data = load_hdf5_compressed(
            data_path, keys_to_load=["latents", "seq"], frame=int(frame)
        )

        # 3. Process distance map
        distance_map = data["latents"]
        # Add channel dimension
        distance_map = torch.from_numpy(distance_map).unsqueeze(0)

        # 4. Process sequence data
        sequence = data["seq"].astype(np.int32)
        valid_indices = sequence != 0  # Identify non-padded positions
        sequence = sequence[valid_indices]  # Remove padding

        # 5. Generate interaction matrix using MPIPI model
        decoded_sequence = self.tokenizer.decoe(sequence)
        interaction_matrix = self.mpipi_150.calculate_pairwise_homotypic_matrix(
            decoded_sequence
        )

        # 6. Normalize interaction matrix and zero out diagonal
        interaction_matrix = row_norm(interaction_matrix)
        np.fill_diagonal(interaction_matrix, 0)

        # 7. Convert sequence to tensor
        sequence = torch.from_numpy(sequence)

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
