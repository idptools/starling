import math

import torch
from torch import nn


# Non-learnable position encodings
class PositionalEncoding1D(nn.Module):
    def __init__(self, embedding_size):
        """
        Positional encoding for 1D data. The positional encoding is added to the input tensor
        to provide information about the position of the elements in the input data. The positional
        encoding is computed using sine and cosine functions.

        Parameters
        ----------
        embedding_size : int
            The number of features of the input data.
        """
        super(PositionalEncoding1D, self).__init__()
        self.embedding_size = embedding_size
        self.cached_encodings = {}  # Cache for previously computed encodings

    def _generate_positional_encoding(self, seq_len, device):
        """
        Generate positional encodings dynamically based on sequence length.

        Parameters
        ----------
        seq_len : int
            The length of the sequence for which to generate positional encodings.
        device : torch.device
            The device on which to create the encodings.

        Returns
        -------
        torch.Tensor
            Positional encodings tensor of shape (1, seq_len, embedding_size)
        """
        # Initialize the positional encoding tensor with 0s
        pe = torch.zeros(seq_len, self.embedding_size, device=device)

        # Get the position tensor (0, 1, 2, ..., seq_len - 1)
        position = torch.arange(
            0, seq_len, dtype=torch.float32, device=device
        ).unsqueeze(1)

        # Compute divisor term for the positional encodings
        div_term = torch.exp(
            torch.arange(0, self.embedding_size, 2, dtype=torch.float32, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / self.embedding_size)
        )

        # Assigns sine values to even indices in the last dimension
        pe[:, 0::2] = torch.sin(position * div_term)

        # Assigns cosine values to odd indices in the last dimension
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        return pe

    def forward(self, x):
        """
        Add positional encodings to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embedding_size)

        Returns
        -------
        torch.Tensor
            Input tensor with positional encodings added
        """
        seq_len = x.size(1)

        # Check if we have cached this sequence length
        cache_key = f"{seq_len}_{x.device}"
        if cache_key not in self.cached_encodings:
            # Generate and cache the positional encoding for this sequence length
            self.cached_encodings[cache_key] = self._generate_positional_encoding(
                seq_len, x.device
            )

            # Limit cache size to prevent memory issues
            if len(self.cached_encodings) > 10:  # Arbitrary limit, adjust as needed
                # Remove a random key (simple approach)
                remove_key = next(iter(self.cached_encodings))
                if remove_key != cache_key:  # Don't remove what we just added
                    del self.cached_encodings[remove_key]

        # Get the positional encoding from cache
        pe = self.cached_encodings[cache_key]

        # Add positional encoding to the input tensor
        return x + pe[:, :seq_len, :]


class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Positional encoding for 2D data using alternating sine and cosine.

        Parameters
        ----------
        embed_dim : int
            The number of embedding dimensions (channels)
        """
        super(PositionalEncoding2D, self).__init__()
        self.embed_dim = embed_dim
        self.cached_encodings = {}  # Cache for previously computed encodings

    def forward(self, x):
        b, c, h, w = x.shape

        # Check cache for this resolution
        cache_key = f"{h}_{w}_{x.device}"
        if cache_key not in self.cached_encodings:
            self.cached_encodings[cache_key] = self.generate_pe(h, w, x.device)

            # Limit cache size
            if len(self.cached_encodings) > 10:
                remove_key = next(iter(self.cached_encodings))
                if remove_key != cache_key:
                    del self.cached_encodings[remove_key]

        pe = self.cached_encodings[cache_key]
        return x + pe

    def generate_pe(self, height, width, device):
        """
        Generate 2D positional encodings with both sine and cosine functions.
        """
        # Make sure embed_dim is divisible by 4
        if self.embed_dim % 4 != 0:
            raise ValueError(
                f"Embedding dimension must be divisible by 4, got {self.embed_dim}"
            )

        # Each dimension gets 1/4 of channels for sin and 1/4 for cos
        dim_t = self.embed_dim // 4

        # Position tensors
        # [height, 1]
        y_pos = torch.arange(height, device=device).float().view(height, 1)
        # [1, width]
        x_pos = torch.arange(width, device=device).float().view(1, width)

        # Frequencies for different dimensions
        freq = torch.exp(
            torch.arange(0, dim_t, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / dim_t)
        ).view(dim_t, 1, 1)  # [dim_t, 1, 1]

        # Calculate encodings
        pos_x_enc = x_pos.expand(height, -1)  # [height, width]
        pos_y_enc = y_pos.expand(-1, width)  # [height, width]

        # Apply frequency bands to positions
        pos_x_enc = pos_x_enc.unsqueeze(0) * freq  # [dim_t, height, width]
        pos_y_enc = pos_y_enc.unsqueeze(0) * freq  # [dim_t, height, width]

        # Initialize positional encoding
        pe = torch.zeros(1, self.embed_dim, height, width, device=device)

        # X dimension - sin and cos
        pe[0, :dim_t] = torch.sin(pos_x_enc)
        pe[0, dim_t : 2 * dim_t] = torch.cos(pos_x_enc)

        # Y dimension - sin and cos
        pe[0, 2 * dim_t : 3 * dim_t] = torch.sin(pos_y_enc)
        pe[0, 3 * dim_t :] = torch.cos(pos_y_enc)

        return pe


# Learnable positional encodings
class LearnablePositionalEncoding1D(nn.Module):
    def __init__(self, sequence_length, embed_dim):
        super(LearnablePositionalEncoding1D, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.positional_encoding = nn.Parameter(
            torch.randn(1, sequence_length, embed_dim)
        )

    def forward(self, x):
        if self.positional_encoding.device != x.device:
            self.positional_encoding = self.positional_encoding.to(x.device)

        return x + self.positional_encoding


class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, height, width, embed_dim):
        super(LearnablePositionalEncoding2D, self).__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        self.positional_encoding = nn.Parameter(
            torch.randn(1, embed_dim, height, width)
        )

    def forward(self, x):
        if self.positional_encoding.device != x.device:
            self.positional_encoding = self.positional_encoding.to(x.device)

        return x + self.positional_encoding
