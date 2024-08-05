import torch
from IPython import embed
from torch import nn

#! Make sure the non-learnable position encodings are properly implemented


# Non-learnable position encodings
class PositionalEncoding1D(nn.Module):
    def __init__(self, max_seq_len, embedding_size):
        """
        Positional encoding for 1D data. The positional encoding is added to the input tensor
        to provide information about the position of the elements in the input data. The positional
        encoding is computed using sine and cosine functions.

        Parameters
        ----------
        max_seq_len : _type_
            Max sequence length of the input data.
        embedding_size : _type_
            The number of features of the input data.
        """
        super(PositionalEncoding1D, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.positional_encoding = self._generate_positional_encoding()

    def _generate_positional_encoding(self):
        # Initialize the positional encoding tensor with 0s
        pe = torch.zeros(self.max_seq_len, self.embedding_size)

        # Get the position tensor (0, 1, 2, ..., max_seq_len - 1)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)

        # Compute divisor term for the positional encodings
        div_term = torch.exp(
            torch.arange(0, self.embedding_size, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / self.embedding_size)
        )

        # Assigns sine values to even indices in the last dimension
        pe[:, 0::2] = torch.sin(position * div_term)

        # Assigns cosine values to odd indices in the last dimension
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        return pe

    def forward(self, x):
        if self.positional_encoding.device != x.device:
            self.positional_encoding = self.positional_encoding.to(x.device)

        # Add positional encoding to the input tensor
        return x + self.positional_encoding


class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Positional encoding for 2D data. The positional encoding is added to the input tensor
        to provide information about the position of the elements in the input data. The positional
        encoding is computed using sine and cosine functions.

        Parameters
        ----------
        embed_dim : int
            The number of features of the input data.
        """
        super(PositionalEncoding2D, self).__init__()

        self.embed_dim = embed_dim

    def forward(self, x):
        b, c, h, w = x.shape
        pe = self.generate_pe(h, w, x.device)
        return x + pe

    def generate_pe(self, height, width, device):
        # Initialize the positional encoding tensor with 0s
        pe = torch.zeros(self.embed_dim, height, width, device=device)

        # Get the position tensors for height and width of the 2D data
        y_position = torch.arange(
            0, height, dtype=torch.float32, device=device
        ).unsqueeze(1)
        x_position = torch.arange(
            0, width, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Compute divisor term for the positional encodings
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / self.embed_dim)
        )

        # Reshape div_term to match (embed_dim/2, height, width) for broadcasting
        div_term = div_term.unsqueeze(1).unsqueeze(1)  # Shape (embed_dim/2, 1, 1)

        # Compute the positional encodings for height and width
        pe_x = torch.sin(x_position.unsqueeze(0) * div_term)
        pe_y = torch.sin(y_position.unsqueeze(0) * div_term)

        # Assign to the positional encoding tensor (even indices for x, odd indices for y)
        pe[0::2, :, :] = pe_x
        pe[1::2, :, :] = pe_y

        # Return and add batch dimension
        return pe.unsqueeze(0)


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
