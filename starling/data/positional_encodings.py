import torch
from torch import nn

#! Make sure the non-learnable position encodings are properly implemented


# Non-learnable position encodings
class PositionalEncoding1D(nn.Module):
    def __init__(self, max_seq_len, embedding_size):
        super(PositionalEncoding1D, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.positional_encoding = self._generate_positional_encoding()

    def _generate_positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.embedding_size)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_size, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / self.embedding_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe

    def forward(self, x):
        # return x + self.positional_encoding[:, : x.size(1), :]
        if self.positional_encoding.device != x.device:
            self.positional_encoding = self.positional_encoding.to(x.device)

        return x + self.positional_encoding


class PositionalEncoding2D(nn.Module):
    def __init__(self, height, width, channels, max_height=24, max_width=24):
        super(PositionalEncoding2D, self).__init__()
        assert (
            height <= max_height and width <= max_width
        ), "Image size exceeds maximum positional encoding size"

        self.height = height
        self.width = width
        self.channels = channels
        self.max_height = max_height
        self.max_width = max_width

        # Compute positional encodings
        self.register_buffer("pos_encoding2D", self._generate_positional_encoding())

    def _generate_positional_encoding(self):
        y = torch.arange(self.height, dtype=torch.float).unsqueeze(1) / self.max_height
        x = torch.arange(self.width, dtype=torch.float).unsqueeze(0) / self.max_width

        sin_y = torch.sin(2 * torch.pi * y)
        cos_x = torch.cos(2 * torch.pi * x)

        sin_y = sin_y.unsqueeze(-1).expand(-1, self.width, self.channels // 2)
        cos_x = cos_x.unsqueeze(-1).expand(self.height, -1, self.channels // 2)

        pos_encoding = torch.cat([sin_y, cos_x], dim=-1)
        return pos_encoding

    def forward(self, x):
        """
        Add positional encodings to input image.

        Args:
        - x: Input image tensor of shape (batch_size, channels, height, width)

        Returns:
        - x_with_pos_encodings: Input image tensor with positional encodings added
        """
        batch_size = x.size(0)
        pos_encoding = self.pos_encoding2D.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x_with_pos_encodings = x + pos_encoding.to(x.device)
        return x_with_pos_encodings


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
