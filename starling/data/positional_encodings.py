import torch
from IPython import embed
from torch import nn

#! Make sure the non-learnable position encodings are properly implemented


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_seq_len, embedding_size, rotary=False):
        """
        Positional encoding for 1D data. The positional encoding is added to the input tensor
        to provide information about the position of the elements in the input data. The positional
        encoding is computed using sine and cosine functions or rotary embeddings.

        Parameters
        ----------
        max_seq_len : int
            Max sequence length of the input data.
        embedding_size : int
            The number of features of the input data.
        rotary : bool
            Whether to use rotary embeddings instead of sine-cosine positional encodings.
        """
        super(PositionalEncoding1D, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.rotary = rotary

        if not self.rotary:
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
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        if self.rotary:
            # Use rotary embeddings
            return self.apply_rotary_embeddings(x)
        else:
            if self.positional_encoding.device != x.device:
                self.positional_encoding = self.positional_encoding.to(x.device)
            return x + self.positional_encoding

    def apply_rotary_embeddings(self, x):
        # Reshape input to (batch_size, seq_len, embedding_size)
        batch_size, seq_len, embedding_size = x.shape
        theta = torch.exp(
            torch.arange(0, embedding_size, 2, device=x.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / embedding_size)
        )
        position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(1)
        angle_rates = position_ids * theta
        sin = torch.sin(angle_rates)
        cos = torch.cos(angle_rates)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim: int, rotary=False):
        """
        Positional encoding for 2D data. The positional encoding is added to the input tensor
        to provide information about the position of the elements in the input data. The positional
        encoding is computed using sine and cosine functions or rotary embeddings.

        Parameters
        ----------
        embed_dim : int
            The number of features of the input data.
        rotary : bool
            Whether to use rotary embeddings instead of sine-cosine positional encodings.
        """
        super(PositionalEncoding2D, self).__init__()
        self.embed_dim = embed_dim
        self.rotary = rotary

    def forward(self, x):
        if self.rotary:
            # Use rotary embeddings
            return self.apply_rotary_embeddings(x)
        else:
            b, c, h, w = x.shape
            pe = self.generate_pe(h, w, x.device)
            return x + pe

    def generate_pe(self, height, width, device):
        pe = torch.zeros(self.embed_dim, height, width, device=device)
        y_position = torch.arange(
            0, height, dtype=torch.float32, device=device
        ).unsqueeze(1)
        x_position = torch.arange(
            0, width, dtype=torch.float32, device=device
        ).unsqueeze(0)
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device)
                * (-torch.log(torch.tensor(10000.0, device=device)) / self.embed_dim)
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        pe[0::2, :, :] = torch.sin(x_position.unsqueeze(0) * div_term)
        pe[1::2, :, :] = torch.sin(y_position.unsqueeze(0) * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def apply_rotary_embeddings(self, x):
        # Apply rotary embeddings across the height and width
        batch_size, channels, height, width = x.shape
        assert (
            channels % 2 == 0
        ), "Embedding dimension must be even for rotary embeddings."
        half_dim = channels // 2

        theta = torch.exp(
            torch.arange(0, half_dim, device=x.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / half_dim)
        )

        y_position_ids = torch.arange(0, height, device=x.device).unsqueeze(1)
        x_position_ids = torch.arange(0, width, device=x.device).unsqueeze(1)

        y_angle_rates = y_position_ids * theta
        x_angle_rates = x_position_ids * theta

        y_sin = torch.sin(y_angle_rates).unsqueeze(-1)
        y_cos = torch.cos(y_angle_rates).unsqueeze(-1)

        x_sin = torch.sin(x_angle_rates).unsqueeze(-2)
        x_cos = torch.cos(x_angle_rates).unsqueeze(-2)

        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotary = torch.cat([x1 * x_cos - x2 * x_sin, x1 * x_sin + x2 * x_cos], dim=-1)
        return x_rotary


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
