import torch
from IPython import embed
from torch import nn
from torch.nn import functional as F

from starling.models.normalization import RMSNorm


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 1) -> None:
        """
        CrossAttention module for use in UNet models. This module is used to
        perform attention on query (distance maps/2D data) using key and value
        (sequence labels). It is used to attend to 2D features using text/sequence
        embeddings, effectively conditioning the model on the text/sequence labels.

        Parameters
        ----------
        embed_dim : int
            Dimension of the input embedding
        num_heads : int
            Number of heads for multi-head attention
        kernel_size : int, optional
            Size of the kernel for generating query, by default 1
        """
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Convolutional projections on 2D data for query matrix, since there are spatial components
        # to this data we are using Conv2s instead of Linear layers
        self.query_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # Sequence labels of shape (batch, sequence_length, embed_dim)
        self.key_conv = nn.Linear(embed_dim, embed_dim)
        self.value_conv = nn.Linear(embed_dim, embed_dim)

        # Output convolutional layer (unclear whether this normalization is necessary)
        # - this is done in transformers
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1), RMSNorm(embed_dim)
        )

    def forward(self, query, key, value):
        batch_size, channels, height, width = query.size()

        # Convolutional projections
        Q = self.query_conv(query)
        K = self.key_conv(key)
        V = self.value_conv(value)

        # Reshape to (batch_size, num_heads, head_dim, height * width)
        Q = Q.view(batch_size, self.num_heads, self.head_dim, -1)
        K = K.view(batch_size, self.num_heads, self.head_dim, -1)
        V = V.view(batch_size, self.num_heads, self.head_dim, -1)

        # Transpose for multi-head attention (batch_size, num_heads, height * width, head_dim)
        Q = Q.transpose(2, 3)
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and reshape back to original dimensions
        attention_output = (
            attention_output.transpose(2, 3)
            .contiguous()
            .view(batch_size, self.embed_dim, height, width)
        )
        attention_output = self.out_conv(attention_output)

        return attention_output


class SelfAttentionConv(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 1) -> None:
        """
        SelfAttentionConv module for use in UNet models. This module is used to
        perform self-attention on 2D data. It is used to attend to spatial features
        in the 2D data, effectively allowing the model to learn spatial relationships
        between pixels.

        Parameters
        ----------
        embed_dim : int
            Dimension of the input embedding
        num_heads : int
            Number of heads for multi-head attention
        kernel_size : int, optional
            Size of the kernel for generating query, key, and value matrices, by default 1
        """
        super(SelfAttentionConv, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.key_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.value_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1), RMSNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor):
        batch_size, channels, height, width = x.size()

        # Convolutional projections
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        # Reshape to (batch_size, num_heads, head_dim, height * width)
        Q = Q.view(batch_size, self.num_heads, self.head_dim, -1)
        K = K.view(batch_size, self.num_heads, self.head_dim, -1)
        V = V.view(batch_size, self.num_heads, self.head_dim, -1)

        # Transpose for multi-head attention (batch_size, num_heads, height * width, head_dim)
        Q = Q.transpose(2, 3)
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and reshape back to original dimensions
        attention_output = (
            attention_output.transpose(2, 3)
            .contiguous()
            .view(batch_size, self.embed_dim, height, width)
        )
        attention_output = self.out_conv(attention_output)

        return attention_output


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, channels_last: bool = False
    ) -> None:
        """
        This is a basic self-attention module. It uses linear layers to project
        the input into query, key, and value matrices, then performs scaled dot-product
        attention on these matrices. The output is then projected back to the original
        embedding dimension. Commonly used in transformer models.

        Parameters
        ----------
        embed_dim : int
            Dimension of the input embedding
        num_heads : int
            Number of heads for multi-head attention
        channels_last : bool, optional
            Whether the input has channels last format, if not it will be rearranged, by default False
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.channels_last = channels_last

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        output_shape = x.size()
        batch_size = output_shape[0]

        # Reshape to put channels last
        if not self.channels_last:
            x = x.permute(0, 2, 3, 1)

        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads and transpose
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and put through final linear layer
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        output = self.out(attention_output).reshape(output_shape)

        return output
