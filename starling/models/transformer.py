import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import nn

from starling.data.positional_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from starling.models.attention import CrossAttention, MultiHeadAttention, SelfAttention


class AdaLayerNorm(nn.Module):
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim

        # Map conditioning (t + c) to scale and shift
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, embed_dim * 2),  # for gamma and beta
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),  # outputs [gamma | beta]
        )

        # no learned gamma/beta
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x, cond):
        """
        x: (B, N, D) - token embeddings
        cond: (B, cond_dim) - conditioning vector (e.g., t_emb + c_emb)
        """
        # Apply vanilla LayerNorm without scale/shift
        x_norm = self.norm(x)

        # Generate dynamic gamma and beta
        gamma_beta = self.cond_mlp(cond)  # (B, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each is (B, D)

        # Expand for broadcasting over sequence
        gamma = gamma.unsqueeze(1)  # (B, 1, D)
        beta = beta.unsqueeze(1)  # (B, 1, D)

        # Apply adaptive scale and shift
        return gamma * x_norm + beta


class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """
        Activation function that combines the concept of gating with the GELU activation function.
        The gating mechanism is used to control the flow of information through the network. The GELU
        activation function is used to introduce non-linearity in the network. The GeGLU activation
        function is often seen in the feed forward layer of transformers.

        The GeGLU activation function
        is defined as follows: x * GELU(gate), where x is the input to the activation function and
        gate is the output of a linear layer.

        Parameters
        ----------
        d_in : int
            The input dimension of the data. Used to initialize the linear layer.
        d_out : int
            The output dimension of the data. Used to initialize the linear layer.
        """
        super().__init__()

        self.proj = nn.Linear(d_in, d_out * 2)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Feed forward layer in the transformer architecture. The feed forward layer consists of
        two linear layers with a GELU activation function in between. The linear layers first
        expand the number of dimensions by a factor of 4 and then reduce the number of dimensions
        back to the original number of dimensions. The GELU activation function is used to introduce
        non-linearity in the network.

        Parameters
        ----------
        embed_dim : int
            The input dimension of the data. Used to initialize the linear layers.
        """
        super().__init__()

        self.net = nn.Sequential(
            GeGLU(embed_dim, embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """
        Transformer encoder layer. The transformer encoder layer consists of a self attention layer
        and a feed forward layer. The self attention layer is used to capture the relationships
        between different elements in the input data. The feed forward layer is used to introduce
        non-linearity in the network.

        Parameters
        ----------
        embed_dim : int
            The input dimension of the data. Used to initialize the self attention and feed forward layers.
        num_heads : int
            The number of heads in the multi-head attention layer. Used to initialize the self attention layer.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        x_normed = self.norm1(x)
        x = x + self.self_attention(
            x_normed, context=x_normed, query_mask=mask, context_mask=mask
        )

        x_normed = self.norm2(x)
        x = x + self.feed_forward(x_normed)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int):
        """
        Sequence encoder layer. The sequence encoder layer consists of a transformer encoder
        and a feed forward layer. The transformer encoder layer is used to capture the relationships
        between different elements in the input data. The feed forward layer is used to introduce
        non-linearity in the network.

        Parameters
        ----------
        num_layers : int
            The number of layers in the transformer encoder.
        embed_dim : int
            The input dimension of the data. Used to initialize the transformer encoder and feed forward layers.
        num_heads : int
            The number of heads in the multi-head attention layer. Used to initialize the transformer encoder.
        """
        super().__init__()

        self.sequence_learned_embedding = nn.Embedding(21, embed_dim)

        self.sequence_positional_encoding = PositionalEncoding1D(embed_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        # Convert input sequence to embeddings
        x = self.sequence_learned_embedding(x)

        # Add positional encodings to the input data
        x = self.sequence_positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        # Apply final normalization
        x = self.final_norm(x)

        return x


class DiTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int):
        """
        Transformer decoder layer. The transformer decoder layer consists of a self attention layer,
        cross attention layer and a feed forward layer. The self attention layer is used to capture the
        relationships between different elements in the input data. The cross attention layer is used to
        capture the relationships between the input data and the context data (usually the transformer
        encoder output). The feed forward layer is used to introduce non-linearity in the network.

        Parameters
        ----------
        embed_dim : int
            The input dimension of the data. Used to initialize the self attention, cross attention and feed forward layers.
        num_heads : int
            The number of heads in the multi-head attention layer. Used to initialize the self attention and cross attention layers.
        context_dim : _type_
            The dimension of the context data. Used to initialize the cross attention layer.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, context_dim)
        self.norm2 = nn.LayerNorm(embed_dim, context_dim)
        self.norm3 = nn.LayerNorm(embed_dim, context_dim)

        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, context_dim)

        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, context, context_mask) -> torch.Tensor:
        # Prenorm the input to the self attention layer
        x_normed = self.norm1(x)
        x = x + self.self_attention(query=x_normed, context=x_normed)

        # Prenorm the input to the cross attention layer (context is layernormed)
        x_normed = self.norm2(x)
        x = x + self.cross_attention(
            query=x_normed, context=context, context_mask=context_mask
        )

        # Prenorm the input to the feed forward layer
        x_normed = self.norm3(x)
        x = x + self.feed_forward(x_normed)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int):
        """
        Spatial transformer network. The spatial transformer network consists of a transformer encoder
        and a transformer decoder. The transformer encoder is used to process the features of the
        context data (in our case protein sequences). The transformer decoder is used to capture the relationships
        between the input data and the context data. The spatial transformer network is used to generate
        the latent space representation of the input data.

        Parameters
        ----------
        embed_dim : int
            The input dimension of the data. Used to initialize the transformer encoder and decoder.
        num_heads : int
            The number of heads in the multi-head attention layer. Used to initialize the transformer encoder and decoder.
        context_dim : int
            The dimension of the context data. Used to initialize the transformer encoder and decoder.
        """
        super().__init__()

        # Add positional encodings to the latent space representation of images (e.i. distance maps)
        self.image_positional_encodings = PositionalEncoding2D(embed_dim)
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.conv_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        # self.transformer_block = TransformerDecoder(embed_dim, num_heads, context_dim)
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, context, mask) -> torch.Tensor:
        # Save the input for the residual connection
        x_in = x

        # Add positional encodings to the latent space representation of images
        x = self.image_positional_encodings(x)
        x = self.group_norm(x)
        x = self.conv_in(x)

        # Transformer block to capture the relationships between the input data and the context data
        x = self.transformer_block(x, context, mask)

        x = self.conv_out(x)

        # Residual connection
        return x + x_in
