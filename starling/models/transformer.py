import torch
from einops import rearrange
from IPython import embed
from torch import nn

from starling.data.positional_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from starling.models.attention import CrossAttention, SelfAttention


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

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = rearrange(x, "b c h w -> b h w c")

        x = self.net(self.norm(x))

        if x.dim() == 4:
            x = rearrange(x, "b h w c -> b c h w")
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, custom_attention: bool):
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

        self.self_attention = SelfAttention(
            embed_dim, num_heads, custom=custom_attention
        )
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prenorm is happening within the self attention layer
        x = x + self.self_attention(x)

        # Prenorm is happening within the feed forward layer
        x = x + self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim, custom_attention):
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

        self.self_attention = SelfAttention(
            embed_dim, num_heads, custom=custom_attention
        )
        self.cross_attention = CrossAttention(
            embed_dim, num_heads, context_dim, custom=custom_attention
        )
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        # Prenorm is happening within the self attention layer
        x = x + self.self_attention(x)

        # Prenorm is happening within the cross attention layer
        x = x + self.cross_attention(x, context)

        # Prenorm is happening within the feed forward layer
        x = x + self.feed_forward(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, context_dim: int, custom_attention: bool
    ):
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

        # Add positional encodings to the context (protein sequence data)
        self.context_positional_encodings = PositionalEncoding1D(384, context_dim)
        self.context_encoder = TransformerEncoder(
            context_dim, num_heads, custom_attention=custom_attention
        )

        # Add positional encodings to the latent space representation of images (e.i. distance maps)
        self.image_positional_encodings = PositionalEncoding2D(embed_dim)
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.conv_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.transformer_block = TransformerDecoder(
            embed_dim, num_heads, context_dim, custom_attention=custom_attention
        )
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        # Save the input for the residual connection
        x_in = x

        # Add positional encodings to the context and process the context features
        context = self.context_positional_encodings(context)
        context = self.context_encoder(context)

        # Add positional encodings to the latent space representation of images
        x = self.image_positional_encodings(x)
        x = self.group_norm(x)
        x = self.conv_in(x)

        # Transformer block to capture the relationships between the input data and the context data
        x = self.transformer_block(x, context)

        x = self.conv_out(x)

        # Residual connection
        return x + x_in
