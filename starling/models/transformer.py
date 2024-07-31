import torch
from einops import rearrange
from IPython import embed
from torch import nn

from starling.data.positional_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from starling.models.attention import CrossAttention, SelfAttention


# Activation function commonly used in the feed forward of transformers
class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.proj = nn.Linear(d_in, d_out * 2)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
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
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prenorm is happening within the self attention layer
        x = x + self.self_attention(x)

        # Prenorm is happening within the feed forward layer
        x = x + self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim):
        super().__init__()

        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.cross_attention = CrossAttention(embed_dim, num_heads, context_dim)
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
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int):
        super().__init__()

        # Add positional encodings to the context
        self.context_positional_encodings = PositionalEncoding1D(384, context_dim)
        self.context_encoder = TransformerEncoder(context_dim, num_heads)

        # Add positional encodings to the latent space representation of images
        self.image_positional_encodings = PositionalEncoding2D(embed_dim)
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.conv_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.transformer_block = TransformerDecoder(embed_dim, num_heads, context_dim)
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        # Save the input for the residual connection
        x_in = x

        # Add positional encodings to the context
        context = self.context_positional_encodings(context)
        context = self.context_encoder(context)

        # Add positional encodings to the latent space representation of images
        x = self.image_positional_encodings(x)
        x = self.group_norm(x)
        x = self.conv_in(x)

        # Transformer
        x = self.transformer_block(x, context)

        x = self.conv_out(x)

        # Residual connection
        return x + x_in
