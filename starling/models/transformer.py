import torch
from einops import rearrange
from IPython import embed
from torch import nn

from starling.data.positional_encodings import PositionalEncoding1D
from starling.models.attention import CrossAttention, SelfAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.net(self.norm(x))
        x = rearrange(x, "b h w c -> b c h w")
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(x)
        x = x + self.feed_forward(x)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim):
        super().__init__()

        # self.self_attention = SelfAttention(embed_dim, num_heads)
        self.cross_attention = CrossAttention(embed_dim, num_heads, context_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        # x = x + self.self_attention(x)
        x = x + self.cross_attention(x, context)
        x = x + self.feed_forward(x)
        return x


# The following block is not currently used, but will be used to replace ESM
class SpatialTransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int):
        super().__init__()

        self.context_positional_encodings = PositionalEncoding1D(384, context_dim)
        # Images are typically treated as fixed-size vectors
        # (e.g., by flattening the spatial dimensions or using convolutional layers to extract features).
        # Positional encodings are not necessary for images because their spatial information is already encoded in their structure,
        # and transformers don't operate directly on pixel values.

        # The encoder block uses multi-head self-attention and a feed-forward network for sequence data
        self.encoder = TransformerEncoder(embed_dim, num_heads)

        self.conv_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.decoder = TransformerDecoder(embed_dim, num_heads, context_dim)
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, context) -> torch.Tensor:
        # Add positional encodings to the context
        context = self.context_positional_encodings(context)
        # Run the context through the encoder to learn representations
        context = self.encoder(context)
        # Run the `x` (image) through the spatial transformer
        x = self.conv_in(x)
        # context from the encoder is used in cross attention in the decoder
        x = self.decoder(x, context)
        # Run the `x` through the final convolutional layer
        x = self.conv_out(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.conv_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.transformer_block = TransformerDecoder(embed_dim, num_heads, context_dim)

        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        x_in = x  # Residual connection
        x = self.group_norm(x)
        x = self.conv_in(x)
        x = self.transformer_block(x, context)
        x = self.conv_out(x)
        return x + x_in
