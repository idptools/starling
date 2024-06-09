import torch
from einops import rearrange
from IPython import embed
from torch import nn

from starling.models.attention import CrossAttention, SelfAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.net(self.norm(x))
        x = rearrange(x, "b h w c -> b c h w")
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim):
        super().__init__()

        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.cross_attention = CrossAttention(embed_dim, num_heads, context_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        x = x + self.self_attention(x)
        x = x + self.cross_attention(x, context)
        x = x + self.feed_forward(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.conv_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.transformer_block = TransformerBlock(embed_dim, num_heads, context_dim)

        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        x_in = x  # Residual connection
        x = self.group_norm(x)
        x = self.conv_in(x)
        x = self.transformer_block(x, context)
        x = self.conv_out(x)
        return x + x_in
