import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from starling.models.transformer import DiTBlock
from starling.models.unet import SinusoidalPosEmb


class PatchEmbed(nn.Module):
    def __init__(self, in_ch, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

        num_tokens = (24 * 24) // patch_size**2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

    def forward(self, x):  # x: (B, C, H, W)
        x = self.proj(x)  # → (B, embed_dim, H/patch, W/patch)
        x = rearrange(x, "b c h w -> b (h w) c")  # → (B, N, embed_dim)
        x = self.norm(x)

        # Add positional embeddings
        x = x + self.pos_embedding
        return x


class ViT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        context_dim: int,
        patch_size: int = 3,
    ):
        """
        Vision transformer model. The vision transformer model consists of a sequence encoder and a spatial transformer.
        The sequence encoder is used to process the features of the context data (in our case protein sequences).
        The spatial transformer is used to capture the relationships between the input data and the context data.

        Parameters
        ----------
        num_layers : int
            The number of layers in the sequence encoder.
        embed_dim : int
            The input dimension of the data. Used to initialize the sequence encoder and spatial transformer.
        num_heads : int
            The number of heads in the multi-head attention layer. Used to initialize the sequence encoder and spatial transformer.
        context_dim : int
            The dimension of the context data. Used to initialize the sequence encoder and spatial transformer.
        """
        super().__init__()
        BASE = 64  # Base dimension for the model
        self.patch_size = patch_size
        self.base = BASE

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(BASE, theta=10000)
        self.time_mlp = nn.Sequential(
            self.time_emb,
            nn.Linear(BASE, embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(embed_dim, embed_dim * 2),
        )

        self.conv_in = nn.Conv2d(1, BASE, kernel_size=3, stride=1, padding=1)

        self.to_patch_embedding = PatchEmbed(
            in_ch=BASE, patch_size=patch_size, embed_dim=embed_dim
        )

        self.transformer_layers = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, context_dim) for _ in range(num_layers)]
        )

        self.out_projection = nn.Sequential(
            nn.Linear(embed_dim, BASE * patch_size * patch_size),
            nn.ReLU(),
            nn.Linear(BASE * patch_size * patch_size, BASE * patch_size * patch_size),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                p1=patch_size,
                p2=patch_size,
                c=BASE,
                h=24 // patch_size,
                w=24 // patch_size,
            ),
        )

        self.conv_out = nn.Conv2d(BASE, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, timestep, sequence, mask) -> torch.Tensor:
        B, C, H, W = x.shape

        # Convert the timestep to the same shape as the input
        timestep = self.time_mlp(timestep).unsqueeze(1)

        # Extract the features from the latent space
        x = self.conv_in(x)

        # Patchify the latent space
        x = self.to_patch_embedding(x)

        # Add time embeddings to the latent space
        scale, shift = timestep.chunk(2, dim=-1)
        x = x * (1 + scale) + shift

        # Run the input through the transformer layers
        for layer in self.transformer_layers:
            x = layer(x, context=sequence, context_mask=mask)

        # Final linear layer to project the output back to the patch size
        x = self.out_projection(x)

        # # Apply the final convolution to get the output
        x = self.conv_out(x)

        return x
