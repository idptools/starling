import math
from typing import Final

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import nn

from starling.data.positional_encodings import (
    PositionalEncoding1D,
)
from starling.models.attention import MultiHeadAttention

# The DiT blocks' LayerNorms were originally built as
# nn.LayerNorm(embed_dim, context_dim), which passes context_dim (512 in the
# released model) as LayerNorm's second positional argument, eps. Every DiT
# LayerNorm has therefore always run with eps = 512 rather than the default
# 1e-5. This is a historical accident, but the released weights were trained
# with it and eps is not stored in checkpoints, so changing it would change the
# shipped model's predictions. Keep it at 512.
DIT_LAYER_NORM_EPS: Final[float] = 512.0


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        """
        Generates sinusoidal positional embeddings that are used in the denoising-diffusion
        models to encode the timestep information (in the ViT) and the ionic strength (in the
        SequenceEncoder). The positional embeddings are generated using sine and cosine
        functions. A new last dimension of size dim is appended to the input, so time of shape
        (batch_size,) gives embeddings of shape (batch_size, dim) and time of shape
        (batch_size, 1) gives (batch_size, 1, dim).

        Parameters
        ----------
        dim : int
            Dimension of the input data.
        theta : int, optional
            A scaling factor for the positional embeddings. The default value is 10000.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional (timestep) embeddings.

        Parameters
        ----------
        time : torch.Tensor
            Timestep information in the shape of (batch_size,) or (batch_size, 1).

        Returns
        -------
        torch.Tensor
            Positional (timestep) embeddings in the shape of (batch_size, dim) for
            input of shape (batch_size,), or (batch_size, 1, dim) for input of shape
            (batch_size, 1). The ionic-strength embedding relies on the latter to get
            a context of one token per sequence.
        """
        device = time.device

        # The number of unique frequencies in the positional embeddings, half
        # will be used for sine and the other half for cosine functions
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, expansion_factor: int = 4):
        """
        A simple Multi-Layer Perceptron with a single hidden layer.

        The MLP first projects the input to a higher dimension (output_dim * expansion_factor),
        applies a ReLU activation, then projects back to the output dimension. There is no
        normalization inside the MLP (the SequenceEncoder applies a LayerNorm after it).

        Parameters
        ----------
        input_dim : int
            The dimension of the input features.
        output_dim : int
            The dimension of the output features.
        expansion_factor : int, optional
            The factor by which to expand the hidden dimension, by default 4.
        """
        super().__init__()

        hidden_dim = output_dim * expansion_factor

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    #     self._init_weights()

    # def _init_weights(self):
    #     # Initialize weights for all linear layers in the sequential
    #     for module in self.net:
    #         if isinstance(module, nn.Linear):
    #             nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        a GeGLU layer, which projects the input to 4 times the number of dimensions and gates
        it with a GELU of a second projection, followed by a linear layer that reduces the number
        of dimensions back to the original number of dimensions. The gated GELU is used to
        introduce non-linearity in the network.

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
        Transformer encoder layer. The transformer encoder layer consists of a self attention layer,
        a cross attention layer to the ionic strength embedding and a feed forward layer, each
        prenormalized and added back to its input. The self attention layer is used to capture the
        relationships between different elements in the input data. The cross attention layer lets
        every residue attend to the embedding of the ionic strength (a context of a single token).
        The feed forward layer is used to introduce non-linearity in the network.

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
        self.norm3 = nn.LayerNorm(embed_dim)

        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, mask, ionic_strengths) -> torch.Tensor:
        x_normed = self.norm1(x)
        x = x + self.self_attention(
            x_normed, context=x_normed, query_mask=mask, context_mask=mask
        )

        x_normed = self.norm2(x)

        x = x + self.cross_attention(x_normed, context=ionic_strengths, query_mask=mask)

        x_normed = self.norm3(x)
        x = x + self.feed_forward(x_normed)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int):
        """
        Sequence encoder. Residues are embedded with a learned embedding plus sinusoidal positional
        encodings and passed through a stack of transformer encoder layers, followed by a final
        LayerNorm. The ionic strength (in mM) is embedded with sinusoidal embeddings, an MLP and
        a LayerNorm, and every encoder layer cross-attends to it. During training only, the
        ionic-strength embedding of a random 20% of the sequences in each batch is replaced by
        zeros, so the model also learns to work without it; in eval mode it is always used.

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

        self.ionic_strength_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            MLP(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.sequence_learned_embedding = nn.Embedding(21, embed_dim)

        self.sequence_positional_encoding = PositionalEncoding1D(embed_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask, ionic_strengths) -> torch.Tensor:
        # Embed ionic strengths and expand for each token
        ionic_strengths = self.ionic_strength_emb(ionic_strengths)

        if self.training:
            # Randomly mask some of the ionic strength values
            mask_ionic = (
                torch.rand(ionic_strengths.shape[0], device=ionic_strengths.device)
                < 0.2
            )
            # masked_fill rather than boolean indexing, which has to count the
            # selected rows on the CPU and so synchronized with the GPU every step
            dropped_rows = mask_ionic.reshape(-1, *[1] * (ionic_strengths.dim() - 1))
            ionic_strengths = ionic_strengths.masked_fill(dropped_rows, 0.0)

        # Convert input sequence to embeddings
        x = self.sequence_learned_embedding(x)

        # Add positional encodings to the input data
        x = self.sequence_positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask=mask, ionic_strengths=ionic_strengths)

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

        # see DIT_LAYER_NORM_EPS for why these run with eps = 512
        self.norm1 = nn.LayerNorm(embed_dim, eps=DIT_LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(embed_dim, eps=DIT_LAYER_NORM_EPS)
        self.norm3 = nn.LayerNorm(embed_dim, eps=DIT_LAYER_NORM_EPS)

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
