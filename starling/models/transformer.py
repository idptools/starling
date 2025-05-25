import math

import torch
from einops import rearrange
from torch import nn

from starling.data.positional_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from starling.models.attention import CrossAttention, SelfAttention


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        """
        Generates sinusoidal positional embeddings that are used in the denoising-diffusion
        models to encode the timestep information. The positional embeddings are generated
        using sine and cosine functions. It takes in time in the shape of (batch_size, 1)
        and returns the positional embeddings in the shape of (batch_size, dim). The positional
        encodings are later used in each of the ResNet blocks to encode the timestep information.

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
            Timestep information in the shape of (batch_size, 1).

        Returns
        -------
        torch.Tensor
            Positional (timestep) embeddings in the shape of (batch_size, dim).
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
        A simple Multi-Layer Perceptron with a single hidden layer and layer normalization.

        The MLP first projects the input to a higher dimension (output_dim * expansion_factor),
        applies a ReLU activation, then projects back to the output dimension. Finally,
        layer normalization is applied to the output.

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

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for all linear layers in the sequential
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

        self._init_weights()

    def _init_weights(self):
        gain = 1 / math.sqrt(2)
        # Initialize the weights of the gate and up projection layers
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

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

        self.up_proj = GeGLU(embed_dim, embed_dim * 4)

        self.down_proj = nn.Linear(embed_dim * 4, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize the weights of the down projection layer
        nn.init.xavier_uniform_(self.down_proj.weight)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = rearrange(x, "b c h w -> b h w c")

        x = self.norm(x)
        x = self.up_proj(x)
        x = self.down_proj(x)

        if x.dim() == 4:
            x = rearrange(x, "b h w c -> b c h w")
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

        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.cross_attention = CrossAttention(embed_dim, num_heads, embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, mask, context) -> torch.Tensor:
        # Prenorm is happening within the self attention layer
        x = x + self.self_attention(x, attention_mask=mask)

        # Cross attend to the context (ionic strength)
        x = x + self.cross_attention(x, context, query_mask=mask)

        # Prenorm is happening within the feed forward layer
        x = x + self.feed_forward(x)
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

        self.ionic_strength_emb = SinusoidalPosEmb(embed_dim)
        self.ionic_strength_mlp = MLP(embed_dim, embed_dim)

        self.cls_token_position = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.sequence_learned_embedding = nn.Embedding(21, embed_dim)

        self.sequence_positional_encoding = PositionalEncoding1D(embed_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize embedding normally with a small stddev.
        nn.init.normal_(self.sequence_learned_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token_position, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask, ionic_strength) -> torch.Tensor:
        # Convert the ionic strength to the same dimension as the input data
        ionic_strength = self.ionic_strength_emb(ionic_strength)

        # Run the MLP on the ionic strength
        ionic_strength = self.ionic_strength_mlp(ionic_strength)

        # Embed the sequences
        x = self.sequence_learned_embedding(x)

        if self.training:
            # Randomly mask some of the ionic strength values
            mask_ionic = (
                torch.rand(ionic_strength.shape[0], device=ionic_strength.device) < 0.2
            )
            ionic_strength[mask_ionic] = torch.zeros_like(ionic_strength[mask_ionic])

        # Give it a unique position (cls token)
        ionic_strength = ionic_strength + self.cls_token_position

        # Add positional encodings to the input data
        x = self.sequence_positional_encoding(x)

        # Run the transformer encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask, context=ionic_strength)

        # Concatenate the cls token to the input data
        x = torch.cat([ionic_strength, x], dim=1)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim):
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

        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.cross_attention = CrossAttention(embed_dim, num_heads, context_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, context, context_mask) -> torch.Tensor:
        # Prenorm is happening within the self attention layer
        x = x + self.self_attention(x)

        # Prenorm is happening within the cross attention layer
        x = x + self.cross_attention(x, context, context_mask=context_mask)

        # Prenorm is happening within the feed forward layer
        x = x + self.feed_forward(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int, num_layers=1):
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
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerDecoder(embed_dim, num_heads, context_dim)
                for _ in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Initialize convolutional layers
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)
        nn.init.xavier_uniform_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, context, mask) -> torch.Tensor:
        # Save the input for the residual connection
        x_in = x

        # Add positional encodings to the latent space representation of images
        x = self.image_positional_encodings(x)
        x = self.group_norm(x)
        x = self.conv_in(x)

        batch_size, *_ = x.shape
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device)
        # Prepend to the existing mask
        mask = torch.cat((cls_mask, mask), dim=1)

        # Transformer block to capture the relationships between the input data and the context data
        for block in self.transformer_blocks:
            x = block(x, context, mask)

        x = self.conv_out(x)

        # Residual connection
        return x + x_in
