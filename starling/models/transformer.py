import torch
from einops import rearrange
from torch import nn

from starling.data.positional_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from starling.models.attention import CrossAttention, SelfAttention


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    FiLM is a conditioning mechanism that applies an affine transformation to features
    based on conditioning information. It modulates each feature map by scaling it with
    a learned parameter gamma and shifting it with a learned parameter beta.

    Formula: y = gamma * x + beta

    References:
        Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the FiLM modulation layer.

        Parameters
        ----------
        input_dim : int
            Dimension of the conditioning input (interaction vector).
        output_dim : int
            Dimension of the features to be modulated.
        """
        super().__init__()
        self.gamma_proj = nn.Linear(input_dim, output_dim)  # Scaling projection
        self.beta_proj = nn.Linear(input_dim, output_dim)  # Shift projection

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to the input features.

        Parameters
        ----------
        features : torch.Tensor
            Input features to be modulated. Shape: [..., output_dim]
        condition : torch.Tensor
            Conditioning information. Shape: [..., input_dim]

        Returns
        -------
        torch.Tensor
            Modulated features with the same shape as input features.
        """
        # Compute scaling factors
        gamma = self.gamma_proj(condition)

        # Compute shifting factors
        beta = self.beta_proj(condition)

        # Apply affine transformation: gamma * features + beta
        return gamma * features + beta


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

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


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


class FiLMFFN(nn.Module):
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

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.film = FiLMModulation(embed_dim, embed_dim)

        self.net = nn.Sequential(
            GeGLU(embed_dim, embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, context) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.film(x, context)
        x = self.net(x)
        return x


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

        self.pre_norm = nn.LayerNorm(embed_dim)

        self.net = nn.Sequential(
            GeGLU(embed_dim, embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = rearrange(x, "b c h w -> b h w c")

        x = self.pre_norm(x)

        x = self.net(x)

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
        self.feed_forward = FiLMFFN(embed_dim)

    def forward(self, x: torch.Tensor, mask, context) -> torch.Tensor:
        # Prenorm is happening within the self attention layer
        x = x + self.self_attention(x, attention_mask=mask)

        # Prenorm is happening within the feed forward layer
        x = x + self.feed_forward(x, context)

        return x


class InteractionMatrixEncoder(nn.Module):
    def __init__(
        self, num_layers: int, embed_dim: int, num_heads: int, context_dim=None
    ):
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

        self.interaction_vector_mlp = MLP(context_dim, embed_dim)

        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.sequence_positional_encoding = PositionalEncoding1D(embed_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, interaction_vector: torch.Tensor, mask) -> torch.Tensor:
        # Run the interaction vector through the MLP
        interaction_vector = self.interaction_vector_mlp(interaction_vector)

        # Get batch size and expand class token to batch dimension
        batch_size = interaction_vector.shape[0]
        cls_tokens = self.class_token.expand(batch_size, -1, -1)

        # Update mask to include class token (always attended to)
        if mask is not None:
            # Create a column of ones for the class token
            cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device)
            # Prepend to the existing mask
            mask = torch.cat((cls_mask, mask), dim=1)

        # Prepend class token to sequence
        interaction_vector = torch.cat((cls_tokens, interaction_vector), dim=1)

        # Add positional encodings to the input data
        interaction_vector = self.sequence_positional_encoding(interaction_vector)

        # Run the sequence through the transformer encoder
        for layer in self.layers:
            interaction_vector = layer(interaction_vector, mask=mask)
        return interaction_vector[:, 0]


class SequenceEncoder(nn.Module):
    def __init__(
        self, num_layers: int, embed_dim: int, num_heads: int, context_dim=None
    ):
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

        self.interaction_vector_mlp = MLP(context_dim, embed_dim)

        self.sequence_learned_embedding = nn.Embedding(21, embed_dim)

        self.sequence_positional_encoding = PositionalEncoding1D(embed_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask, interaction_vector=None) -> torch.Tensor:
        # Run the interaction vector through an MLP
        interaction_vector = self.interaction_vector_mlp(interaction_vector)

        # Turn the sequence into a learned embedding
        x = self.sequence_learned_embedding(x)

        # Add positional encodings to the combined sequence
        x = self.sequence_positional_encoding(x)

        # Run the sequence through the transformer encoder
        for layer in self.layers:
            x = layer(x, mask=mask, context=interaction_vector)
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
        self.transformer_block = TransformerDecoder(embed_dim, num_heads, context_dim)
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
