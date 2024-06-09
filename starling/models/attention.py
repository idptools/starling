import torch
from einops import rearrange
from IPython import embed
from torch import nn
from torch.nn import functional as F

from starling.models.normalization import RMSNorm


class CrossAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, context_dim: int, channel_last=False
    ) -> None:
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
        self.channel_last = channel_last
        self.context_dim = context_dim

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(self.context_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(self.context_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context):
        batch_size, channels, height, width = query.size()

        if not self.channel_last:
            query = rearrange(query, "b c h w -> b h w c")

        # Prenormalization
        query = self.query_norm(query)
        context = self.context_norm(context)

        # Linear projection for the query (image features) - might be useful to change to Conv2d
        Q = self.query_proj(query)  # [batch_size, height, width, channels]

        # Linear projections for the key and value (text embeddings)
        K = self.key_proj(context)  # [batch_size, seq_len, head_dim * num_heads]
        V = self.value_proj(context)  # [batch_size, seq_len, head_dim * num_heads]

        # Reshape query (image features) to match multi-head attention dimensions
        # [batch_size, num_heads, height*width, head_dim]
        Q = rearrange(Q, "b x y (h d) -> b h (x y) d", h=self.num_heads)

        # Reshape key and value (text embeddings) for multi-head attention
        # [batch_size, num_heads, seq_len, head_dim]
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        # [batch_size, num_heads, seq_len, head_dim]
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and reshape back to original dimensions
        attention_output = rearrange(
            attention_output, "b h (x y) d -> b x y (h d)", x=height, y=width
        )
        attention_output = self.out_proj(attention_output)

        if not self.channel_last:
            attention_output = rearrange(attention_output, "b h w c -> b c h w")

        return attention_output


# The attention pooling could be used as an additional conditioning mechanism where its concatenated with
# timestep embeddings and then added to ResNet blocks (either in the middle or at the beginning)
# - Imagen seems to this at the beginning of the ResNet blocks
class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.SiLU(),  # Swish activation function
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: input features of shape (batch_size, num_features, feature_dim)
        batch_size, num_features, feature_dim = x.size()

        # Compute attention scores
        attention_scores = self.attention(x)  # shape: (batch_size, num_features, 1)
        attention_weights = torch.softmax(
            attention_scores, dim=1
        )  # shape: (batch_size, num_features, 1)

        # Compute weighted sum of features
        pooled_features = torch.sum(
            attention_weights * x, dim=1
        )  # shape: (batch_size, feature_dim)

        return pooled_features


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

        self.query_norm = nn.LayerNorm(embed_dim)

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        if not self.channels_last:
            x = rearrange(x, "b c h w -> b h w c")

        # Prenormalization
        x = self.query_norm(x)

        # Linear projection for the query (image features) - might be useful to change to Conv2d
        Q = self.query_proj(x)  # [batch_size, height, width, channels]

        # Linear projections for the key and value (text embeddings)
        K = self.key_proj(x)  # [batch_size, seq_len, head_dim * num_heads]
        V = self.value_proj(x)  # [batch_size, seq_len, head_dim * num_heads]

        # Reshape query (image features) to match multi-head attention dimensions
        # [batch_size, num_heads, height*width, head_dim]
        Q = rearrange(Q, "b x y (h d) -> b h (x y) d", h=self.num_heads)
        K = rearrange(K, "b x y (h d) -> b h (x y) d", h=self.num_heads)
        V = rearrange(V, "b x y (h d) -> b h (x y) d", h=self.num_heads)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and reshape back to original dimensions
        attention_output = rearrange(
            attention_output, "b h (x y) d -> b x y (h d)", x=height, y=width
        )
        attention_output = self.out_proj(attention_output)

        if not self.channels_last:
            attention_output = rearrange(attention_output, "b h w c -> b c h w")

        return attention_output
