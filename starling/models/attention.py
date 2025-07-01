import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from starling.models.normalization import RMSNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context_dim: int = None):
        """
        Multi-head attention module supporting both self- and cross-attention.

        Parameters
        ----------
        embed_dim : int
            Dimension of the query input (and output)
        num_heads : int
            Number of attention heads
        context_dim : int, optional
            Dimension of context input. If None, defaults to `embed_dim` (i.e., self-attention).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.context_dim = context_dim or embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Normalizations
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)

        # Projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(self.context_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(self.context_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context=None, query_mask=None, context_mask=None):
        """
        query:   (B, N, Dq) — tokens to be conditioned
        context: (B, S, Dc) — conditioning source (or None for self-attention)
        """
        if context is None:
            context = query
            context_mask = query_mask

        B, N, _ = query.shape
        _, S, _ = context.shape

        # Normalize
        query = self.query_norm(query)
        context = self.context_norm(context)

        # Project to Q, K, V
        Q = self.query_proj(query)
        K = self.key_proj(context)
        V = self.value_proj(context)

        # Reshape for multi-head attention
        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Build attention mask (broadcasted)
        if query_mask is not None or context_mask is not None:
            if query_mask is None:
                query_mask = torch.ones(B, N, device=query.device, dtype=torch.bool)
            if context_mask is None:
                context_mask = torch.ones(B, S, device=query.device, dtype=torch.bool)

            attn_mask = rearrange(query_mask, "b n -> b 1 n 1") & rearrange(
                context_mask, "b s -> b 1 1 s"
            )  # (B, 1, N, S)
            attn_mask = repeat(attn_mask, "b 1 n s -> b h n s", h=self.num_heads)
            attn_mask = attn_mask.bool()
        else:
            attn_mask = None

        # Scaled dot-product attention (Fused version in PyTorch ≥2.0)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)

        # Merge heads
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.out_proj(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_dim: int,
    ) -> None:
        """
        Cross-attention between query (tokens) and context (e.g., protein sequence).

        Parameters
        ----------
        embed_dim : int
            Dimensionality of the query tokens
        num_heads : int
            Number of attention heads
        context_dim : int
            Dimensionality of the context (keys/values)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.context_dim = context_dim

        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(context_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(context_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context, query_mask=None, context_mask=None):
        """
        query:   (B, N, D)     — tokens to be conditioned
        context: (B, S, C)     — context (e.g., sequence embeddings)
        """
        B, N, D = query.shape
        _, S, _ = context.shape

        # Normalize and project
        query = self.query_norm(query)
        context = self.context_norm(context)

        Q = self.query_proj(query)  # (B, N, D)
        K = self.key_proj(context)  # (B, S, D)
        V = self.value_proj(context)  # (B, S, D)

        # Multi-head reshape
        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Attention masks
        if query_mask is not None or context_mask is not None:
            if query_mask is None:
                query_mask = torch.ones((B, N), device=query.device)
            if context_mask is None:
                context_mask = torch.ones((B, S), device=query.device)

            query_mask = rearrange(query_mask, "b n -> b 1 n 1")
            context_mask = rearrange(context_mask, "b s -> b 1 1 s")
            attention_mask = query_mask * context_mask  # (B, 1, N, S)
            attention_mask = repeat(
                attention_mask, "b 1 n s -> b h n s", h=self.num_heads
            )
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)
        out = rearrange(out, "b h n d -> b n (h d)")  # back to (B, N, D)
        return self.out_proj(out)


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

        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.query_norm = nn.LayerNorm(embed_dim)

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None):
        input_dim = x.dim()

        if input_dim == 4:
            batch_size, channels, height, width = x.size()
        elif input_dim == 3:
            batch_size, seq_len, channels = x.size()
        else:
            raise ValueError("Input dimension not supported")

        if not self.channels_last and input_dim == 4:
            x = rearrange(x, "b c h w -> b h w c")

        # Prenormalization
        x = self.query_norm(x)

        # Linear projection for the query
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        if input_dim == 4:
            # If input is 4D (images)
            Q = rearrange(Q, "b x y (h d) -> b h (x y) d", h=self.num_heads)
            K = rearrange(K, "b x y (h d) -> b h (x y) d", h=self.num_heads)
            V = rearrange(V, "b x y (h d) -> b h (x y) d", h=self.num_heads)
        elif input_dim == 3:
            # If input is 3D (text)
            Q = rearrange(Q, "b x (h d) -> b h x d", h=self.num_heads)
            K = rearrange(K, "b x (h d) -> b h x d", h=self.num_heads)
            V = rearrange(V, "b x (h d) -> b h x d", h=self.num_heads)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(
                    batch_size, self.num_heads, seq_len, seq_len
                )

        attention_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask
        )

        # Concatenate heads and reshape back to original dimensions
        if input_dim == 4:
            attention_output = rearrange(
                attention_output, "b h (x y) d -> b x y (h d)", x=height, y=width
            )
        elif input_dim == 3:
            attention_output = rearrange(
                attention_output, "b h x d -> b x (h d)", x=seq_len
            )

        attention_output = self.out_proj(attention_output)

        if not self.channels_last and input_dim == 4:
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

        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

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
