import torch
from IPython import embed
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, channels_last=False):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.channels_last = channels_last

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        output_shape = query.size()
        batch_size = output_shape[0]

        # Reshape to put channels last
        if not self.channels_last:
            query = query.permute(0, 2, 3, 1)

        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split into multiple heads and transpose
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and put through final linear layer
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )
        output = self.out(attention_output).reshape(output_shape)

        return output
