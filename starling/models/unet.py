import math

import torch
from IPython import embed
from torch import nn

from starling.models.attention import CrossAttention
from starling.models.blocks import ResBlockEncBasic, ResizeConv2d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalSequential(nn.Sequential):
    def forward(self, x, condition):
        for module in self._modules.values():
            x = module(x, condition)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm):
        super().__init__()

        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
        }

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            normalization[norm](out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResnetLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        num_blocks,
        timestep_dim,
    ):
        super().__init__()

        self.layer = nn.ModuleList()

        self.in_channels = in_channels

        for block in range(num_blocks):
            self.layer.append(
                ResBlockEncBasic(
                    self.in_channels, out_channels, 1, norm, timestep_dim, kernel_size=3
                )
            )

            self.in_channels = out_channels

    def forward(self, x, time):
        for layer in self.layer:
            x = layer(x, time)
        return x


class AttentionResnetLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        num_blocks,
        attention_heads,
        timestep_dim,
    ):
        super().__init__()

        self.linear_projection = nn.Linear(timestep_dim, out_channels)

        self.layer = nn.ModuleList()
        self.attention = nn.ModuleList()

        self.in_channels = in_channels

        for block in range(num_blocks):
            self.layer.append(
                ResBlockEncBasic(
                    in_channels, out_channels, 1, norm, timestep_dim, kernel_size=3
                )
            )

            self.attention.append(
                CrossAttention(out_channels, attention_heads),
            )

            self.in_channels = out_channels

        # LayerNorm is more common for this purpose, think about switching that
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x, time, sequence_label):
        # Project the sequence label to the correct dimension
        sequence_label = self.linear_projection(sequence_label)

        for layer, attention in zip(self.layer, self.attention):
            x = layer(x, time)

            # I believe that the following is correct but I am not 100% sure, in transformers the input
            # to attention is summed with the output of attention (which is what I'm doing here)
            x_attention = attention(query=x, key=sequence_label, value=sequence_label)
            x = self.norm(x + x_attention)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base,
        norm,
        blocks=[2, 2, 2],
        middle_blocks=2,
        sinusoidal_pos_emb_theta=10000,
    ):
        super().__init__()

        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = base * 4
        self.base = base

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(self.time_dim, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(
            self.time_emb,
            nn.Linear(self.time_dim, self.time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Initial convolution

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.in_channels = base
        all_in_channels = [base * (2**i) for i in range(len(blocks) + 1)]

        # Encoder part of UNet

        self.encoder_layer1 = ResnetLayer(
            self.in_channels, all_in_channels[0], self.norm, blocks[0], self.time_dim
        )

        self.downsample1 = Downsample(all_in_channels[0], all_in_channels[1], norm)

        self.encoder_layer2 = ResnetLayer(
            all_in_channels[1], all_in_channels[1], self.norm, blocks[1], self.time_dim
        )

        self.downsample2 = Downsample(all_in_channels[1], all_in_channels[2], norm)

        self.encoder_layer3 = ResnetLayer(
            all_in_channels[2], all_in_channels[2], self.norm, blocks[2], self.time_dim
        )

        self.downsample3 = Downsample(all_in_channels[2], all_in_channels[3], norm)

        # Middle convolution of the UNet

        self.middle = ResnetLayer(
            all_in_channels[3],
            all_in_channels[3],
            self.norm,
            middle_blocks,
            self.time_dim,
        )

        # Decoder part of UNet

        self.upconv1 = ResizeConv2d(
            all_in_channels[3],
            all_in_channels[2],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=nn.InstanceNorm2d,
            activation="relu",
        )

        self.decoder_layer1 = ResnetLayer(
            all_in_channels[2] * 2,
            all_in_channels[2],
            self.norm,
            blocks[2],
            self.time_dim,
        )

        self.upconv2 = ResizeConv2d(
            all_in_channels[2],
            all_in_channels[1],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=nn.InstanceNorm2d,
            activation="relu",
        )

        self.decoder_layer2 = ResnetLayer(
            all_in_channels[1] * 2,
            all_in_channels[1],
            self.norm,
            blocks[1],
            self.time_dim,
        )

        self.upconv3 = ResizeConv2d(
            all_in_channels[1],
            all_in_channels[0],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=nn.InstanceNorm2d,
            activation="relu",
        )

        self.decoder_layer3 = ResnetLayer(
            all_in_channels[0] * 2,
            all_in_channels[0],
            self.norm,
            blocks[0],
            self.time_dim,
        )

        self.final_conv = nn.Conv2d(base, out_channels, kernel_size=7, padding=3)

    def forward(self, x, time, labels=None):
        # Get the time embeddings
        time = self.time_mlp(time)

        # Add the labels to time embeddings if they are provided
        if labels is not None:
            time += labels

        # Start the UNet pass
        x = self.initial_conv(x)

        # Encoder forward passes
        x = self.encoder_layer1(x, time)
        x_layer1 = x.clone()
        x = self.downsample1(x)

        x = self.encoder_layer2(x, time)
        x_layer2 = x.clone()
        x = self.downsample2(x)

        x = self.encoder_layer3(x, time)
        x_layer3 = x.clone()
        x = self.downsample3(x)

        # Mid UNet
        x = self.middle(x, time)

        # Decoder forward passes
        x = self.upconv1(x)
        x = torch.cat((x, x_layer3), dim=1)
        x = self.decoder_layer1(x, time)

        x = self.upconv2(x)
        x = torch.cat((x, x_layer2), dim=1)
        x = self.decoder_layer2(x, time)

        x = self.upconv3(x)
        x = torch.cat((x, x_layer1), dim=1)
        x = self.decoder_layer3(x, time)

        # Final convolutions
        x = self.final_conv(x)

        return x
