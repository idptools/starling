import math

import torch
from IPython import embed
from torch import nn

from starling.models.attention import (
    AttentionPooling,
    CrossAttention,
    SelfAttentionConv,
)
from starling.models.blocks import ResBlockEncBasic, ResizeConv2d
from starling.models.normalization import RMSNorm
from starling.models.transformer import SpatialTransformer


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
            "rms": RMSNorm,
            "group": nn.GroupNorm,
        }

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            normalization[norm](out_channels)
            if norm != "group"
            else normalization[norm](32, out_channels),
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
        class_dim=None,
    ):
        super().__init__()

        self.layer = nn.ModuleList()

        self.in_channels = in_channels

        for block in range(num_blocks):
            self.layer.append(
                ResBlockEncBasic(
                    self.in_channels, out_channels, 1, norm, timestep_dim, class_dim
                )
            )

            self.in_channels = out_channels

    def forward(self, x, time):
        for layer in self.layer:
            x = layer(x, time)
        return x


class CrossAttentionResnetLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        num_blocks,
        attention_heads,
        timestep_dim,
        label_dim,
    ):
        super().__init__()

        self.layer = nn.ModuleList()
        self.transformer = nn.ModuleList()

        self.in_channels = in_channels

        for block in range(num_blocks):
            self.layer.append(
                ResBlockEncBasic(self.in_channels, out_channels, 1, norm, timestep_dim)
            )
            self.transformer.append(
                CrossAttention(out_channels, attention_heads, label_dim),
            )

            self.in_channels = out_channels

    def forward(self, x, time, sequence_label):
        for layer, transformer in zip(self.layer, self.transformer):
            x = layer(x, time)
            x = x + transformer(x, context=sequence_label)
        return x


class UNetConditional(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base,
        norm,
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=512,
        sinusoidal_pos_emb_theta=10000,
    ):
        super().__init__()

        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "rms": RMSNorm,
            "group": nn.GroupNorm,
        }

        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = base * 4
        self.base = base
        self.labels_dim = labels_dim

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(self.base, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(
            self.time_emb,
            nn.Linear(self.base, self.time_dim),
            nn.SiLU(inplace=False),
            nn.Linear(self.time_dim, self.time_dim),
        )

        all_in_channels = [base * (2**i) for i in range(len(blocks) + 1)]

        # Encoder part of UNet

        self.conv_in = CrossAttentionResnetLayer(
            in_channels,
            all_in_channels[0],
            self.norm,
            blocks[0],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.encoder_layer1 = CrossAttentionResnetLayer(
            all_in_channels[0],
            all_in_channels[0],
            self.norm,
            blocks[0],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.downsample1 = Downsample(all_in_channels[0], all_in_channels[1], norm)

        self.encoder_layer2 = CrossAttentionResnetLayer(
            all_in_channels[1],
            all_in_channels[1],
            self.norm,
            blocks[1],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.downsample2 = Downsample(all_in_channels[1], all_in_channels[2], norm)

        self.encoder_layer3 = CrossAttentionResnetLayer(
            all_in_channels[2],
            all_in_channels[2],
            self.norm,
            blocks[2],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.downsample3 = Downsample(all_in_channels[2], all_in_channels[3], norm)

        # Middle convolution of the UNet

        self.middle = CrossAttentionResnetLayer(
            all_in_channels[3],
            all_in_channels[3],
            self.norm,
            middle_blocks,
            8,
            self.time_dim,
            self.labels_dim,
        )

        # Decoder part of UNet

        self.upconv1 = ResizeConv2d(
            all_in_channels[3],
            all_in_channels[2],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=normalization[norm],
            activation="relu",
        )

        self.decoder_layer1 = CrossAttentionResnetLayer(
            all_in_channels[2] * 2,
            all_in_channels[2],
            self.norm,
            blocks[2],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.upconv2 = ResizeConv2d(
            all_in_channels[2],
            all_in_channels[1],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=normalization[norm],
            activation="relu",
        )

        self.decoder_layer2 = CrossAttentionResnetLayer(
            all_in_channels[1] * 2,
            all_in_channels[1],
            self.norm,
            blocks[1],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.upconv3 = ResizeConv2d(
            all_in_channels[1],
            all_in_channels[0],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=normalization[norm],
            activation="relu",
        )

        self.decoder_layer3 = CrossAttentionResnetLayer(
            all_in_channels[0] * 2,
            all_in_channels[0],
            self.norm,
            blocks[1],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.conv_out = nn.Conv2d(all_in_channels[0], out_channels, kernel_size=1)

    def forward(self, x, time, labels=None):
        # Get the time embeddings
        time = self.time_mlp(time)

        # Initial convolution
        x = self.conv_in(x, time, labels)

        # Encoder forward passes
        x = self.encoder_layer1(x, time, labels)
        x_layer1 = x.clone()
        x = self.downsample1(x)

        x = self.encoder_layer2(x, time, labels)
        x_layer2 = x.clone()
        x = self.downsample2(x)

        x = self.encoder_layer3(x, time, labels)
        x_layer3 = x.clone()
        x = self.downsample3(x)

        # Mid UNet
        x = self.middle(x, time, labels)

        # Decoder forward passes
        x = self.upconv1(x)
        x = torch.cat((x, x_layer3), dim=1)
        x = self.decoder_layer1(x, time, labels)

        x = self.upconv2(x)
        x = torch.cat((x, x_layer2), dim=1)
        x = self.decoder_layer2(x, time, labels)

        x = self.upconv3(x)
        x = torch.cat((x, x_layer1), dim=1)
        x = self.decoder_layer3(x, time, labels)

        # Final convolutions
        x = self.conv_out(x)

        return x


class UNetConditionalTest(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base,
        norm,
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=512,
        num_heads=8,
        sinusoidal_pos_emb_theta=10000,
    ):
        super().__init__()

        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "rms": RMSNorm,
            "group": nn.GroupNorm,
        }

        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = base
        self.labels_dim = labels_dim

        all_in_channels = [base * (2**i) for i in range(len(blocks) + 1)]
        self.time_dim = all_in_channels[-1]

        # Timestep embeddings
        self.time_emb = SinusoidalPosEmb(self.base, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(
            self.time_emb,
            nn.Linear(self.base, self.time_dim),
            nn.SiLU(inplace=False),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # First convolution of the UNet
        # self.conv_in = nn.Sequential(
        #     nn.Conv2d(in_channels, all_in_channels[0], kernel_size=3, padding=1),
        #     normalization[norm](all_in_channels[0])
        #     if norm != "group"
        #     else normalization[norm](32, all_in_channels[0]),
        # )

        self.conv_in = CrossAttentionResnetLayer(
            in_channels,
            all_in_channels[0],
            self.norm,
            blocks[0],
            num_heads,
            self.time_dim,
            self.labels_dim,
        )

        # Encoder part of UNet

        self.encoder_layer1 = CrossAttentionResnetLayer(
            all_in_channels[0],
            all_in_channels[0],
            self.norm,
            blocks[0],
            num_heads,
            self.time_dim,
            self.labels_dim,
        )

        self.downsample1 = Downsample(all_in_channels[0], all_in_channels[0], norm)

        self.encoder_layer2 = CrossAttentionResnetLayer(
            all_in_channels[0],
            all_in_channels[1],
            self.norm,
            blocks[1],
            num_heads,
            self.time_dim,
            self.labels_dim,
        )

        self.downsample2 = Downsample(all_in_channels[1], all_in_channels[1], norm)

        self.encoder_layer3 = CrossAttentionResnetLayer(
            all_in_channels[1],
            all_in_channels[2],
            self.norm,
            blocks[2],
            num_heads,
            self.time_dim,
            self.labels_dim,
        )

        self.downsample3 = Downsample(all_in_channels[2], all_in_channels[2], norm)

        # Middle convolution of the UNet (no spatial dimension changes here)

        self.middle = nn.ModuleList()
        self.middle.append(
            CrossAttentionResnetLayer(
                all_in_channels[2],
                all_in_channels[3],
                self.norm,
                middle_blocks,
                num_heads,
                self.time_dim,
                self.labels_dim,
            )
        )
        self.middle.append(
            CrossAttentionResnetLayer(
                all_in_channels[3],
                all_in_channels[2],
                self.norm,
                middle_blocks,
                num_heads,
                self.time_dim,
                self.labels_dim,
            )
        )

        # Decoder part of UNet

        self.upconv1 = ResizeConv2d(
            all_in_channels[2],
            all_in_channels[2],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=normalization[norm],
            activation="relu",
        )

        # (I could probably include cross attention in the first ResnetLayer as well, but I'm not sure if it's necessary)
        self.decoder_layer1 = nn.ModuleList()
        self.decoder_layer1.append(
            ResnetLayer(
                all_in_channels[2] * 2,
                all_in_channels[2],
                self.norm,
                2,
                self.time_dim,
            )
        )
        self.decoder_layer1.append(
            CrossAttentionResnetLayer(
                all_in_channels[2],
                all_in_channels[1],
                self.norm,
                blocks[2],
                num_heads,
                self.time_dim,
                self.labels_dim,
            )
        )

        self.upconv2 = ResizeConv2d(
            all_in_channels[1],
            all_in_channels[1],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=normalization[norm],
            activation="relu",
        )

        self.decoder_layer2 = nn.ModuleList()
        self.decoder_layer2.append(
            ResnetLayer(
                all_in_channels[1] * 2,
                all_in_channels[1],
                self.norm,
                2,
                self.time_dim,
            )
        )
        self.decoder_layer2.append(
            CrossAttentionResnetLayer(
                all_in_channels[1],
                all_in_channels[0],
                self.norm,
                blocks[1],
                num_heads,
                self.time_dim,
                self.labels_dim,
            )
        )

        self.upconv3 = ResizeConv2d(
            all_in_channels[0],
            all_in_channels[0],
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=normalization[norm],
            activation="relu",
        )

        self.decoder_layer3 = nn.ModuleList()
        self.decoder_layer3.append(
            ResnetLayer(
                all_in_channels[0] * 2,
                all_in_channels[0],
                self.norm,
                2,
                self.time_dim,
            )
        )
        self.decoder_layer3.append(
            CrossAttentionResnetLayer(
                all_in_channels[0],
                all_in_channels[0],
                self.norm,
                blocks[0],
                num_heads,
                self.time_dim,
                self.labels_dim,
            )
        )

        self.conv_out = nn.Conv2d(all_in_channels[0], out_channels, kernel_size=1)
        # self.conv_out = nn.Conv2d(
        #     all_in_channels[0], out_channels, kernel_size=3, padding=1
        # )

    def forward(self, x, time, labels=None):
        # Get the time embeddings
        time = self.time_mlp(time)

        # Initial convolution
        x = self.conv_in(x, time, labels)
        # x = self.conv_in(x)

        # Encoder forward passes
        x = self.encoder_layer1(x, time, labels)
        x_layer1 = x.clone()
        x = self.downsample1(x)

        x = self.encoder_layer2(x, time, labels)
        x_layer2 = x.clone()
        x = self.downsample2(x)

        x = self.encoder_layer3(x, time, labels)
        x_layer3 = x.clone()
        x = self.downsample3(x)

        # Mid UNet
        for layer in self.middle:
            x = layer(x, time, labels)

        # Decoder forward passes
        x = self.upconv1(x)
        x = torch.cat((x, x_layer3), dim=1)
        for num, layer in enumerate(self.decoder_layer1):
            if num == 0:
                x = layer(x, time)
            else:
                x = layer(x, time, labels)

        x = self.upconv2(x)
        x = torch.cat((x, x_layer2), dim=1)
        for num, layer in enumerate(self.decoder_layer2):
            if num == 0:
                x = layer(x, time)
            else:
                x = layer(x, time, labels)

        x = self.upconv3(x)
        x = torch.cat((x, x_layer1), dim=1)
        for num, layer in enumerate(self.decoder_layer3):
            if num == 0:
                x = layer(x, time)
            else:
                x = layer(x, time, labels)

        # Final convolutions
        x = self.conv_out(x)

        return x
