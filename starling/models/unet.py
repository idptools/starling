import math

import torch
from IPython import embed
from torch import nn

from starling.models.attention import CrossAttention, SelfAttentionConv
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


class SelfAttentionResnetLayer(nn.Module):
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

        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "rms": RMSNorm,
            "group": nn.GroupNorm,
        }

        self.layer = nn.ModuleList()
        self.attention = nn.ModuleList()

        self.in_channels = in_channels

        for block in range(num_blocks):
            self.layer.append(
                ResBlockEncBasic(
                    self.in_channels, out_channels, 1, norm, timestep_dim, kernel_size=3
                )
            )

            # In self-attention kernel size is 1 to preserve spatial information and avoid spatial mixing
            self.attention.append(
                SelfAttentionConv(out_channels, attention_heads, kernel_size=1),
            )

            self.in_channels = out_channels

        # This is equivalent to LayerNorm but faster (no centering) - commonly used in transformers
        self.norm = (
            normalization[norm](out_channels)
            if norm != "group"
            else normalization[norm](32, out_channels)
        )

    def forward(self, x, time):
        for layer, attention in zip(self.layer, self.attention):
            x = layer(x, time)
            # I believe that the following is correct but I am not 100% sure, in transformers the input
            # to attention is summed with the output of attention (which is what I'm doing here)

            x = self.norm(x)
            x_attention = attention(x=x)
            x = self.norm(x + x_attention)
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

        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "rms": RMSNorm,
            "group": nn.GroupNorm,
        }

        self.label_mlp = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(label_dim, out_channels),
        )

        self.layer = nn.ModuleList()
        self.transformer = nn.ModuleList()

        self.in_channels = in_channels

        for block in range(num_blocks):
            self.layer.append(
                ResBlockEncBasic(
                    self.in_channels, out_channels, 1, norm, timestep_dim, kernel_size=3
                )
            )

            self.transformer.append(
                SpatialTransformer(out_channels, attention_heads),
            )

            self.in_channels = out_channels

    def forward(self, x, time, sequence_label):
        # Project the sequence label to the correct dimension
        sequence_label = self.label_mlp(sequence_label)
        for layer, transformer in zip(self.layer, self.transformer):
            x = layer(x, time)
            x = transformer(x, context=sequence_label)
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

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(self.base, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(
            self.time_emb,
            nn.Linear(self.base, self.time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Initial convolution

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=7, stride=1, padding=3),
            normalization[norm](base)
            if norm != "group"
            else normalization[norm](32, base),
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
            norm=normalization[norm],
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
            norm=normalization[norm],
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
            norm=normalization[norm],
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


class UNetSelfAttention(nn.Module):
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

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(self.base, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(
            self.time_emb,
            nn.Linear(self.base, self.time_dim),
            nn.SiLU(inplace=False),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Initial convolution

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=7, stride=1, padding=3),
            normalization[norm](base)
            if norm != "group"
            else normalization[norm](32, base),
            nn.ReLU(inplace=True),
        )

        self.in_channels = base
        all_in_channels = [base * (2**i) for i in range(len(blocks) + 1)]

        # Encoder part of UNet

        self.encoder_layer1 = SelfAttentionResnetLayer(
            self.in_channels, all_in_channels[0], self.norm, blocks[0], 8, self.time_dim
        )

        self.downsample1 = Downsample(all_in_channels[0], all_in_channels[1], norm)

        self.encoder_layer2 = SelfAttentionResnetLayer(
            all_in_channels[1],
            all_in_channels[1],
            self.norm,
            blocks[1],
            8,
            self.time_dim,
        )

        self.downsample2 = Downsample(all_in_channels[1], all_in_channels[2], norm)

        self.encoder_layer3 = SelfAttentionResnetLayer(
            all_in_channels[2],
            all_in_channels[2],
            self.norm,
            blocks[2],
            8,
            self.time_dim,
        )

        self.downsample3 = Downsample(all_in_channels[2], all_in_channels[3], norm)

        # Middle convolution of the UNet

        self.middle = SelfAttentionResnetLayer(
            all_in_channels[3],
            all_in_channels[3],
            self.norm,
            middle_blocks,
            8,
            self.time_dim,
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

        self.decoder_layer1 = SelfAttentionResnetLayer(
            all_in_channels[2] * 2,
            all_in_channels[2],
            self.norm,
            blocks[2],
            8,
            self.time_dim,
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

        self.decoder_layer2 = SelfAttentionResnetLayer(
            all_in_channels[1] * 2,
            all_in_channels[1],
            self.norm,
            blocks[1],
            8,
            self.time_dim,
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

        self.decoder_layer3 = SelfAttentionResnetLayer(
            all_in_channels[0] * 2,
            all_in_channels[0],
            self.norm,
            blocks[0],
            8,
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


class UNetConditional(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base,
        norm,
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=384,
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

        self.label_mlp = nn.Sequential(
            nn.Linear(self.labels_dim, self.labels_dim),
            nn.SiLU(inplace=False),
            nn.Linear(self.labels_dim, self.labels_dim),
        )

        all_in_channels = [base * (2**i) for i in range(len(blocks) + 1)]

        # Encoder part of UNet

        self.encoder_layer1 = CrossAttentionResnetLayer(
            self.in_channels,
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
            blocks[0],
            8,
            self.time_dim,
            self.labels_dim,
        )

        self.final_conv = nn.Conv2d(
            all_in_channels[0], out_channels, kernel_size=1, padding=0
        )

    def forward(self, x, time, labels=None):
        # Get the time embeddings
        time = self.time_mlp(time)

        if labels is not None:
            labels = self.label_mlp(labels)

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
        x = self.final_conv(x)

        return x


class UNetConditional2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base,
        norm,
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=384,
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

        self.label_mlp = nn.Sequential(
            nn.Linear(self.labels_dim, self.labels_dim),
            nn.SiLU(inplace=False),
            nn.Linear(self.labels_dim, self.labels_dim),
        )

        all_in_channels = [base * (2**i) for i in range(len(blocks) + 1)]

        # Encoder part of UNet

        self.encoder_layer1 = ResnetLayer(
            self.in_channels,
            all_in_channels[0],
            self.norm,
            blocks[0],
            self.time_dim,
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

        self.decoder_layer3 = ResnetLayer(
            all_in_channels[0] * 2,
            all_in_channels[0],
            self.norm,
            blocks[0],
            self.time_dim,
        )

        self.final_conv = nn.Conv2d(
            all_in_channels[0], out_channels, kernel_size=1, padding=0
        )

    def forward(self, x, time, labels=None):
        # Get the time embeddings
        time = self.time_mlp(time)

        if labels is not None:
            labels = self.label_mlp(labels)

        # Encoder forward passes
        x = self.encoder_layer1(x, time)
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
        x = self.decoder_layer3(x, time)

        # Final convolutions
        x = self.final_conv(x)

        return x
