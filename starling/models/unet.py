import math
from typing import List

import torch
from torch import nn

from starling.models.blocks import ResBlockEncBasic, ResizeConv2d
from starling.models.normalization import RMSNorm
from starling.models.transformer import SpatialTransformer


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


class ConditionalSequential(nn.Sequential):
    def forward(self, x, condition):
        for module in self._modules.values():
            x = module(x, condition)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str):
        """
        A convolutional block that reduces the spatial dimensions of the input tensor by a factor of 2.
        The block consists of a convolutional layer with a kernel size of 3, stride of 2, and padding of 1.
        The convolutional layer is followed by a normalization layer and a ReLU activation function.

        Parameters
        ----------
        in_channels : int
            The number of features in the input tensor.
        out_channels : int
            The number of features in the output tensor.
        norm : str
            The normalization layer to be used in the block. Choose from batch, instance, rms, or group.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        in_channels: int,
        out_channels: int,
        norm: str,
        num_blocks: int,
        attention_heads: int,
        timestep_dim: int,
        label_dim: int,
        custom_attention: bool,
    ):
        """
        A combination of ResNet blocks followed by spatial transformer blocks. The ResNet block
        processes the input tensor and the spatial transformer block captures the relationships
        between the input tensor and the context data (protein sequences).

        Parameters
        ----------
        in_channels : int
            The number of features in the input tensor.
        out_channels : int
            The number of features in the output tensor.
        norm : str
            The normalization layer to be used in the block. Choose from batch, instance, rms, or group.
        num_blocks : int
            The number of ResNet + spatial transformer blocks in the layer.
        attention_heads : int
            The number of heads in the multi-head attention layer.
        timestep_dim : int
            The dimension of the timestep embeddings.
        label_dim : int
            The dimension of the context data (protein sequences).
        """
        super().__init__()

        self.layer = nn.ModuleList()
        self.transformer = nn.ModuleList()

        self.in_channels = in_channels

        for block in range(num_blocks):
            self.layer.append(
                ResBlockEncBasic(self.in_channels, out_channels, 1, norm, timestep_dim)
            )
            self.transformer.append(
                SpatialTransformer(
                    out_channels, attention_heads, label_dim, custom_attention
                ),
            )

            self.in_channels = out_channels

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, sequence_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ResNet + spatial transformer blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be processed by the ResNet + spatial transformer blocks.
        time : torch.Tensor
            Timestep embeddings to be used by the network.
        sequence_label : torch.Tensor
            Context data (protein sequences) to guide the prediction.

        Returns
        -------
        torch.Tensor
            Output of the ResNet + spatial transformer blocks.
        """
        for layer, transformer in zip(self.layer, self.transformer):
            x = layer(x, time)
            x = transformer(x, context=sequence_label)
        return x


class UNetConditional(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base: int,
        norm: str,
        blocks: List = [2, 2, 2],
        middle_blocks: int = 2,
        labels_dim: int = 512,
        sinusoidal_pos_emb_theta: int = 10000,
        custom_attention: bool = True,
    ):
        """
        A U-Net architecture that uses ResNet blocks with spatial transformer blocks to process the input
        tensor and the context data (protein sequences). The U-Net architecture consists of an encoder,
        a middle section, and a decoder. The spatial transformer blocks are used to capture the relationships
        between the input tensor and the context data (in our case protein sequences).

        Parameters
        ----------
        in_channels : int
            The number of features in the input tensor.
        out_channels : int
            The number of features in the output tensor.
        base : int
            The base number of features in the U-Net architecture.
        norm : str
            The normalization layer to be used in the block. Choose from batch, instance, rms, or group
        blocks : List, optional
            The number of ResNet + spatial transformer blocks in each section of the U-Net architecture, by default [2, 2, 2]
        middle_blocks : int, optional
            The number of ResNet + spatial transformer blocks in the middle section of the U-Net architecture, by default 2
        labels_dim : int, optional
            The dimension of the context data (i.e., protein sequences), by default 512
        sinusoidal_pos_emb_theta : int, optional
            A scaling factor for the positional (timestep) embeddings, by default 10000
        """
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
            custom_attention,
        )

        self.encoder_layer1 = CrossAttentionResnetLayer(
            all_in_channels[0],
            all_in_channels[0],
            self.norm,
            blocks[0],
            8,
            self.time_dim,
            self.labels_dim,
            custom_attention,
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
            custom_attention,
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
            custom_attention,
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
            custom_attention,
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
            custom_attention,
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
            custom_attention,
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
            custom_attention,
        )

        self.conv_out = nn.Conv2d(all_in_channels[0], out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the UNet architecture.

        Parameters
        ----------
        x : torch.Tensor
            Data to pass through the UNet architecture.
        time : torch.Tensor
            Timestep embeddings.
        labels : torch.Tensor, optional
            Context data (protein sequences) to guide the prediction, by default None

        Returns
        -------
        torch.Tensor
            Output of the UNet architecture.
        """
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

        # Decoder forward passes with skip connections from the encoder
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
