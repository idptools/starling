import math

import torch
from IPython import embed
from torch import nn

from starling.models.blocks import ResBlockDecBasic, ResBlockEncBasic, ResizeConv2d


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


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base,
        dimension,
        norm,
        blocks=[2, 2, 2, 2],
        sinusoidal_pos_emb_theta=10000,
        time_dim=320,
    ):
        super().__init__()

        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = base
        self.dimension = dimension

        # Time embeddings
        self.time_emb = SinusoidalPosEmb(time_dim, theta=sinusoidal_pos_emb_theta)

        # Encoder part of UNet

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.in_channels = base
        layer_in_channels = [base * (2**i) for i in range(len(blocks))]

        self.encoder_layer1 = self._make_encoder_layer(
            ResBlockEncBasic, layer_in_channels[0], blocks[0], True, stride=2
        )
        self.encoder_layer2 = self._make_encoder_layer(
            ResBlockEncBasic, layer_in_channels[1], blocks[1], True, stride=2
        )
        self.encoder_layer3 = self._make_encoder_layer(
            ResBlockEncBasic, layer_in_channels[2], blocks[2], True, stride=2
        )
        self.encoder_layer4 = self._make_encoder_layer(
            ResBlockEncBasic, layer_in_channels[3], blocks[3], True, stride=2
        )

        # Middle convolution of the UNet
        self.middle = nn.Sequential(
            nn.Conv2d(
                layer_in_channels[3],
                layer_in_channels[3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(layer_in_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                layer_in_channels[3],
                layer_in_channels[3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(layer_in_channels[3]),
            nn.ReLU(inplace=True),
        )

        # Decoder part of UNet

        # Reverse the blocks to make the decoder
        blocks.reverse()

        layer_in_channels = [base * (2**i) for i in range(len(blocks))]
        self.in_channels = layer_in_channels[-1]

        self.decoder_layer1 = self._make_decoder_layer(
            ResBlockDecBasic, layer_in_channels[-1], blocks[0], True, stride=2
        )
        self.decoder_layer2 = self._make_decoder_layer(
            ResBlockDecBasic, layer_in_channels[-2], blocks[1], True, stride=2
        )
        self.decoder_layer3 = self._make_decoder_layer(
            ResBlockDecBasic, layer_in_channels[-3], blocks[2], True, stride=2
        )
        self.decoder_layer4 = self._make_decoder_layer(
            ResBlockDecBasic,
            layer_in_channels[-4],
            blocks[3],
            True,
            stride=1,
            last_layer=True,
        )

        self.final_conv = ResizeConv2d(
            in_channels=base,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            scale_factor=2,
            norm=None,
            activation="relu",
        )

        # self.final2_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def _make_encoder_layer(self, block, out_channels, blocks, conditional, stride=1):
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, self.norm, conditional)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, 1, self.norm, conditional)
            )
        return ConditionalSequential(*layers)

    def _make_decoder_layer(
        self, block, out_channels, blocks, conditional, stride=1, last_layer=False
    ):
        layers = []
        self.in_channels = out_channels * block.contraction
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, 1, self.norm, conditional)
            )
        if stride > 1 and block == ResBlockDecBasic:
            out_channels = int(out_channels / 2)
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                last_layer=last_layer,
                norm=self.norm,
                conditional=conditional,
            )
        )
        return ConditionalSequential(*layers)

    def forward(self, x, time, labels=None):
        # Get the time embeddings
        time = self.time_emb(time)

        # Add the labels to time embeddings if they are provided
        if labels is not None:
            time += labels

        # embed()
        # Start the UNet pass
        x = self.init_conv(x)

        # Encoder forward passes
        x = self.encoder_layer1(x, time)
        x_layer1 = x.clone()

        x = self.encoder_layer2(x, time)
        # Clone for residual connection to the decoder
        x_layer2 = x.clone()

        x = self.encoder_layer3(x, time)
        # Clone for residual connection to the decoder
        x_layer3 = x.clone()

        x = self.encoder_layer4(x, time)
        # Clone for residual connection to the decoder
        x_layer4 = x.clone()

        # Two convolution in the middle of the UNet
        x = self.middle(x)

        # Decoder forward passes
        x = x + x_layer4
        x = self.decoder_layer1(x, time)

        # Residual connection from the encoder
        x = x + x_layer3
        x = self.decoder_layer2(x, time)

        # Residual connection from the encoder
        x = x + x_layer2
        x = self.decoder_layer3(x, time)

        # Residual connection from the encoder
        x = x + x_layer1
        x = self.decoder_layer4(x, time)

        # Residual connection from the encoder
        x = x + x_layer1
        x = self.final_conv(x)

        return x
