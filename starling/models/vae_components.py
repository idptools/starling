import torch.nn.functional as F
from IPython import embed
from torch import nn

from starling.models.blocks import (
    LayerNorm,
    ResBlockDecBasic,
    ResBlockEncBasic,
    ResizeConv2d,
)


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_blocks,
        norm,
        kernel_size=None,
        base=64,
        block_type=ResBlockEncBasic,
    ) -> None:
        super().__init__()

        self.block_type = block_type
        self.norm = norm
        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "layer": LayerNorm,
        }

        # First convolution of the ResNet Encoder reduction in the spatial dimensions / 2
        # with kernel=7 and stride=2 AvgPool2d reduces spatial dimensions by / 2
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=base,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            normalization[norm](base),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = base

        layer_in_channels = [base * (2**i) for i in range(len(num_blocks))]

        # Setting up the layers for the encoder
        self.layer1 = self._make_layer(
            self.block_type, layer_in_channels[0], num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            self.block_type, layer_in_channels[1], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.block_type, layer_in_channels[2], num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.block_type, layer_in_channels[3], num_blocks[3], stride=2
        )

        # self.max_features = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample_cnn = nn.Conv2d(
            in_channels=layer_in_channels[3],
            out_channels=layer_in_channels[3],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.activation = nn.ReLU(inplace=True)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, norm=self.norm))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, stride=1, norm=self.norm)
            )
        return nn.Sequential(*layers)

    def forward(self, data):
        data = self.first_conv(data)
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = self.layer4(data)
        data = self.activation(self.downsample_cnn(data))
        return data


class ResNet_Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        num_blocks,
        kernel_size,
        dimension,
        block_type=ResBlockDecBasic,
        base=64,
    ) -> None:
        super().__init__()

        # Calculate the input channels from the encoder, assuming
        # symmetric encoder and decoder setup
        self.block_type = block_type
        if self.block_type == ResBlockDecBasic:
            layer_in_channels = [base * (2**i) for i in range(len(num_blocks))]
            self.in_channels = layer_in_channels[-1]
        else:
            layer_in_channels = [base * (4**i) for i in range(len(num_blocks))]
            self.in_channels = layer_in_channels[-1]

        self.interpolate = int(dimension / (2 ** (len(num_blocks) + 1)))

        # Setting up the layers for the decoder

        self.layer1 = self._make_layer(
            self.block_type, layer_in_channels[-1], num_blocks[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.block_type, layer_in_channels[-2], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.block_type, layer_in_channels[-3], num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.block_type,
            layer_in_channels[-4],
            num_blocks[3],
            stride=1,
            last_layer=True,
        )

        in_channels_post_resnets = layer_in_channels[-4]

        # # This part could be done through interpolation (analogous to MaxPool)
        self.reshaping_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels_post_resnets,
                out_channels=in_channels_post_resnets,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(in_channels_post_resnets),
            # nn.LayerNorm([64, int(dimension / 2), int(dimension / 2)]),
            nn.ReLU(inplace=True),
        )

        # Final output layer that looks similar to the first layer of
        # the ResNet Encoder
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels_post_resnets,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, out_channels, blocks, stride=1, last_layer=False):
        layers = []
        self.in_channels = out_channels * block.contraction
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        if stride > 1 and block == ResBlockDecBasic:
            out_channels = int(out_channels / 2)
        layers.append(
            block(self.in_channels, out_channels, stride, last_layer=last_layer)
        )
        return nn.Sequential(*layers)

    def forward(self, data):
        # data = self.resize_conv(data)
        data = F.interpolate(data, size=(self.interpolate, self.interpolate))
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = self.layer4(data)
        data = self.reshaping_conv(data)
        data = self.output_layer(data)
        return data


# Current implementations of ResNets


def Resnet18_Encoder(in_channels, kernel_size, dimension, base):
    return ResNet_Encoder(
        in_channels,
        num_blocks=[2, 2, 2, 2],
        kernel_size=kernel_size,
        dimension=dimension,
        base=base,
    )


def Resnet18_Decoder(out_channels, kernel_size, dimension, base):
    return ResNet_Decoder(
        out_channels,
        num_blocks=[2, 2, 2, 2],
        kernel_size=kernel_size,
        dimension=dimension,
        base=base,
    )


def Resnet34_Encoder(in_channels, kernel_size, dimension, base):
    return ResNet_Encoder(
        in_channels,
        num_blocks=[3, 4, 6, 3],
        kernel_size=kernel_size,
        dimension=dimension,
        base=base,
    )


def Resnet34_Decoder(out_channels, kernel_size, dimension, base):
    return ResNet_Decoder(
        out_channels,
        num_blocks=[3, 4, 6, 3],
        kernel_size=kernel_size,
        dimension=dimension,
        base=base,
    )
