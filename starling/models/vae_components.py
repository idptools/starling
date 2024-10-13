from typing import List

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
            "group": nn.GroupNorm,
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
            normalization[norm](base)
            if norm != "group"
            else normalization[norm](32, base),
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

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = nn.ModuleList()
        # layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                norm=self.norm,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    norm=self.norm,
                )
            )
        return layers
        # return nn.Sequential(*layers)

    def forward(self, data):
        data = self.first_conv(data)

        for block in self.layer1:
            data = block(data)

        for block in self.layer2:
            data = block(data)

        for block in self.layer3:
            data = block(data)

        for block in self.layer4:
            data = block(data)

        return data


class ResNet_Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        num_blocks: List,
        dimension: int,
        norm: str,
        block_type=ResBlockDecBasic,
        base=64,
    ) -> None:
        super().__init__()

        self.norm = norm

        # Calculate the input channels from the encoder, assuming
        # symmetric encoder and decoder setup
        self.block_type = block_type
        if self.block_type == ResBlockDecBasic:
            layer_in_channels = [base * (2**i) for i in range(len(num_blocks))]
            self.in_channels = layer_in_channels[-1]
        else:
            layer_in_channels = [base * (4**i) for i in range(len(num_blocks))]
            self.in_channels = layer_in_channels[-1]

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

        self.output_layer = ResizeConv2d(
            in_channels=in_channels_post_resnets,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
            norm=None,
            activation="relu",
            scale_factor=2,
        )

    def _make_layer(self, block, out_channels, blocks, stride=1, last_layer=False):
        layers = nn.ModuleList()
        self.in_channels = out_channels * block.contraction
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    norm=self.norm,
                )
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
            )
        )

        return layers

    def forward(self, data):
        for block in self.layer1:
            data = block(data)

        for block in self.layer2:
            data = block(data)

        for block in self.layer3:
            data = block(data)

        for block in self.layer4:
            data = block(data)

        data = self.output_layer(data)
        return data


class ConditionalSequential(nn.Sequential):
    def forward(self, x, condition=None):
        if condition is None:
            for module in self._modules.values():
                x = module(x)
        else:
            for module in self._modules.values():
                x = module(x, condition)
        return x


# Current implementations of ResNets


def Resnet18_Encoder(in_channels, norm, base):
    return ResNet_Encoder(
        in_channels,
        num_blocks=[2, 2, 2, 2],
        base=base,
        norm=norm,
    )


def Resnet18_Decoder(out_channels, dimension, base, norm):
    return ResNet_Decoder(
        out_channels,
        num_blocks=[2, 2, 2, 2],
        dimension=dimension,
        base=base,
        norm=norm,
    )


def Resnet34_Encoder(in_channels, base, norm):
    return ResNet_Encoder(
        in_channels,
        num_blocks=[3, 4, 6, 3],
        base=base,
        norm=norm,
    )


def Resnet34_Decoder(out_channels, dimension, base, norm):
    return ResNet_Decoder(
        out_channels,
        num_blocks=[3, 4, 6, 3],
        dimension=dimension,
        base=base,
        norm=norm,
    )
