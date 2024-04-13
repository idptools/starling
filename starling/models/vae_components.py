import torch.nn.functional as F
from IPython import embed
from torch import nn

from starling.models.blocks import ResBlockDecBasic, ResBlockEncBasic, ResizeConv2d


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_blocks,
        kernel_size=None,
        dimension=None,
        block_type=ResBlockEncBasic,
    ) -> None:
        super().__init__()

        self.block_type = block_type

        # First convolution of the ResNet Encoder reduction in the spatial dimensions / 2
        # with kernel=7 and stride=2 AvgPool2d reduces spatial dimensions by / 2
        self.in_channels = 64
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(self.block_type, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block_type, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block_type, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block_type, 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.block_type, 512, num_blocks[3], stride=2)

        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, data):
        data = self.first_conv(data)
        # The final adaptive average can also be done through convolution
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = self.layer4(data)
        data = self.layer5(data)
        data = self.average_pool(data)
        return data


class ResNet_Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        num_blocks,
        kernel_size,
        dimension,
        block_type=ResBlockDecBasic,
    ) -> None:
        super().__init__()

        # Calculate the input channels from the encoder, assuming
        # symmetric encoder and decoder setup
        self.block_type = block_type
        if self.block_type == ResBlockDecBasic:
            self.in_channels = 512
        else:
            self.in_channels = 4096

        self.interpolate = int(dimension / (2 ** (len(num_blocks) + 1)))

        # This part can be done in many ways, this is just one of them
        # It adds some number of parameters
        # self.resize_conv = ResizeConv2d(
        #     in_channels=self.in_channels,
        #     out_channels=self.in_channels,
        #     kernel_size=kernel_size,
        #     size=(self.interpolate, self.interpolate),
        #     mode="nearest",
        # )

        self.layers = nn.ModuleList()

        self.layer0 = self._make_layer(self.block_type, 512, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(self.block_type, 512, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(self.block_type, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block_type, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            self.block_type, 64, num_blocks[3], stride=1, last_layer=True
        )

        # # This part could be done through interpolation (analogous to MaxPool)
        self.reshaping_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(64),
            # nn.LayerNorm([64, int(dimension / 2), int(dimension / 2)]),
            nn.ReLU(inplace=True),
        )

        # Final output layer that looks similar to the first layer of
        # the ResNet Encoder
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
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
        data = self.layer0(data)
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = self.layer4(data)
        data = self.reshaping_conv(data)
        data = self.output_layer(data)
        return data


# Current implementations of ResNets


def Resnet18_Encoder(in_channels, kernel_size, dimension):
    return ResNet_Encoder(
        in_channels,
        num_blocks=[2, 2, 2, 2],
        kernel_size=kernel_size,
        dimension=dimension,
    )


def Resnet18_Decoder(out_channels, kernel_size, dimension):
    return ResNet_Decoder(
        out_channels,
        num_blocks=[2, 2, 2, 2],
        kernel_size=kernel_size,
        dimension=dimension,
    )


def Resnet34_Encoder(in_channels, kernel_size, dimension):
    return ResNet_Encoder(
        in_channels,
        num_blocks=[3, 4, 6, 3],
        kernel_size=kernel_size,
        dimension=dimension,
    )


def Resnet34_Decoder(out_channels, kernel_size, dimension):
    return ResNet_Decoder(
        out_channels,
        num_blocks=[3, 4, 6, 3],
        kernel_size=kernel_size,
        dimension=dimension,
    )
