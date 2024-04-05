import torch.nn.functional as F
from IPython import embed
from torch import nn

from starling.models.blocks import ResBlockDecBasic, ResBlockEncBasic, ResizeConv2d


class ResNet_Encoder_Original(nn.Module):
    def __init__(self, in_channels, num_blocks, kernel_size, dimension) -> None:
        super().__init__()

        # First convolution of the ResNet Encoder reduction in the spatial dimensions / 2
        # with kernel=7 and stride=2 AvgPool2d reduces spatial dimensions by / 2
        # self.dimension = int(dimension / 2)
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

        self.layers = nn.ModuleList()

        for num, block in enumerate(num_blocks):
            if num == 0:
                self.layers.append(
                    self._make_layer(
                        ResBlockEncBasic,
                        self.in_channels,
                        block,
                        kernel_size,
                        dimension,
                        stride=1,
                    )
                )
            else:
                self.layers.append(
                    self._make_layer(
                        ResBlockEncBasic,
                        int(self.in_channels * 2),
                        block,
                        kernel_size,
                        dimension,
                        stride=2,
                    )
                )

        self.average_pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(
        self, residual_block, out_channels, num_blocks, kernel_size, dimension, stride
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for num, block in enumerate(range(num_blocks)):
            layers += [
                residual_block(
                    self.in_channels, out_channels, kernel_size, dimension, strides[num]
                )
            ]
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, data):
        data = self.first_conv(data)
        for layer in self.layers:
            data = layer(data)
        # The final adaptive average can also be done through convolution
        data = self.average_pool(data)
        return data


class ResNet_Decoder_Original(nn.Module):
    def __init__(
        self,
        out_channels,
        num_blocks,
        kernel_size,
        dimension,
    ) -> None:
        super().__init__()

        # Calculate the input channels from the encoder, assuming
        # symmetric encoder and decoder setup
        self.in_channels = int(64 * 2 ** (len(num_blocks) - 1))
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

        for num, block in enumerate(num_blocks):
            if num == len(num_blocks) - 1:
                self.layers.append(
                    self._make_layer(
                        ResBlockDecBasic,
                        self.in_channels,
                        block,
                        kernel_size,
                        dimension,
                        stride=1,
                    )
                )
            else:
                self.layers.append(
                    self._make_layer(
                        ResBlockDecBasic,
                        int(self.in_channels / 2),
                        block,
                        kernel_size,
                        dimension,
                        stride=2,
                    )
                )

        # This part could be done through interpolation (analogous to MaxPool)
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
            nn.ReLU(),
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
            nn.ReLU(),
        )

    def _make_layer(
        self, residual_block, out_channels, num_blocks, kernel_size, dimension, stride
    ):
        layers = []
        for block in range(num_blocks - 1):
            layers += [
                residual_block(
                    self.in_channels, self.in_channels, kernel_size, dimension, stride=1
                )
            ]

        layers += [
            residual_block(
                self.in_channels, out_channels, kernel_size, dimension, stride
            )
        ]

        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, data):
        # data = self.resize_conv(data)
        data = F.interpolate(data, size=(self.interpolate, self.interpolate))
        for layer in self.layers:
            data = layer(data)
        data = self.reshaping_conv(data)
        data = self.output_layer(data)
        return data


# Current implementations of ResNets


def Resnet18_Encoder(in_channels, kernel_size, dimension):
    return ResNet_Encoder_Original(
        in_channels,
        num_blocks=[2, 2, 2, 2],
        kernel_size=kernel_size,
        dimension=dimension,
    )


def Resnet18_Decoder(out_channels, kernel_size, dimension):
    return ResNet_Decoder_Original(
        out_channels,
        num_blocks=[2, 2, 2, 2],
        kernel_size=kernel_size,
        dimension=dimension,
    )


def Resnet34_Encoder(in_channels, kernel_size, dimension):
    return ResNet_Encoder_Original(
        in_channels,
        num_blocks=[3, 4, 6, 3],
        kernel_size=kernel_size,
        dimension=dimension,
    )


def Resnet34_Decoder(out_channels, kernel_size, dimension):
    return ResNet_Decoder_Original(
        out_channels,
        num_blocks=[3, 4, 6, 3],
        kernel_size=kernel_size,
        dimension=dimension,
    )
