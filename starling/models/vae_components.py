import torch.nn.functional as F
from IPython import embed
from torch import nn

from starling.models.blocks import ResBlockDecBasic, ResBlockEncBasic, ResizeConv2d


class ResNet_Encoder(nn.Module):
    def __init__(self, in_channels, num_blocks, kernel_size, dimension) -> None:
        super().__init__()

        self.in_channels = 64

        # This block is here instead of the original convolution
        # with 7 kernel and stride=2
        self.first_ResBlock = ResBlockEncBasic(
            in_channels, self.in_channels, kernel_size, dimension, stride=2
        )
        # This block is here instead of MaxPool2d/AvgPool2d
        self.second_ResBlock = ResBlockEncBasic(
            self.in_channels, self.in_channels, kernel_size, dimension, stride=2
        )

        # ResNet Layers
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

        # This reduces the spatial dimension from 3x3 to 1x1 instead of AvgPool2d
        self.final_convolution = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=0
            ),
            nn.ReLU(),
        )

        # This averages (b, c, h, w) to (b, c, 1, 1)
        # There might be place for improvement here
        # self.average_pool = nn.AdaptiveAvgPool2d(1)

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
        # data = self.first_conv(data)
        data = self.first_ResBlock(data)
        data = self.second_ResBlock(data)
        for layer in self.layers:
            data = layer(data)
        # The final adaptive average can also be done through convolution
        data = self.final_convolution(data)
        return data


class ResNet_Decoder(nn.Module):
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

        # This calculates the final spatial sizes in the encoder block
        # before the averaging takes place
        self.interpolate = int(dimension / (2 ** (len(num_blocks) + 1)))

        self.resize_conv = nn.ModuleList()

        num_upsampling_layers = int(self.interpolate / (3 * 2) + 1)
        for num, _ in enumerate(range(num_upsampling_layers)):
            self.resize_conv.append(
                ResizeConv2d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=kernel_size,
                    scale_factor=3 if num == 0 else 2,
                    mode="nearest",
                )
            )

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

        # Upsamples the spatial dimensions without changing the
        # the number of channels
        self.reshaping_resnet = ResBlockDecBasic(
            self.in_channels,
            self.in_channels,
            kernel_size=kernel_size,
            stride=2,
            dimension=dimension,
        )

        # Upsamples the spatial dimensions and downsamples the channels to be
        # equal to the out_channels
        self.final_resnet = ResBlockDecBasic(
            self.in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            dimension=dimension,
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        # # This part could be done through interpolation (analogous to MaxPool)
        # self.reshaping_conv = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=64,
        #         out_channels=64,
        #         kernel_size=kernel_size,
        #         stride=2,
        #         padding=1,
        #         output_padding=1,
        #     ),
        #     nn.BatchNorm2d(64),
        #     # nn.LayerNorm([64, int(dimension / 2), int(dimension / 2)]),
        #     nn.ReLU(),
        # )

        # # Final output layer that looks similar to the first layer of
        # # the ResNet Encoder
        # self.output_layer = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=64,
        #         out_channels=out_channels,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         output_padding=1,
        #     ),
        #     nn.ReLU(),
        # )

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
        # Resizing back to the shapes before averaging step
        # in the encoder
        for layer in self.resize_conv:
            data = layer(data)
        for layer in self.layers:
            data = layer(data)
        data = self.reshaping_resnet(data)
        data = self.final_resnet(data)
        data = self.output_layer(data)
        return data


# Current implementations of ResNets


def Resnet18_Encoder(in_channels, kernel_size, dimension):
    return ResNet_Encoder(
        in_channels,
        num_blocks=[2, 2, 2, 2, 2],
        kernel_size=kernel_size,
        dimension=dimension,
    )


def Resnet18_Decoder(out_channels, kernel_size, dimension):
    return ResNet_Decoder(
        out_channels,
        num_blocks=[2, 2, 2, 2, 2],
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
