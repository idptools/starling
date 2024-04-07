import torch.nn.functional as F
from IPython import embed
from torch import nn

from starling.models.blocks import ResBlockDecBasic, ResBlockEncBasic, ResizeConv2d


class ResNet_Encoder(nn.Module):
    def __init__(self, in_channels, num_blocks, kernel_size, dimension) -> None:
        super().__init__()

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
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=int(self.in_channels * 2),
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(int(self.in_channels * 2)),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = 128

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

        # This averages (b, c, h, w) to (b, c, 1, 1)
        # There might be place for improvement here
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
        # data = self.first_ResBlock(data)
        # data = self.second_ResBlock(data)
        for layer in self.layers:
            data = layer(data)
        # The final adaptive average can also be done through convolution
        # data = self.final_convolutions(data)
        data = self.average_pool(data)
        return data


class ResNet_Decoder(nn.Module):
    def __init__(
        self, out_channels, num_blocks, kernel_size, dimension, in_channels=128
    ) -> None:
        super().__init__()

        # Calculate the input channels from the encoder, assuming
        # symmetric encoder and decoder setup
        self.in_channels = int(in_channels * 2 ** (len(num_blocks) - 1))

        # This calculates the final spatial sizes in the encoder block
        # before the averaging takes place
        self.interpolate = int(dimension / (2 ** (len(num_blocks) + 1)))

        # self.resize_conv = nn.ModuleList()

        # num_upsampling_layers = int(self.interpolate / (3 * 2) + 1)
        # for num, _ in enumerate(range(num_upsampling_layers)):
        #     self.resize_conv.append(
        #         ResizeConv2d(
        #             in_channels=self.in_channels,
        #             out_channels=self.in_channels,
        #             kernel_size=kernel_size,
        #             scale_factor=3 if num == 0 else 2,
        #             mode="nearest",
        #         )
        #     )

        self.resize_conv = ResizeConv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=kernel_size,
            size=(self.interpolate, self.interpolate),
            mode="nearest",
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

        # This part could be done through interpolation (analogous to MaxPool)
        self.reshaping_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(self.in_channels),
            # nn.LayerNorm([64, int(dimension / 2), int(dimension / 2)]),
            nn.ReLU(),
        )

        # Final output layer that looks similar to the first layer of
        # the ResNet Encoder
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
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
        # Resizing back to the shapes before averaging step
        # in the encoder
        # for layer in self.resize_conv:
        #     data = layer(data)
        data = self.resize_conv(data)
        for layer in self.layers:
            data = layer(data)
        data = self.reshaping_conv(data)
        # data = self.final_resnet(data)
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
