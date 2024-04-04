import torch.nn.functional as F
from IPython import embed
from torch import nn

from starling.models.blocks import ResBlockDecBasic, ResBlockEncBasic, ResizeConv2d


class ResNet_Encoder(nn.Module):
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
            # nn.BatchNorm2d(starting_feature_extraction),
            nn.LayerNorm([self.in_channels, int(dimension / 2), int(dimension / 2)]),
            # AvgPool2d, might not be the best thing one can do here,
            # We might need a convolution layer instead
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
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
                    )
                )

        self.average_pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(
        self, residual_block, out_channels, num_blocks, kernel_size, dimension
    ):
        layers = []
        for block in range(num_blocks):
            layers += [
                residual_block(self.in_channels, out_channels, kernel_size, dimension)
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
        self.interpolate = int(dimension / (2 ** (len(num_blocks) + 1)))
        self.resize_conv = ResizeConv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
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
            # nn.BatchNorm2d(64),
            nn.LayerNorm([64, int(dimension / 2), int(dimension / 2)]),
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
        self, residual_block, out_channels, num_blocks, kernel_size, dimension
    ):
        layers = []
        for block in range(num_blocks - 1):
            layers += [
                residual_block(
                    self.in_channels, self.in_channels, kernel_size, dimension
                )
            ]

        layers += [
            residual_block(self.in_channels, out_channels, kernel_size, dimension)
        ]

        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, data):
        data = self.resize_conv(data)
        for layer in self.layers:
            data = layer(data)
        data = self.reshaping_conv(data)
        data = self.output_layer(data)
        return data
