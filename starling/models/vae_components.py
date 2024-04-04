import torch.nn.functional as F
from torch import nn

from starling.models.blocks import ResBlockDecBasic, ResBlockEncBasic


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_blocks,
        kernel_size,
    ) -> None:
        super().__init__()

        # First convolution of the ResNet Encoder reduction in the
        # spatial dimensions / 2 with kernel=7 and stride=2
        # maxpool reduces spatial dimensions by / 2
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64

        self.layers = nn.ModuleList()

        for num, block in enumerate(num_blocks):
            if num == 0:
                self.layers.append(
                    self._make_layer(
                        ResBlockEncBasic, self.in_channels, block, kernel_size
                    )
                )
            else:
                self.layers.append(
                    self._make_layer(
                        ResBlockEncBasic, int(self.in_channels * 2), block, kernel_size
                    )
                )

        # self.layer1 = self._make_layer(ResBlockEnc, 64, num_blocks[0], kernel_size)
        # self.layer2 = self._make_layer(ResBlockEnc, 128, num_blocks[1], kernel_size)
        # self.layer3 = self._make_layer(ResBlockEnc, 256, num_blocks[2], kernel_size)
        # self.layer4 = self._make_layer(ResBlockEnc, 512, num_blocks[3], kernel_size)

        self.average_pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, residual_block, out_channels, num_blocks, kernel_size):
        layers = []
        for block in range(num_blocks):
            layers += [residual_block(self.in_channels, out_channels, kernel_size)]
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
    ) -> None:
        super().__init__()

        # Calculate the input channels from the encoder, assuming
        # symmetric encoder and decoder setup
        self.in_channels = int(64 * 2 ** (len(num_blocks) - 1))

        #! This is hard coding the input dimensions in
        #! We should do this differently, so its general
        #! Could be given as an argument in Resnet_Decoder
        self.spatial_dims = int(384 / (2 ** (len(num_blocks) + 1)))

        self.layers = nn.ModuleList()

        for num, block in enumerate(num_blocks):
            if num == len(num_blocks) - 1:
                self.layers.append(
                    self._make_layer(
                        ResBlockDecBasic, self.in_channels, block, kernel_size
                    )
                )
            else:
                self.layers.append(
                    self._make_layer(
                        ResBlockDecBasic, int(self.in_channels / 2), block, kernel_size
                    )
                )

        # self.layer4 = self._make_layer(ResBlockDec, 256, num_blocks[0], kernel_size)
        # self.layer3 = self._make_layer(ResBlockDec, 128, num_blocks[1], kernel_size)
        # self.layer2 = self._make_layer(ResBlockDec, 64, num_blocks[2], kernel_size)
        # self.layer1 = self._make_layer(ResBlockDec, 64, num_blocks[3], kernel_size)

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

    def _make_layer(self, residual_block, out_channels, num_blocks, kernel_size):
        layers = []
        for block in range(num_blocks - 1):
            layers += [residual_block(self.in_channels, self.in_channels, kernel_size)]
        layers += [residual_block(self.in_channels, out_channels, kernel_size)]
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, data):
        # We could maybe do this with a linear layer instead
        data = F.interpolate(data, size=(self.spatial_dims, self.spatial_dims))
        for layer in self.layers:
            data = layer(data)
        # data = self.layer4(data)
        # data = self.layer3(data)
        # data = self.layer2(data)
        # data = self.layer1(data)
        data = self.reshaping_conv(data)
        data = self.output_layer(data)
        return data
