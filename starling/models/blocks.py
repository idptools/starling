import torch.nn.functional as F
from IPython import embed
from torch import nn


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor=2, mode="bilinear"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            instance_norm(out_channels),
            nn.ReLU(),
        )

    def forward(self, data):
        return self.conv(data)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=1,
            ),
            instance_norm(out_channels),
            nn.ReLU(),
        )

    def forward(self, data):
        return self.conv_transpose(data)


class ResBlockDec(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        upsample=False,
    ) -> None:
        super().__init__()

        self.upsample = upsample
        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                padding=padding,
                kernel_size=kernel_size,
            ),
            instance_norm(in_channels),
            nn.ReLU(),
        )

        if self.upsample:
            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    kernel_size=kernel_size,
                    padding=padding,
                    output_padding=1,
                ),
                instance_norm(out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    kernel_size=kernel_size,
                    padding=padding,
                    output_padding=1,
                ),
                instance_norm(out_channels),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    padding=padding,
                    kernel_size=kernel_size,
                ),
                instance_norm(out_channels),
            )

        self.activation = nn.ReLU()

    def forward(self, data):
        if self.upsample:
            identity = self.shortcut(data)
        else:
            identity = data

        out = self.conv1(data)
        out = self.conv2(out)
        out += identity
        return self.activation(out)


class ResBlockEnc(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, downsample=False
    ) -> None:
        super().__init__()

        self.downsample = downsample
        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels if self.downsample else in_channels,
                stride=2 if self.downsample else 1,
                padding=padding,
                kernel_size=kernel_size,
            ),
            instance_norm(out_channels if self.downsample else in_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels if self.downsample else in_channels,
                out_channels=out_channels if self.downsample else in_channels,
                stride=1,
                padding=padding,
                kernel_size=kernel_size,
            ),
            instance_norm(out_channels if self.downsample else in_channels),
        )

        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                ),
                instance_norm(out_channels),
            )
        self.activation = nn.ReLU()

    def forward(self, data):
        # Set up the shortcut connection if necessary
        if self.downsample:
            identity = self.shortcut(data)
        else:
            identity = data

        # First convolution
        out = self.conv1(data)
        # Second convolution
        out = self.conv2(out)
        # Add the input and run it through activation function
        out += identity
        return self.activation(out)


def instance_norm(features, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(features, affine=True, eps=eps, **kwargs)


class vanilla_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        modules = []
        for num, hidden_dim in enumerate(out_channels):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    instance_norm(hidden_dim),
                    nn.ReLU(),
                )
            )
            in_channels = hidden_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels[-1],
                    kernel_size=3,
                    stride=1,
                    padding=0,
                ),
                # instance_norm(out_channels[-1] * 2),
                nn.ReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, data):
        return self.encoder(data)


class vanilla_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        modules = []

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=0,
                ),
                instance_norm(out_channels[0]),
                nn.ReLU(),
            )
        )

        num_layers = len(out_channels) - 1
        for num in range(num_layers):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels[num],
                        out_channels[num + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=1,
                    ),
                    instance_norm(out_channels[num + 1]),
                    nn.ReLU(),
                )
            )

        # Final output layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels[-1],
                    out_channels=out_channels[-1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
                nn.ReLU(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, data):
        return self.decoder(data)


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dims,
        kernel_size,
    ) -> None:
        super().__init__()

        # First convolution of the ResNet Encoder
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dims[0],
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            instance_norm(hidden_dims[0]),
            nn.ReLU(),
        )

        modules = []

        # The rest of the blocks
        num_layers = len(hidden_dims)
        for num in range(num_layers):
            if num == 0:
                # First Layer of the ResNet
                modules.append(
                    nn.Sequential(
                        ResBlockEnc(
                            in_channels=hidden_dims[num],
                            out_channels=hidden_dims[num],
                            kernel_size=kernel_size,
                            downsample=False,
                        ),
                        ResBlockEnc(
                            in_channels=hidden_dims[num],
                            out_channels=hidden_dims[num],
                            kernel_size=kernel_size,
                            downsample=False,
                        ),
                    )
                )
            else:
                # The rest of the layers of the ResNet
                modules.append(
                    nn.Sequential(
                        ResBlockEnc(
                            in_channels=hidden_dims[num - 1],
                            out_channels=hidden_dims[num],
                            kernel_size=kernel_size,
                            downsample=True,
                        ),
                        ResBlockEnc(
                            in_channels=hidden_dims[num],
                            out_channels=hidden_dims[num],
                            kernel_size=kernel_size,
                            downsample=False,
                        ),
                    )
                )

        # Final output layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dims[-1],
                    out_channels=hidden_dims[-1],
                    kernel_size=3,
                    stride=1,
                    padding=0,
                ),
                # instance_norm(out_channels[-1] * 2),
                nn.ReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, data):
        data = self.first_conv(data)
        return self.encoder(data)


class ResNet_Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_dims,
        kernel_size,
    ) -> None:
        super().__init__()

        modules = []

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[0],
                    out_channels=hidden_dims[0],
                    kernel_size=3,
                    stride=1,
                    padding=0,
                ),
                instance_norm(hidden_dims[0]),
                nn.ReLU(),
            )
        )

        # The rest of the blocks
        num_layers = len(hidden_dims) - 1
        for num in range(num_layers):
            # First blocks of the decoder
            modules.append(
                nn.Sequential(
                    ResBlockDec(
                        in_channels=hidden_dims[num],
                        out_channels=hidden_dims[num],
                        kernel_size=kernel_size,
                        upsample=False,
                    ),
                    ResBlockDec(
                        in_channels=hidden_dims[num],
                        out_channels=hidden_dims[num + 1],
                        kernel_size=kernel_size,
                        upsample=True,
                    ),
                )
            )

        # The Final ResNet Block of the Decoder
        modules.append(
            nn.Sequential(
                ResBlockDec(
                    in_channels=hidden_dims[-1],
                    out_channels=hidden_dims[-1],
                    kernel_size=kernel_size,
                    upsample=False,
                ),
                ResBlockDec(
                    in_channels=hidden_dims[-1],
                    out_channels=hidden_dims[-1],
                    kernel_size=kernel_size,
                    upsample=False,
                ),
            )
        )

        # Last convolution of the ResNet Decoder
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dims[-1],
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, data):
        data = self.decoder(data)
        return self.last_conv(data)
