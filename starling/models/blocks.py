import torch.nn.functional as F
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        modules = []

        for num, layers in enumerate(range(num_layers)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        padding=padding,
                        kernel_size=kernel_size,
                    ),
                    instance_norm(out_channels),
                    nn.ReLU() if num == layers else nn.Identity(),
                )
            )
        self.conv = nn.Sequential(*modules)
        self.activation = nn.ReLU()

    def forward(self, data):
        identity = data
        data = self.conv(data)
        return self.activation(identity + data)


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
                    out_channels=out_channels[-1] * 2,
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
                    in_channels=out_channels[0] * 2,
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
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels[0],
            stride=2,
            kernel_size=7,
            padding=3,
        )

        in_channels = out_channels[0]

        modules = []

        for num, hidden_dim in enumerate(out_channels[1:]):
            modules.append(
                nn.Sequential(
                    ResBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        num_layers=3,
                    ),
                    DownsampleBlock(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
                )
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, data):
        data = self.first_conv(data)
        return self.encoder(data)


class ResNet_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        modules = []
        num_layers = len(out_channels) - 1
        for num in range(num_layers):
            modules.append(
                nn.Sequential(
                    ResBlock(
                        in_channels=out_channels[num],
                        out_channels=out_channels[num],
                        kernel_size=kernel_size,
                        num_layers=3,
                    ),
                    UpsampleBlock(
                        in_channels=out_channels[num],
                        out_channels=out_channels[num + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
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
