from torch import nn


def conv2d_block(in_channels, out_channels, kernel_size, stride, norm, activation):
    padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)
    nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
    )


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
                    # nn.BatchNorm2d(hidden_dim),
                    instance_norm(hidden_dim),
                    nn.ReLU(),
                )
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, data):
        return self.encoder(data)


class vanilla_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        modules = []
        for num in range(len(out_channels) - 1):
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
                    # nn.BatchNorm2d(out_channels[num + 1]),
                    instance_norm(out_channels[num + 1]),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

    def forward(self, data):
        return self.decoder(data)
