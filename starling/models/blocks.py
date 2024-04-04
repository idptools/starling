import torch.nn.functional as F
from IPython import embed
from torch import nn


class MinPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MinPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        # Perform min pooling using torch.min and torch.nn.functional.max_pool2d
        unpool = nn.functional.max_pool2d(
            -1 * x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=False,
        )
        return -1 * unpool


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
            nn.BatchNorm2d(out_channels),
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, data):
        return self.conv_transpose(data)


class ResBlockDecBasic(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        # First convolution which doesn't change the shape of the tensor
        # (b, c, h, w) -> (b, c, h, w) stride = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                padding=padding,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        # Here we figure out whether the spatial dimensions
        # of the tensor need to be upsampled
        # (b, c, h, w) -> (b, c, h, w) stride = 1, conv2d
        # (b, c, h, w) -> (b, c/2, h*2, w*2 ) stride = 2, convtranspose2d
        if in_channels != out_channels:
            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    kernel_size=kernel_size,
                    padding=padding,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
            )
            # Setup a shortcut connection
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
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
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential()

        self.activation = nn.ReLU()

    def forward(self, data):
        # Setup the shortcut connection if necessary
        identity = self.shortcut(data)
        # First convolution of the data
        out = self.conv1(data)
        # Second convolution of the data
        out = self.conv2(out)
        # Connect the input data to the
        # output of convolutions
        out += identity
        # Run it through the activation function
        return self.activation(out)


class ResBlockEncBasic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        # First convolution of the ResNet with or without downsampling
        # depending on the downsample flag (stride=1 or 2)
        # (b, c, h, w) -> (b, c, h, w) stride = 1
        # (b, c, h, w) -> (b, c*2, h /2, w /2 ) stride = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2 if in_channels != out_channels else 1,
                padding=padding,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Second convolution which doesn't do any downsampling, but needs
        # to setup in_channels and out_channels according to self.conv1
        # (b, c, h, w) -> (b, c, h, w) stride = 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                padding=padding,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Set up the shortcut if downsampling is done
        # (b, c, h, w) -> (b, c*2, h /2, w /2 ) stride = 2
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()
        self.activation = nn.ReLU()

    def forward(self, data):
        # Set up the shortcut connection if necessary
        identity = self.shortcut(data)
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
                    nn.BatchNorm2d(hidden_dim),
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
                nn.BatchNorm2d(out_channels[-1]),
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
                nn.BatchNorm2d(out_channels[0]),
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
                    nn.BatchNorm2d(out_channels[num + 1]),
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
