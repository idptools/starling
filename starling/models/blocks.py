import torch
import torch.nn.functional as F
from einops import rearrange
from IPython import embed
from torch import nn

from starling.models.normalization import RMSNorm


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Modified from:
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L119
    It increases memory requirements substantially, unclear if that can be changed
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)

            return x


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


#! Fix how the activation function is passed in (should be torch.nn.Module not str)
class ResizeConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm: torch.nn.Module,
        activation: str,
        padding: int,
        size: int = None,
        scale_factor: int = None,
        mode: str = "nearest",
    ):
        """
        Instead of ConvTranspose2d, this module uses F.interpolate to upsample the input
        tensor and then applies a convolutional layer. This is useful when the output of
        the upsampled tensor is not the same as the input tensor. This module can be used
        in the decoder part of the network. This type of upsampling is known to fix checkerboard
        artifacts that are common in ConvTranspose2d (https://distill.pub/2016/deconv-checkerboard/).

        Parameters
        ----------
        in_channels : int
            The number of input channels
        out_channels : int
            The number of output channels
        kernel_size : int
            The size of the kernel for the convolutional layer
        norm : torch.nn.Module
            The normalization layer to use. Pass in the class of the normalization layer (example nn.InstanceNorm2d)
        activation : str
            The activation function to use. Pass in the class of the activation function (example nn.ReLU)
        padding : int
            The padding to use for the convolutional layer
        size : int, optional
            The spatial size of the output tensor. If None, the scale_factor is used, by default None
        scale_factor : int, optional
            The scale factor to use for the upsampling, by default None
        mode : str, optional
            The mode to use for the upsampling, by default "nearest"
        """
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

        if norm is not None:
            normalization = (
                norm(out_channels) if norm != nn.GroupNorm else norm(32, out_channels)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            nn.Identity() if norm is None else normalization,
            nn.Identity() if activation is None else nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(
            x, size=self.size, scale_factor=self.scale_factor, mode=self.mode
        )
        x = self.conv(x)
        return x


class ResBlockEncBasic(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm: str,
        timestep: int = None,
        kernel_size: int = 3,
    ) -> None:
        """
        A basic residual block. This block is often used in the ResNet architecture,
        specifically in ResNet18 and ResNet34. It consists of two convolutional layers
        with a ReLU activation function in between. The input is added to the output of
        the second convolutional layer. The output is then passed through another ReLU
        activation function. The block can also be conditioned on some class labels or
        other information that is summed to the output of first convolutional output before
        the normalization and activation function.

        Parameters
        ----------
        in_channels : int
            The number of input channels
        out_channels : int
            The number of output channels
        stride : int
            The stride of the first convolutional layer
        norm : str
            The normalization layer to use. Options are "batch", "instance", "layer", "rms" and "group"
        timestep : int, optional
            The dimension of the class labels/timesteps to condition the block on, if None no
            conditioning is done, by default None
        kernel_size : int, optional
            The size of the kernel for convolutional layers, by default 3
        """
        super().__init__()

        kernel_size = 3 if kernel_size is None else kernel_size
        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "layer": LayerNorm,
            "rms": RMSNorm,
            "group": nn.GroupNorm,
        }

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            kernel_size=kernel_size,
        )
        self.norm1 = (
            normalization[norm](out_channels)
            if norm != "group"
            else normalization[norm](32, out_channels)
        )

        self.activation1 = nn.ReLU(inplace=True)

        if timestep is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(timestep, out_channels * 2),
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                padding=padding,
                kernel_size=kernel_size,
            ),
            normalization[norm](out_channels)
            if norm != "group"
            else normalization[norm](32, out_channels),
        )

        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                normalization[norm](out_channels)
                if norm != "group"
                else normalization[norm](32, out_channels),
            )
        else:
            self.shortcut = nn.Sequential()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, data, timestep=None):
        # Set up the shortcut connection if necessary
        identity = self.shortcut(data)

        # First convolution
        data = self.conv1(data)

        # Add timestep conditioning if provided using FiLM
        if timestep is not None:
            timestep = self.time_mlp(timestep)
            timestep = rearrange(timestep, "b c -> b c 1 1")
            # See the following link for explanation of scale, shift for timestep/class conditioning
            # https://distill.pub/2018/feature-wise-transformations/
            scale, shift = timestep.chunk(2, dim=1)
            data = data * (scale + 1) + shift

        # Add normalization and activation function after timestep conditioning
        data = self.norm1(data)
        data = self.activation1(data)

        # Second convolution
        data = self.conv2(data)

        # Add the input and run it through activation function
        data += identity
        return self.activation(data)


class ResBlockDecBasic(nn.Module):
    contraction = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        norm,
        last_layer=None,
        kernel_size=None,
    ) -> None:
        super().__init__()

        kernel_size = 3 if kernel_size is None else kernel_size
        padding = 2 if kernel_size == 5 else (3 if kernel_size == 7 else 1)

        normalization = {
            "batch": nn.BatchNorm2d,
            "instance": nn.InstanceNorm2d,
            "layer": LayerNorm,
        }

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
            normalization[norm](in_channels),
            nn.ReLU(inplace=True),
        )

        if stride > 1:
            self.conv2 = ResizeConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                norm=normalization[norm],
                activation=None,
                padding=padding,
                scale_factor=stride,
                mode="nearest",
            )

            self.shortcut = ResizeConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                norm=normalization[norm],
                activation=None,
                padding=0,
                scale_factor=stride,
                mode="nearest",
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
                normalization[norm](out_channels),
            )
            self.shortcut = nn.Sequential()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, data):
        # Setup the shortcut connection if necessary
        identity = self.shortcut(data)
        # First convolution of the data
        data = self.conv1(data)
        # Second convolution of the data
        data = self.conv2(data)
        # Connect the input data to the output of convolutions
        data += identity
        # Run it through the activation function
        return self.activation(data)


class ResBlockEncBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expansion=4,
    ) -> None:
        super().__init__()
        self.expansion = expansion

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=int(out_channels * self.expansion),
                kernel_size=1,
            ),
            nn.BatchNorm2d(int(out_channels * self.expansion)),
            nn.ReLU(inplace=True),
        )

        if stride != 1 or in_channels != int(out_channels * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(out_channels * self.expansion),
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(int(out_channels * self.expansion)),
            )
        else:
            self.shortcut = nn.Sequential()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, data):
        identity = self.shortcut(data)
        out = self.conv1(data)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return self.activation(out)


class ResBlockDecBottleneck(nn.Module):
    contraction = 4

    def __init__(
        self, in_channels, out_channels, stride, contraction=4, last_layer=False
    ) -> None:
        super().__init__()

        self.contraction = contraction

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if stride != 1:
            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    output_padding=1,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        expansion = (
            self.contraction
            if stride == 1 and not last_layer
            else (1 if last_layer else int(self.contraction / 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * expansion,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels * expansion),
            nn.ReLU(inplace=True),
        )

        if stride != 1 or last_layer:
            expansion = 1 if last_layer else int(self.contraction / 2)
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=int(out_channels * expansion),
                    kernel_size=1,
                    stride=stride,
                    output_padding=1 if stride > 1 else 0,
                ),
                nn.BatchNorm2d(int(out_channels * expansion)),
            )
        else:
            self.shortcut = nn.Sequential()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, data):
        identity = self.shortcut(data)
        out = self.conv1(data)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return self.activation(out)


def instance_norm(features, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(features, affine=True, eps=eps, **kwargs)


def layer_norm(out_channels, starting_dimension, **kwargs):
    denominator = 4 * (out_channels / 64)
    dimension = int(starting_dimension / denominator)
    return nn.LayerNorm([out_channels, dimension, dimension])


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
                    nn.ReLU(inplace=True),
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
                nn.ReLU(inplace=True),
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
                nn.ReLU(inplace=True),
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
                    nn.ReLU(inplace=True),
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
                nn.ReLU(inplace=True),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, data):
        return self.decoder(data)


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
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )

    def forward(self, data):
        return self.conv_transpose(data)
