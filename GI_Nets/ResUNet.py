import enum
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from torch import cat
from torch.tensor import Tensor


# Network structure follows the description in the paper. For implementation we consult the Keras
# implementation https://github.com/nikhilroxtomar/Deep-Residual-Unet which somewhat differs than
# the paper description


class Activation(enum.Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    LRELU = 4

def create_norm_block(num_features: int):
    num_groups = num_features / 32
    return nn.GroupNorm(num_channels=num_features)

def create_conv_block(in_channels: int, out_channels: int, stride=1) -> nn.Module:
    return nn.Sequential(
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False
        ),
    )


class InitialResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        conv_block = create_conv_block(in_channels=out_channels, out_channels=out_channels)

        self.res = nn.Sequential(OrderedDict([("l0_res_conv_layer", conv_layer), ("l0_res_conv_block", conv_block)]))

        conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.skip = nn.Sequential(
            OrderedDict(
                [
                    ("l0_skip_1x1", conv_1x1),
                    ("l0_skip_bn", nn.BatchNorm2d(num_features=out_channels)),
                ]
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.res(input) + self.skip(input)


class ResBlock(nn.Module):
    def __init__(self, layer_name: str, in_channels: int, out_channels: int = -1, stride=1):
        super().__init__()

        if out_channels == -1:
            out_channels = in_channels * 2

        conv_1 = create_conv_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        conv_2 = create_conv_block(in_channels=out_channels, out_channels=out_channels)
        self.res = nn.Sequential(
            OrderedDict(
                [
                    ("l{}_res_conv_1".format(layer_name), conv_1),
                    ("l{}_res_conv_2".format(layer_name), conv_2),
                ]
            )
        )
        conv_1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False
        )
        self.skip = nn.Sequential(
            OrderedDict(
                [
                    ("l{}_skip_conv_1x1".format(layer_name), conv_1x1),
                    ("l{}_skip_bn".format(layer_name), nn.BatchNorm2d(num_features=out_channels)),
                ]
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.res(input) + self.skip(input)


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        out_channels = 2 * in_channels
        self.conv_block_1 = create_conv_block(in_channels=in_channels, out_channels=out_channels, stride=2)
        self.conv_block_2 = create_conv_block(in_channels=out_channels, out_channels=out_channels)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv_block_2(self.conv_block_1(input))


class ResUNet(nn.Module):
    def __init__(self, num_input_channels: int, final_activation: Activation):
        super(ResUNet, self).__init__()

        self.non_linearity = nn.LeakyReLU(negative_slope=0.01)

        if final_activation == Activation.SIGMOID:
            self.final_nonlinearity = nn.Sigmoid()
        elif final_activation == Activation.TANH:
            self.final_nonlinearity = nn.Tanh()
        else:
            raise Exception("Non-expected final non-linear activation!")

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Contracting part

        # Initial block parts
        # input 512x512
        self.down_0 = InitialResBlock(in_channels=num_input_channels, out_channels=32)  # 32 x 512 x 512
        self.down_1 = ResBlock(layer_name="d1", in_channels=32, out_channels=64, stride=2)  # 64 x 256 x 256
        self.down_2 = ResBlock(layer_name="d2", in_channels=64, out_channels=128, stride=2)  # 128 x 128 x 128
        self.down_3 = ResBlock(layer_name="d3", in_channels=128, out_channels=256, stride=2)  # 256 x 64 x 64

        self.bottleneck = Bottleneck(in_channels=256)  # 512 x 32 x 32

        self.up_3 = ResBlock(layer_name="u3", in_channels=512 + 256, out_channels=256)  # 256 x 64 x 64
        self.up_2 = ResBlock(layer_name="u2", in_channels=256 + 128, out_channels=128)  # 128 x 128 x 128
        self.up_1 = ResBlock(layer_name="u1", in_channels=128 + 64, out_channels=64)  # 64 x 256 x 256
        self.up_0 = ResBlock(layer_name="u0", in_channels=64 + 32, out_channels=32)  # 32 x 512 x 512

        self.output_1x1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

        # for layer in down_layers:
        #     nn.init.normal_(layer.weight, std=0.011)
        #     nn.init.constant_(layer.bias, 0.0)

        # for layer in up_layers:
        #     nn.init.normal_(layer.weight, std=0.01)
        #     nn.init.constant_(layer.bias, 0.0)

    def upsample_concat(self, input: Tensor, skip: Tensor) -> Tensor:
        t = self.upsample(input)
        return torch.cat((t, skip), 1)

    def forward(self, input):
        d_0 = self.down_0(input)
        d_1 = self.down_1(d_0)
        d_2 = self.down_2(d_1)
        d_3 = self.down_3(d_2)

        b = self.bottleneck(d_3)

        u_3 = self.up_3(self.upsample_concat(b, d_3))
        u_2 = self.up_2(self.upsample_concat(u_3, d_2))
        u_1 = self.up_1(self.upsample_concat(u_2, d_1))
        u_0 = self.up_0(self.upsample_concat(u_1, d_0))

        o = self.final_nonlinearity(self.output_1x1(u_0))

        return o
