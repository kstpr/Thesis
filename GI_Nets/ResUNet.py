import enum
from collections import OrderedDict
from enums import Activation, NormType
from typing import Callable, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch import cat
from torch.tensor import Tensor


# Network structure follows the description in the paper. For implementation we consult the Keras
# implementation https://github.com/nikhilroxtomar/Deep-Residual-Unet which somewhat differs than
# the paper description


def create_norm_layer(num_features: int, norm_type: NormType) -> nn.Module:
    if norm_type == NormType.BATCH:
        return nn.BatchNorm2d(num_features)
    elif norm_type == NormType.GROUP:
        num_groups = num_features // 32
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    else:
        raise Exception("Unexpected input.")


def create_conv_block(
    in_channels: int,
    out_channels: int,
    create_norm: Callable[[int], nn.Module],
    stride=1,
) -> nn.Module:
    return nn.Sequential(
        create_norm(num_features=in_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False
        ),
    )


class InitialResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, create_norm: Callable[[int], nn.Module]):
        super().__init__()
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        conv_block = create_conv_block(in_channels=out_channels, out_channels=out_channels, create_norm=create_norm)

        self.res = nn.Sequential(OrderedDict([("l0_res_conv_layer", conv_layer), ("l0_res_conv_block", conv_block)]))

        conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.skip = nn.Sequential(
            OrderedDict(
                [
                    ("l0_skip_1x1", conv_1x1),
                    ("l0_skip_bn", create_norm(num_features=out_channels)),
                ]
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.res(input) + self.skip(input)


class ResBlock(nn.Module):
    def __init__(
        self,
        layer_name: str,
        in_channels: int,
        create_norm: Callable[[int], nn.Module],
        out_channels: int = -1,
        stride=1,
    ):
        super().__init__()

        if out_channels == -1:
            out_channels = in_channels * 2

        conv_1 = create_conv_block(
            in_channels=in_channels, out_channels=out_channels, create_norm=create_norm, stride=stride
        )
        conv_2 = create_conv_block(in_channels=out_channels, out_channels=out_channels, create_norm=create_norm)
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
                    ("l{}_skip_bn".format(layer_name), create_norm(num_features=out_channels)),
                ]
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.res(input) + self.skip(input)


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, create_norm: Callable[[int], nn.Module]):
        super().__init__()
        out_channels = 2 * in_channels
        self.conv_block_1 = create_conv_block(
            in_channels=in_channels, out_channels=out_channels, create_norm=create_norm, stride=2
        )
        self.conv_block_2 = create_conv_block(
            in_channels=out_channels, out_channels=out_channels, create_norm=create_norm
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv_block_2(self.conv_block_1(input))


class ResUNet(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        final_activation: Activation,
        levels: int = 4,
        norm_type: NormType = NormType.BATCH,
    ):
        super(ResUNet, self).__init__()

        self.levels = levels

        if final_activation == Activation.SIGMOID:
            self.final_nonlinearity = nn.Sigmoid()
        elif final_activation == Activation.TANH:
            self.final_nonlinearity = nn.Tanh()
        else:
            raise Exception("Non-expected final non-linear activation!")

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        create_norm = partial(create_norm_layer, norm_type=norm_type)

        # input 512x512
        self.down_blocks = nn.ModuleList(
            [InitialResBlock(in_channels=num_input_channels, out_channels=32, create_norm=create_norm)]
        )  # 32 x 512 x 512
        for i in range(0, levels - 1):
            factor = 2 ** i
            self.down_blocks.append(
                # Reduces resoultion twice, increases number of channels twice
                ResBlock(
                    layer_name="d{}".format(i),
                    in_channels=32 * factor,
                    out_channels=64 * factor,
                    create_norm=create_norm,
                    stride=2,
                )
            )

        self.bottleneck = Bottleneck(
            in_channels=64 * (2 ** (levels - 2)), create_norm=create_norm
        )  # out channels = 2 * in channels

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            factor = 2 ** i
            self.up_blocks.append(
                # Increases resoultion twice, reduces number of channels twice
                ResBlock(
                    layer_name="u{}".format(i),
                    in_channels=64 * factor + 32 * factor,
                    out_channels=32 * factor,
                    create_norm=create_norm,
                )
            )

        self.output_1x1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def upsample_concat(self, input: Tensor, skip: Tensor) -> Tensor:
        t = self.upsample(input)
        return torch.cat((t, skip), 1)

    def forward(self, input):
        down_outputs = []
        for i, down_blk in enumerate(self.down_blocks):
            in_tensor = input if i == 0 else down_outputs[i - 1]
            down_outputs.append(down_blk(in_tensor))

        b = self.bottleneck(down_outputs[self.levels - 1])

        down_outputs.reverse()

        up_outputs = []
        for i, up_blk in enumerate(self.up_blocks):
            skip = down_outputs[i]
            in_tensor = b if i == 0 else up_outputs[i - 1]
            up_outputs.append(up_blk(self.upsample_concat(in_tensor, skip)))

        o = self.final_nonlinearity(self.output_1x1(up_outputs[-1]))

        return o
