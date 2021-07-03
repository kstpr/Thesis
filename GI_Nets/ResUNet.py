import enum
from collections import OrderedDict

from numpy import pad
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
    elif norm_type == NormType.INSTANCE:
        return nn.InstanceNorm2d(num_features)
    elif norm_type == NormType.NONE:
        return nn.Identity()
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
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        ),
    )


class InitialResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        create_norm: Callable[[int], nn.Module],
    ):
        super().__init__()
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        conv_block = create_conv_block(
            in_channels=out_channels, out_channels=out_channels, create_norm=create_norm
        )

        self.res = nn.Sequential(
            OrderedDict(
                [("l0_res_conv_layer", conv_layer), ("l0_res_conv_block", conv_block)]
            )
        )

        conv_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
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
            in_channels=in_channels,
            out_channels=out_channels,
            create_norm=create_norm,
            stride=stride,
        )
        conv_2 = create_conv_block(
            in_channels=out_channels, out_channels=out_channels, create_norm=create_norm
        )
        self.res = nn.Sequential(
            OrderedDict(
                [
                    ("l{}_res_conv_1".format(layer_name), conv_1),
                    ("l{}_res_conv_2".format(layer_name), conv_2),
                ]
            )
        )
        conv_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.skip = nn.Sequential(
            OrderedDict(
                [
                    ("l{}_skip_conv_1x1".format(layer_name), conv_1x1),
                    (
                        "l{}_skip_bn".format(layer_name),
                        create_norm(num_features=out_channels),
                    ),
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
            in_channels=in_channels,
            out_channels=out_channels,
            create_norm=create_norm,
            stride=2,
        )
        self.conv_block_2 = create_conv_block(
            in_channels=out_channels, out_channels=out_channels, create_norm=create_norm
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv_block_2(self.conv_block_1(input))


# Following the official Keras implementation https://github.com/DebeshJha/ResUNetPlusPlus/blob/master/m_resunet.py
# and an unofficial PyTorch one - https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py
class SEBlock(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 8):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.W1 = nn.Linear(
            in_features=in_channels, out_features=in_channels // ratio, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.W2 = nn.Linear(
            in_features=in_channels // ratio, out_features=in_channels, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.kaiming_normal(self.W1.weight)
        torch.nn.init.kaiming_normal(self.W2.weight)

    def forward(self, input: Tensor) -> Tensor:
        b, c, _, _ = input.size()
        avgs = self.global_avg_pool(input).view(b, c)
        weights_vector = self.sigmoid(self.W2(self.relu(self.W1(avgs)))).view(
            b, c, 1, 1
        )

        return input * weights_vector.expand_as(input)


class ASPPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels,
        create_norm: Callable[[int], nn.Module],
        dilation_rates=[6, 12, 18],
    ):
        super().__init__()
        self.conv_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            create_norm(num_features=out_channels),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation_rates[0],
                dilation=dilation_rates[0],
            ),
            create_norm(num_features=out_channels),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation_rates[1],
                dilation=dilation_rates[1],
            ),
            create_norm(num_features=out_channels),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation_rates[2],
                dilation=dilation_rates[2],
            ),
            create_norm(num_features=out_channels),
        )

        self.conv_1x1 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv_1x1(
            self.conv_0(input)
            + self.conv_1(input)
            + self.conv_2(input)
            + self.conv_3(input)
        )


class AttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels_decoder: int,
        in_channels_skip: int,
        out_channles: int,
        create_norm: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.block_skip = nn.Sequential(
            create_norm(num_features=in_channels_skip),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels_skip,
                out_channels=out_channles,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block_decoder = nn.Sequential(
            create_norm(num_features=in_channels_decoder),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels_decoder,
                out_channels=out_channles,
                kernel_size=3,
                padding=1,
            ),
        )

        self.block_add = nn.Sequential(
            create_norm(num_features=out_channles),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channles,
                out_channels=out_channles,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, input_decoder: Tensor, input_skip: Tensor) -> Tensor:
        addition_res = self.block_skip(input_skip) + self.block_decoder(input_decoder)
        return input_decoder * self.block_add(addition_res)


class AttnUpsampleConcatBlock(nn.Module):
    def __init__(
        self,
        in_channels_decoder: int,
        in_channels_skip: int,
        out_channles_attn: int,
        create_norm: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.attn = AttentionBlock(
            in_channels_decoder=in_channels_decoder,
            in_channels_skip=in_channels_skip,
            out_channles=out_channles_attn,
            create_norm=create_norm,
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input_decoder: Tensor, input_skip: Tensor) -> Tensor:
        attn_out = self.attn(input_decoder, input_skip)
        upsampled = self.upsample(attn_out)
        return torch.cat((upsampled, input_skip), 1)


class ResUNetPlusPlus(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        final_activation: Activation,
        levels: int = 4,
        norm_type: NormType = NormType.BATCH,
    ):
        super(ResUNetPlusPlus, self).__init__()

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
            [
                InitialResBlock(
                    in_channels=num_input_channels,
                    out_channels=16,
                    create_norm=create_norm,
                )
            ]
        )  # output 16 x 512 x 512

        for i in range(0, levels - 1):
            factor = 2 ** i
            self.down_blocks.append(
                # Reduces resoultion twice, increases number of channels twice
                nn.Sequential(
                    SEBlock(in_channels=16 * factor),
                    ResBlock(
                        layer_name="d{}".format(i),
                        in_channels=16 * factor,
                        out_channels=32 * factor,
                        create_norm=create_norm,
                        stride=2,
                    ),
                )
            )

        self.bottleneck = nn.Sequential(
            ResBlock(
                layer_name="BottleneckRes",
                in_channels=32 * (2 ** (levels - 2)),
                out_channels=64 * (2 ** (levels - 2)),
                create_norm=create_norm,
                stride=2,
            ),
            ASPPBlock(
                in_channels=64 * (2 ** (levels - 2)),
                out_channels=128 * (2 ** (levels - 2)),
                create_norm=create_norm
            ),
        )

        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()

        for i in reversed(range(levels)):
            factor = 2 ** i
            self.up_attentions.append(
                AttnUpsampleConcatBlock(
                    in_channels_decoder=64 * factor,
                    in_channels_skip=16 * factor,
                    out_channles_attn=64 * factor,
                    create_norm=create_norm,
                )
            )
            self.up_blocks.append(
                ResBlock(
                    layer_name="u{}".format(i),
                    in_channels=64 * factor + 16 * factor,
                    out_channels=32 * factor,
                    create_norm=create_norm,
                )
            )

        self.aspp_output = ASPPBlock(
            in_channels=32, out_channels=16, create_norm=create_norm
        )
        self.output_1x1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)

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
            up_outputs.append(up_blk(self.up_attentions[i](in_tensor, skip)))

        o = self.final_nonlinearity(self.output_1x1(self.aspp_output(up_outputs[-1])))

        return o


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
            [
                InitialResBlock(
                    in_channels=num_input_channels,
                    out_channels=32,
                    create_norm=create_norm,
                )
            ]
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
