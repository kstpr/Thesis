import torch
import torch.nn as nn
from torch import cat
from torch.tensor import Tensor


# Network structure follows the description in the paper. For implementation we consult the Keras
# implementation https://github.com/nikhilroxtomar/Deep-Residual-Unet which somewhat differs than
# the paper description


class ResUNet(nn.Module):
    def __init__(self, num_input_channels: int):
        super(ResUNet, self).__init__()

        self.non_linearity = nn.LeakyReLU(negative_slope=0.01)
        self.final_nonlinearity = nn.Sigmoid()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        # Contracting part

        # Initial block parts
        # input 512x512
        self.initial_conv_layer = nn.Conv2d(in_channels=num_input_channels, out_channels=32, kernel_size=3, padding=1)
        self.initial_conv_block = self.create_conv_block(in_channels=32, out_channels=32)
        self.initial_1x1 = nn.Conv2d(in_channels=num_input_channels, out_channels=32, kernel_size=1)
        self.initial_bn = nn.BatchNorm2d(num_features=32)

        self.res_down_modules_1: nn.ModuleDict = self.create_res_block_modules(in_channels=32, out_channels=64, stride=2)
        self.res_down_modules_2: nn.ModuleDict = self.create_res_block_modules(in_channels=64, out_channels=128, stride=2)

        # self.bottleneck_conv_block_1 = self.create_conv_block(in_channels=128, out_channels=256, stride=2)
        # self.bottleneck_conv_block_2 = self.create_conv_block(in_channels=256, out_channels=256)

        self.bottleneck_conv_block_1 = self.create_conv_block(in_channels=64, out_channels=128, stride=2)
        self.bottleneck_conv_block_2 = self.create_conv_block(in_channels=128, out_channels=128)

        self.res_up_modules_3: nn.ModuleDict = self.create_res_block_modules(in_channels=128 + 256, out_channels=128)
        self.res_up_modules_2: nn.ModuleDict = self.create_res_block_modules(in_channels=64 + 128, out_channels=64)
        self.res_up_modules_1: nn.ModuleDict = self.create_res_block_modules(in_channels=32 + 64, out_channels=32)

        self.output_1x1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

        # for layer in down_layers:
        #     nn.init.normal_(layer.weight, std=0.011)
        #     nn.init.constant_(layer.bias, 0.0)

        # for layer in up_layers:
        #     nn.init.normal_(layer.weight, std=0.01)
        #     nn.init.constant_(layer.bias, 0.0)

    def forward_initial_res_block(self, input: Tensor) -> Tensor:
        t = self.initial_conv_layer(input)
        t = self.initial_conv_block(t)

        shortcut = self.initial_1x1(input)
        shortcut = self.initial_bn(shortcut)

        return t + shortcut

    def forward_res_block(self, input: Tensor, res_block_modules: nn.ModuleDict) -> Tensor:
        t = res_block_modules["conv_1"](input)
        t = res_block_modules["conv_2"](t)

        shortcut = res_block_modules["bn"](self.non_linearity(res_block_modules["conv_1x1"](input)))

        return t + shortcut

    def forward_bottleneck(self, input: Tensor) -> Tensor:
        t = self.bottleneck_conv_block_1(input)
        t = self.bottleneck_conv_block_2(t)

        return t

    def forward_upsample_concat(self, input: Tensor, skip: Tensor) -> Tensor:
        t = self.upsample(input)
        return torch.cat((t, skip), 1)

    def forward(self, input):
        e_1 = self.forward_initial_res_block(input) # 512

        e_2 = self.forward_res_block(e_1, self.res_down_modules_1) # 256
        #e_3 = self.forward_res_block(e_2, self.res_down_modules_2) # 128

        b = self.forward_bottleneck(e_2) # 64

        #u_3 = self.forward_res_block(self.forward_upsample_concat(b, e_3), self.res_up_modules_3) # 128
        u_2 = self.forward_res_block(self.forward_upsample_concat(b, e_2), self.res_up_modules_2) # 256
        u_1 = self.forward_res_block(self.forward_upsample_concat(u_2, e_1), self.res_up_modules_1)

        o = self.final_nonlinearity(self.output_1x1(u_1))

        return o

    def create_conv_block(self, in_channels: int, out_channels: int, stride=1) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
        )

    def create_res_block_modules(self, in_channels: int, out_channels: int = -1, stride=1) -> nn.ModuleDict:
        if out_channels == -1:
            out_channels = in_channels * 2
        return nn.ModuleDict(
            {
                "conv_1": self.create_conv_block(in_channels=in_channels, out_channels=out_channels, stride=stride),
                "conv_2": self.create_conv_block(in_channels=out_channels, out_channels=out_channels),
                "conv_1x1": nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                "bn": nn.BatchNorm2d(num_features=out_channels),
            }
        )
