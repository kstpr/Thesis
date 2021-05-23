from typing import Tuple
import torch.nn as nn
from torch import tensor, cat

# Network structure is strictly translated to PyTorch from the Caffe description, provided by the
# authors here - http://deep-shading-datasets.mpi-inf.mpg.de/ in section Indirect Light,
# Network definition and solver (2.0KB)
#
# We follow an approach in translating the Caffe description to PyTorch similar, but not identical
# to the one found here - https://github.com/wjw12/Deep-Shading/blob/master/GI_model.py


class UNet(nn.Module):
    def __init__(self, num_input_channels: int):
        super(UNet, self).__init__()
        self.non_linearity = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        # In the Caffe description the authors use Deconvolution layers with weights initialized by a bilinear
        # filler and learning rate set to 0 that acts per channel (num groups = num input channels). In the other
        # PyTorch source ConvTranspose2D layers are used, initialized with weights 0.25 and frozen during learning.
        # We use a standard UpsamplingBilinear2d layer from PyTorch that acts per channel too.
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Contracting part
        self.down_0 = nn.Conv2d(in_channels=num_input_channels, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.down_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, groups=2)
        self.down_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, groups=4)
        self.down_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, groups=8)

        # Bottleneck layer
        self.bottleneck = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, groups=16)

        # Expanding part
        # Input of each of the up_n layers is the concatenated output of self.upsample and (self.down_n o ReLU)
        self.up_3 = nn.Conv2d(in_channels=(256 + 128), out_channels=128, kernel_size=3, padding=1, stride=1, groups=8)
        self.up_2 = nn.Conv2d(in_channels=(128 + 64), out_channels=64, kernel_size=3, padding=1, stride=1, groups=4)
        self.up_1 = nn.Conv2d(in_channels=(64 + 32), out_channels=32, kernel_size=3, padding=1, stride=1, groups=2)
        self.up_0 = nn.Conv2d(in_channels=(32 + 16), out_channels=3, kernel_size=3, padding=1, stride=1)

        up_layers = [self.up_3, self.up_2, self.up_1, self.up_0]
        down_layers = [self.down_0, self.down_1, self.down_2, self.down_3, self.bottleneck]

        for layer in down_layers:
            nn.init.normal_(layer.weight, std=0.011)
            nn.init.constant_(layer.bias, 0.0)

        for layer in up_layers:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, input):
        t = input

        (t_0, t) = self.down_block(t, self.down_0)
        (t_1, t) = self.down_block(t, self.down_1)
        (t_2, t) = self.down_block(t, self.down_2)
        (t_3, t) = self.down_block(t, self.down_3)

        t = self.non_linearity(self.bottleneck(t))

        t = self.up_block(t, t_3, self.up_3)
        t = self.up_block(t, t_2, self.up_2)
        t = self.up_block(t, t_1, self.up_1)
        t = self.up_block_tanh(t, t_0, self.up_0) # In order for the network to return bounded values

        return t

    def down_block(self, input: tensor, down_conv_layer: nn.Module) -> Tuple[tensor, tensor]:
        t = self.non_linearity(down_conv_layer(input))
        t_cloned = t.clone()
        t = self.pooling(t)

        return (t_cloned, t)

    def up_block(self, input: tensor, skip: tensor, up_layer: nn.Module) -> tensor:
        t = cat((self.upsample(input), skip), 1)
        t = self.non_linearity(up_layer(t))
        return t
    
    def up_block_tanh(self, input: tensor, skip: tensor, up_layer: nn.Module) -> tensor:
        t = cat((self.upsample(input), skip), 1)
        t = self.tanh(up_layer(t))
        # [-1, 1] -> [0, 1]
        t = (t + 1.0) / 2.0
        return t
