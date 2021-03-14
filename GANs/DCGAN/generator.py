import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_gpu, latent_size, feat_maps_size, num_channels, image_size):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        base = nn.Sequential(
            self.inner_block(feat_maps_size * 4),
            self.inner_block(feat_maps_size * 2),
            self.inner_block(feat_maps_size),
            self.output_block(in_channels=feat_maps_size, out_channels=num_channels),
        )
        if image_size == 64:
            self.main = nn.Sequential(
                self.input_block(latent_size, feat_maps_size * 8),
                base,
            )
        elif image_size == 128:
            self.main = nn.Sequential(
                self.input_block(latent_size, feat_maps_size * 16),
                self.inner_block(out_channels=feat_maps_size * 8),
                base,
            )
        elif image_size == 256:
            self.main = nn.Sequential(
                self.input_block(latent_size, feat_maps_size * 32),
                self.inner_block(out_channels=feat_maps_size * 16),
                self.inner_block(out_channels=feat_maps_size * 8),
                base,
            )

    def input_block(self, latent_size, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(latent_size, out_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    # in_channels = out_channels * 2
    def inner_block(self, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(out_channels * 2, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def output_block(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        return self.main(input)
