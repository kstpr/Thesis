import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_gpu, latent_size, feat_maps_size, num_channels, image_size):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        base = nn.Sequential(
            # second
            nn.ConvTranspose2d(feat_maps_size * 8, feat_maps_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 4),
            nn.ReLU(True),
            # third
            nn.ConvTranspose2d(feat_maps_size * 4, feat_maps_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 2),
            nn.ReLU(True),
            # fourth
            nn.ConvTranspose2d(feat_maps_size * 2, feat_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size),
            nn.ReLU(True),
            ## last
            nn.ConvTranspose2d(feat_maps_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        if image_size == 64:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_size, feat_maps_size * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feat_maps_size * 8),
                nn.ReLU(True),
                base,
            )
        elif image_size == 128:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_size, feat_maps_size * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feat_maps_size * 16),
                nn.ReLU(True),
                #
                nn.ConvTranspose2d(feat_maps_size * 16, feat_maps_size * 8, 4, 2, 1, bias=False),  # 128
                nn.BatchNorm2d(feat_maps_size * 8),
                nn.ReLU(True),
                base,
            )

    def forward(self, input):
        return self.main(input)
