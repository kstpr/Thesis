import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_gpu, num_channels, feat_maps_size, image_size):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        base = nn.Sequential(
            # L1
            nn.Conv2d(num_channels, feat_maps_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # L2
            nn.Conv2d(feat_maps_size, feat_maps_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # L3
            nn.Conv2d(feat_maps_size * 2, feat_maps_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # L4
            nn.Conv2d(feat_maps_size * 4, feat_maps_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if image_size == 64:
            self.main = nn.Sequential(
                base,
                # Last
                nn.Conv2d(feat_maps_size * 8, 1, 4, 1, 0, bias=False),  # 64
                nn.Sigmoid(),
            )
        elif image_size == 128:
            self.main = nn.Sequential(
                base,
                # L5 Last inner block for resulution 128 x 128
                nn.Conv2d(feat_maps_size * 8, feat_maps_size * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feat_maps_size * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # Last
                nn.Conv2d(feat_maps_size * 16, 1, 4, 1, 0, bias=False),  # 128
                nn.Sigmoid(),
            )

    def forward(self, input):
        return self.main(input)
