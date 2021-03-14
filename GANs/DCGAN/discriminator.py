import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_gpu: int, num_channels: int, feat_maps_size: int, image_size: int, use_amp: bool):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.use_amp = use_amp

        base = nn.Sequential(
            self.input_block(in_channels=num_channels, out_channels=feat_maps_size),
            self.inner_block(feat_maps_size),
            self.inner_block(feat_maps_size * 2),
            self.inner_block(feat_maps_size * 4),
        )

        # TODO - depending on the image size dynamically build the rest of the network
        if image_size == 64:
            self.main = nn.Sequential(base, self.output_block(feat_maps_size * 8))
        elif image_size == 128:
            self.main = nn.Sequential(
                base, self.inner_block(feat_maps_size * 8), self.output_block(feat_maps_size * 16)
            )
        elif image_size == 256:
            self.main = nn.Sequential(
                base,
                self.inner_block(feat_maps_size * 8),
                self.inner_block(feat_maps_size * 16),
                self.output_block(feat_maps_size * 32),
            )

    def input_block(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    # out_channels = in_channels * 2
    def inner_block(self, in_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def output_block(self, in_channels) -> nn.Module:
        # If using mixed precission the sigmoid is concatenated by default with the BCELossWithLogits
        # that is compatible with amp. Otherwise the standard incompatible BCELoss is used.
        return nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False) if self.use_amp else  nn.Sequential(
            nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        ) 

    def forward(self, input):
        return self.main(input)
