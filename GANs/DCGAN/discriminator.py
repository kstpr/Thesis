import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_gpu: int, num_channels: int, feat_maps_size: int, image_size: int, use_sigmoid: bool):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.use_sigmoid = use_sigmoid
        self.is_wgan = False

        self.main = nn.Sequential()
        self.add_input_block(main=self.main, in_channels=num_channels, out_channels=feat_maps_size)
        self.add_inner_block(main=self.main, num=1, in_channels=feat_maps_size)
        self.add_inner_block(main=self.main, num=2, in_channels=feat_maps_size * 2)
        self.add_inner_block(main=self.main, num=3, in_channels=feat_maps_size * 4)

        # TODO - depending on the image size dynamically build the rest of the network
        if image_size == 64:
            self.add_output_block(main=self.main, num=4, in_channels=feat_maps_size * 8)
        elif image_size == 128:
            self.add_inner_block(main=self.main, num=4, in_channels=feat_maps_size * 8)
            self.add_output_block(main=self.main, num=5, in_channels=feat_maps_size * 16)
        elif image_size == 256:
            self.add_inner_block(main=self.main, num=4, in_channels=feat_maps_size * 8),
            self.add_inner_block(main=self.main, num=5, in_channels=feat_maps_size * 16),
            self.add_output_block(main=self.main, num=6, in_channels=feat_maps_size * 32),


    def add_input_block(self, main: nn.Module, in_channels: int, out_channels: int) -> None:
        main.add_module("Conv-0", nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
        main.add_module("LReLU-0", nn.LeakyReLU(0.2, inplace=True))

    # out_channels = in_channels * 2
    def add_inner_block(self, main: nn.Module, num: int, in_channels: int) -> None:
        main.add_module("Conv-%d" % (num,), nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False))
        main.add_module("BN-%d" % (num,), nn.BatchNorm2d(in_channels * 2))
        main.add_module("LRelU-%d" % (num,), nn.LeakyReLU(0.2, inplace=True))

    def add_output_block(self, main: nn.Module, num: int, in_channels: int) -> None:
        # If using mixed precission the sigmoid is concatenated by default with the BCELossWithLogits
        # that is compatible with amp. Otherwise the standard incompatible BCELoss is used.
        # For WGAN the sigmoid is also not used.
        main.add_module("Conv-%d" % (num,), nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False))
        if self.use_sigmoid:
            main.add_module("Sigmoid-%d" % (num,), nn.Sigmoid())

    def forward(self, input):
        return self.main(input)
