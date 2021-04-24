# %%
from GIDatasetCached import GIDatasetCached
import matplotlib
from GIDataset import BufferType, GIDataset
from DeepShadingUNet import UNet
from torchsummary import summary
import torch
from PIL import Image

from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt

from timeit import default_timer as timer
import sys


def run():
    # dataset = GIDataset(
    #     root_dir="/media/ksp/424C0DBB4C0DAB2D/Thesis/Dataset/Validate/",
    #     input_buffers=[
    #         BufferType.ALBEDO,
    #         BufferType.DI,
    #         BufferType.CS_NORMALS,
    #         BufferType.WS_NORMALS,
    #         BufferType.CS_POSITIONS,
    #         BufferType.DEPTH,
    #     ],
    #     useHDR=True,
    #     resolution=512,
    # )

    dataset = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/asd/",
        input_buffers=[
            BufferType.ALBEDO,
            BufferType.DI,
            BufferType.CS_NORMALS,
            BufferType.WS_NORMALS,
            BufferType.CS_POSITIONS,
            BufferType.DEPTH,
        ],
        use_hdr=True,
        resolution=512,
    )

    def tosRGB(image_arr):
        image_arr = np.clip(image_arr, 0, 1)
        # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
        # from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
        return np.where(image_arr <= 0.0031308, 12.92 * image_arr, 1.055 * np.power(image_arr, 1 / 2.4) - 0.055)

    begin1 = timer()
    some_item = dataset.__getitem__(315)
    end1 = timer()
    ellapsed1 = end1 - begin1

    print("Time for single item: " + str(ellapsed1))

    # num_channels = some_item.shape[0]
    #
    # print(num_channels)

    device = torch.device("cuda:0")  # if (torch.cuda.is_available() and self.config.num_gpu > 0) else "cpu")
    # net = UNet(num_channels).to(device)
    # summary(net, input_size=(num_channels, 512, 512))

    # dataset.transform_and_save_dataset_as_tensors("/home/ksp/Thesis/data/GI/asd/")

    for num_workers in range(13):
        print("Init DataLoader")
        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=num_workers)
        print("Ready")

        print("Initializing iter")
        begin = timer()
        data_iter = iter(dataloader)
        print("Iter initialization took {} s".format(timer() - begin))
        print("Begin loop")
        batch_num = 0
        begin = timer()
        while batch_num < min(50, len(dataloader)):
            sys.stdout.write("#")
            sys.stdout.flush()
            data = data_iter.next()
            batch_num += 1
        print("\nAverage time for a single batch with {} workers: {} s".format(num_workers, (timer() - begin)/batch_num))

    # print(data.shape)
    # data = data.to(device)
    # for i in range(110):
    #     if i == 10:
    #         begin1 = time.time_ns()
    #     forward = net(data)
    # print("Time for forward pass: {} ms".format((time.time_ns() - begin1) / 1_000_000.0))
    # i = 0

    """ for image_arr in asd:
        i += 1
        image_arr = 
        print(image_arr.shape)
        print(image_arr)
        matplotlib.image.imsave('name' + str(i) +'.png', tosRGB(image_arr))
    """
    # print(dataset.calculate_dataset_size())


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    run()


# loss = (1.0 - SSIMLoss(x, y))/ 2.0
# %%
