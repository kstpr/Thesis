# %%
import matplotlib
from GIDataset import BufferType, GIDataset
from DeepShadingUNet import UNet
from torchsummary import summary
import torch
from PIL import Image

from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt

import time


def run():
    dataset = GIDataset(
        root_dir="/media/ksp/424C0DBB4C0DAB2D/Thesis/Dataset/Validate/",
        input_buffers=[
            BufferType.ALBEDO,
            BufferType.DI,
            BufferType.CS_NORMALS,
            BufferType.WS_NORMALS,
            BufferType.CS_POSITIONS,
            BufferType.DEPTH,
            BufferType.GT_RTGI,
        ],
        useHDR=True,
        resolution=512,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    data_iter = iter(dataloader)

    def tosRGB(image_arr):
        image_arr = np.clip(image_arr, 0, 1)
        # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
        # from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
        return np.where(image_arr <= 0.0031308, 12.92 * image_arr, 1.055 * np.power(image_arr, 1 / 2.4) - 0.055)

    begin1 = time.time()
    some_item = dataset.__getitem__(315)
    end1 = time.time()
    ellapsed1 = end1 - begin1

    print("Time for single image: " + str(ellapsed1))

    num_channels = some_item.shape[0]
    print(num_channels)

    device = torch.device("cuda:0")  # if (torch.cuda.is_available() and self.config.num_gpu > 0) else "cpu")
    net = UNet(num_channels).to(device)
    summary(net, input_size=(num_channels, 512, 512))

    begin = time.time()
    data = data_iter.next()
    end = time.time()

    ellapsed = end - begin

    print("Time for batch: " + str(ellapsed))
    print(data.shape)
    data = data.to(device)

    print(net(data))

    # i = 0
    """ for image_arr in asd:
        i += 1
        image_arr = 
        print(image_arr.shape)
        print(image_arr)

        matplotlib.image.imsave('name' + str(i) +'.png', tosRGB(image_arr))
        plt.imshow(tosRGB(image_arr))
        plt.show() """

    print(dataset.calculate_dataset_size())


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    run()


# loss = (1.0 - SSIMLoss(x, y))/ 2.0
# %%
