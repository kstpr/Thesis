# %%
import sys
import random
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.utils as vutils

from torchsummary import summary
from piq import SSIMLoss

from GIDatasetCached import GIDatasetCached
from GIDataset import ALL_INPUT_BUFFERS_CANONICAL, BufferType, GIDataset
from DeepShadingUNet import UNet
from config import Config


def cache_dataset_as_tensors(dir_name: str):
    dataset = GIDataset(
        root_dir="/media/ksp/424C0DBB4C0DAB2D/Thesis/Dataset/{}/".format(dir_name),
        input_buffers=ALL_INPUT_BUFFERS_CANONICAL,
        useHDR=True,
        resolution=512,
    )
    dataset.transform_and_save_dataset_as_tensors("/home/ksp/Thesis/data/GI/PytorchTensors/{}/".format(dir_name))


def tosRGB(image_arr):
    image_arr = np.clip(image_arr, 0, 1)
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    # from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
    return np.where(image_arr <= 0.0031308, 12.92 * image_arr, 1.055 * np.power(image_arr, 1 / 2.4) - 0.055)


def scale_0_1(t: torch.Tensor):
    size = t.size()
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    if not torch.all(t == 0.0):
        t /= t.max(1, keepdim=True)[0] + 0.0000001
    t = t.view(size)


class Trainer:
    def __init__(self, config: Config, train_dataset: Dataset) -> None:
        self.init_torch()

        self.config: Config = config

        self.train_dataset = train_dataset
        self.train_dataloader: DataLoader = DataLoader(
            dataset=train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers
        )
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.config.num_gpu > 0) else "cpu")

        self.net: UNet = UNet(16).to(self.device)
        summary(self.net, input_size=(16, 512, 512))

        self.optimizer = optim.Adam(self.net.parameters())

        self.samples_indices = torch.randint(0, len(train_dataset), (10,))

    def init_torch(self) -> None:
        manualSeed = 999
        print(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def benchmark_num_workers(self, dataset):
        for num_workers in range(2, 13):
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
            while batch_num < len(dataloader):
                sys.stdout.write("#")
                sys.stdout.flush()
                data = data_iter.next()
                batch_num += 1
            epoch_time = timer() - begin
            print(
                "\nAverage time for a single batch with {} workers: {} s. Time for an entire epoch: {}".format(
                    num_workers, epoch_time / batch_num, epoch_time
                )
            )

    def structural_dissimilarity_loss(
        self, ssim_loss: nn.Module, output: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Structural Dissimilarity as described by the auhtors"""
        if (torch.any(output < 0.0)) or (torch.any(ground_truth < 0.0)):
            print("\n\n\Will EXPLODE!\n\n")
        one_minus_ssim_val: torch.Tensor = ssim_loss(output, ground_truth)

        # TODO these should be constants
        return one_minus_ssim_val / torch.full(one_minus_ssim_val.size(), 2.0)

    def train(self):
        self.save_snapshots(1)
        train_size = len(self.train_dataloader)
        loss_fn = SSIMLoss()

        for epoch_num in range(1, self.config.num_epochs + 1):
            begin_epoch = timer()
            for batch_num, (input, gt) in enumerate(self.train_dataloader):
                # Skip last incomplete batch
                if batch_num == train_size - 1:
                    continue

                begin_batch = timer()

                gt = gt.to(self.device)
                # torch.cuda.synchronize()
                # gt_to_device = timer()

                self.optimizer.zero_grad()
                # torch.cuda.synchronize()
                # opt_zero_grad = timer()

                output: torch.Tensor = self.net(input.to(self.device))
                # torch.cuda.synchronize()
                # forward_pass = timer()

                scale_0_1(output)
                scale_0_1(gt)
                # torch.cuda.synchronize()
                # scale_tensors = timer()

                loss = self.structural_dissimilarity_loss(loss_fn, output, gt)
                # torch.cuda.synchronize()
                # loss_time = timer()
                loss.backward()

                self.optimizer.step()
                # torch.cuda.synchronize()
                # back_and_step = timer()

                # if batch_num == 50 and epoch_num == 1:
                #     print(
                #         "Times:\n\tGT to device: {} s\n\tOptimizer zero grad: {} s\n\tForward pass: {} s\n\tScale tensors: {} s\n\tLoss calc: {}\n\tBackprop and step: {}".format(
                #             gt_to_device - begin_batch,
                #             opt_zero_grad - gt_to_device,
                #             forward_pass - opt_zero_grad,
                #             scale_tensors - forward_pass,
                #             loss_time - scale_tensors,
                #             back_and_step - loss_time
                #         )
                #     )

                if batch_num == train_size - 2 or batch_num % 50 == 0:
                    print("Batch {} took {} s. Loss = {}".format(batch_num, timer() - begin_batch, loss))

            print("----------------------------------------------------------------")
            print("Epoch {} took {} s.".format(epoch_num, timer() - begin_epoch))
            print("----------------------------------------------------------------")
            print()

            if (epoch_num % 5 == 0) or (epoch_num == self.config.num_epochs):
                self.save_snapshots(epoch_num)

    def save_snapshots(self, epoch_num):
        save_dir = "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepShadingBased/results/"
        image_path = save_dir + "snapshot_{}.png".format(epoch_num)

        tensors = []
        with torch.no_grad():
            for index in self.samples_indices:
                (sample_input, sample_gt) = self.train_dataset.__getitem__(index)
                di = sample_input[0:3, :]
                tensors.append(di.clone())
                sample_input = sample_input.to(self.device)
                sample_output = self.net(torch.unsqueeze(sample_input, 0)).detach().cpu()
                sample_output = sample_output.squeeze()
                tensors.append(sample_gt)
                tensors.append(sample_output)

        grid_tensor = vutils.make_grid(tensors, nrow=3).cpu()
        vutils.save_image(grid_tensor, image_path)

    def save_networks_every_nth_epoch(self, epoch):
        """Save N network snapshots during the whole training"""
        save_interval_in_epochs = int(self.config.num_epochs / 5)
        if self.config.num_epochs <= 100:
            save_interval_in_epochs = 1
         
        if epoch != 0 and epoch % save_interval_in_epochs == 0:
            torch.save(self.G.state_dict(), self.config.netowrk_snapshots_root + "netG_epoch_%d.pth" % epoch)


def run():
    # cache_dataset_as_tensors("Test")

    buffers_list = [
        BufferType.ALBEDO,
        BufferType.DI,
        BufferType.WS_NORMALS,
        BufferType.CS_NORMALS,
        BufferType.CS_POSITIONS,
        BufferType.DEPTH,
    ]

    train_dataset = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/PytorchTensors/Train/",
        input_buffers=buffers_list,
        use_hdr=True,
        resolution=512,
    )

    # benchmark_num_workers(train_dataset)

    config: Config = Config(
        num_gpu=1, num_workers=6, batch_size=8, num_epochs=50, learning_rate=0.01, use_validation=True
    )

    trainer = Trainer(config=config, train_dataset=train_dataset)
    trainer.train()

    # tensors = dataset.__getitem__(147)
    # matplotlib.image.imsave('gt.png', tosRGB(tensors[1].permute(1,2,0).numpy()))


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    run()


# loss = (1.0 - SSIMLoss(x, y))/ 2.0
# %%
