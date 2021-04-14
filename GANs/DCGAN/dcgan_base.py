# We follow the DCGAN tutorial from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Main change is refactoring into a class as a base for pipeline, saving of intermediate results and
# visualizing - kpresnakov

from __future__ import print_function

from abc import ABC, abstractmethod

#%matplotlib inline
import random
from timeit import default_timer as timer

import torch

import torch.nn as nn
import torch.nn.parallel

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ColorJitter, RandomHorizontalFlip, RandomRotation
import torchvision.utils as vutils

from torch.cuda.amp import autocast, GradScaler

from torchsummary import summary

from IPython.display import HTML

from config import Config
from discriminator import Discriminator
from generator import Generator
import util.visualization as viz_util
import util.logging as log_util


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN_base(ABC):
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.init_torch()

        self.use_amp = False

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.config.num_gpu > 0) else "cpu")

        self.init_dataset_and_loader()
        self.init_validation_dataset_and_loader()

        self.G = self.init_generator()
        self.D = self.init_discriminator()

        self.init_loss_and_optimizer()
        self.init_lr_decay_deltas()

        # Training data
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.D_x_vals = []
        self.G_D_z_vals = []

        # Validation data
        self.D_x_validation_vals = []
        self.D_losses_validation = []

    def init_torch(self) -> None:
        manualSeed = 999
        print(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def init_dataset_and_loader(self):
        dataset = dset.ImageFolder(
            root=self.config.dataroot.dataset_root,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.config.image_size),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        self.dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
        )

    def init_validation_dataset_and_loader(self):
        validation_dataset = dset.ImageFolder(
            root=self.config.dataroot.validation_dataset_root,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.config.image_size),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        self.validation_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
        )

    def init_dataset_and_loader_transformed(self):
        dataset = dset.ImageFolder(
            root=self.config.dataroot.dataset_root,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.config.image_size),
                    # transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        self.dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
        )

    def init_generator(self) -> Generator:
        generator = Generator(
            num_gpu=self.config.num_gpu,
            latent_size=self.config.latent_size,
            feat_maps_size=self.config.g_feat_maps,
            num_channels=self.config.num_channels,
            image_size=self.config.image_size,
        ).to(self.device)

        generator.apply(weights_init_normal)

        print(generator)
        print("\n")
        summary(generator, input_size=(self.config.latent_size, 1, 1))

        return generator

    @abstractmethod
    def init_discriminator(self) -> Discriminator:
        discriminator = Discriminator(
            num_gpu=self.config.num_gpu,
            num_channels=self.config.num_channels,
            feat_maps_size=self.config.d_feat_maps,
            image_size=self.config.image_size,
            use_sigmoid=self.use_amp,
        ).to(self.device)

        discriminator.apply(weights_init_normal)

        print(discriminator)
        print("\n")
        summary(discriminator, input_size=(self.config.num_channels, self.config.image_size, self.config.image_size))

        return discriminator

    @abstractmethod
    def init_loss_and_optimizer(self) -> None:
        pass

    def init_lr_decay_deltas(self) -> None:
        if self.config.lr_linear_decay_enabled:
            self.lr_g_decay_delta = self.config.g_learning_rate / (
                self.config.num_epochs - self.config.g_lr_decay_start_epoch
            )
            self.lr_d_decay_delta = self.config.d_learning_rate / (
                self.config.num_epochs - self.config.d_lr_decay_start_epoch
            )

    def train_and_plot(self):
        self.train()
        self.plot_results()

    @abstractmethod
    def train(self):
        pass

    def update_learning_rate_linear_decay(self, current_epoch):
        if current_epoch < self.config.d_lr_decay_start_epoch:
            return

        for param_group in self.optimizerD.param_groups:
            param_group["lr"] -= self.lr_d_decay_delta
        for param_group in self.optimizerG.param_groups:
            param_group["lr"] -= self.lr_g_decay_delta

    ################################ Visualization and logging ################################

    @abstractmethod
    def plot_results(self) -> None:
        pass

    def persist_training_data(self, loss_G, loss_D, D_x, D_G_z) -> None:
        self.G_losses.append(loss_G.item())
        self.D_losses.append(loss_D.item())
        self.D_x_vals.append(D_x)
        self.G_D_z_vals.append(D_G_z)

    def persist_validation_data(self, loss_D_val, D_x_val) -> None:
        self.D_losses_validation.append(loss_D_val.item())
        self.D_x_validation_vals.append(D_x_val)

    def save_fakes_snapshot_every_nth_epoch(self, epoch, batch_num):
        output_name = self.config.intermediates_root + "epoch%d.png" % (epoch)

        if (epoch % 5 == 0) and (batch_num >= len(self.dataloader) - 1):
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                viz_util.save_current_fakes_snapshot(fake, epoch, False, self.device, output_name)

        if (epoch == self.config.num_epochs - 1) and (batch_num >= len(self.dataloader) - 1):
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                viz_util.save_current_fakes_snapshot(fake, epoch, True, self.device, output_name)

    def save_networks_every_nth_epoch(self, epoch):
        n = 50
        if epoch != 0 and epoch % n == 0:
            torch.save(self.G.state_dict(), self.config.netowrk_snapshots_root + "netG_epoch_%d.pth" % epoch)
            torch.save(self.D.state_dict(), self.config.netowrk_snapshots_root + "netD_epoch_%d.pth" % epoch)

    # Generates and saves 2048 fake images for running the FID score script
    def generate_fake_results(self):
        noise_batch_size = self.config.batch_size
        num_noise_batches = 2048 // noise_batch_size

        print("Generating random fakes: %d batches with size %d" % (num_noise_batches, noise_batch_size))
        noise_batches = torch.randn(
            num_noise_batches, noise_batch_size, self.config.latent_size, 1, 1, device=self.device
        )

        with torch.no_grad():
            for i, noise_batch in enumerate(noise_batches):
                generated_set = self.G(noise_batch).detach().cpu()
                for j, generated_img in enumerate(generated_set):
                    vutils.save_image(
                        generated_img,
                        self.config.output_root + "fake_%d.png" % (i * noise_batch_size + j),
                        normalize=True,
                        padding=0,
                    )