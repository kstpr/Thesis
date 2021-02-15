# We follow the DCGAN tutorial from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Main change is refactoring into a class as a base for pipeline, saving of intermediate results and
# visualizing - kpresnakov

from __future__ import print_function

#%matplotlib inline
import random
from timeit import default_timer as timer

# mport wandb

import torch

import torch.nn as nn
import torch.nn.parallel

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML

from config import Config
from discriminator import Discriminator
from generator import Generator


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN:
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.init_torch()

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.config.num_gpu > 0) else "cpu")

        self.init_dataset_and_loader()

        self.G = Generator(
            num_gpu=self.config.num_gpu,
            latent_size=self.config.latent_size,
            feat_maps_size=self.config.g_feat_maps,
            num_channels=self.config.num_channels,
        ).to(self.device)

        self.G.apply(weights_init_normal)
        print(self.G)

        self.D = Discriminator(
            num_gpu=self.config.num_gpu,
            num_channels=self.config.num_channels,
            feat_maps_size=self.config.d_feat_maps,
        ).to(self.device)

        self.D.apply(weights_init_normal)
        print(self.D)

        self.init_loss_and_optimizer()

    def init_torch(self) -> None:
        manualSeed = 999
        print(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def init_dataset_and_loader(self):
        dataset = dset.ImageFolder(
            root=self.config.dataroot,
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

    def init_loss_and_optimizer(self) -> None:
        self.criterion: nn.BCELoss = nn.BCELoss()

        self.fixed_noise = torch.randn(64, self.config.latent_size, 1, 1, device=self.device)

        self.optimizerD = optim.Adam(
            self.D.parameters(),
            lr=self.config.d_learning_rate,
            betas=(self.config.d_beta_1, self.config.d_beta_2),
        )
        self.optimizerG = optim.Adam(
            self.G.parameters(),
            lr=self.config.g_learning_rate,
            betas=(self.config.g_beta_1, self.config.d_beta_2),
        )

    def plot_training_examples(self):
        real_batch = next(iter(self.dataloader))
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),
                (1, 2, 0),
            )
        )
        plt.close(fig)

    def train_and_plot(self):
        self.plot_training_examples()
        self.train()
        self.plot_losses()

    def train(self):
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        real_label = 1
        fake_label = 0

        print("Starting training loop...")
        start_training = timer()
        for epoch in range(self.config.num_epochs):
            start_epoch = timer()
            for batch_num, data in enumerate(self.dataloader, 0):
                self.train_batch(real_label, fake_label, epoch, batch_num, data)
                self.plot_fakes_sample(epoch=epoch, batch_num=batch_num)
                iters += 1
            end_epoch = timer()
            print("Epoch %d took %.4fs." % (epoch, end_epoch - start_epoch))
        end_training = timer()
        print("Training for %d took %.4fs." % (self.config.num_epochs, end_training - start_training))

    def train_batch(self, real_label, fake_label, epoch, i, data):
        # Update D - max log(D(x)) + log(1 - D(G(z))
        ############################################
        # Train with real batch
        self.D.zero_grad()
        real = data[0].to(self.device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float32, device=self.device)

        # Forward pass real batch through D
        output = self.D(real).view(-1)
        # Loss on real batch
        errD_real = self.criterion(output, label)

        # Calculate gradients for D
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake batch
        noise = torch.randn(b_size, self.config.latent_size, 1, 1, device=self.device)

        # Generate fake batch using G
        fake = self.G(noise)
        label.fill_(fake_label)

        # Forward pass fake batch through D
        output = self.D(fake.detach()).view(-1)

        # Loss on fake batch
        errD_fake = self.criterion(output, label)

        # Calculate gradients for D
        errD_fake.backward()

        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake  # how

        self.optimizerD.step()

        # Update G - max log(D(G(z)))
        #############################
        self.G.zero_grad()

        label.fill_(real_label)
        output = self.D(fake).view(-1)

        errG = self.criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()

        # Update G
        self.optimizerG.step()

        self.log_batch_stats(epoch, i, D_x, D_G_z1, errD, errG, D_G_z2)

    def log_batch_stats(self, epoch, i, D_x, D_G_z1, errD, errG, D_G_z2):
        if i % 50 == 0:
            print(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                % (
                    epoch,
                    self.config.num_epochs,
                    i,
                    len(self.dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )

        self.G_losses.append(errG.item())
        self.D_losses.append(errD.item())

    def plot_fakes_sample(self, epoch, batch_num):
        if (epoch % 5 == 0) and (batch_num == len(self.dataloader) - 1):
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                self.plot_current_fake(fake, epoch, False)

        if (epoch == self.config.num_epochs - 1) and (batch_num == len(self.dataloader) - 1):
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                self.plot_current_fake(fake, epoch, True)

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 5))
        plt.title("G and D loss during trainning")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.close(fig)

    def plot_current_fake(self, fake_batch, epoch, isFinal):
        fig = plt.figure(figsize=(16, 16))
        plt.axis("off")
        plt.title("Images in" + ("final epoch" if isFinal else "epoch %d" % (epoch)))
        plt.imshow(
            np.transpose(
                vutils.make_grid(fake_batch.to(self.device)[:64], padding=2, normalize=True).cpu(),
                (1, 2, 0),
            )
        )
        plt.savefig(self.config.intermediates_root + "epoch%d.png" % (epoch))
        plt.close(fig)
        print("Figure saved.")

    def plot_results_animation(self):
        import matplotlib

        matplotlib.rcParams["animation.embed_limit"] = 64

        fig = plt.figure(figsize=(12, 12))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

    def generate_fake_results(self):
        self.fixed_noise = torch.randn(2048, self.config.latent_size, 1, 1, device=self.device)

        with torch.no_grad():
            generated_set = self.G(self.fixed_noise).detach().cpu()
            for i, generated_img in enumerate(generated_set):
                vutils.save_image(
                    generated_img,
                    self.config.output_root + "fake_%d.png" % (i),
                    normalize=True,
                    padding=0,
                )
