# We follow the DCGAN tutorial from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Main change is refactoring into a class as a base for pipeline, saving of intermediate results and
# visualizing - kpresnakov

from __future__ import print_function

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


class DCGAN:
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.init_torch()

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
            image_size=self.config.image_size
        ).to(self.device)

        generator.apply(weights_init_normal)

        print(generator)
        print('\n')
        summary(generator, input_size=(self.config.latent_size, 1, 1))

        return generator

    def init_discriminator(self) -> Discriminator:
        discriminator = Discriminator(
            num_gpu=self.config.num_gpu,
            num_channels=self.config.num_channels,
            feat_maps_size=self.config.d_feat_maps,
            image_size=self.config.image_size
        ).to(self.device)

        discriminator.apply(weights_init_normal)

        print(discriminator)
        print('\n')
        summary(discriminator, input_size=(self.config.num_channels, self.config.image_size, self.config.image_size))

        return discriminator

    def init_loss_and_optimizer(self) -> None:
        self.loss: nn.BCELoss = nn.BCELoss()

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

    def train(self):
        real_label = 1
        real_label_smoothed = 0.9
        fake_label = 0

        print("Starting training loop...")
        start_training = timer()
        for epoch in range(self.config.num_epochs):
            start_epoch = timer()
            for batch_num, data in enumerate(self.dataloader, 0):
                self.train_batch(real_label, real_label_smoothed, fake_label, epoch, batch_num, data)
                self.save_fakes_snapshot_every_nth_epoch(epoch=epoch, batch_num=batch_num)
            if self.config.use_validation:
                for val_batch_num, val_data in enumerate(self.validation_dataloader, 0):
                    self.validate_batch(real_label, fake_label, epoch, val_batch_num, val_data)
            if self.config.lr_linear_decay_enabled:
                self.update_learning_rate_linear_decay(epoch)
            print("Epoch %d took %.4fs." % (epoch, timer() - start_epoch))

        training_time = timer() - start_training
        print("Training for %d took %.4fs." % (self.config.num_epochs, training_time))

    def train_batch(self, real_label, real_label_smoothed, fake_label, epoch, batch_num, data):
        # Update D - max log(D(x)) + log(1 - D(G(z))
        ############################################
        # Train with real batch
        self.D.zero_grad()
        real = data[0].to(self.device)
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label_smoothed, dtype=torch.float32, device=self.device)

        # Forward pass real batch through D
        output = self.D(real).view(-1)
        # Loss on real batch
        loss_D_real = self.loss(output, label)

        # Calculate gradients for D
        loss_D_real.backward()
        D_x = output.mean().item()

        # Train with fake batch
        noise = torch.randn(batch_size, self.config.latent_size, 1, 1, device=self.device)

        # Generate fake batch using G
        fake = self.G(noise)
        label.fill_(fake_label)

        # Forward pass fake batch through D
        output = self.D(fake.detach()).view(-1)

        # Loss on fake batch
        lost_D_fake = self.loss(output, label)

        # Calculate gradients for D
        lost_D_fake.backward()

        D_G_z1 = output.mean().item()

        loss_D = loss_D_real + lost_D_fake  # how

        self.optimizerD.step()

        # Update G - max log(D(G(z)))
        #############################
        self.G.zero_grad()

        label.fill_(real_label)
        output = self.D(fake).view(-1)

        loss_G = self.loss(output, label)
        loss_G.backward()

        D_G_z2 = output.mean().item()

        # Update G
        self.optimizerG.step()

        log_util.log_batch_stats(
            epoch, self.config.num_epochs, batch_num, len(self.dataloader), D_x, D_G_z1, loss_D, loss_G, D_G_z2
        )
        self.persist_training_data(loss_G, loss_D, D_x, D_G_z2)

    # Validates performance of the discriminator - total loss (val real + fake) and probability on validation setN[]
    def validate_batch(self, real_label, fake_label, epoch, batch_num, data):
        with torch.no_grad():
            self.D.zero_grad()
            real_validation = data[0].to(self.device)
            batch_size = real_validation.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float32, device=self.device)

            # Forward pass real batch through D
            output = self.D(real_validation).view(-1)
            # Loss on real batch
            loss_D_real_validation = self.loss(output, label)

            D_x_validation = output.mean().item()

            noise = torch.randn(batch_size, self.config.latent_size, 1, 1, device=self.device)

            # Generate fake batch using G
            fake = self.G(noise)
            label.fill_(fake_label)

            # Forward pass fake batch through D
            output = self.D(fake.detach()).view(-1)

            # Loss on fake batch
            lost_D_fake_validation = self.loss(output, label)

            loss_D_validation = loss_D_real_validation + lost_D_fake_validation

            log_util.log_validation_batch_stats(
                epoch, self.config.num_epochs, batch_num, len(self.dataloader), D_x_validation, loss_D_validation
            )
            self.persist_validation_data(loss_D_validation, D_x_validation)

    def update_learning_rate_linear_decay(self, current_epoch):
        if current_epoch < self.config.d_lr_decay_start_epoch:
            return

        for param_group in self.optimizerD.param_groups:
            param_group["lr"] -= self.lr_d_decay_delta
        for param_group in self.optimizerG.param_groups:
            param_group["lr"] -= self.lr_g_decay_delta

    ################################ Visualization and logging ################################

    def plot_results(self) -> None:
        losses_figure_path = self.config.experiment_output_root + "losses.png"
        viz_util.plot_and_save_losses(self.G_losses, self.D_losses, losses_figure_path)

        val_loss_figure_path = self.config.experiment_output_root + "val_loss.png"
        viz_util.plot_and_save_val_loss(self.D_losses_validation, val_loss_figure_path)

        probabilities_figure_path = self.config.experiment_output_root + "probs.png"
        viz_util.plot_and_save_discriminator_probabilities(self.D_x_vals, self.G_D_z_vals, probabilities_figure_path)

        val_prob_figure_path = self.config.experiment_output_root + "val_prob.png"
        viz_util.plot_and_save_discriminator_val_probs(self.D_x_validation_vals, val_prob_figure_path)

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

        if (epoch % 5 == 0) and (batch_num == len(self.dataloader) - 1):
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                viz_util.save_current_fakes_snapshot(fake, epoch, False, self.device, output_name)

        if (epoch == self.config.num_epochs - 1) and (batch_num == len(self.dataloader) - 1):
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                viz_util.save_current_fakes_snapshot(fake, epoch, True, self.device, output_name)

    # Generates and saves 2048 fake images for running the FID score script
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

    # def plot_results_animation(self):
    #     import matplotlib

    #     matplotlib.rcParams["animation.embed_limit"] = 64

    #     fig = plt.figure(figsize=(12, 12))
    #     plt.axis("off")
    #     ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.img_list]
    #     ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    #     HTML(ani.to_jshtml())