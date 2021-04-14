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

from torch.cuda.amp import autocast, GradScaler

from torchsummary import summary

from IPython.display import HTML

from config import Config
from discriminator import Discriminator
from generator import Generator
from dcgan_base import DCGAN_base
import util.visualization as viz_util
import util.logging as log_util


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN_tryout(DCGAN_base):

    def init_loss_and_optimizer(self) -> None:
        self.loss_func: nn.Module = nn.BCEWithLogitsLoss() if self.use_amp else nn.BCELoss()

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

    def train(self):
        real_label = 1.0
        real_label_smoothed = 0.9 if self.config.use_label_smoothing else 1.0
        fake_label = 0.0

        if self.use_amp:
            self.scaler = GradScaler()

        print("Starting training loop...")
        start_training = timer()
        for epoch in range(self.config.num_epochs):
            start_epoch = timer()

            for batch_num, data in enumerate(self.dataloader, 0):
                if self.use_amp:
                    self.train_batch_amp(real_label, real_label_smoothed, fake_label, epoch, batch_num, data)
                else:
                    self.train_batch(real_label, real_label_smoothed, fake_label, epoch, batch_num, data)
                self.save_fakes_snapshot_every_nth_epoch(epoch=epoch, batch_num=batch_num)

            if self.config.use_validation:
                for val_batch_num, val_data in enumerate(self.validation_dataloader, 0):
                    self.validate_batch(real_label, fake_label, epoch, val_batch_num, val_data)

            if self.config.lr_linear_decay_enabled:
                self.update_learning_rate_linear_decay(epoch)
            self.save_networks_every_nth_epoch(epoch=epoch)
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
        loss_D_real = self.loss_func(output, label)

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
        lost_D_fake = self.loss_func(output, label)

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

        loss_G = self.loss_func(output, label)
        loss_G.backward()

        D_G_z2 = output.mean().item()

        # Update G
        self.optimizerG.step()

        log_util.log_batch_stats(
            epoch, self.config.num_epochs, batch_num, len(self.dataloader), D_x, D_G_z1, loss_D, loss_G, D_G_z2
        )
        self.persist_training_data(loss_G, loss_D, D_x, D_G_z2)

    def train_batch_amp(self, real_label, real_label_smoothed, fake_label, epoch, batch_num, data):
        # Update D - max log(D(x)) + log(1 - D(G(z))
        ############################################
        # Train with real batch
        self.D.zero_grad()
        real = data[0].to(self.device)
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label_smoothed, dtype=torch.float32, device=self.device)

        with autocast():
            # Forward pass real batch through D
            output = self.D(real).view(-1)
            # Loss on real batch
            loss_D_real = self.loss_func(output, label)

        # Calculate gradients for D
        self.scaler.scale(loss_D_real).backward()
        D_x = output.mean().item()

        # Train with fake batch
        noise = torch.randn(batch_size, self.config.latent_size, 1, 1, device=self.device)

        label.fill_(fake_label)

        with autocast():
            # Generate fake batch using G
            fake = self.G(noise)
            # Forward pass fake batch through D
            output = self.D(fake.detach()).view(-1)
            # Loss on fake batch
            lost_D_fake = self.loss_func(output, label)

        # Calculate gradients for D
        self.scaler.scale(lost_D_fake).backward()

        D_G_z1 = output.mean().item()

        loss_D = loss_D_real + lost_D_fake  # how

        self.scaler.step(self.optimizerD)

        # Update G - max log(D(G(z)))
        #############################
        self.G.zero_grad()

        label.fill_(real_label)

        with autocast():
            output = self.D(fake).view(-1)
            loss_G = self.loss_func(output, label)

        self.scaler.scale(loss_G).backward()

        D_G_z2 = output.mean().item()

        # Update G
        self.scaler.step(self.optimizerG)

        self.scaler.update()

        log_util.log_batch_stats(
            epoch, self.config.num_epochs, batch_num, len(self.dataloader), D_x, D_G_z1, loss_D, loss_G, D_G_z2
        )
        self.persist_training_data(loss_G, loss_D, D_x, D_G_z2)

    # TODO - not amp
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
            loss_D_real_validation = self.loss_func(output, label)

            D_x_validation = output.mean().item()

            noise = torch.randn(batch_size, self.config.latent_size, 1, 1, device=self.device)

            # Generate fake batch using G
            fake = self.G(noise)
            label.fill_(fake_label)

            # Forward pass fake batch through D
            output = self.D(fake.detach()).view(-1)

            # Loss on fake batch
            lost_D_fake_validation = self.loss_func(output, label)

            loss_D_validation = loss_D_real_validation + lost_D_fake_validation

            log_util.log_validation_batch_stats(
                epoch, self.config.num_epochs, batch_num, len(self.dataloader), D_x_validation, loss_D_validation
            )
            self.persist_validation_data(loss_D_validation, D_x_validation)

    def plot_results(self) -> None:
        losses_figure_path = self.config.experiment_output_root + "losses.png"
        viz_util.plot_and_save_losses(self.G_losses, self.D_losses, losses_figure_path)

        val_loss_figure_path = self.config.experiment_output_root + "val_loss.png"
        viz_util.plot_and_save_val_loss(self.D_losses_validation, val_loss_figure_path)

        probabilities_figure_path = self.config.experiment_output_root + "probs.png"
        viz_util.plot_and_save_discriminator_probabilities(self.D_x_vals, self.G_D_z_vals, probabilities_figure_path)

        val_prob_figure_path = self.config.experiment_output_root + "val_prob.png"
        viz_util.plot_and_save_discriminator_val_probs(self.D_x_validation_vals, val_prob_figure_path)