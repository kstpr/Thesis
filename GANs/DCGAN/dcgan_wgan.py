from timeit import default_timer as timer

from typing import Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable

from torchsummary import summary

from dcgan_base import DCGAN_base, weights_init_normal
from discriminator import Discriminator
import util.logging as log_util

# Refactored version of https://github.com/martinarjovsky/WassersteinGAN
class DCGAN_WGAN(DCGAN_base):

    # Break Liskov! Loss is not initialized here
    def init_loss_and_optimizer(self) -> None:
        self.fixed_noise = torch.randn(64, self.config.latent_size, 1, 1, device=self.device)

        self.optimizerD = optim.RMSprop(self.D.parameters(), lr=self.config.d_learning_rate)
        self.optimizerG = optim.RMSprop(
            self.G.parameters(),
            lr=self.config.g_learning_rate,
        )

    def init_discriminator(self) -> Discriminator:
        discriminator = Discriminator(
            num_gpu=self.config.num_gpu,
            num_channels=self.config.num_channels,
            feat_maps_size=self.config.d_feat_maps,
            image_size=self.config.image_size,
            use_sigmoid=False,
        ).to(self.device)

        discriminator.apply(weights_init_normal)

        print(discriminator)
        print("\n")

        summary(discriminator, input_size=(self.config.num_channels, self.config.image_size, self.config.image_size))
        return discriminator

    def train(self):
        # TODO pass from config
        self.num_critic_iters = 5
        self.weights_clip_param = 0.01

        generator_iterations = 0

        self.tensor_one = torch.FloatTensor([1]).to(self.device)
        self.tensor_minus_one = torch.FloatTensor([-1]).to(self.device)

        if self.use_amp:
            self.scaler = GradScaler()

        print("Starting training loop...")
        start_training = timer()
        for epoch in range(self.config.num_epochs):
            start_epoch = timer()

            data_iter = iter(self.dataloader)
            batch_num = 0

            while batch_num < len(self.dataloader):
                # This is noted in https://github.com/martinarjovsky/WassersteinGAN
                if generator_iterations < 25 or generator_iterations % 500 == 0:
                    max_critic_iterations = 100
                else:
                    max_critic_iterations = self.num_critic_iters

                current_critic_iterations = 0
                while current_critic_iterations < max_critic_iterations and batch_num < len(self.dataloader):
                    data = data_iter.next()
                    wass_loss, real_loss, fake_loss = self.train_critic(data)

                    current_critic_iterations += 1
                    batch_num += 1
                    # if batch_num % 10 == 0:
                    #    print("Critic loss: %.4f" % (wass_loss,))

                self.unfreeze_network(self.G)
                self.freeze_network(self.D)
                gen_loss = self.train_generator()
                generator_iterations += 1
                # if batch_num % 50 == 0:
                #    print("Generator loss: %.4f" % (gen_loss,))
                self.unfreeze_network(self.D)
                self.freeze_network(self.G)

                log_util.log_wgan_batch_stats(
                    epoch=epoch,
                    num_epochs=self.config.num_epochs,
                    batch_num=batch_num,
                    num_batches=len(self.dataloader),
                    loss_C=wass_loss,
                    loss_G=gen_loss,
                    C_x=real_loss,
                    C_G_z=fake_loss
                )
                self.save_fakes_snapshot_every_nth_epoch(epoch=epoch, batch_num=batch_num)

            # if self.config.use_validation:
            #     for val_batch_num, val_data in enumerate(self.validation_dataloader, 0):
            #         self.validate_batch(real_label, fake_label, epoch, val_batch_num, val_data)

            # if self.config.lr_linear_decay_enabled:
            #     self.update_learning_rate_linear_decay(epoch)
            self.save_networks_every_nth_epoch(epoch=epoch)
            print("Epoch %d took %.4fs." % (epoch, timer() - start_epoch))

        training_time = timer() - start_training
        print("Training for %d took %.4fs." % (self.config.num_epochs, training_time))

    def train_critic(self, data) -> Tuple[float, float, float]:
        # We presume Critic parameters are unfrozen
        self.clip_weights(self.D)

        data, _ = data
        self.D.zero_grad()

        # Train on real images
        real = data.to(self.device)

        loss_D_real = self.D(real).mean(0).view(1)
        loss_D_real.backward(self.tensor_one)

        # Train on generated images
        with torch.no_grad():
            noise = torch.randn(self.config.batch_size, self.config.latent_size, 1, 1, device=self.device)
            fake = self.G(noise).data

        loss_D_fake = self.D(fake).mean(0).view(1)
        loss_D_fake.backward(self.tensor_minus_one)

        wasserstein_D_loss = loss_D_real - loss_D_fake

        self.optimizerD.step()

        return (wasserstein_D_loss.data[0], loss_D_real.data[0], loss_D_fake.data[0])

    def train_generator(self) -> float:
        # We presume critic parameters are frozen
        self.G.zero_grad()

        noise = torch.randn(self.config.batch_size, self.config.latent_size, 1, 1, device=self.device)
        noise = noise
        fake = self.G(noise)

        loss_G = self.D(fake).mean(0).view(1)
        loss_G.backward(self.tensor_one)

        self.optimizerG.step()

        return loss_G.data[0]

    def plot_results(self) -> None:
        pass

    def freeze_network(self, net: nn.Module):
        for param in net.parameters():
            param.requires_grad = False

    def unfreeze_network(self, net: nn.Module):
        for param in net.parameters():
            param.requires_grad = True

    def clip_weights(self, net: nn.Module):
        for param in net.parameters():
            param.data.clamp_(-self.weights_clip_param, self.weights_clip_param)
