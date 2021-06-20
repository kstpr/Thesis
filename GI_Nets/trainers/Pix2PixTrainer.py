from trainers.GANTrainer import GANTrainer
from typing import List, Tuple
from utils.logger import log_batch_stats_gan

from torch.nn.modules.loss import L1Loss
from torch.tensor import Tensor
import torch

from torch import nn
from timeit import default_timer as timer
from DeepIllumination.networks import GANLoss


class Pix2PixTrainer(GANTrainer):
    def initialize_losses(self) -> List[nn.Module]:
        gan_loss_fn = GANLoss("vanilla").to(self.device)
        l1_loss_fn = L1Loss()
        return [gan_loss_fn, l1_loss_fn]

    def train_epoch(
        self, epoch_num: int, loss_functions: List[nn.Module]
    ) -> Tuple[List[float], List[float]]:
        gan_loss_fn: GANLoss = loss_functions[0]
        l1_loss_fn: L1Loss = loss_functions[1]

        losses_d: List[float] = []
        losses_g: List[float] = []

        batch_load_start = timer()

        for batch_num, (input, gt) in enumerate(self.train_dataloader):
            # Skip last incomplete batch
            if batch_num == self.train_size - 1:
                continue
            should_log = (
                batch_num == self.train_size - 2
                or batch_num % self.config.batches_log_interval == 0
            )

            input: Tensor = self.io_transform.transform_input(input)
            gt: Tensor = self.io_transform.transform_gt(gt)
            fake: Tensor = self.io_transform.transform_output(output=self.netG(input))

            self.io_transform.clear()

            # Code is adapted from
            # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/pix2pix_model.py

            # I. Train D:
            # 1. Train D with fake
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()

            fake_cond = torch.cat((input, fake), 1)
            pred_fake = self.netD(fake_cond.detach())

            loss_D_fake = gan_loss_fn(prediction=pred_fake, target_is_real=False)

            # 2. Train D with real
            real_cond = torch.cat((input, gt), 1)
            pred_real = self.netD(real_cond)

            loss_D_real = gan_loss_fn(prediction=pred_real, target_is_real=True)

            loss_D: Tensor = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()

            self.optimizer_D.step()

            # II. Train G
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()

            fake_cond = torch.cat((input, fake), 1)
            pred_fake = self.netD(fake_cond)

            # Reverses labels, as in DCGAN I guess
            loss_G_GAN = gan_loss_fn(prediction=pred_fake, target_is_real=True)
            loss_L1 = l1_loss_fn(fake, gt)

            loss_G: Tensor = loss_G_GAN + (loss_L1 * self.config.lambda_l1)
            loss_G.backward()

            self.optimizer_G.step()

            losses_g.append(loss_G.item())
            losses_d.append(loss_D.item())

            if should_log:
                batch_end = timer()
                log_batch_stats_gan(
                    epoch=epoch_num,
                    batch_num=batch_num,
                    batch_train_time=batch_end - batch_load_start,
                    d_loss_real=loss_D_real.item(),
                    d_loss_fake=loss_D_fake.item(),
                    d_total_loss=loss_D.item(),
                    g_gan_loss=loss_G_GAN.item(),
                    g_l1_loss=loss_L1.item(),
                    g_total_loss=loss_G.item(),
                )

            batch_load_start = timer()

        self.scheduler_D.step()
        self.scheduler_G.step()

        return (losses_d, losses_g)

    # From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/base_model.py#L219
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad