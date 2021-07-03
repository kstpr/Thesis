from copy import deepcopy
import dataclasses
from re import S
from typing import List, Optional, Tuple
from utils.logger import (
    log_batch_stats_gan,
    log_validation_stats,
    log_validation_stats_gan,
)
from piq.ssim import ssim

from torch.nn.modules.loss import L1Loss
from torch.tensor import Tensor
from utils.datawriter import serialize_and_save_config
import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchsummary.torchsummary import summary

from configs.config import ConfigGAN
from trainers.BaseTrainer import BaseTrainer
import netutils
from torch import nn
from timeit import default_timer as timer
from DeepIllumination.networks import GANLoss

from dacite import from_dict


class GANTrainer(BaseTrainer):
    """Base class for training of different GANs"""

    def __init__(
        self,
        config: ConfigGAN,
        device: torch.device,
        train_dataset: Dataset,
        io_transform: netutils.IOTransform,
        validation_dataset: Optional[Dataset],
    ):
        super().__init__(
            config,
            device,
            train_dataset,
            io_transform,
            validation_dataset=validation_dataset,
        )

        self.config.__class__ = ConfigGAN
 
        self.initialize_optimizers(config)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - (config.num_epochs // 2)) / float(
                config.num_epochs // 2 + 1
            )
            return lr_l
        self.intialize_schedulers(lambda_rule)

    # Terrible design. Call BEFORE __init__ in child classes.
    def initialize_networks(self, nets: List[nn.Module], num_channels: int):
        self.netG = nets[0]
        self.netD = nets[1]

        summary(self.netG, input_size=(num_channels, 512, 512))
        summary(self.netD, input_size=(num_channels + 3, 512, 512))

    def initialize_optimizers(self, config: ConfigGAN):
        self.optimizer_G = optim.Adam(
            params=self.netG.parameters(),
            betas=(config.optim_G_beta_1, config.optim_G_beta_2),
            lr=config.lr_optim_G,
        )
        self.optimizer_D = optim.Adam(
            params=self.netD.parameters(),
            betas=(config.optim_D_beta_1, config.optim_D_beta_2),
            lr=config.lr_optim_D,
        )

    def intialize_schedulers(self, lambda_rule):
        # Similar to the one from Pix2Pix code - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L38
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer_G, lr_lambda=lambda_rule
        )
        self.scheduler_D = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer_D, lr_lambda=lambda_rule
        )

    def load_saved_model(self, model_path: str):
        """Unsafe. Python is a nuisance and can't have a second constructor for this path. Make sure all the parameters are as expected."""
        checkpoint: dict = torch.load(model_path)
        self.netG.load_state_dict(checkpoint["model_G_state_dict"])
        self.netD.load_state_dict(checkpoint["model_D_state_dict"])

        if "optimizer_G_state_dict" in checkpoint:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        else:
            print("Warning! Optimizer state was not recovered!")

        if "scheduler_G_state_dict" in checkpoint:
            self.schedulers_OK = True
            self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
            self.scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])
        else:
            print("Warning! Scheduler state was not recovered!")

        self.resume_epoch = checkpoint["epoch"]
        self.config = from_dict(data_class=ConfigGAN, data=checkpoint["config"])
        # TODO - hacked up for a failed training, delete
        self.config.num_epochs = 200

    def continue_training_loaded_model(self):
        if self.resume_epoch == self.config.num_epochs:
            print("Tried to continue training but it was already over.")
            return

        if not self.schedulers_OK:
            # Hack! to make the scheduler catch up with the training.
            for i in range(self.resume_epoch):
                self.scheduler_D.step()
                self.scheduler_G.step()

        print(
            "Resume training from epoch {}, lr_D: {}, lr_G: {}".format(
                self.resume_epoch,
                self.scheduler_D.get_last_lr(),
                self.scheduler_G.get_last_lr(),
            )
        )
        self.train(start_epoch=self.resume_epoch + 1)

    # implement to return the needed loss functions
    def initialize_losses(self) -> List[nn.Module]:
        pass

    def train(self, start_epoch=1):
        if start_epoch == 1:  # otherwise training is resumed and this is not needed
            serialize_and_save_config(self.config)

        loss_functions: List[nn.Module] = self.initialize_losses()

        d_losses: List[float] = []
        g_losses: List[float] = []

        val_ssims: List[float] = []
        min_val_ssim = -1000.0
        best_generator_state: Optional[dict] = None
        best_generator_epoch: int = 0

        for epoch_num in range(start_epoch, self.config.num_epochs + 1):
            begin_epoch = timer()
            self.print_begin_epoch(epoch_num)

            (d_losses_e, g_losses_e) = self.train_epoch(epoch_num, loss_functions)

            print("Epoch completed for {0:.4f} s".format(timer() - begin_epoch))
            print(
                "Learning rates: G_lr = {0}, D_lr = {1}".format(
                    self.scheduler_G.get_last_lr(), self.scheduler_D.get_last_lr()
                )
            )

            d_losses.extend(d_losses_e)
            g_losses.extend(g_losses_e)

            # Validate
            if self.config.use_validation:
                val_ssim = self.validate_epoch(epoch_num)
                val_ssims.append(val_ssim)

                if val_ssim < min_val_ssim:
                    min_val_ssim = val_ssim
                    best_generator_epoch = epoch_num
                    best_generator_state = deepcopy(self.netG.state_dict())

            self.print_end_epoch(epoch_num, begin_epoch)

            if (epoch_num % self.config.image_snapshots_interval == 0) or (
                epoch_num == self.config.num_epochs
            ):
                self.save_snapshots(self.netG, epoch_num)

            self.save_networks_every_nth_epoch(epoch=epoch_num)

            if (epoch_num % 10 == 0) and self.config.use_validation:
                self.save_best_network(best_generator_epoch, best_generator_state)

    # implement the training algorithm for the gan in place
    def train_epoch(
        self, epoch_num: int, loss_functions: List[nn.Module]
    ) -> Tuple[List[float], List[float]]:
        pass

    def validate_epoch(self, epoch_num: int) -> float:
        self.netG.eval()

        begin_validation = timer()

        ssim_accum = 0.0

        with torch.no_grad():
            for batch_num, (input, gt) in enumerate(self.validation_dataloader):
                input: Tensor = self.io_transform.transform_input(input)
                gt: Tensor = self.io_transform.transform_gt(gt)
                output: Tensor = self.io_transform.transform_output(
                    output=self.netG(input)
                )

                self.io_transform.clear()
                ssim_accum += ssim(gt, output, kernel_size=7, data_range=1.0)

        num_batches = len(self.validation_dataloader)

        avg_ssim = ssim_accum / num_batches

        log_validation_stats_gan(epoch_num, timer() - begin_validation, avg_ssim)

        self.netG.train()

        return avg_ssim

    def save_networks_every_nth_epoch(self, epoch):
        """Save (at most) N network snapshots during the whole training."""
        if self.config.num_network_snapshots <= 0:
            return

        save_interval_in_epochs = int(
            self.config.num_epochs / self.config.num_network_snapshots
        )
        if save_interval_in_epochs == 0:
            return

        if (epoch != 0 and epoch % save_interval_in_epochs == 0) or (
            epoch == self.config.num_epochs
        ):
            file_path = (
                self.config.dirs.network_snapshots_dir + "snapshot_epoch_%d.tar" % epoch
            )
            torch.save(
                {
                    "epoch": epoch,
                    "config": dataclasses.asdict(self.config),
                    "model_G_state_dict": self.netG.state_dict(),
                    "model_D_state_dict": self.netD.state_dict(),
                    "optimizer_G_state_dict": self.optimizer_G.state_dict(),
                    "optimizer_D_state_dict": self.optimizer_D.state_dict(),
                    "scheduler_G_state_dict": self.scheduler_G.state_dict(),
                    "scheduler_D_state_dict": self.scheduler_D.state_dict(),
                },
                file_path,
            )
            print("Model snapshot saved in {}.".format(file_path))

    #TODO this is broken as is.
    def save_best_network(self, epoch, network_state_dict):
        file_path = (
            self.config.dirs.network_snapshots_dir + "best_net_G_epoch_%d.tar" % epoch
        )
        torch.save(
            {
                "epoch": epoch,
                "config": dataclasses.asdict(self.config),
                "model_G_state_dict": network_state_dict,
            },
            file_path,
        )
