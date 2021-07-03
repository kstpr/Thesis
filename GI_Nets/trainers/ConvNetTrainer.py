from trainers.BaseTrainer import BaseTrainer
import dataclasses

from enums import OptimizerType
from copy import deepcopy
from os.path import join
from timeit import default_timer as timer
from typing import List, Optional

import torch
from torch import nn, optim
from torch.nn.modules.loss import L1Loss, MSELoss
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset

from torchsummary import summary
from piq import SSIMLoss, LPIPS
from dacite import from_dict

from configs.config import Config
from utils.datawriter import serialize_and_save_config
import utils.visualization as viz
from utils.logger import log_batch_stats, log_validation_stats
import netutils


class ConvNetTrainer(BaseTrainer):
    def __init__(
        self,
        config: Config,
        train_dataset: Dataset,
        net: nn.Module,
        num_channels: int,
        device: torch.device,
        io_transform: netutils.IOTransform,
        validation_dataset: Optional[Dataset] = None,
        optimizer_type: OptimizerType = OptimizerType.ADAM,  # TODO add to config
    ) -> None:
        super().__init__(config, device, train_dataset, io_transform, validation_dataset)

        self.net: nn.Module = net
        summary(self.net, input_size=(num_channels, 512, 512))

        self.optimizer: optim.Optimizer = self.create_optimizer(optimizer_type)

        # TODO make this more polite
        if config.use_lr_scheduler:
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 - (config.num_epochs // 2)) / float(
                    config.num_epochs // 2 + 1
                )
                return lr_l
            self.scheduler = optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer, lr_lambda=lambda_rule
            ) 
            #optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 20, 40, 70], gamma=0.2)
        else:
            self.scheduler = None

        self.zero_tensor = torch.tensor(0.0)

    ### =================================== PUBLIC API ===================================

    ### Use the following two methods to resume training from a serialized network snapshot
    def load_saved_model(self, model_path: str):
        """Unsafe. Python is a nuisance and can't have a second constructor for this path. Make sure all the parameters are as expected."""
        checkpoint: dict = torch.load(model_path)
        self.net.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.resume_epoch = checkpoint["epoch"]
        self.config = from_dict(data_class=Config, data=checkpoint["config"])

    def continue_training_loaded_model(self):
        if self.resume_epoch == self.config.num_epochs:
            print("Tried to continue training but it was already over.")
            return

        print(
            "Resume training from epoch {}, lr_D: {}, lr_G: {}".format(
                self.resume_epoch,
                self.scheduler.get_last_lr(),
            )
        )

        self.train(start_epoch=self.resume_epoch + 1)

    def train(self, start_epoch=1):
        if start_epoch == 1:  # otherwise training is resumed and this is not needed
            serialize_and_save_config(self.config)

        ssim_loss_fn = SSIMLoss(kernel_size=7, data_range=1.0)
        additional_loss_fn = LPIPS()
        l2_loss_fn = MSELoss()
        l1_loss_fn = L1Loss()

        train_losses: List[float] = []
        val_losses: List[float] = []

        min_val_loss = 1000.0
        best_model_state: Optional[dict] = None
        best_model_epoch: int = 0 if start_epoch == 1 else start_epoch

        for epoch_num in range(start_epoch, self.config.num_epochs + 1):
            begin_epoch = timer()
            self.print_begin_epoch(epoch_num)

            # Train
            losses = self.train_epoch(epoch_num, l1_loss_fn, l2_loss_fn, ssim_loss_fn, additional_loss_fn)
            print("Train completed for {0:.4f} s".format(timer() - begin_epoch))

            train_losses.extend(losses)
            # Validate
            if self.config.use_validation:
                val_loss = self.validate_epoch(epoch_num, l1_loss_fn, l2_loss_fn, ssim_loss_fn, additional_loss_fn)
                val_losses.append(val_loss)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_model_epoch = epoch_num
                    best_model_state = deepcopy(self.net.state_dict())

            self.print_end_epoch(epoch_num, begin_epoch)

            if (epoch_num % self.config.image_snapshots_interval == 0) or (epoch_num == self.config.num_epochs):
                self.save_snapshots(self.net, epoch_num)

            self.save_networks_every_nth_epoch(epoch=epoch_num)

            if (epoch_num % 10 == 0) and self.config.use_validation:
                self.save_best_network(best_model_epoch, best_model_state)

            if epoch_num - best_model_epoch > self.config.num_epochs // 5:
                # Stop training if the results don't improve for some epochs
                break

        if self.config.use_validation:
            self.save_best_network(best_model_epoch, best_model_state)

        losses_figure_path = join(self.config.dirs.experiment_results_root, "losses.png")
        viz.plot_and_save_losses(
            train_losses,
            val_losses,
            self.config.num_epochs,
            training_batches_per_epoch=len(self.train_dataloader),
            output_full_path=losses_figure_path,
        )

    ### =================================== PRIVATE ===================================

    def create_optimizer(self, optimizer_type: OptimizerType) -> torch.optim.Optimizer:
        if optimizer_type == OptimizerType.ADADELTA:
            return optim.Adadelta(self.net.parameters())
        elif optimizer_type == OptimizerType.ADAM:
            return optim.Adam(self.net.parameters(), lr=self.config.learning_rate)

    def train_epoch(self, epoch, l1_loss_fn, l2_loss_fn, ssim_loss_fn, additional_loss_fn) -> List[float]:
        losses: List[float] = []

        batch_load_start = timer()
        for batch_num, (input, gt) in enumerate(self.train_dataloader):
            # Skip last incomplete batch
            if batch_num == self.train_size - 1:
                continue
            should_log = batch_num == self.train_size - 2 or batch_num % self.config.batches_log_interval == 0

            batch_train_start = timer()

            self.optimizer.zero_grad()

            input: Tensor = self.io_transform.transform_input(input)
            gt: Tensor = self.io_transform.transform_gt(gt)
            output: Tensor = self.io_transform.transform_output(output=self.net(input))

            self.io_transform.clear()

            sdsim_loss = (
                self.structural_dissimilarity_loss(ssim_loss_fn, output, gt)
                if self.config.alpha != 0.0
                else self.zero_tensor
            )
            l1_loss = l1_loss_fn(output, gt) if self.config.beta != 0.0 else self.zero_tensor
            l2_loss = l2_loss_fn(output, gt) if self.config.gamma != 0.0 else self.zero_tensor
            additional_loss = additional_loss_fn(output, gt) if self.config.delta != 0.0 else self.zero_tensor

            loss: Tensor = (
                (self.config.alpha * sdsim_loss)
                + (self.config.beta * l1_loss)
                + (self.config.gamma * l2_loss)
                + (self.config.delta * additional_loss)
            )

            loss.backward()

            self.optimizer.step()

            # TODO may be a bottleneck
            loss_val = loss.detach().cpu().item()
            losses.append(loss_val)

            if should_log:
                batch_end = timer()
                log_batch_stats(
                    epoch,
                    batch_num,
                    batch_end - batch_load_start,
                    batch_end - batch_train_start,
                    loss_val,
                    sdsim_loss.detach().cpu().item(),
                    l1_loss.detach().cpu().item(),
                    additional_loss=additional_loss.detach().cpu().item()
                    # l2_loss.detach().cpu().item(),
                )

            batch_load_start = timer()

        if not self.scheduler is None:
            self.scheduler.step()

        return losses

    def validate_epoch(self, epoch_num: int, l1_loss_fn, l2_loss_fn, ssim_loss_fn, additional_loss_fn) -> float:
        self.net.eval()

        total_loss: float = 0.0
        total_l1_loss: float = 0.0
        total_additional_loss: float = 0.0
        total_sdsim_loss: float = 0.0

        begin_validation = timer()

        with torch.no_grad():
            for batch_num, (input, gt) in enumerate(self.validation_dataloader):
                input: Tensor = self.io_transform.transform_input(input)
                gt: Tensor = self.io_transform.transform_gt(gt)
                output: Tensor = self.io_transform.transform_output(output=self.net(input))

                self.io_transform.clear()

                sdsim_loss = (
                    self.structural_dissimilarity_loss(ssim_loss_fn, output, gt)
                    if self.config.alpha != 0.0
                    else self.zero_tensor
                )
                l1_loss = l1_loss_fn(output, gt) if self.config.beta != 0.0 else self.zero_tensor
                l2_loss = l2_loss_fn(output, gt) if self.config.gamma != 0.0 else self.zero_tensor
                additional_loss = additional_loss_fn(output, gt) if self.config.delta != 0.0 else self.zero_tensor

                loss: Tensor = (
                    (self.config.alpha * sdsim_loss)
                    + (self.config.beta * l1_loss)
                    + (self.config.gamma * l2_loss)
                    + (self.config.delta * additional_loss)
                )

                total_sdsim_loss += sdsim_loss.detach().cpu().item()
                total_l1_loss += l1_loss.detach().cpu().item()
                total_additional_loss += additional_loss.detach().cpu().item()
                total_loss += loss.detach().cpu().item()

        num_batches = len(self.validation_dataloader)
        total_loss /= num_batches
        total_l1_loss /= num_batches
        total_sdsim_loss /= num_batches
        total_additional_loss /= num_batches

        log_validation_stats(
            epoch_num, timer() - begin_validation, total_loss, total_sdsim_loss, total_l1_loss, total_additional_loss
        )

        self.net.train()

        return total_loss

    # Only in case of need
    def ssim_sanity_check(self, output: Tensor, ground_truth: Tensor):
        if (torch.any(output < 0.0)) or (torch.any(ground_truth < 0.0)):
            print(
                "\n\n\Will EXPLODE! output min = {0:.4f}, output max = {1:.4f}, gt min = {2:.4f}, gt max = {3:.4f}\n\n".format(
                    output.min().item(), output.max().item(), ground_truth.min().item(), ground_truth.max().item()
                )
            )

    def structural_dissimilarity_loss(self, ssim_loss: nn.Module, output: Tensor, ground_truth: Tensor) -> Tensor:
        """Structural Dissimilarity as described by the auhtors"""
        one_minus_ssim_val: Tensor = ssim_loss(output, ground_truth)

        # TODO these should be constants
        return one_minus_ssim_val / torch.full(one_minus_ssim_val.size(), 2.0)

    def save_networks_every_nth_epoch(self, epoch):
        """Save (at most) N network snapshots during the whole training."""
        if self.config.num_network_snapshots <= 0:
            return

        save_interval_in_epochs = int(self.config.num_epochs / self.config.num_network_snapshots)
        if save_interval_in_epochs == 0:
            return

        if (epoch != 0 and epoch % save_interval_in_epochs == 0) or (epoch == self.config.num_epochs):
            file_path = self.config.dirs.network_snapshots_dir + "snapshot_epoch_%d.tar" % epoch
            torch.save(
                {
                    "epoch": epoch,
                    "config": dataclasses.asdict(self.config),
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict()
                },
                file_path,
            )
            print("Model snapshot saved in {}.".format(file_path))

    def save_best_network(self, epoch, network_state_dict):
        file_path = self.config.dirs.network_snapshots_dir + "best_net_epoch_%d.tar" % epoch
        torch.save(
            {
                "epoch": epoch,
                "config": dataclasses.asdict(self.config),
                "model_state_dict": network_state_dict,
            },
            file_path,
        )