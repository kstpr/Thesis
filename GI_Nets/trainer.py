import dataclasses
import logging
import sys
import random
from os.path import join
from timeit import default_timer as timer
from typing import List, Optional

import torch
from torch import nn, optim
from torch.nn.modules.loss import L1Loss, MSELoss
from torch.tensor import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.utils as vutils

from torchsummary import summary
from piq import SSIMLoss
from dacite import from_dict

from config import Config
from utils import scale_0_1_
from datawriter import serialize_and_save_config
import visualization as viz
from logger import log_batch_stats, log_validation_stats


class Trainer:
    def __init__(
        self,
        config: Config,
        train_dataset: Dataset,
        net: nn.Module,
        num_channels: int,
        device: torch.device,
        validation_dataset: Optional[Dataset] = None,
    ) -> None:
        self.init_torch()

        self.config: Config = config

        self.train_dataset = train_dataset
        self.train_dataloader: DataLoader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers_train,
        )

        self.init_validation_dataloader_if_needed(validation_dataset)

        self.train_size = len(self.train_dataloader)
        if self.config.use_validation:
            self.validate_size = len(self.validation_dataloader)

        self.device = device

        # TODO get num channels from dataset
        self.net: nn.Module = net
        summary(self.net, input_size=(num_channels, 512, 512))

        self.optimizer = optim.Adadelta(self.net.parameters())

        self.samples_indices = torch.randint(0, len(train_dataset), (10,))
        self.zero_tensor = torch.tensor(0.0)

    ### =================================== PUBLIC API ===================================

    ### Use the following two methods to resume training from a serialized network snapshot
    def load_saved_model(self, model_path: str):
        """Unsafe. Python is a nuisance and can't have a second constructor for this path. Make sure all the parameters are as expected."""
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.resume_epoch = checkpoint["epoch"]
        self.config = from_dict(data_class=Config, data=checkpoint["config"])

    def continue_training_loaded_model(self):
        if self.resume_epoch == self.config.num_epochs:
            print("Tried to continue training but it was already over.")
            return

        self.train(start_epoch=self.resume_epoch + 1)

    ### Use to find optimal number of workers for the used dataset and CPU
    def benchmark_num_workers(self, dataset) -> int:
        best_time: float = 10000.0
        best_num_workers = 0

        for num_workers in range(2, 13):
            print("Init DataLoader")
            dataloader = DataLoader(
                dataset=dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=num_workers
            )
            print("Ready")

            print("Initializing iter")
            begin = timer()
            for (i, data) in enumerate(dataloader):
                sys.stdout.write("#")
                sys.stdout.flush()
            epoch_time = timer() - begin

            num_batches = len(dataloader)

            if epoch_time < best_time:
                best_time = epoch_time
                best_num_workers = num_workers
            print(
                "\nAverage time for a single batch with {} workers: {} s. Time for {} batches: {}".format(
                    num_workers, epoch_time / num_batches, num_batches, epoch_time
                )
            )

        return best_num_workers

    def train(self, start_epoch=1):
        if start_epoch == 1:  # otherwise training is resumed and this is not needed
            serialize_and_save_config(self.config)

        ssim_loss_fn = SSIMLoss()
        l2_loss_fn = MSELoss()
        l1_loss_fn = L1Loss()

        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch_num in range(start_epoch, self.config.num_epochs + 1):
            begin_epoch = timer()
            self.print_begin_epoch(epoch_num)

            # Train
            losses = self.train_epoch(epoch_num, l1_loss_fn, l2_loss_fn, ssim_loss_fn)
            train_losses.extend(losses)

            # Validate
            if self.config.use_validation:
                with torch.no_grad():
                    val_loss = self.validate_epoch(epoch_num, l1_loss_fn, l2_loss_fn, ssim_loss_fn)
                    val_losses.append(val_loss)

            self.print_end_epoch(epoch_num, begin_epoch)

            if (epoch_num % self.config.image_snapshots_interval == 0) or (epoch_num == self.config.num_epochs):
                self.save_snapshots(epoch_num)

            self.save_networks_every_nth_epoch(epoch=epoch_num)

        losses_figure_path = join(self.config.dirs.experiment_results_root, "losses.png")
        viz.plot_and_save_losses(
            train_losses,
            val_losses,
            self.config.num_epochs,
            training_batches_per_epoch=len(self.train_dataloader),
            output_full_path=losses_figure_path,
        )

    ### =================================== PRIVATE ===================================

    def init_torch(self) -> None:
        manualSeed = 999
        print(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def init_validation_dataloader_if_needed(self, validation_dataset: Optional[Dataset]):
        if self.config.use_validation:
            if validation_dataset == None:
                raise Exception("Expected to use validation dataset but None is provided.")
            self.validation_dataloader: DataLoader = DataLoader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers_train,
            )

    def train_epoch(self, epoch, l1_loss_fn, l2_loss_fn, ssim_loss_fn) -> List[float]:
        losses: List[float] = []
        for batch_num, (input, gt) in enumerate(self.train_dataloader):
            # Skip last incomplete batch
            if batch_num == self.train_size - 1:
                continue

            begin_batch = timer()

            gt = gt.to(self.device)
            self.optimizer.zero_grad()

            output: Tensor = self.net(input.to(self.device))

            # TODO a limitation
            # SDSIM requires normalized input between 0 and 1
            # scale_0_1_(output)
            gt = torch.clamp_max(gt, 1.0)
            output = (output + 1.0) / 2.0  # torch.clamp_max(output, 1.0)

            sdsim_loss = self.structural_dissimilarity_loss(ssim_loss_fn, output, gt)
            l1_loss = l1_loss_fn(output, gt) if self.config.beta != 0.0 else self.zero_tensor
            l2_loss = l2_loss_fn(output, gt) if self.config.gamma != 0.0 else self.zero_tensor
            loss: Tensor = (
                (self.config.alpha * sdsim_loss) + (self.config.beta * l1_loss) + (self.config.gamma * l2_loss)
            )

            loss.backward()

            self.optimizer.step()

            if batch_num == self.train_size - 2 or batch_num % self.config.batches_log_interval == 0:
                log_batch_stats(
                    epoch,
                    batch_num,
                    timer() - begin_batch,
                    loss.detach().cpu().item(),
                    sdsim_loss.detach().cpu().item(),
                    l1_loss.detach().cpu().item(),
                    l2_loss.detach().cpu().item(),
                )

            losses.append(loss.cpu().item())

        return losses

    def validate_epoch(self, epoch_num: int, l1_loss_fn, l2_loss_fn, ssim_loss_fn) -> float:
        self.net.eval()
        total_loss: float = 0.0
        total_l1_loss: float = 0.0
        total_sdsim_loss: float = 0.0

        begin_validation = timer()

        for batch_num, (input, gt) in enumerate(self.validation_dataloader):
            gt = gt.to(self.device)
            output: Tensor = self.net(input.to(self.device))
            gt = torch.clamp_max(gt, 1.0)
            output = (output + 1.0) / 2.0

            l1_loss = l1_loss_fn(output, gt) if self.config.beta != 0.0 else self.zero_tensor
            l2_loss = l2_loss_fn(output, gt) if self.config.gamma != 0.0 else self.zero_tensor
            sdsim_loss = self.structural_dissimilarity_loss(ssim_loss_fn, output, gt)
            loss: Tensor = (
                (self.config.alpha * sdsim_loss) + (self.config.beta * l1_loss) + (self.config.gamma * l2_loss)
            )
            total_sdsim_loss += sdsim_loss.detach().cpu().item()
            total_l1_loss += l1_loss.detach().cpu().item()
            total_loss += loss.detach().cpu().item()

        num_batches = len(self.validation_dataloader)
        total_loss /= num_batches
        total_l1_loss /= num_batches
        total_sdsim_loss /= num_batches

        log_validation_stats(epoch_num, timer() - begin_validation, total_loss, total_sdsim_loss, total_l1_loss)

        return total_loss

    def structural_dissimilarity_loss(self, ssim_loss: nn.Module, output: Tensor, ground_truth: Tensor) -> Tensor:
        """Structural Dissimilarity as described by the auhtors"""
        if (torch.any(output < 0.0)) or (torch.any(ground_truth < 0.0)):
            print(
                "\n\n\Will EXPLODE! output min = {0:.4f}, output max = {1:.4f}, gt min = {2:.4f}, gt max = {1:.4f}\n\n".format(
                    output.min().item(), output.max().item(), ground_truth.min().item(), ground_truth.max().item()
                )
            )

        one_minus_ssim_val: Tensor = ssim_loss(output, ground_truth)

        # TODO these should be constants
        return one_minus_ssim_val / torch.full(one_minus_ssim_val.size(), 2.0)

    def save_snapshots(self, epoch_num):
        save_dir = self.config.dirs.result_snapshots_dir
        image_path = save_dir + "snapshot_{}.png".format(epoch_num)

        tensors = []
        with torch.no_grad():
            for index in self.samples_indices:
                (sample_input, sample_gt) = self.train_dataset.__getitem__(index)
                albedo = sample_input[0:3, :]
                tensors.append(albedo.clone())
                di = sample_input[3:6, :]
                tensors.append(di.clone())
                sample_input = sample_input.to(self.device)
                sample_output = self.net(torch.unsqueeze(sample_input, 0)).detach().cpu()
                sample_output = sample_output.squeeze()
                sample_output = (sample_output + 1.0) / 2.0
                tensors.append(sample_gt)
                tensors.append(sample_output)

        grid_tensor = vutils.make_grid(tensors, nrow=4).cpu()
        vutils.save_image(grid_tensor, image_path)

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
                },
                file_path,
            )
            print("Model snapshot saved in {}.".format(file_path))

    def print_begin_epoch(self, epoch_num):
        print()
        print("----------------------------------------------------------------")
        print("Epoch {0}.".format(epoch_num))
        print("----------------------------------------------------------------")

    def print_end_epoch(self, epoch_num, begin_epoch):
        print("----------------------------------------------------------------")
        print("Epoch {0} took {1:.2f} s.".format(epoch_num, timer() - begin_epoch))
        print("----------------------------------------------------------------")
        print()
