from abc import ABC, abstractmethod
from typing import Optional
import random
from utils.utils import tosRGB_tensor
from timeit import default_timer as timer

import torch
from torch.tensor import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torchvision.utils as vutils

from configs.config import Config
import netutils

# Training set sample indices, hardcoded for reproducibility
SAMPLE_INDICES = [
    2319,
    3225,
    3371,
    2636,
    4512,
    984,
    733,
    649,
    4741,
    4333,
]



""" More code can be extracted here but there's no time."""
class BaseTrainer(ABC):
    def __init__(
        self,
        config: Config,
        device: torch.device,
        train_dataset: Dataset,
        io_transform: netutils.IOTransform,
        validation_dataset: Optional[Dataset] = None,
    ):
        self.init_torch()

        self.config: Config = config
        self.train_dataset: Dataset = train_dataset
        self.validation_dataset: Dataset = validation_dataset

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
        self.io_transform = io_transform

    ### =================================== PUBLIC API ===================================

    ### Use the following two methods to resume training from a serialized network snapshot
    @abstractmethod
    def load_saved_model(self, model_path: str):
        pass

    @abstractmethod
    def continue_training_loaded_model(self):
        pass

    @abstractmethod
    def train(self, start_epoch=1):
        pass

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

    def save_snapshots(self, net: nn.Module, epoch_num: int):
        save_dir = self.config.dirs.result_snapshots_dir
        image_path = save_dir + "snapshot_{}.png".format(epoch_num)

        tensors = []
        with torch.no_grad():
            for index in SAMPLE_INDICES:
                sample_input: Tensor
                sample_gt: Tensor
                (sample_input, sample_gt) = self.train_dataset.__getitem__(index)

                input_clone = sample_input.clone().to(self.device)

                transformed_input: Tensor = self.io_transform.transform_input(sample_input.unsqueeze(0)).squeeze()
                transformed_gt: Tensor = self.io_transform.transform_gt_eval(
                    sample_gt.unsqueeze(0), visualize=True
                ).squeeze()
                transformed_output: Tensor = (
                    self.io_transform.transform_output_eval(output=net(transformed_input.unsqueeze(0)), visualize=True)
                    .detach()
                    .squeeze()
                )

                self.io_transform.clear()

                diffuse_albedo = tosRGB_tensor(input_clone[0:3, :].clone())
                tensors.append(diffuse_albedo)

                di = tosRGB_tensor(input_clone[3:6, :].clone())
                tensors.append(di)

                tensors.append(transformed_gt)
                tensors.append(transformed_output)

        num_images_per_row = 4
        grid_tensor = vutils.make_grid(tensors, nrow=num_images_per_row).cpu()
        vutils.save_image(grid_tensor, image_path)

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