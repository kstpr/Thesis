from abc import ABC, abstractmethod
import torch
from torch import device
from torch.tensor import Tensor
from typing import Tuple

ZERO_TENSOR = torch.tensor(0.0)


class IOTransform(ABC):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device: torch.device = device

    @abstractmethod
    def transform_input(self, input: Tensor) -> Tensor:
        """Input is transformed in the form ready to be fed in the network. """
        pass

    @abstractmethod
    def transform_output(self, input: Tensor, output: Tensor) -> Tensor:
        """Output is transformed in form compatible with GT and is ready to be used in
        losses and metrics."""
        pass

    @abstractmethod
    def transform_gt(self, gt: Tensor, input: Tensor = None) -> Tensor:
        """GT is transformed in form form compatible with output and is ready to be used
        in losses and metrics."""
        pass


class ClampGtTransform(IOTransform):
    """Clamps gt to [0, 1] and moves some tensors to device. Otherwise leaves the i/o unchanged."""

    def transform_input(self, input: Tensor) -> Tensor:
        return input.to(self.device)

    def transform_output(self, input: Tensor, output: Tensor) -> Tensor:
        return output

    def transform_gt(self, gt: Tensor, input: Tensor = None) -> Tensor:
        return torch.clamp_max(gt.to(self.device), 1.0)


class AdditiveDiMaskTransform(IOTransform):
    """Transforms di to be a mask = di - albedo, in range [-1, 1]. Network output is regarded also as
    a mask in [-1, 1], added to albedo (result in [-1, 2]) and clamped to [0,1] for a realistic
    image output that is compared with clamped gt by the losses and metrics."""

    def transform_input(self, input: Tensor) -> Tensor:
        input = input.to(self.device)

        albedo = input[:, 0:3, :].clamp(0.0, 1.0)  # in [0, 1]

        mask = input[:, 3:6, :].clamp(0.0, 1.0) - albedo  # in [-1, 1]
        input[:, 3:6, :] = mask

        return input

    def transform_output(self, input: Tensor, output: Tensor) -> Tensor:
        output = output + input[:, 0:3, :].clamp(0.0, 1.0)  # output mask + albedo, in [-1, 2]
        output = output.clamp(0.0, 1.0)
        return output

    def transform_gt(self, gt: Tensor, input: Tensor = None) -> Tensor:
        torch.clamp_max(gt.to(self.device), 1.0)


class AdditiveDiGtMaskTransform(IOTransform):
    """As above but the output of the network is not added directly to albedo but compared
    with gt - albedo in [-1, 1]. To be able to use SSIM (images in [0,1]) we apply the usual
    (1 + x) / 2 transform to both masks.
    """

    def transform_input(self, input: Tensor) -> Tensor:
        input = input.to(self.device)

        albedo = input[:, 0:3, :].clamp(0.0, 1.0)  # in [0, 1]

        mask = input[:, 3:6, :].clamp(0.0, 1.0) - albedo  # in [-1, 1]
        input[:, 3:6, :] = mask

        return input

    def transform_output(self, input: Tensor, output: Tensor) -> Tensor:
        output = output + input[:, 0:3, :].clamp(0.0, 1.0)  # output mask + albedo, in [-1, 2]
        output = output.clamp(0.0, 1.0)
        return output

    def transform_gt(self, gt: Tensor, input: Tensor = None) -> Tensor:
        torch.clamp_max(gt.to(self.device), 1.0)