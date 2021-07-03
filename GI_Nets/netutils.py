from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import remove
import torch
from torch import device
from torch.tensor import Tensor
from torchvision.transforms import Normalize
from typing import List, Tuple

from enums import OutputType
from utils.utils import tosRGB_tensor
from GIDataset import ALL_INPUT_BUFFERS_CANONICAL, BufferType

ZERO_TENSOR = torch.tensor(0.0)


class IOTransform(ABC):
    """Base class for transforms that act on input/output of the nets."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device: torch.device = device
        self.min_positive = torch.finfo(torch.float32).tiny

    def transform_input(self, input: Tensor) -> Tensor:
        """Input is transformed in the form ready to be fed in the network. """
        transformed_input = input.to(self.device)
        self.unmodified_input = transformed_input.clone()
        return transformed_input

    @abstractmethod
    def transform_output(self, output: Tensor) -> Tensor:
        """Output is transformed in form compatible with GT and is ready to be used in
        losses and metrics."""
        pass

    @abstractmethod
    def transform_gt(self, gt: Tensor) -> Tensor:
        """GT is transformed in form form compatible with output and is ready to be used
        in losses and metrics."""
        pass

    def transform_output_eval(self, output: Tensor, visualize: bool = False) -> Tensor:
        """Like transform_output but returns tensor ready for evaluation/visualization."""
        result = self.transform_output(output)
        if visualize:
            result = tosRGB_tensor(result)

        return result

    def transform_gt_eval(self, gt: Tensor, visualize: bool = False) -> Tensor:
        """Like transform_gt but returns tensor ready for evaluation/visualization."""
        result = self.transform_gt(gt)
        if visualize:
            result = tosRGB_tensor(result)

        return result

    def get_albedo(self) -> Tensor:
        return self.unmodified_input[:, 0:3, :].clamp(self.min_positive, 1.0)

    def clear(self):
        self.unmodified_input = None


class ClampGtTransform(IOTransform):
    """Clamps gt to [0, 1] and moves some tensors to device. Otherwise leaves the i/o unchanged."""

    def transform_input(self, input: Tensor) -> Tensor:
        return super().transform_input(input)

    def transform_output(self, output: Tensor) -> Tensor:
        return (output.nan_to_num())# + 1.0) / 2.0 $ TODO - I comment and uncomment this manually depending on the net. NOT good!

    def transform_gt(self, gt: Tensor) -> Tensor:
        return gt.to(self.device).clamp(0.0, 1.0)


class AdditiveDiMaskTransform(IOTransform):
    """Transforms di to be a mask = di - albedo, in range [-1, 1]. Network output is regarded also as
    a mask in [-1, 1], added to albedo (result in [-1, 2]) and clamped to [0,1] for a realistic
    image output that is compared with clamped gt by the losses and metrics."""

    def __init__(self, device: torch.device, remove_albedo: bool = False) -> None:
        super().__init__(device)
        self.remove_albedo = remove_albedo

    def transform_input(self, input: Tensor) -> Tensor:
        input = super().transform_input(input)

        # albedo = input[:, 0:3, :].clamp(0.0, 1.0)  # in [0, 1]

        # mask = input[:, 3:6, :].clamp(0.0, 1.0) - albedo  # in [-1, 1]
        # input[:, 3:6, :] = mask

        if self.remove_albedo:
            input = input[:, 3:, :]

        return input

    def transform_output(self, output: Tensor) -> Tensor:
        output = output + self.unmodified_input[:, 0:3, :].clamp(
            0.0, 1.0
        )  # output mask + albedo, in [-1, 2]
        output = output.clamp(0.0, 1.0)
        return output

    def transform_gt(self, gt: Tensor) -> Tensor:
        return gt.to(self.device).clamp(0.0, 1.0)


# Do not use
class AdditiveDiGtMaskTransform(IOTransform):
    """As above but the output of the network is not added directly to albedo but compared
    with gt - albedo in [-1, 1]. To be able to use SSIM (images in [0,1]) we apply the usual
    (1 + x) / 2 transform to both masks.
    """

    def __init__(self, device: torch.device, remove_albedo=False) -> None:
        super().__init__(device)
        self.remove_albedo = remove_albedo

    def transform_input(self, input: Tensor) -> Tensor:
        input = super().transform_input(input)

        albedo = input[:, 0:3, :].clamp(0.0, 1.0)  # in [0, 1]

        mask = input[:, 3:6, :].clamp(0.0, 1.0) - albedo  # in [-1, 1]
        input[:, 3:6, :] = mask
        if self.remove_albedo:
            input = input[:, 3:, :]

        return input

    def transform_output(self, output: Tensor) -> Tensor:
        return (output + 1.0) / 2.0  # mask in [0, 1]

    def transform_gt(self, gt: Tensor) -> Tensor:
        gt = gt.to(self.device).clamp(0.0, 1.0)  # gt in [0, 1]
        albedo = self.unmodified_input[:, 0:3, :].clamp(0.0, 1.0)  # albedo in [0, 1]
        mask_gt = (gt - albedo + 1.0) / 2.0  # gt mask from [-1, 1] to [0, 1]

        return mask_gt

    def transform_output_eval(self, output: Tensor, visualize: bool = False) -> Tensor:
        albedo = self.unmodified_input[:, 0:3, :].clamp(0.0, 1.0)  # in [0, 1]
        mask_output = output + albedo  # mask + albedo in [-1, 2]

        if visualize:
            mask_output = tosRGB_tensor(mask_output)  # tosRGB_tensor clamps internally
        else:
            mask_output = mask_output.clamp(0.0, 1.0)

        return mask_output

    def transform_gt_eval(self, gt: Tensor, visualize: bool = False) -> Tensor:
        gt = gt.to(self.device)

        if visualize:
            gt = tosRGB_tensor(gt)
        else:
            gt = gt.clamp(0.0, 1.0)

        return gt


PI_OVER_2 = 3.1415926535 / 2.0
TWO_OVER_PI = 2.0 / 3.1415926535
EPS = 0.00000001
C = 0.992824927


class InputDIAsMaskTransform(IOTransform):
    """Clamps gt to [0, 1] and moves some tensors to device. Otherwise leaves the i/o unchanged."""

    def transform_input(self, input: Tensor) -> Tensor:
        input = super().transform_input(input)
        albedo = input[:, 0:3, :].clamp(0.0, 1.0)

        di_mask = torch.div(
            input[:, 3:6, :].clamp(0.0, 1.0), albedo
        ).nan_to_num()  # in [0, infty)

        # di -> log-space mask in input
        input[:, 3:6, :] = transform_mask_to_log_space(di_mask)  # in (-1, 1)
        return input

    def transform_output(self, output: Tensor) -> Tensor:
        return (output.nan_to_num() + 1.0) / 2.0

    def transform_gt(self, gt: Tensor) -> Tensor:
        return gt.to(self.device).clamp(0.0, 1.0)

@dataclass
class MMTConfig:
    input_as_mask: bool = False,
    output_type: OutputType = OutputType.DIRECT_IMAGE,
    gt_as_mask: bool = False,

class MultMaskTransform(IOTransform):
    def __init__(
        self,
        device: torch.device,
        config: MMTConfig,
        remove_albedo: bool = False,
        normalize_input: bool = False,

    ) -> None:
        super().__init__(device)
        self.remove_albedo = remove_albedo
        self.normalize_input = normalize_input
        self.min_positive = torch.finfo(torch.float32).tiny
        self.config : MMTConfig = config
        # if self.normalize_input:
        #     self.normalizer = Normalize(DATASET_MEAN_MULT_MASK, DATASET_STD_MULT_MASK)


    def transform_input(self, input: Tensor) -> Tensor:
        input = super().transform_input(input)

        if self.config.input_as_mask:
            albedo = input[:, 0:3, :].clamp(self.min_positive, 1.0)
            di_mask = torch.div(
                input[:, 3:6, :].clamp(0.0, 1.0), albedo
            ).nan_to_num()  # in (0, infty)

            # di -> mask in input
            input[:, 3:6, :] = transform_mask_to_log_space(di_mask)  # in (-1, 1)

        if self.normalize_input:
            input = self.normalizer(input)

        if self.remove_albedo:
            input = input[:, 3:, :]
        return input

    def transform_output(self, output: Tensor) -> Tensor:
        output_type = self.config.output_type
        if output_type == OutputType.ALBEDO_MULT_W_MASK:
            return (
                (
                    transform_mask_to_exp_space(output)
                    * self.unmodified_input[:, 0:3, :].clamp(self.min_positive, 1.0)
                )
                .nan_to_num()
                .clamp(0.0, 1.0)
            )
        elif output_type == OutputType.DIRECT_IMAGE or output_type == OutputType.MASK:
            return (output + 1.0) / 2.0

    def transform_gt(self, gt: Tensor) -> Tensor:
        gt = gt.to(self.device)
        if self.config.gt_as_mask:
            albedo = self.unmodified_input[:, 0:3, :].clamp(self.min_positive, 1.0)
            gt_mask = torch.div(
                gt.clamp(0.0, 1.0), albedo
            ).nan_to_num()  # in [0, infty)
            gt_mask_log = transform_mask_to_log_space(gt_mask)
            return (gt_mask_log + 1.0) / 2.0
        else:
            return gt.to(self.device).clamp(0.0, 1.0)

    def transform_output_eval(self, output: Tensor, visualize: bool = False) -> Tensor:
        """Like transform_output but returns tensor ready for evaluation/visualization."""

        result = self.transform_output(output)

        if self.config.output_type == OutputType.MASK:
            result = (transform_mask_to_exp_space(result) * self.unmodified_input[:, 0:3, :].clamp(self.min_positive, 1.0)).nan_to_num().clamp(0.0, 1.0)
 
        if visualize:
            result = tosRGB_tensor(result)
        else:
            result = result.clamp(0.0, 1.0)

        return result

    def transform_gt_eval(self, gt: Tensor, visualize: bool = False) -> Tensor:
        """Like transform_gt but returns tensor ready for evaluation/visualization."""
        result = gt.to(self.device)

        if visualize:
            result = tosRGB_tensor(result)
        else:
            result = result.clamp(0.0, 1.0)

        return result


def transform_mask_to_log_space(mask: Tensor) -> Tensor:
    """Takes a mask in [0, infty) as an input and transforms it to (-1, 1). The transforms
    transform the output range as follows:
    # [0, infty) -> log -> (-infty, infty) -> atan -> (- pi/2, pi/2) -> * 2/pi -> (-1, 1)
    """
    mask_norm = TWO_OVER_PI * torch.atan(torch.log(mask))
    return mask_norm


def transform_mask_to_exp_space(mask_log: Tensor) -> Tensor:
    """ mask_log (-1, 1)"""
    return torch.exp(torch.tan((PI_OVER_2) * mask_log * C))