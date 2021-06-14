import torch
from torch.tensor import Tensor


def mae(output: Tensor, gt: Tensor) -> float:
    mae = torch.mean(torch.abs(output - gt))
    return mae.item()


def mse(output: Tensor, gt: Tensor) -> float:
    mse = torch.mean((output - gt) ** 2)
    return mse.item()


def psnr(output: Tensor, gt: Tensor) -> float:
    mse = torch.mean((output - gt) ** 2)
    # We expect our tensors to have values in [0, 1], hence 1the 1.0 constant in the psnr expression
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # 1.0 = max allowed dynamic range
    return psnr.item()