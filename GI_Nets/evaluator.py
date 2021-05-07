from timeit import default_timer as timer

from torch import nn
import torch
from torch.tensor import Tensor
from torch.utils.data import DataLoader, dataloader
from torch.utils.data.dataset import Dataset


from piq import ssim

from config import Config
from measures import mae, mse, psnr


class Evaluator:
    def __init__(self, config: Config, net: nn.Module, test_dataset: Dataset) -> None:
        self.config = config

        self.net: nn.Module = net
        self.net.eval()

        self.dataset: Dataset = test_dataset
        self.dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers_test,
        )

        self.sample_indices = torch.randint(0, len(test_dataset), (20,))

    def eval(self, device: torch.device) -> None:
        begin_eval = timer()

        mae_vals = []
        mse_vals = []
        psnr_vals = []
        ssim_vals = []
        lpips_vals = []

        train_size = len(self.dataloader)

        with torch.no_grad():
            for batch_num, (input, gt) in enumerate(self.dataloader):
                # Skip last incomplete batch
                if batch_num == train_size - 1:
                    continue

                gt = gt.to(device)
                output: torch.Tensor = self.net(input.to(device))

                # TODO a big limitation
                gt = torch.clamp_max(gt, 1.0)
                output = (output + 1.0) / 2.0  # torch.clamp_max(output, 1.0)

                for (output_img, gt_img) in zip(output, gt):
                    mae_vals.append(mae(output_img, gt_img))
                    mse_vals.append(mse(output_img, gt_img))
                    psnr_vals.append(psnr(output_img, gt_img))
                    ssim_vals.append(ssim(output_img, gt_img))

        avg_mae = sum(mae_vals) / len(mae_vals)
        avg_mse = sum(mse_vals) / len(mse_vals)
        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)
        print(
            "Evaluation ended for {0:.2f} s. Results:\n\tMAE = {1:.4f}\n\tMSE = {2:.4f}\n\tPSNR = {3:.4f}\n\tSSIM = {4:.4f}".format(
                timer() - begin_eval, avg_mae, avg_mse, avg_psnr, avg_ssim
            )
        )
