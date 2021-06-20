import os
from numpy.core.fromnumeric import std
from piq.perceptual import DISTS
from test import as_color_mapped_image
from timeit import default_timer as timer
from typing import List
from statistics import stdev

from torch import nn
import torch
from torch.tensor import Tensor
from torch.utils.data import DataLoader, dataloader
from torch.utils.data.dataset import Dataset
import torchvision.utils as vutils

from piq import ssim, LPIPS, multi_scale_ssim, fsim, gmsd, multi_scale_gmsd, haarpsi, mdsi
import matplotlib

matplotlib.use("Agg")
from matplotlib.cm import get_cmap

from configs.config import Config
from measures import mae, mse, psnr
from utils.datawriter import serialize_and_save_results
import netutils
from utils.utils import tosRGB_tensor


class Evaluator:
    def __init__(
        self,
        config: Config,
        net: nn.Module,
        dataset: Dataset,
        device: torch.device,
        io_transform: netutils.IOTransform,
        save_results: bool = True,
        uses_secondary_dataset: bool = False,
    ) -> None:
        self.config = config
        self.save_results = save_results

        self.net: nn.Module = net
        self.net.eval()

        self.dataset: Dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers_test,
        )

        self.device: torch.device = device
        self.io_transform: netutils.IOTransform = io_transform

        self.sample_indices = torch.randint(0, len(dataset), (40,))
        self.uses_secondary_dataset = uses_secondary_dataset

    def eval(self) -> None:
        begin_eval = timer()

        mae_vals = []
        mse_vals = []
        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        ms_ssim_vals = []
        gmsd_vals = []
        ms_gmsd_vals = []
        haar_vals = []
        mdsi_vals = []
        dists_vals = []

        test_size = len(self.dataloader)  # in batches

        def avg(lst: List[float]) -> float:
            return sum(lst) / len(lst)

        print("\nEvaluating network performance for test set {}...".format("B" if self.uses_secondary_dataset else "A"))
        with torch.no_grad():
            lpips = LPIPS()
            dists = DISTS()
            for batch_num, (input, gt) in enumerate(self.dataloader):
                # Skip last incomplete batch
                if batch_num == test_size - 1:
                    continue

                input: Tensor = self.io_transform.transform_input(input)
                gt: Tensor = self.io_transform.transform_gt_eval(gt, visualize=False)
                output: Tensor = self.io_transform.transform_output_eval(output=self.net(input), visualize=False)

                self.io_transform.clear()

                for (output_img, gt_img) in zip(output, gt):
                    mae_vals.append(mae(output_img, gt_img))
                    mse_vals.append(mse(output_img, gt_img))
                    psnr_vals.append(psnr(output_img, gt_img))
                    ssim_vals.append(ssim(output_img, gt_img, data_range=1.0, kernel_size=7).detach().item())
                    lpips_vals.append(lpips(output_img, gt_img).detach().item())

                    ms_ssim_vals.append(multi_scale_ssim(output_img, gt_img).detach().item())
                    # fsim_vals.append(fsim(output_img, gt_img))
                    gmsd_vals.append(gmsd(output_img, gt_img).detach().item())
                    ms_gmsd_vals.append(multi_scale_gmsd(output_img, gt_img).detach().item())
                    haar_vals.append(haarpsi(output_img, gt_img).detach().item())
                    mdsi_vals.append(mdsi(output_img, gt_img).detach().item())
                    dists_vals.append(dists(output_img, gt_img).detach().item())

        print("Evaluating inference time...")
        # evaluate inference time
        with torch.no_grad():
            times: List[float] = []
            skip_items = 50  # skip the first few items because they tend to be slower
            for i in range(self.dataset.__len__()):
                if i < skip_items:
                    continue

                input: Tensor = self.dataset.__getitem__(i)

                input = input[0].unsqueeze(0).to(self.device)
                input = self.io_transform.transform_input(input)

                begin = timer()

                output: torch.Tensor = self.net(input)

                t = timer() - begin
                times.append(t)

        print("Processing results...")
        self.process_and_save_results(avg, mae_vals, mse_vals, psnr_vals, ssim_vals, lpips_vals, times, begin_eval)
        self.process_secondary(avg, ms_ssim_vals, gmsd_vals, ms_gmsd_vals, haar_vals, mdsi_vals, dists_vals)
        if self.save_results:
            print("Saving snapshots with diff...")
            self.save_snapshots()

    def test_inference(self):
        times: List[float] = []

        with torch.no_grad():
            skip_num = 100
            skip = 0
            for i in range(2):
                begin_epoch = timer()

                for i in range(len(self.dataset)):
                    #begin = timer()

                    input = self.dataset[i][0].unsqueeze(0).to(self.device)
                    output: Tensor = self.net(input)

                    #t = timer() - begin
                    # if skip < skip_num:
                    #     skip = skip + 1
                    # else:
                    #     times.append(t)

                epoch_time = timer() - begin_epoch
                print(
                    "Epoch for {0:.4f} s, single item for {1:.4f} s".format(
                        epoch_time, epoch_time / len(self.dataloader)
                    )
                )

            print("Single item avg: {0:.4f} s".format(sum(times) / len(times)))

    def process_and_save_results(self, avg, mae_vals, mse_vals, psnr_vals, ssim_vals, lpips_vals, times, begin_eval):
        # Measures
        avg_mae = avg(mae_vals)
        std_mae = stdev(mae_vals)
        avg_mse = avg(mse_vals)
        std_mse = stdev(mse_vals)
        avg_psnr = avg(psnr_vals)
        std_psnr = stdev(psnr_vals)
        avg_ssim = avg(ssim_vals)
        std_ssim = stdev(ssim_vals)
        avg_lpips = avg(lpips_vals)
        std_lpips = stdev(lpips_vals)
        # Inference time
        avg_time = avg(times)
        std_time = stdev(times)

        if self.save_results:
            filename = "results_secondary_test_set.json" if self.uses_secondary_dataset else "results.json"
            filepath = os.path.join(self.config.dirs.experiment_results_root, filename)
            serialize_and_save_results(
                filepath,
                avg_mae,
                std_mae,
                avg_mse,
                std_mse,
                avg_ssim,
                std_ssim,
                avg_psnr,
                std_psnr,
                avg_lpips,
                std_lpips,
                avg_time,
                std_time,
            )
        print(
            """Evaluation ended for {0:.2f} s. Results:
            MAE = {1:.4f} +/- {2:.4f}
            MSE = {3:.4f} +/- {4:.4f}
            PSNR = {5:.4f} +/- {6:.4f}
            SSIM = {7:.4f} +/- {8:.4f}
            LPIPS = {9:.4f}  +/- {10:.4f}
            Avg. inference time = {11:.4f}  +/- {12:.4f}""".format(
                timer() - begin_eval,
                avg_mae,
                std_mae,
                avg_mse,
                std_mse,
                avg_psnr,
                std_psnr,
                avg_ssim,
                std_ssim,
                avg_lpips,
                std_lpips,
                avg_time,
                std_time,
            )
        )

    def process_secondary(self, avg, ms_ssim_vals, gmsd_vals, ms_gmsd_vals, haar_vals, mdsi_vals, dists_vals):
        avg_ms_ssim = avg(ms_ssim_vals)
        std_ms_ssim = std(ms_ssim_vals)
        avg_gmsd = avg(gmsd_vals)
        std_gmsd = std(gmsd_vals)
        avg_ms_gmsd = avg(ms_gmsd_vals)
        std_ms_gmsd = std(ms_gmsd_vals)
        avg_haar = avg(haar_vals)
        std_haar = std(haar_vals)
        avg_mdsi = avg(mdsi_vals)
        std_mdsi = std(mdsi_vals)
        avg_dists = avg(dists_vals)
        std_dists = std(dists_vals)
        print(
            """Secondary Results {0}:
            MS-SSIM = {1:.4f} +/- {2:.4f}
            FSIM = {3:.4f} +/- {4:.4f}
            GMSD = {5:.4f} +/- {6:.4f}
            MS-GMSD = {7:.4f} +/- {8:.4f}
            HAAR = {9:.4f}  +/- {10:.4f}
            MDSI = {11:.4f}  +/- {12:.4f}
            DISTS = {13:.4f}  +/- {14:.4f}""".format(
                "",
                avg_ms_ssim,
                std_ms_ssim,
                99999.9,
                99999.9,
                avg_gmsd,
                std_gmsd,
                avg_ms_gmsd,
                std_ms_gmsd,
                avg_haar,
                std_haar,
                avg_mdsi,
                std_mdsi,
                avg_dists,
                std_dists,
            )
        )

    def save_snapshots(self):
        save_dir = self.config.dirs.test_output_samples_dir

        if self.uses_secondary_dataset:
            save_dir = save_dir[:-2] + "_secondary/"
            os.mkdir(save_dir)

        tensors = []
        with torch.no_grad():
            for index in self.sample_indices:
                image_path = os.path.join(save_dir, "sample_{}.png".format(index))

                (sample_input, sample_gt) = self.dataset.__getitem__(index)

                di = sample_input[3:6, :].to(self.device).clamp(0.0, 1.0)

                tensors.append(tosRGB_tensor(di.clone()))

                sample_gt: Tensor = self.io_transform.transform_gt_eval(sample_gt, visualize=True)
                sample_input: Tensor = self.io_transform.transform_input(sample_input.unsqueeze(0))
                sample_output: Tensor = self.io_transform.transform_output_eval(
                    output=self.net(sample_input), visualize=True
                ).squeeze()

                self.io_transform.clear()

                tensors.append(tosRGB_tensor(sample_gt))
                tensors.append(tosRGB_tensor(sample_output))

                tensors.extend(self.make_diff_images(sample_output, di, sample_gt))

                grid_tensor = vutils.make_grid(tensors, nrow=3)
                vutils.save_image(grid_tensor, image_path)
                tensors.clear()

    def make_diff_images(self, target: Tensor, source_1: Tensor, source_2) -> List[Tensor]:
        diff_1: Tensor = source_1 - target
        diff_2: Tensor = source_2 - target

        # colors = {0: "Reds", 1: "Greens", 2: "Blues"}

        # per_channel_diffs = [self.make_monochrome_diff_per_channel(diff_1[i], colors[i]) for i in range(3)]
        # per_channel_diffs.extend([self.make_monochrome_diff_per_channel(diff_2[i], colors[i]) for i in range(3)])

        # return per_channel_diffs
        diff_1_map = self.make_distance_based_diff_image(diff_1)
        diff_2_map = self.make_distance_based_diff_image(diff_2)
        empty_tensor = torch.ones_like(diff_2_map)

        return [diff_1_map, diff_2_map, empty_tensor]

    def make_monochrome_diff_per_channel(self, diff_t: Tensor, colormap_name: str) -> Tensor:
        return self.as_color_mapped_image(diff_t, colormap_name)

    def make_distance_based_diff_image(self, diff_t: Tensor) -> Tensor:
        xs_sq: Tensor = diff_t[0].pow(2)
        ys_sq: Tensor = diff_t[1].pow(2)
        zs_sq: Tensor = diff_t[2].pow(2)
        normed_distances_map = torch.sqrt(xs_sq + ys_sq + zs_sq) / 1.73205080757  # sqrt(3)

        return self.as_color_mapped_image(normed_distances_map, "jet")

    def as_color_mapped_image(self, t: Tensor, colormap_name: str) -> Tensor:
        cm_hot = get_cmap(colormap_name)
        t = t.cpu()
        t_np = t.numpy()
        t_np = cm_hot(t_np)
        t_ten = torch.from_numpy(t_np)
        t_ten = t_ten.to(self.device)
        return t_ten.permute((2, 0, 1))[0:3, :]
