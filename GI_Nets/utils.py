from typing import List
import os
from os import listdir, makedirs
from os.path import join, isfile
import shutil
import sys
from timeit import default_timer as timer
import json
from datetime import datetime

from numpy.lib import utils
import numpy as np

import torch
import torchvision.utils as vutils
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from GIDataset import ALL_INPUT_BUFFERS_CANONICAL, BufferType, GIDataset
import netutils
from config import Directories


def tosRGB(image_arr):
    image_arr = np.clip(image_arr, 0, 1)
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    # from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
    return np.where(image_arr <= 0.0031308, 12.92 * image_arr, 1.055 * np.power(image_arr, 1 / 2.4) - 0.055)


def tosRGB_tensor(image_tensor: Tensor) -> Tensor:
    image_tensor = image_tensor.clamp(0.0, 1.0)
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    # from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
    return torch.where(
        image_tensor <= 0.0031308, 12.92 * image_tensor, 1.055 * torch.pow(image_tensor, 1 / 2.4) - 0.055
    )


def scale_0_1_(t: torch.Tensor):
    size = t.size()
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    if not torch.all(t == 0.0):
        t /= t.max(1, keepdim=True)[0] + 0.0000001
    t = t.view(size)


def shuffle_datasets(dataset_1_path: str, dataset_2_path: str, ratio: float = 0.9) -> None:
    paths_1 = [
        (1, name)
        for name in os.listdir(dataset_1_path)
        if isfile(join(dataset_1_path, name)) and (not name == "dataset_descr.json")
    ]
    len_ds_1 = len(paths_1)
    paths_2 = [
        (0, name)
        for name in os.listdir(dataset_2_path)
        if isfile(join(dataset_2_path, name)) and (not name == "dataset_descr.json")
    ]
    len_ds_2 = len(paths_2)

    paths_combined = paths_1 + paths_2
    print(paths_combined[0:100])
    combined_size = len_ds_1 + len_ds_2

    mask = np.random.choice([0, 1], size=combined_size, p=[ratio, 1.0 - ratio])
    zipped = zip(paths_combined, mask)

    paths = {1: dataset_1_path, 0: dataset_2_path}

    for (from_ds, name), to_ds in zipped:
        if from_ds == to_ds:
            continue

        from_path = join(paths[from_ds], name)
        to_path = join(paths[to_ds], name)

        print("Move {} to {}".format(from_path, to_path))
        shutil.move(from_path, to_path)


def cache_dataset_as_tensors(dir_name: str):
    dataset = GIDataset(
        root_dir="/media/ksp/424C0DBB4C0DAB2D/Thesis/Dataset/{}/".format(dir_name),
        input_buffers=ALL_INPUT_BUFFERS_CANONICAL,
        useHDR=True,
        resolution=512,
    )
    dataset.transform_and_save_dataset_as_tensors("/home/ksp/Thesis/data/GI/PytorchTensors/{}/".format(dir_name))


def cache_single_scene_as_tensors():
    dataset_files = GIDataset(
        "/media/ksp/424C0DBB4C0DAB2D/Thesis/Dataset/Test/",
        input_buffers=ALL_INPUT_BUFFERS_CANONICAL,
        useHDR=True,
        resolution=512,
    )

    dataset_files.transform_and_save_single_scene_buffers_as_tensors(
        scene_path="/media/ksp/424C0DBB4C0DAB2D/Thesis/Dataset/Test/clothes_shop_aug/",
        target_dir="/home/ksp/Thesis/data/GI/PytorchTensors/Test/clothes_shop_aug/",
        add_to_descr=False,
    )


def setup_directories_with_timestamp(results_root: str, experiment_name: str, additional_data: str = "") -> Directories:
    timestamp: str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    experiment_root = join(
        results_root,
        "{}_{}_{}".format(experiment_name, timestamp, additional_data),
    )

    makedirs(experiment_root)

    intermediates_root = experiment_root + "/intermediates/"
    test_output_samlpes_root = experiment_root + "/test_output_samples/"
    network_snapshots_root = experiment_root + "/network_snapshots/"

    makedirs(intermediates_root, exist_ok=True)
    makedirs(test_output_samlpes_root, exist_ok=True)
    makedirs(network_snapshots_root, exist_ok=True)

    # TODO output is not used yet
    return Directories(
        experiment_results_root=experiment_root,
        result_snapshots_dir=intermediates_root,
        network_snapshots_dir=network_snapshots_root,
        test_output_samples_dir=test_output_samlpes_root,
    )


def sanity_check(num_sample_indices: int, train_dataset: Dataset, buffers_list: List[BufferType]):
    print("Sanity check...")

    sample_indices = torch.randint(0, len(train_dataset), (num_sample_indices,))

    save_dir = "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepShadingBased/sanity_check/"
    image_path = save_dir + "sanity_check.png"
    tensors = []

    with torch.no_grad():
        for index in sample_indices:
            (sample_input, sample_gt) = train_dataset.__getitem__(index)
            start_index = 0
            for buffer_type in buffers_list:
                num_channels = 3 if buffer_type != BufferType.DEPTH else 1
                image_tensor = sample_input[start_index : (start_index + num_channels), :]
                if buffer_type == BufferType.DEPTH:
                    # Depth is 1 channel but we need 3 channels to show as an image
                    image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], 0)
                tensors.append(image_tensor.clone())
                start_index += num_channels
            tensors.append(sample_gt)

    grid_tensor = vutils.make_grid(tensors, nrow=(len(buffers_list) + 1))
    vutils.save_image(grid_tensor, image_path)

    print("Sanity check image saved in {}".format(image_path))


def sanity_check_2(dataset: Dataset, device):
    image_path = "sanity_test.png"

    tensors = []
    sample_indices = torch.randint(0, len(dataset), (30,))

    for index in sample_indices:
        (sample_input, sample_gt) = dataset.__getitem__(index)

        sample_gt: Tensor = sample_gt

        io_transform = netutils.ClampGtTransform(device)

        sample_input: Tensor = io_transform.transform_input(sample_input)

        diffuse_albedo = sample_input.squeeze()[0:3, :].clamp(0.0, 1.0)
        tensors.append(diffuse_albedo)

        di = sample_input.squeeze()[3:6, :]
        di_2 = di.clone().clamp(1.0, 2.0) - 1.0
        di_4 = tosRGB_tensor(di.clone())

        tensors.append(di)  # di clamped
        tensors.append(di_2)  # over-values
        tensors.append(di_4)  # sRGB
        sample_gt_srgb = tosRGB_tensor(sample_gt.to(device))
        tensors.append(sample_gt_srgb)

        grid_tensor = vutils.make_grid(tensors, nrow=5)

    vutils.save_image(grid_tensor, image_path)


def reserialize_tensors_dataset_as_ndarrays(tensors_dataset_path: str, ndarrays_dataset_path: str) -> None:
    name_path_pairs = [(name, join(tensors_dataset_path, name)) for name in listdir(tensors_dataset_path)]
    for name, path in name_path_pairs:
        if name.endswith("json"):
            shutil.copyfile(path, join(ndarrays_dataset_path, name))
        elif name.endswith("pt"):
            cached_tensor: torch.Tensor = torch.load(path)
            nd_arr = cached_tensor.numpy()
            pre, ext = os.path.splitext(name)
            nd_arr_name = "{}.npy".format(pre)
            nd_arr_path = join(ndarrays_dataset_path, nd_arr_name)
            with open(nd_arr_path, "wb") as f:
                np.save(f, nd_arr)
                print("Saved {}".format(nd_arr_path))


def undo_export_transforms_for_nd_arrays(nd_arrays_path: str, ndarrays_original_coords_path: str) -> None:
    name_path_pairs = [(name, join(nd_arrays_path, name)) for name in listdir(nd_arrays_path)]
    for name, path in name_path_pairs:
        if name.endswith("json"):
            shutil.copyfile(path, join(ndarrays_original_coords_path, name))
        elif name.endswith("npy"):
            with open(path, "rb") as f:
                array = np.load(f)
                array[6:9, :, :] = (array[6:9, :, :] - 0.5) * 2.0  # ws normals
                array[9:12, :, :] = (array[9:12, :, :] - 0.5) * 2.0  # cs normals
                array[12:14, :, :] = (array[12:14, :, :] - 0.5) * 20.0  # cs positions xy
                array[14, :, :] = array[14, :, :] * 20.0  # cs depth

                nd_arr_path = join(ndarrays_original_coords_path, name)

                with open(nd_arr_path, "wb") as f:
                    np.save(f, array)
                    print("Saved {}".format(nd_arr_path))


### Use to find optimal number of workers for the used dataset and CPU
def benchmark_num_workers(batch_size: int, dataset: Dataset) -> int:
    best_time: float = 10000.0
    best_num_workers = 0

    for num_workers in range(0, 13):
        print("Init DataLoader")
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        print("Ready")

        print("Initializing iter")
        begin = timer()
        for (i, data) in enumerate(dataloader):
            sys.stdout.write("#")
            sys.stdout.flush()
        epoch_time = timer() - begin
        print("Epoch time 1: {0:.4f}".format(epoch_time))

        # simple_iter_start = timer()
        # for i in range(len(dataset)):
        #     data = dataset.__getitem__(i)
        #     if i % 16 == 0:
        #         sys.stdout.write("#")
        #         sys.stdout.flush()
        # print("Simple iter epoch: {0:.4f}".format(timer() - simple_iter_start))

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


def results_to_latex(net_name: str):
    results_path = (
        "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepShadingBased/results/buffer_combinations/{}/results.json".format(
            net_name
        )
    )

    with open(results_path, "r") as read_file:
        results_dict = json.load(read_file)

    results_latex = "\\textbf{{}}          & {0}     & {1}     & {2}     & {3}    & {4}     \\\\ \hdashline".format(
        results_dict["mae"], results_dict["mse"], results_dict["ssim"], results_dict["psnr"], results_dict["lpips"]
    )
    return results_latex


def find_mean_and_stdev(dataset: Dataset) -> None:  # Tuple[Tensor, Tensor]:
    dataloader: DataLoader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=6,
    )

    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for batch, data in enumerate(dataloader):
        data = data[0]
        albedo = data[:, 0:3, :].clamp(0.0, 1.0) + 1.0  # in [0, 1]

        di_mask = torch.div(data[:, 3:6, :].clamp(0.0, 1.0), albedo)  # in [0.5, 2.0]

        data[:, 3:6, :] = di_mask

        batch_samples = data.size(0)
        data: Tensor = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print("Mean: ")
    print(mean)
    print("Std: ")
    print(std)


DATASET_MEAN = torch.tensor(
    [
        0.5569,
        0.4677,
        0.4190,
        0.2718,
        0.2258,
        0.2023,
        0.4931,
        0.4419,
        0.4973,
        0.5009,
        0.4892,
        0.1986,
        0.4995,
        0.5025,
        0.1398,
        0.1559,
    ]
)

DATASET_STD = torch.tensor(
    [
        0.2797,
        0.2778,
        0.2975,
        0.2165,
        0.2012,
        0.2009,
        0.1867,
        0.2166,
        0.1895,
        0.1956,
        0.2342,
        0.1489,
        0.0487,
        0.0443,
        0.0537,
        0.0535,
    ]
)

DATASET_MEAN_MULT_MASK = torch.tensor(
    [
        0.5569,
        0.4677,
        0.4190,
        0.8399,
        0.8594,
        0.8763,
        0.4931,
        0.4419,
        0.4973,
        0.5009,
        0.4892,
        0.1986,
        0.4995,
        0.5025,
        0.1398,
        0.1559,
    ]
)

DATASET_STD_MULT_MASK = torch.tensor(
    [
        0.2797,
        0.2778,
        0.2975,
        0.1187,
        0.1110,
        0.1131,
        0.1867,
        0.2166,
        0.1895,
        0.1956,
        0.2342,
        0.1489,
        0.0487,
        0.0443,
        0.0537,
        0.0535,
    ]
)


if __name__ == "__main__":
    net_name = "unet_05_14_2021__17_19_36_normals_only"
    print(results_to_latex(net_name))