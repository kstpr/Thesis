import numpy as np
import torch
import os
from os import listdir
from os.path import join, isfile
import shutil
from torch.utils.data import DataLoader
import sys
from timeit import default_timer as timer

from torch.utils.data.dataset import Dataset


def tosRGB(image_arr):
    image_arr = np.clip(image_arr, 0, 1)
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    # from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
    return np.where(image_arr <= 0.0031308, 12.92 * image_arr, 1.055 * np.power(image_arr, 1 / 2.4) - 0.055)


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
