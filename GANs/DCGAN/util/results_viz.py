#%%
import matplotlib.pyplot as plt

from os import listdir
from os import path
import json

from dataclasses import dataclass
from itertools import groupby
from typing import List

@dataclass
class ExperimentData:
    dataset: str
    batch_size: int
    num_epochs: int
    fid_score: float
    time: float

############################################################################
# Methods for automatic building of plots and figures for the results data #
############################################################################

def gather_results_by_dataset(results_dir: str, dataset_keyword: str):
    dirs = [f for f in listdir(results_dir) if path.isdir(path.join(results_dir, f))]
    experiment_datas: list[ExperimentData] = []

    for dir in dirs:
        config_filename = path.join(results_dir, dir, "config.json")
        results_filename = path.join(results_dir, dir, "results.json")
        with open(config_filename) as json_config, open(results_filename) as json_results:
            config = json.load(json_config)
            results = json.load(json_results)
            # new version patch
            dataroot = config["dataroot"]["dataset_root"] if isinstance(config["dataroot"], dict) else config["dataroot"]
            data = ExperimentData(
                dataroot, config["batch_size"], config["num_epochs"], results["fid"], results["time"]
            )
            experiment_datas.append(data)

    target_data = [data for data in experiment_datas if dataset_keyword in data.dataset]
    plot_fid_by_epochs(target_data)
    plot_training_time_by_batch(target_data)


def plot_fid_by_epochs(data: List[ExperimentData]):
    fig = plt.figure(figsize=(9, 5))
    plt.title("FID score by epoch")
    data_sorted = sorted(data, key=lambda x: x.batch_size)
    for batch_size, group in groupby(data_sorted, lambda x: x.batch_size):
        epochs_fids = [(p.num_epochs, p.fid_score) for p in list(group)]
        epochs_fids_sorted = sorted(epochs_fids, key=lambda x : x[0])
        epochs, fid_scores = zip(*epochs_fids_sorted)
        plt.plot(epochs, fid_scores, label=batch_size)

    plt.legend(title="Batch size")
    plt.show()
    plt.close(fig)

def plot_training_time_by_batch(data: List[ExperimentData]):
    fig = plt.figure(figsize=(9, 5))
    plt.title("Training time by batch size")
    data_sorted = sorted(data, key=lambda x: x.batch_size)
    for batch_size, group in groupby(data_sorted, lambda x: x.batch_size):
        epochs_times = [(p.num_epochs, p.time) for p in list(group)]
        epochs_times_sorted = sorted(epochs_times, key=lambda x : x[0])
        epochs, times = zip(*epochs_times_sorted)
        plt.plot(epochs, times, label=batch_size)

    plt.legend(title="Batch size")
    plt.show()
    plt.close(fig)

#%%
gather_results_by_dataset("/home/ksp/Thesis/src/Thesis/GANs/DCGAN/results/ttur/", "cat")


# %%
