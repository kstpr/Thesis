from datawriter import serialize_and_save_config
from typing import List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import roll
from numpy.core.shape_base import stack
import wandb
import queue


def plot_and_save_losses(
    train_losses,
    validation_losses,
    num_epochs: int,
    training_batches_per_epoch: int,
    output_full_path: str,
    show_train_loss_avg: bool = True,
):
    fig = plt.figure(figsize=(10, 5))
    plt.title("Train and validation losses")

    x1 = np.linspace(0, num_epochs, len(train_losses))
    plt.plot(x1, train_losses, label="Train loss")

    if show_train_loss_avg:
        window_size = 40
        avgs_train_losses = get_rolling_averages(train_losses, window_size=window_size)
        start = window_size / training_batches_per_epoch
        x2 = np.linspace(start, num_epochs, len(avgs_train_losses))
        plt.plot(x2, avgs_train_losses, label="Avg. Train loss")

    x3 = np.linspace(0, num_epochs, len(validation_losses))
    plt.plot(x3, validation_losses, color="red", label="Validation loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_full_path)
    plt.close(fig)

    # wandb.log({"Losses" : wandb.Image(output_full_path)})


def get_rolling_averages(values: List[float], window_size: int) -> List[float]:
    rolling_averages: List[float] = []
    window_queue: List[float] = []

    for i in range(len(values)):
        if len(window_queue) == window_size:
            rolling_averages.append(sum(window_queue) / window_size)
            window_queue.pop(0)
            window_queue.append(values[i])
        else:
            window_queue.append(values[i])

    return rolling_averages
