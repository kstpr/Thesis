# %%
import dataclasses
from datetime import datetime
from os import makedirs, path
from timeit import default_timer as timer

import torch

from config import Config, DataRoot
from dcgan import DCGAN
from dcgan_test import DCGAN_tryout
from dcgan_wgan import DCGAN_WGAN
from datawriter import DataWriter

from pytorch_fid.fid_score import calculate_fid_given_paths
import wandb

general_dataroot = "/home/ksp/Thesis/data/"

cats_dataroot = DataRoot(
    dataset_root=path.join(general_dataroot, "cats_only"),
    validation_dataset_root=path.join(general_dataroot, "cats_only_val"),
    immediate_dir=path.join(general_dataroot, "cats_only", "cat"),
)
dogs_dataroot = DataRoot(
    dataset_root=path.join(general_dataroot, "dogs_only"),
    validation_dataset_root=path.join(general_dataroot, "dogs_only_val"),
    immediate_dir=path.join(general_dataroot, "dogs_only", "dog"),
)
wildlife_dataroot = DataRoot(
    dataset_root=path.join(general_dataroot, "wildlife_only"),
    validation_dataset_root=path.join(general_dataroot, "wildlife_only_val"),
    immediate_dir=path.join(general_dataroot, "wildlife_only", "wild"),
)

results_root = "/home/ksp/Thesis/src/Thesis/GANs/DCGAN/results/"


def setup_directories_with_timestamp(experiment_batch_name: str = ""):
    timestamp: str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    experiment_root = "%s%s/" % (
        results_root,
        experiment_batch_name + timestamp,
    )

    intermediates_root = experiment_root + "intermediates/"
    output_root = experiment_root + "output/"
    network_snapshots_root = experiment_root + "network_snapshots/"

    makedirs(intermediates_root, exist_ok=True)
    makedirs(output_root, exist_ok=True)
    makedirs(network_snapshots_root, exist_ok=True)

    return {
        "experiment_root": experiment_root,
        "intermediates": intermediates_root,
        "output": output_root,
        "network_snapshots": network_snapshots_root,
    }


def run_dcgan(config: Config) -> float:
    """Trains the GAN, saves intermediate results and generates 2048 fake samples from the generator
    Returns total training time
    """
    dcgan = DCGAN_WGAN(config=config)
    start = timer()
    dcgan.train_and_plot()
    elapsed_time: float = timer() - start
    dcgan.generate_fake_results()

    return elapsed_time


blueprint_config: Config = Config(
    num_gpu=1,
    # network params
    g_feat_maps=64,# * 2,
    d_feat_maps=64,
    num_channels=3,
    # directories
    dataroot=cats_dataroot,
    experiment_output_root="placeholder",
    intermediates_root="placeholder",
    output_root="placeholder",
    netowrk_snapshots_root="placeholder",
    # ...
    dataloader_num_workers=12,
    use_label_smoothing=False,
    image_size=64,
    latent_size=100,
    # training params
    use_validation=False,
    batch_size=64,
    num_epochs=1000,
    # learning rate
    g_learning_rate=0.0002,
    d_learning_rate=0.0002,
    # Adam
    g_beta_1=0.5,
    g_beta_2=0.999,
    d_beta_1=0.5,
    d_beta_2=0.999,
    # linear decay block
    lr_linear_decay_enabled=False,
    g_lr_decay_start_epoch=100,
    d_lr_decay_start_epoch=100,
)

batch_sizes = [-1]
latent_sizes = [-1]  # [25, 75, 125, 150, 175, 200]
# d_learning_rates = [0.0001, 0.0002, 0.0005, 0.001]
# g_learning_rates = [0.0001, 0.0002, 0.0005, 0.001]


def setup_wandb(config: Config):
    wandb.init(settings=wandb.Settings(start_method="fork"), entity="kpresnakov", project="test")

    wandb_config = wandb.config  # Initialize config
    wandb_config.batch_size = config.batch_size  # input batch size for training (default: 64)
    wandb_config.epochs = config.num_epochs  # number of epochs to train (default: 10)
    wandb_config.g_lr = config.g_learning_rate
    wandb_config.d_lr = config.d_learning_rate  # learning rate (default: 0.01)
    wandb_config.log_interval = 10  # how many batches to wait before logging training status


def run_experiments(tag: str) -> None:
    for latent_size in latent_sizes:
        for batch_size in batch_sizes:
            config = dataclasses.replace(blueprint_config)

            # config.latent_size = latent_size

            dirs = setup_directories_with_timestamp("%s_bs_%d_epochs_%d" % (tag, 16, 200))

            config.intermediates_root = dirs["intermediates"]
            config.output_root = dirs["output"]
            config.experiment_output_root = dirs["experiment_root"]
            config.netowrk_snapshots_root = dirs["network_snapshots"]

            setup_wandb(config)

            data_writer = DataWriter(config=config)
            data_writer.serialize_config()

            print(config)

            training_time = run_dcgan(config=config)
            fid = calculate_fid_given_paths([config.dataroot.immediate_dir, config.output_root], 50, "cuda:0", 2048)

            data_writer.serialize_results(time=training_time, fid=fid)

            print("FID: %f" % fid)


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    run_experiments("cats_wgan_")


# %%
