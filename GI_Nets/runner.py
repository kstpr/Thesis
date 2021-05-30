# %%
from ResUNet import ResUNet
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

import wandb
from dacite import from_dict

from DeepShadingUNetBN import NormType, UNetNorm
from evaluator import Evaluator
from DeepShadingUNet import UNet
from GIDatasetCached import DatasetType, GIDatasetCached
from GIDataset import BufferType
from config import Config, Directories
from trainer import Trainer
import netutils
from utils import (
    setup_directories_with_timestamp,
    undo_export_transforms_for_nd_arrays,
    sanity_check,
    sanity_check_2,
    find_mean_and_stdev,
)


def setup_wandb(config: Config):
    wandb.init(settings=wandb.Settings(start_method="fork"), entity="kpresnakov", project="gi")

    wandb_config = wandb.config  # Initialize config
    wandb_config.batch_size = config.batch_size  # input batch size for training (default: 64)
    wandb_config.epochs = config.num_epochs  # number of epochs to train (default: 10)
    wandb_config.log_interval = config.batches_log_interval  # how many batches to wait before logging training status


def load_config_from_saved_net_descr(checkpoint: dict) -> Config:
    config: Config = from_dict(data_class=Config, data=checkpoint["config"])
    return config


def load_net_from_saved_net_descr(checkpoint: dict, blueprint_net: nn.Module) -> nn.Module:
    blueprint_net.load_state_dict(checkpoint["model_state_dict"])
    return blueprint_net


def load_saved_network(checkpoint_path: str, num_channels=16) -> Tuple[nn.Module, torch.device, Config]:
    print("Loading {}...".format(checkpoint_path))
    checkpoint: dict = torch.load(checkpoint_path)
    config = load_config_from_saved_net_descr(checkpoint)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.num_gpu > 0) else "cpu")
    # TODO Manually instantiates the network, should be automated
    net = UNet(num_channels).to(device)
    net = load_net_from_saved_net_descr(checkpoint, net)

    return (net, device, config)


def new_config(alpha: float, beta: float, gamma: float, name: str, descr: str) -> Config:
    dirs: Directories = setup_directories_with_timestamp(
        results_root="/home/ksp/Thesis/src/Thesis/GI_Nets/DeepShadingBased/results/masks/",
        experiment_name="unet",
        additional_data=name,
    )

    config: Config = Config(
        num_gpu=1,
        num_workers_train=6,
        num_workers_validate=6,
        num_workers_test=6,
        batch_size=8,
        num_epochs=200,
        learning_rate=0.01,
        use_validation=True,
        alpha=alpha,  # SDSIM weight
        beta=beta,  # L1 weight
        gamma=gamma,  # L2 weight
        dirs=dirs,
        num_network_snapshots=5,
        image_snapshots_interval=1,
        descr=descr,
    )

    return config


def run():
    buffers_list = [
        BufferType.ALBEDO,
        BufferType.DI,
        BufferType.WS_NORMALS,
        BufferType.CS_NORMALS,
        BufferType.CS_POSITIONS,
        BufferType.DEPTH,
    ]

    train_dataset_cached = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/NdArrays/Train/",
        input_buffers=buffers_list,
        use_hdr=True,
        resolution=512,
        type=DatasetType.ND_ARRAYS,
    )

    validation_dataset_cached = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/NdArrays/Validate/",
        input_buffers=buffers_list,
        use_hdr=True,
        resolution=512,
        type=DatasetType.ND_ARRAYS,
    )

    test_dataset_cached = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/NdArrays/Test/",
        input_buffers=buffers_list,
        use_hdr=True,
        resolution=512,
        type=DatasetType.ND_ARRAYS,
    )
    # find_mean_and_stdev(test_dataset_cached)
    # sanity_check_2(train_dataset_cached, device)
    # sanity_check(25, train_dataset_cached, buffers_list)

    #    utils.benchmark_num_workers(16, train_dataset_cached)
    # utils.benchmark_num_workers(16, train_dataset_cached_arrays)

    load_net: bool = False

    if load_net:
        run_saved(test_dataset_cached)
    else:
        run_new(train_dataset_cached, validation_dataset_cached, test_dataset_cached, buffers_list)

    # trainer.continue_training_loaded_model()


def run_saved(test_dataset: Dataset):
    net, device, config = load_saved_network(
        "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepShadingBased/results/masks/unet_05_30_2021__04_08_15_plain/network_snapshots/best_net_epoch_160.tar"
    )
    io_transform: netutils.IOTransform = netutils.ClampGtTransform(device=device)
    Evaluator(config, net, test_dataset, device, io_transform, save_results=True, uses_secondary_dataset=False).eval()
    Evaluator(config, net, test_dataset, device, io_transform, save_results=True, uses_secondary_dataset=True).eval()


def run_new(train_dataset: Dataset, validation_dataset: Dataset, test_dataset: Dataset, buffers_list: List[BufferType]):
    configs = [
        new_config(
            alpha=1.0,
            beta=0.5,
            gamma=0.0,
            name="plain",
            descr="""Plain Deep shading, ResUNet, with Loss = SSIM + 0.5 L1. Adam optimizer.""",
        ),
        # new_config(
        #     alpha=1.0,
        #     beta=0.5,
        #     gamma=0.0,
        #     name="add_oc_no_albedo",
        #     descr="""Plain Deep shading, last activation is tanh, full buffer OC with Loss = SSIM + 0.5 L1. Adadelta optimizer.""",
        # ),
        # new_config(
        #     alpha=1.0,
        #     beta=0.5,
        #     gamma=0.0,
        #     name="add_mask_batch",
        #     descr="""Plain Deep shading, BN, last activation is tanh, full buffer with add mask instead of DI,
        #     network output interpreted as mask and
        #     compared to gi mask directly, SSIM kernel size 7, Loss = SSIM + 0.5 L1. Adadelta optimizer.""",
        # ),
        # new_config(
        #     alpha=1.0,
        #     beta=0.5,
        #     gamma=0.0,
        #     name="add_mask_instance",
        #     descr="""Plain Deep shading, BN, last activation is tanh, full buffer with add mask instead of DI,
        #     network output interpreted as mask and
        #     compared to gi mask directly, SSIM kernel size 7, Loss = SSIM + 0.5 L1. Adadelta optimizer.""",
        # ),
        # new_config(
        #     alpha=1.0,
        #     beta=0.5,
        #     gamma=0.0,
        #     name="no_mask_batch",
        #     descr="""Plain Deep shading, BN, last activation is tanh, full buffer with mult mask instead of DI,
        #     network output interpreted as mask and
        #     compared to gi mask directly, SSIM kernel size 7, Loss = SSIM + 0.5 L1. Adadelta optimizer.""",
        # ),
        # new_config(
        #     alpha=1.0,
        #     beta=0.5,
        #     gamma=0.0,
        #     name="no_mask_instance",
        #     descr="""Plain Deep shading, BN, last activation is tanh, full buffer with mult mask instead of DI,
        #     network output interpreted as mask and
        #     compared to gi mask directly, SSIM kernel size 7, Loss = SSIM + 0.5 L1. Adadelta optimizer.""",
        # ),
    ]

    num_channels = get_num_channels(buffers_list)
    device = torch.device("cuda:0")  # if (torch.cuda.is_available() and config.num_gpu > 0) else "cpu")

    for (num, config) in enumerate(configs):
        setup_wandb(config)
        if num == 0:
            io_transform: netutils.IOTransform = netutils.ClampGtTransform(device=device)#, remove_albedo=False)
            net = ResUNet(num_channels)
        # elif num == 1:
        #     io_transform: netutils.IOTransform = netutils.AdditiveDiMaskTransform(device=device, remove_albedo=True)
        #     num_channels = num_channels - 3
        #     net = UNet(num_channels)
        # elif num == 2:
        #     io_transform: netutils.IOTransform = netutils.AdditiveDiGtMaskTransform(device=device)
        #     net = UNetNorm(num_channels, norm_type=NormType.BATCH)
        # elif num == 3:
        #     io_transform: netutils.IOTransform = netutils.AdditiveDiGtMaskTransform(device=device)
        #     net = UNetNorm(num_channels, norm_type=NormType.INSTANCE)
        # elif num == 4:
        #     io_transform: netutils.IOTransform = netutils.ClampGtTransform(device=device)
        #     net = UNetNorm(num_channels, norm_type=NormType.BATCH)
        # elif num == 5:
        #     io_transform: netutils.IOTransform = netutils.ClampGtTransform(device=device)
        #     net = UNetNorm(num_channels, norm_type=NormType.INSTANCE)

        net = net.to(device)

        trainer = Trainer(
            config=config,
            train_dataset=train_dataset,
            net=net,
            num_channels=num_channels,
            device=device,
            io_transform=io_transform,
            validation_dataset=validation_dataset,
            outputs_masks=False,
        )

        trainer.train()

        Evaluator(
            config, net, test_dataset, device, io_transform, save_results=True, uses_secondary_dataset=False
        ).eval()
        Evaluator(
            config, net, test_dataset, device, io_transform, save_results=True, uses_secondary_dataset=True
        ).eval()


# tensors = dataset.__getitem__(147)
# matplotlib.image.imsave('gt.png', tosRGB(tensors[1].permute(1,2,0).numpy()))


def get_num_channels(buffers: List[BufferType]) -> int:
    return sum([3 if buffer != BufferType.DEPTH else 1 for buffer in buffers])


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    run()

# %%
