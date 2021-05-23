# %%
from test import scale_0_1
from numpy.lib import utils
from torch import nn
from torch.tensor import Tensor
from evaluator import Evaluator
from DeepShadingUNet import UNet
from os import makedirs
from typing import Dict, List, Tuple
from datetime import datetime
from os.path import join

import torch
from torch.utils.data.dataset import Dataset
import torchvision.utils as vutils

import wandb
from dacite import from_dict

from GIDatasetCached import DatasetType, GIDatasetCached
from GIDataset import ALL_INPUT_BUFFERS_CANONICAL, BufferType, GIDataset
from config import Config, Directories
from trainer import Trainer
import utils
import netutils


def setup_wandb(config: Config):
    wandb.init(settings=wandb.Settings(start_method="fork"), entity="kpresnakov", project="gi")

    wandb_config = wandb.config  # Initialize config
    wandb_config.batch_size = config.batch_size  # input batch size for training (default: 64)
    wandb_config.epochs = config.num_epochs  # number of epochs to train (default: 10)
    wandb_config.log_interval = config.batches_log_interval  # how many batches to wait before logging training status


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

def load_config_from_saved_net_descr(checkpoint: dict) -> Config:
    config: Config = from_dict(data_class=Config, data=checkpoint["config"])
    return config


def load_net_from_saved_net_descr(checkpoint: dict, blueprint_net: nn.Module) -> nn.Module:
    blueprint_net.load_state_dict(checkpoint["model_state_dict"])
    return blueprint_net


def train_new_network(
    config: Config, train_dataset: Dataset, validation_dataset: Dataset, num_channels: int
) -> Tuple[nn.Module, torch.device, netutils.IOTransform]:
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.num_gpu > 0) else "cpu")

    net = UNet(num_channels).to(device)

    # sanity_check(50, train_dataset_cached, buffers_list)
    io_transform: netutils.IOTransform = netutils.AdditiveDiMaskTransform(device=device)

    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        net=net,
        num_channels=num_channels,
        device=device,
        io_transform=io_transform,
        validation_dataset=validation_dataset,
    )

    # nw = trainer.benchmark_num_workers(train_dataset_cached)
    # print("Best num workers for bs {} : {}".format(config.batch_size, nw))

    trainer.train()

    return (net, device, io_transform)


def load_saved_network(checkpoint_path: str) -> Tuple[nn.Module, torch.device, Config]:
    print("Loading {}...".format(checkpoint_path))
    checkpoint: dict = torch.load(checkpoint_path)
    config = load_config_from_saved_net_descr(checkpoint)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.num_gpu > 0) else "cpu")
    num_channels = 16
    net = UNet(num_channels).to(device)
    net = load_net_from_saved_net_descr(checkpoint, net)

    return (net, device, config)


# implement def load_and_continue_train_network()...


def sanity_check_2(dataset: Dataset, device):
    image_path = "sanity_test.png"

    tensors = []
    sample_indices = torch.randint(0, len(dataset), (30,))

    for index in sample_indices:
        (sample_input, sample_gt) = dataset.__getitem__(index)

        sample_gt: Tensor = sample_gt

        sample_input: Tensor = netutils.transform_input(sample_input, device=device)

        diffuse_albedo = sample_input.squeeze()[0:3, :].clamp(0.0, 1.0)
        tensors.append(diffuse_albedo)

        di = sample_input.squeeze()[3:6, :]
        di_2 = di.clone().clamp(1.0, 2.0) - 1.0
        di_3 = di.clone()
        utils.scale_0_1_(di_3)
        di_4 = di.clone().cpu().numpy()
        di_4 = torch.from_numpy(utils.tosRGB(di_4)).to(device)

        tensors.append(di)  # di clamped
        tensors.append(di_3)  # scaled
        tensors.append(di_2)  # over-values
        tensors.append(di_4)  # sRGB
        sample_gt_srgb = torch.from_numpy(utils.tosRGB(sample_gt.cpu())).to(device)
        tensors.append(sample_gt_srgb)

        # di_mask = di_mask + torch.div((torch.max(di_mask) - torch.min(di_mask)), 2.0)
        # tensors.append(di)  # di or di mask

        grid_tensor = vutils.make_grid(tensors, nrow=6)

    vutils.save_image(grid_tensor, image_path)


def new_config(alpha: float, beta: float, gamma: float, name: str) -> Config:
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
        batch_size=16,
        num_epochs=200,
        learning_rate=0.01,
        use_validation=True,
        alpha=alpha,  # SDSIM weight
        beta=beta,  # L1 weight
        gamma=gamma,  # L2 weight
        dirs=dirs,
        num_network_snapshots=5,
        image_snapshots_interval=2,
        descr="Plain Deep shading, last activation is tanh, full buffer with additive mask (di - albedo) instead of DI, output = clamp(output + albedo), SSIM kernel size 7, Loss = SSIM + 0.5 L1. Adadelta optimizer.".format(
            name
        ),
    )

    return config

def run():
    # cache_dataset_as_tensors("Test")
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

    #    utils.benchmark_num_workers(16, train_dataset_cached)
    # utils.benchmark_num_workers(16, train_dataset_cached_arrays)

    ########################## LOAD ##########################
    # net, device, config = load_saved_network(
    #     "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepShadingBased/results/unet_05_13_2021__02_33_19_ssim_l1_l2/network_snapshots/snapshot_epoch_200.tar"
    # )
    # io_transform = netutils.ClampGtTransform(device)
    # Evaluator(config, net, test_dataset_cached, device, io_transform=io_transform, save_results=True, uses_secondary_dataset=False).eval()
    ##########################################################
    
    ########################## TRAIN ##########################
    configs = [
        new_config(alpha=1.0, beta=0.5, gamma=0.0, name="additive_mask"),
    ]

    num_channels = get_num_channels(buffers_list)

    for config in configs:
        setup_wandb(config)
        net, device, io_transform = train_new_network(config, train_dataset_cached, validation_dataset_cached, num_channels)

        Evaluator(config, net, test_dataset_cached, device, io_transform, save_results=True, uses_secondary_dataset=False).eval()
    ###########################################################

    # sanity_check_2(train_dataset_cached, device) 

    # trainer.continue_training_loaded_model()


# tensors = dataset.__getitem__(147)
# matplotlib.image.imsave('gt.png', tosRGB(tensors[1].permute(1,2,0).numpy()))


def get_num_channels(buffers: List[BufferType]) -> int:
    return sum([3 if buffer != BufferType.DEPTH else 1 for buffer in buffers])


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    run()

# %%