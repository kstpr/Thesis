# %%
from cgi import test
from torchsummary.torchsummary import summary
from trainers.Pix2PixTrainer import Pix2PixTrainer
from trainers.Pix2PixHDTrainer import Pix2PixHDTrainer
from DeepIllumination.networks import (
    define_D as pix2pix_define_D,
    define_G as pix2pix_define_G,
)
from DeepCG.networks import (
    define_D as pix2pixHD_define_D,
    define_G as pix2pixHD_define_G,
)
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

import wandb
from dacite import from_dict
from wandb.sdk.wandb_run import Run

from evaluator import Evaluator
from DeepShadingBased.DS_UNet import UNet
from GIDatasetCached import DatasetType, GIDatasetCached
from GIDataset import ALL_INPUT_BUFFERS_CANONICAL, BufferType
from configs.config import Config, ConfigGAN, Directories
from trainers.ConvNetTrainer import ConvNetTrainer
import netutils
from ResUNet import ResUNet
from enums import Activation, NormType, OptimizerType
from utils.utils import (
    setup_directories_with_timestamp,
    undo_export_transforms_for_nd_arrays,
    sanity_check,
    # sanity_check_2,
    find_mean_and_stdev,
)


def setup_wandb(config: Config) -> Run:
    run = wandb.init(
        settings=wandb.Settings(start_method="fork"),
        entity="kpresnakov",
        project="gi",
        reinit=True,
    )

    wandb_config = wandb.config  # Initialize config
    wandb_config.batch_size = (
        config.batch_size
    )  # input batch size for training (default: 64)
    wandb_config.epochs = config.num_epochs  # number of epochs to train (default: 10)
    wandb_config.log_interval = (
        config.batches_log_interval
    )  # how many batches to wait before logging training status
    wandb_config.decr = config.descr

    return run


def load_config_from_saved_net_descr(checkpoint: dict) -> Config:
    config: Config = from_dict(data_class=Config, data=checkpoint["config"])
    return config


def load_net_from_saved_net_descr(
    checkpoint: dict, blueprint_net: nn.Module
) -> nn.Module:
    blueprint_net.load_state_dict(checkpoint["model_state_dict"])
    return blueprint_net


def load_saved_network(
    checkpoint_path: str, num_channels=16
) -> Tuple[nn.Module, torch.device, Config]:
    print("Loading {}...".format(checkpoint_path))
    checkpoint: dict = torch.load(checkpoint_path)
    config = load_config_from_saved_net_descr(checkpoint)

    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and config.num_gpu > 0) else "cpu"
    )
    # TODO Manually instantiates the network, should be automated
    # net =  ResUNet(num_channels, final_activation=Activation.TANH, levels=6).to(device)
    net = UNet(num_channels)
    net = load_net_from_saved_net_descr(checkpoint, net)
    net = net.to(device)

    return (net, device, config)


def new_config(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    dirs: Directories,
    descr: str,
    num_epochs: int = 100,
    use_lr_scheduler: bool = False,
) -> Config:
    config: Config = Config(
        num_gpu=1,
        num_workers_train=6,
        num_workers_validate=6,
        num_workers_test=6,
        batch_size=8,
        num_epochs=num_epochs,
        learning_rate=0.01,
        use_validation=True,
        use_lr_scheduler=use_lr_scheduler,
        alpha=alpha,  # SDSIM weight
        beta=beta,  # L1 weight
        gamma=gamma,  # L2 weight
        delta=delta,  # Additional term weight
        dirs=dirs,
        num_network_snapshots=5,
        image_snapshots_interval=1,
        descr=descr,
    )

    return config


def new_gan_config(
    dirs: Directories, descr: str, num_epochs: int = 100, log_interval=50
) -> ConfigGAN:
    config: ConfigGAN = ConfigGAN(
        num_gpu=1,
        num_workers_train=6,
        num_workers_validate=6,
        num_workers_test=6,
        batch_size=7,
        num_epochs=num_epochs,
        learning_rate=0.01,  # Not used for GANs
        use_validation=True,
        use_lr_scheduler=True,
        alpha=-1,  # SDSIM weight Not used for GANs
        beta=-1,  # L1 weight Not used for GANs
        gamma=-1,  # L2 weight Not used for GANs
        delta=-1,  # Additional term weight Not used for GANs
        dirs=dirs,
        num_network_snapshots=5,
        image_snapshots_interval=1,
        batches_log_interval=log_interval,
        descr=descr,
        lr_optim_D=0.0002,
        lr_optim_G=0.0002,
        optim_D_beta_1=0.5,
        optim_D_beta_2=0.999,
        optim_G_beta_1=0.5,
        optim_G_beta_2=0.999,
        lambda_l1=100.0,
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

    reduced_buffer = [
        BufferType.ALBEDO,
        BufferType.DI,
        BufferType.CS_NORMALS
    ]

    train_dataset_cached = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/NdArrays/Train/",
        input_buffers=reduced_buffer,
        use_hdr=True,
        resolution=512,
        type=DatasetType.ND_ARRAYS,
    )

    validation_dataset_cached = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/NdArrays/Validate/",
        input_buffers=reduced_buffer,
        use_hdr=True,
        resolution=512,
        type=DatasetType.ND_ARRAYS,
    )

    test_dataset_cached = GIDatasetCached(
        "/home/ksp/Thesis/data/GI/NdArrays/Test/",
        input_buffers=reduced_buffer,
        use_hdr=True,
        resolution=512,
        type=DatasetType.ND_ARRAYS,
    )
    # find_mean_and_stdev(test_dataset_cached)
    # sanity_check_2(train_dataset_cached, device)
    # sanity_check(25, train_dataset_cached, buffers_list)

    #    utils.benchmark_num_workers(16, train_dataset_cached)
    # utils.benchmark_num_workers(16, train_dataset_cached_arrays)

    kk: int = 1

    if kk == 0:
        run_saved(train_dataset_cached, validation_dataset_cached, test_dataset_cached)
    elif kk == 1:
        run_new(
            train_dataset_cached,
            validation_dataset_cached,
            test_dataset_cached,
            reduced_buffer,
        )
    elif kk == 2:
        eval_saved(test_dataset_cached, validation_dataset_cached)

def eval_saved(test_dataset: Dataset, validation_dataset: Dataset):
    model_path = "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepCG/results/pix2pixHD_06_16_2021__00_13_25_pix2pixHD all losses vgg + feat + gan/network_snapshots/snapshot_epoch_100.tar"
    num_channels = get_num_channels(ALL_INPUT_BUFFERS_CANONICAL)
    device = torch.device("cuda:0")
    io_transform: netutils.IOTransform = netutils.ClampGtTransform(device=device)
    
    checkpoint: dict = torch.load(model_path)

    netG = pix2pixHD_define_G(
        input_nc=num_channels, output_nc=3, ngf=64, netG="global", gpu_ids=[0]
    )
    netG.load_state_dict(checkpoint["model_G_state_dict"])

    config = from_dict(data_class=ConfigGAN, data=checkpoint["config"])
    Evaluator(
        config,
        netG,
        test_dataset,
        device,
        io_transform,
        save_results=True,
        uses_secondary_dataset=False,
    ).eval()
    Evaluator(
        config,
        netG,
        validation_dataset,
        device,
        io_transform,
        save_results=True,
        uses_secondary_dataset=True,
    ).eval()

def run_saved(
    train_dataset: Dataset, validation_dataset: Dataset, test_dataset: Dataset
):
    model_path = "/home/ksp/Thesis/src/Thesis/GI_Nets/DeepCG/results/pix2pixHD_06_16_2021__00_13_25_pix2pixHD all losses vgg + feat + gan/network_snapshots/snapshot_epoch_80.tar"

    num_channels = get_num_channels(ALL_INPUT_BUFFERS_CANONICAL)
    device = torch.device("cuda:0")
    io_transform: netutils.IOTransform = netutils.ClampGtTransform(device=device)

    netD = pix2pixHD_define_D(
        input_nc=num_channels + 3,
        ndf=64,
        num_D=3,
        n_layers_D=3,
        getIntermFeat=True,
        gpu_ids=[0],
    )
    netG = pix2pixHD_define_G(
        input_nc=num_channels, output_nc=3, ngf=64, netG="global", gpu_ids=[0]
    )

    checkpoint: dict = torch.load(model_path)
    config = from_dict(data_class=ConfigGAN, data=checkpoint["config"])
    config.batch_size = 6

    run = setup_wandb(config)

    trainer = Pix2PixHDTrainer(
        config=config,
        device=device,
        train_dataset=train_dataset,
        num_channels=num_channels,
        netG=netG,
        netD=netD,
        io_transform=io_transform,
        validation_dataset=validation_dataset,
    )

    trainer.load_saved_model(model_path)
    trainer.continue_training_loaded_model()

    Evaluator(
        config,
        netG,
        test_dataset,
        device,
        io_transform,
        save_results=True,
        uses_secondary_dataset=False,
    ).eval()
    Evaluator(
        config,
        validation_dataset,
        device,
        io_transform,
        save_results=True,
        uses_secondary_dataset=True,
    ).eval()

    run.finish()


def run_new(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    buffers_list: List[BufferType],
):
    dirs3: Directories = setup_directories_with_timestamp(
        results_root="/home/ksp/Thesis/src/Thesis/GI_Nets/DeepCG/results/",
        experiment_name="pix2pixHD_AShNCs",
        additional_data="reduced_buffer_albedo_shading_mask_normalsCS",
    )

    log_interval = len(train_dataset) // (8 * 10)
    configs = [
        new_gan_config(dirs3, "pix2pixHD_DeepCG_buffer", num_epochs=100, log_interval=log_interval)
        # new_config(
        #     alpha=1.0,
        #     beta=0.5,
        #     gamma=0.0,
        #     delta=0.0,
        #     dirs=dirs3,
        #     descr="""ResUNet 6 layers, full buffer with Loss = SDSIM + 0.5 L1 + 0.5 LPIPS. Batch Norm. Adam optimizer with lr sched [6, 20, 40, 70], gamma = 0.2.""",
        #     use_lr_scheduler=True,
        #     num_epochs=1,
        # ),
    ]

    device = torch.device(
        "cuda:0"
    )  # if (torch.cuda.is_available() and config.num_gpu > 0) else "cpu")

    for (num, config) in enumerate(configs):
        run = setup_wandb(config)
        num_channels = get_num_channels(buffers_list)

        if num == 0:
            io_transform: netutils.IOTransform = netutils.InputDIAsMaskTransform(
                device=device
            )
            # net = UNet(num_channels)
            # net = ResUNet(num_channels, final_activation=Activation.SIGMOID, levels=6, norm_type=NormType.BATCH)

            # netG = pix2pix_define_D(input_nc=num_channels, netG="unet_256", norm="instance", output_nc=3, ngf=64, use_dropout=True, use_upsampling=False)
            # netD = pix2pix_define_G(input_nc=num_channels + 3, ndf=64, norm="instance", netD="basic")
            netD = pix2pixHD_define_D(
                input_nc=num_channels + 3,
                ndf=64,
                num_D=3,
                n_layers_D=3,
                getIntermFeat=True,
                gpu_ids=[0],
            )
            netG = pix2pixHD_define_G(
                input_nc=num_channels, output_nc=3, ngf=64, netG="global", gpu_ids=[0]
            )

        # net = net.to(device)
        # netG = netG.to(device)
        # netD = netD.to(device)

        trainer = Pix2PixHDTrainer(
            config=config,
            device=device,
            train_dataset=train_dataset,
            num_channels=num_channels,
            netG=netG,
            netD=netD,
            io_transform=io_transform,
            validation_dataset=validation_dataset,
        )

        trainer.train()

        # trainer = ConvNetTrainer(
        #     config=config,
        #     train_dataset=train_dataset,
        #     net=net,
        #     num_channels=num_channels,
        #     device=device,
        #     io_transform=io_transform,
        #     validation_dataset=validation_dataset,
        #     optimizer_type=OptimizerType.ADAM
        # )

        # trainer2 = Pix2PixTrainer(
        #     config=config,
        #     device=device,
        #     train_dataset=train_dataset,
        #     num_channels=num_channels,
        #     netG=net2,
        #     netD=net2,
        #     io_transform=io_transform,
        #     validation_dataset=validation_dataset,
        # )

        # Evaluator(
        #     config, net, test_dataset, device, io_transform, save_results=True, uses_secondary_dataset=False
        # ).test_inference()

        # trainer.train()

        Evaluator(
            config,
            netG,
            test_dataset,
            device,
            io_transform,
            save_results=True,
            uses_secondary_dataset=False,
        ).eval()
        Evaluator(
            config,
            netG,
            validation_dataset,
            device,
            io_transform,
            save_results=True,
            uses_secondary_dataset=True,
        ).eval()

        run.finish()


# tensors = dataset.__getitem__(147)
# matplotlib.image.imsave('gt.png', tosRGB(tensors[1].permute(1,2,0).numpy()))


def get_num_channels(buffers: List[BufferType]) -> int:
    return sum([3 if buffer != BufferType.DEPTH else 1 for buffer in buffers])


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    run()

# %%
