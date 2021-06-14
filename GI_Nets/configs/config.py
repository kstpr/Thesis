from dataclasses import dataclass

@dataclass
class Directories:
    experiment_results_root: str = ""  # root dir for all outputs
    result_snapshots_dir: str = ""  # snapshots of input + output images during training
    network_snapshots_dir: str = ""  # saved network snapshots
    test_output_samples_dir: str = ""  # samples of the network applied on the test dataset


@dataclass
class Config:
    # 1 if you have a capable GPU else 0 for CPU. Multiple GPUs not tested.
    num_gpu: int
    # Num workers to load the data. Benchmark to find best values.
    num_workers_train: int
    num_workers_validate: int
    num_workers_test: int

    # Training parameters
    batch_size: int
    num_epochs: int
    learning_rate: float
    use_validation: bool
    use_lr_scheduler: bool
    
    # Losses Weights
    alpha: float
    beta: float
    gamma: float
    delta: float

    # Directories
    dirs: Directories

    # Input Buffers
    # input_buffers: List[BufferType]

    # Misc
    num_network_snapshots: int = 5
    image_snapshots_interval: int = 5  # in epochs
    batches_log_interval: int = 50

    descr: str = ""


@dataclass
class ConfigGAN(Config):
    lr_optim_G: float = 0.0002
    lr_optim_D: float = 0.0002

    optim_G_beta_1: float = -1
    optim_G_beta_2: float = -1

    optim_D_beta_1: float = -1
    optim_D_beta_2: float = -1

    lambda_l1: float = 100.0


# Config used up to ResUNet

# @dataclass
# class Config:
#     # 1 if you have a capable GPU else 0 for CPU. Multiple GPUs not tested.
#     num_gpu: int
#     # Num workers to load the data. Benchmark to find best values.
#     num_workers_train: int
#     num_workers_validate: int
#     num_workers_test: int

#     # Training parameters
#     batch_size: int
#     num_epochs: int
#     learning_rate: float
#     use_validation: bool

#     # Losses Weights
#     alpha: float
#     beta: float
#     gamma: float

#     # Directories
#     dirs: Directories

#     # Input Buffers
#     # input_buffers: List[BufferType]

#     # Misc
#     num_network_snapshots: int = 5
#     image_snapshots_interval: int = 5  # in epochs
#     batches_log_interval: int = 50

#     descr: str = ""
