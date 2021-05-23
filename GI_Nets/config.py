from dataclasses import dataclass
from typing import List
from GIDataset import BufferType


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

    # Losses Weights
    alpha: float
    beta: float
    gamma: float

    # Directories
    dirs: Directories

    # Input Buffers
    # input_buffers: List[BufferType]

    # Misc
    num_network_snapshots: int = 5
    image_snapshots_interval: int = 5  # in epochs
    batches_log_interval: int = 50

    descr: str = ""

    # # Adam hyperparams
    # g_beta_1: float
    # g_beta_2: float
    # d_beta_1: float
    # d_beta_2: float

    # # Learning rate linear decay parameters
    # g_lr_decay_start_epoch: int
    # d_lr_decay_start_epoch: int
    # lr_linear_decay_enabled: bool = False
