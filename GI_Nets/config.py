from dataclasses import dataclass

@dataclass
class Config:
    num_gpu: int
    num_workers: str  # num threads to load the data

    # Training parameters
    batch_size: int
    num_epochs: int
    learning_rate: float
    use_validation: bool

    # Directories
    results_root: str
    #result_snapshots_dir: str
    #network_snapshots_dir: str

    # Misc
    num_network_snapshots: int

    # # Adam hyperparams
    # g_beta_1: float
    # g_beta_2: float
    # d_beta_1: float
    # d_beta_2: float

    # # Learning rate linear decay parameters
    # g_lr_decay_start_epoch: int
    # d_lr_decay_start_epoch: int
    # lr_linear_decay_enabled: bool = False