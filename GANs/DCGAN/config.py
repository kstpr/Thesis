from dataclasses import dataclass

@dataclass
class DataRoot:
    dataset_root: str  # in a format expected by the dataloader; contain classes folders, unused by our needs
    validation_dataset_root: str # the same format as above; used for validation 
    immediate_dir: str  # folder directly containing the actual images; needed for calculating the FID score

@dataclass
class Config:
    num_gpu: int

    # Network parameters
    g_feat_maps: int  # number of feature maps before the end of the generator
    d_feat_maps: int  # number of feature maps after the input of the discriminator
    num_channels: int  # the number of channels for G's output and D's input

    # Data parameters
    dataroot: DataRoot
    experiment_output_root: str # parent output root - save everything that's not images here
    intermediates_root: str  # intermediate results in the form of generator samples and loss history
    output_root: str  # final results as individual fake images generated from fixed noise
    dataloader_num_workers: str  # num threads to load the data

    # Instance parameters
    image_size: int  # implicitly w = h
    latent_size: int  # size of the latent noise vector

    # Training parameters
    use_validation: bool
    batch_size: int
    num_epochs: int
    g_learning_rate: float
    d_learning_rate: float

    # Adam hyperparams
    g_beta_1: float
    g_beta_2: float
    d_beta_1: float
    d_beta_2: float

    # Learning rate linear decay parameters
    g_lr_decay_start_epoch: int
    d_lr_decay_start_epoch: int
    lr_linear_decay_enabled: bool = False