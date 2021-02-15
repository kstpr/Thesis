from dataclasses import dataclass

@dataclass
class Config:
    num_gpu: int

    # Network parameters
    g_feat_maps: int  # number of feature maps before the end of the generator
    d_feat_maps: int  # number of feature maps after the input of the discriminator
    num_channels: int  # the number of channels for G's output and D's input

    # Data parameters
    dataroot: str
    intermediates_root: str  # intermediate results in the form of generator samples and loss history
    output_root: str  # final results as individual fake images generated from fixed noise
    dataloader_num_workers: str  # num threads to load the data

    # Instance parameters
    image_size: int  # implicitly w = h
    latent_size: int  # size of the latent noise vector

    # Training parameters
    batch_size: int
    num_epochs: int
    g_learning_rate: float
    d_learning_rate: float

    # Adam hyperparams
    g_beta_1: float
    g_beta_2: float
    d_beta_1: float
    d_beta_2: float