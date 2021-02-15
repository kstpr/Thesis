# %%
import dataclasses
import json
from datetime import datetime
from os import makedirs

from config import Config
from dcgan import DCGAN

dataroot = "~/Thesis/data/"
cats_dataroot = dataroot + "cats_only/"
dogs_dataroot = dataroot + "dogs_only/"
wildlife_dataroot = dataroot + "wildlife_only/"

results_root = "/home/ksp/Thesis/src/Thesis/GANs/DCGAN/results/"


def setup_directories_with_timestamp():
    now = datetime.now()  # current date and time

    timestamp: str = now.strftime("%m_%d_%Y__%H_%M_%S")
    experiment_root = "%s%s/" % (
        results_root,
        timestamp,
    )

    intermediates_root = experiment_root + "intermediates/"
    output_root = experiment_root + "output/"

    makedirs(intermediates_root, exist_ok=True)
    makedirs(output_root, exist_ok=True)

    return {
        "experiment_root": experiment_root,
        "intermediates": intermediates_root,
        "output": output_root,
    }


def run_dcgan(config: Config):
    dcgan = DCGAN(config=config)
    dcgan.train_and_plot()
    dcgan.generate_fake_results()


blueprint_config: Config = Config(
    num_gpu=1,
    g_feat_maps=64,
    d_feat_maps=64,
    num_channels=3,
    dataroot=dogs_dataroot,
    intermediates_root="placeholder",
    output_root="placeholder",
    dataloader_num_workers=12,
    image_size=64,
    latent_size=100,
    batch_size=64,
    num_epochs=100,
    g_learning_rate=0.0002,
    d_learning_rate=0.0002,
    g_beta_1=0.5,
    g_beta_2=0.999,
    d_beta_1=0.5,
    d_beta_2=0.999,
)

num_epochs_list = [5, 5, 5, 5]

for num_epochs in num_epochs_list:
    dirs = setup_directories_with_timestamp()
    config = dataclasses.replace(blueprint_config)
    config.num_epochs = num_epochs
    config.intermediates_root = dirs["intermediates"]
    config.output_root = dirs["output"]

    with open(dirs["experiment_root"] + "config.json", "w") as text_file:
        text_file.write(json.dumps(dataclasses.asdict(config), indent=4))

    run_dcgan(config=config)


# %%
import gc
import torch

def memReport():
    objs = gc.get_objects()
    print(len(objs))
    for obj in objs:
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

memReport()
# %%
