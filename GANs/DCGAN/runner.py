# %%
import dataclasses
from datetime import datetime
from os import makedirs, path
from timeit import default_timer as timer

from config import Config, DataRoot
from dcgan import DCGAN
from datawriter import DataWriter

from pytorch_fid.fid_score import calculate_fid_given_paths


general_dataroot = "/home/ksp/Thesis/data/"

cats_dataroot = DataRoot(
    dataset_root=path.join(general_dataroot, "cats_only"), immediate_dir=path.join(general_dataroot, "cats_only", "cat")
)
dogs_dataroot = DataRoot(
    dataset_root=path.join(general_dataroot, "dogs_only"), immediate_dir=path.join(general_dataroot, "dogs_only", "dog")
)
wildlife_dataroot = DataRoot(
    dataset_root=path.join(general_dataroot, "wildlife_only"), immediate_dir=path.join(general_dataroot, "wildlife_only", "wild")
)

results_root = "/home/ksp/Thesis/src/Thesis/GANs/DCGAN/results/"


def setup_directories_with_timestamp(experiment_batch_name: str = ""):
    now = datetime.now()  # current date and time

    timestamp: str = now.strftime("%m_%d_%Y__%H_%M_%S")
    experiment_root = "%s%s/" % (
        results_root,
        experiment_batch_name + timestamp,
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


def run_dcgan(config: Config) -> float:
    """Trains the GAN, saves intermediate results and generates 2048 fake samples from the generator
    Return elapsed time for training the net
    """
    dcgan = DCGAN(config=config)
    start = timer()
    dcgan.train_and_plot()
    elapsed_time: float = timer() - start
    dcgan.generate_fake_results()

    return elapsed_time


blueprint_config: Config = Config(
    num_gpu=1,
    g_feat_maps=64,
    d_feat_maps=64,
    num_channels=3,
    dataroot=wildlife_dataroot,
    experiment_output_root="placeholder",
    intermediates_root="placeholder",
    output_root="placeholder",
    dataloader_num_workers=12,  # yes
    image_size=64,
    latent_size=100,
    batch_size=8,  # yes
    num_epochs=100,  # yes
    g_learning_rate=0.0002,
    d_learning_rate=0.0002,
    g_beta_1=0.5,
    g_beta_2=0.999,
    d_beta_1=0.5,
    d_beta_2=0.999,
)

num_epochs_list = [200]
batch_sizes = [2, 4]


def run_experiments(tag: str) -> None:
    for num_epochs in num_epochs_list:
        for batch_size in batch_sizes:
            dirs = setup_directories_with_timestamp("%s_sat_bs_%d_epochs_%d_" % (tag, batch_size, num_epochs))
            config = dataclasses.replace(blueprint_config)
            config.num_epochs = num_epochs
            config.batch_size = batch_size
            config.intermediates_root = dirs["intermediates"]
            config.output_root = dirs["output"]
            config.experiment_output_root = dirs["experiment_root"]

            data_writer = DataWriter(config=config)
            data_writer.serialize_config()

            print(config)
            training_time = run_dcgan(config=config)
            fid = calculate_fid_given_paths([config.dataroot.immediate_dir, config.output_root], 50, "cuda:0", 2048)

            data_writer.serialize_results(time=training_time, fid=fid)

            print("FID: %f" % fid)


run_experiments("wild")


# %%
