import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.core.numeric import full
import wandb
import numpy as np
import torchvision.utils as vutils


def save_current_fakes_snapshot(fake_batch, epoch, isFinal, device, full_output_path):
    fig = plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Images in " + ("final epoch" if isFinal else "epoch %d" % (epoch)))
    plt.imshow(
        np.transpose(
            vutils.make_grid(fake_batch.to(device)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig(full_output_path)
    plt.close(fig)
    print("Figure saved.")

def plot_and_save_losses(g_losses, d_losses, output_full_path):
    fig = plt.figure(figsize=(10, 5))
    plt.title("G and D loss during trainning")
    plt.plot(g_losses, label="G loss")
    plt.plot(d_losses, label="D loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_full_path)
    plt.close(fig)

    wandb.log({"Losses" : wandb.Image(output_full_path)})

def plot_and_save_val_loss(d_losses_val, output_full_path):
    fig = plt.figure(figsize=(10, 5))
    plt.title("Validation loss of D")
    plt.plot(d_losses_val, label="D loss val")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_full_path)
    plt.close(fig)

    wandb.log({"Validation loss" : wandb.Image(output_full_path)})

def plot_and_save_discriminator_probabilities(d_x, d_g_z, output_full_path) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.title("Real and fake probabilities during training")
    plt.plot(d_x, label="Real: D(x)")
    plt.plot(d_g_z, label="Fake: D(G(z))")
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(output_full_path)
    plt.close(fig)

    wandb.log({"Real-Fake probabilities" : wandb.Image(output_full_path)})

def plot_and_save_discriminator_val_probs(d_x_val, output_full_path) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.title("Validation probability during training")
    plt.plot(d_x_val, label="Validation D(x)")
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(output_full_path)
    plt.close(fig)

    wandb.log({"Train/Val D(x) probabilities" : wandb.Image(output_full_path)})

def plot_training_examples(dataloader, device, path):
    real_batch = next(iter(dataloader))
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig(path)
    plt.close(fig)

def plot_jittery_examples(dataloader, device, path):
    real_batch = next(iter(dataloader))
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Jittery Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig(path)
    plt.close(fig)