import matplotlib.pyplot as plt
import wandb
import numpy as np
import torchvision.utils as vutils


def plot_and_save_losses(g_losses, d_losses, output_full_path):
    fig = plt.figure(figsize=(10, 5))
    plt.title("G and D loss during trainning")
    plt.plot(g_losses, label="G loss")
    plt.plot(d_losses, label="D loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_full_path)
    plt.close(fig)

    wandb.log({"Losses" : wandb.Image(output_full_path)})

def plot_and_save_discriminator_results(d_x, d_g_z, output_full_path) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.title("Real and fake probabilities during trainning")
    plt.plot(d_x, label="Real: D(x)")
    plt.plot(d_g_z, label="Fake: D(G(z))")
    plt.xlabel("iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(output_full_path)
    plt.close(fig)

    wandb.log({"Real-Fake probabilities" : wandb.Image(output_full_path)})

def plot_training_examples(dataloader, device):
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
    plt.close(fig)