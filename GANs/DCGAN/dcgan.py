# We follow the DCGAN tutorial from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Main change is refactoring into a class as a base for pipeline and intermediate results saving and
# visualizing

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import print_function
#%matplotlib inline
import random
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import ConvTranspose2d
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, num_gpu, latent_size, feat_maps_size, num_channels):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            # first
            nn.ConvTranspose2d(
                in_channels=latent_size,
                out_channels=feat_maps_size * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(feat_maps_size * 8),
            nn.ReLU(True),
            # second
            nn.ConvTranspose2d(
                in_channels=feat_maps_size * 8,
                out_channels=feat_maps_size * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feat_maps_size * 4),
            nn.ReLU(True),
            # third
            nn.ConvTranspose2d(
                in_channels=feat_maps_size * 4,
                out_channels=feat_maps_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feat_maps_size * 2),
            nn.ReLU(True),
            # fourth
            nn.ConvTranspose2d(
                in_channels=feat_maps_size * 2,
                out_channels=feat_maps_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feat_maps_size),
            nn.ReLU(True),
            ## last
            nn.ConvTranspose2d(feat_maps_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, num_gpu, num_channels, feat_maps_size):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            # L1
            nn.Conv2d(num_channels, feat_maps_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # L2
            nn.Conv2d(feat_maps_size, feat_maps_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # L3
            nn.Conv2d(feat_maps_size * 2, feat_maps_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # L4
            nn.Conv2d(feat_maps_size * 4, feat_maps_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Last
            nn.Conv2d(feat_maps_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input)

class DCGAN:
    def __init__(self, num_gpu=1) -> None:
        self.init_torch()

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

        self.init_parameters()
        self.init_dataset_and_loader()

        g_feat_maps = 64
        d_feat_maps = 64
        num_channels = 3

        self.G = Generator(
            num_gpu=num_gpu,
            latent_size=self.latent_size,
            feat_maps_size=g_feat_maps,
            num_channels=num_channels
        ).to(self.device)
        
        self.G.apply(weights_init)
        print(self.G)

        self.D = Discriminator(
            num_gpu=num_gpu,
            num_channels=num_channels,
            feat_maps_size=d_feat_maps
        ).to(self.device)

        self.D.apply(weights_init)
        print(self.D)

        self.init_loss_and_optimizer()

    def init_torch(self) -> None:
        manualSeed = 999
        print(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def init_parameters(self) -> None:
        self.dataroot = '~/Thesis/data/'
        self.results_root = '/home/ksp/Thesis/src/Thesis/GANs/DCGAN/results/'
        self.cats_dataroot = self.dataroot + 'cats_only/'
        self.wildlife_dataroot = self.dataroot + 'wildlife_only/'

        self.workers = 12

        self.batch_size = 16
        self.image_size = 64
        self.latent_size = 100
        
        self.num_epochs = 100
        self.learning_rate = 0.0002

        # Adam hyperparams
        self.beta_1 = 0.5
        self.num_gpu = 1

    def init_dataset_and_loader(self):
        dataset = dset.ImageFolder(
            root=self.wildlife_dataroot,
            transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )

        self.dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )

    def init_loss_and_optimizer(self) -> None:
        self.criterion : nn.BCELoss = nn.BCELoss()

        self.fixed_noise = torch.randn(64, self.latent_size, 1, 1, device=self.device)

        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.beta_1, 0.999) )
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.beta_1, 0.999) )

    def plot_training_examples(self):
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    real_batch[0].to(self.device)[:64],
                    padding=2,
                    normalize=True
                ).cpu(),
                (1,2,0)
            )
        )

    def train_and_plot(self):
        self.plot_training_examples()
        self.train()
        self.plot_losses()
        # self.plot_results_animation()

    def train(self):
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        real_label = 1
        fake_label = 0

        print("Starting training loop...")
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.dataloader, 0):
                self.train_batch(iters, real_label, fake_label, epoch, i, data)
                iters += 1

    def train_batch(self, iters, real_label, fake_label, epoch, i, data):
        # Update D - max log(D(x)) + log(1 - D(G(z))
        ############################################
        # Train with real batch
        self.D.zero_grad()
        real = data[0].to(self.device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float32, device=self.device)

        # Forward pass real batch through D
        output = self.D(real).view(-1)
        # Loss on real batch
        errD_real = self.criterion(output, label)

        # Calculate gradients for D
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake batch
        noise = torch.randn(b_size, self.latent_size, 1, 1, device=self.device)

        # Generate fake batch using G
        fake = self.G(noise)
        label.fill_(fake_label)

        # Forward pass fake batch through D
        output = self.D(fake.detach()).view(-1)

        # Loss on fake batch
        errD_fake = self.criterion(output, label)

        # Calculate gradients for D
        errD_fake.backward()

        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake # how

        self.optimizerD.step()

        # Update G - max log(D(G(z)))
        #############################
        self.G.zero_grad()

        label.fill_(real_label)
        output = self.D(fake).view(-1)

        errG = self.criterion(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()

        # Update G
        self.optimizerG.step()

        self.log_batch_stats(iters, epoch, i, D_x, D_G_z1, errD, errG, D_G_z2)

    def log_batch_stats(self, iters, epoch, i, D_x, D_G_z1, errD, errG, D_G_z2):
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
                (epoch, self.num_epochs, i, len(self.dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
            )

        self.G_losses.append(errG.item())
        self.D_losses.append(errD.item())

        if (epoch % 5 == 0) and (i == len(self.dataloader) - 1):
            print('End of %d-th epoch:' % epoch)
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                self.plot_current_fake(fake, epoch, False)

        if (epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1):
            print ("End result:")
            with torch.no_grad():
                fake = self.G(self.fixed_noise).detach().cpu()
                self.plot_current_fake(fake, epoch, True)

        # if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1)):
        #     with torch.no_grad():
        #         fake = self.G(self.fixed_noise).detach().cpu()
        #         self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    def plot_losses(self):
        plt.figure(figsize=(10,5))
        plt.title("G and D loss during trainning")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    def plot_current_fake(self, fake_batch, epoch, isFinal):
        plt.figure(figsize=(16,16))
        plt.axis("off")
        plt.title('Images in' + ("final epoch" if isFinal else 'epoch %d' % (epoch)))
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    fake_batch.to(self.device)[:64],
                    padding=2,
                    normalize=True
                ).cpu(),
                (1,2,0)
            )
        )
        plt.savefig(self.results_root + 'epoch%d.png' % (epoch))
        print("Figure saved.")

    # %%capture
    def plot_results_animation(self):
        import matplotlib
        matplotlib.rcParams['animation.embed_limit'] = 64

        fig=plt.figure(figsize=(12,12))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in self.img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

# %%capture
dcgan = DCGAN()
dcgan.train_and_plot()

# %%
torch.cuda.empty_cache() 
# %%
