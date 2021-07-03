from trainers.BaseTrainer import BaseTrainer
from DeepCG.buffer_pool import BufferPool
import netutils
from configs.config import ConfigGAN
from typing import List, Optional, Tuple
from utils.logger import log_batch_stats_gan, log_batch_stats_pix2pixHD

from torch import nn
import torch
from torch.tensor import Tensor
from trainers.GANTrainer import GANTrainer
from DeepCG.networks import GANLoss, VGGLoss

from torch.utils.data.dataset import Dataset

from torch.cuda.amp.autocast_mode import autocast
from timeit import default_timer as timer

class Pix2PixHDTrainer(GANTrainer):
    def __init__(
        self,
        config: ConfigGAN,
        device: torch.device,
        train_dataset: Dataset,
        num_channels: int,
        netG: nn.Module,
        netD: nn.Module,
        io_transform: netutils.IOTransform,
        validation_dataset: Optional[Dataset],
        use_amp: bool = True
    ):
        super().initialize_networks([netG, netD], num_channels)
        super().__init__(
            config,
            device,
            train_dataset,
            io_transform,
            validation_dataset,
        )
        batch_size = self.train_dataloader.batch_size
        
        self.fake_buffers_pool = BufferPool(batch_size * 10)
        self.lambda_feat_loss = 10.0

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def initialize_losses(self) -> List[nn.Module]:
        gan_loss = GANLoss(tensor=torch.cuda.FloatTensor).to(self.device)
        feature_matching_loss = nn.L1Loss()
        vgg_loss = VGGLoss([0])

        return [gan_loss, feature_matching_loss, vgg_loss]

    def train_epoch(
        self, epoch_num: int, loss_functions: List[nn.Module]
    ) -> Tuple[List[float], List[float]]:
        least_squares_loss_fn: GANLoss = loss_functions[0]
        feature_matching_loss_fn: nn.L1Loss = loss_functions[1]
        vgg_loss_fn: VGGLoss = loss_functions[2]

        losses_d: List[float] = []
        losses_g: List[float] = []

        batch_load_start = timer()

        # Code is adapted from Pix2PixHD official repository - https://github.com/NVIDIA/pix2pixHD
        for batch_num, (input, gt) in enumerate(self.train_dataloader):
            # Skip last incomplete batch
            if batch_num == self.train_size - 1:
                continue
            should_log = (
                batch_num == self.train_size - 2
                or batch_num % self.config.batches_log_interval == 0
            )

            input: Tensor = self.io_transform.transform_input(input)
            gt: Tensor = self.io_transform.transform_gt(gt)
            fake: Tensor = self.io_transform.transform_output(output=self.netG(input))

            self.io_transform.clear()

            # D GAN losses
            fake_cond = torch.cat((input, fake), 1)
            pred_fake_pool = self.discriminate(fake_cond, use_pool=True)
            loss_D_fake = least_squares_loss_fn(pred_fake_pool, False)

            real_cond = torch.cat((input, gt), 1)
            pred_real = self.discriminate(real_cond)
            loss_D_real = least_squares_loss_fn(pred_real, True)

            # G GAN loss
            pred_fake = self.netD.forward(fake_cond)
            loss_G_GAN = least_squares_loss_fn(pred_fake, True)

            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            feat_weights = 1.0
            D_weights = 0.333 # 3 discriminators

            # TODO - debug
            for i in range(3): # 3 Discriminators
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        feature_matching_loss_fn(pred_fake[i][j], pred_real[i][j].detach()) * self.lambda_feat_loss

            
            loss_G_VGG = vgg_loss_fn(fake, gt) * self.lambda_feat_loss

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG

            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

            losses_g.append(loss_G.item())
            losses_d.append(loss_D.item())

            if should_log:
                batch_end = timer()
                log_batch_stats_pix2pixHD(
                    epoch=epoch_num,
                    batch_num=batch_num,
                    batch_train_time=batch_end - batch_load_start,
                    d_loss_real=loss_D_real.item(),
                    d_loss_fake=loss_D_fake.item(),
                    d_loss_total=loss_D.item(),
                    g_gan_loss=loss_G_GAN.item(),
                    g_feat_loss=loss_G_GAN_Feat.item(),
                    g_vgg_loss=loss_G_VGG.item(),
                    g_total_loss=loss_G.item()
                )

            batch_load_start = timer()

        self.scheduler_D.step()
        self.scheduler_G.step()

        return (losses_d, losses_g)

    def discriminate(self, fake_batch: Tensor, use_pool=False):
        fake_batch = fake_batch.detach()
        if use_pool:
            fake_query = self.fake_buffers_pool.query(fake_batch)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(fake_batch)