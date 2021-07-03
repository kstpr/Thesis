from utils.logger import log_batch_stats_deepCG
from DeepCG import buffer_pool
import netutils
from configs.config import ConfigGAN
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.tensor import Tensor
from torch.utils.data.dataset import Dataset
import torch.optim as optim

from trainers.GANTrainer import GANTrainer
from DeepCG.networks import GANLoss, VGGLoss
from torchsummary import summary
from timeit import default_timer as timer
from netutils import transform_mask_to_exp_space


class DeepCGTrainer(GANTrainer):
    def __init__(
        self,
        config: ConfigGAN,
        device: torch.device,
        train_dataset: Dataset,
        num_channels: int,
        netG: nn.Module,
        netD: nn.Module,
        netD_2: nn.Module,
        io_transform: netutils.IOTransform,
        validation_dataset: Optional[Dataset],
        use_amp: bool = False,
    ):
        self.initialize_networks([netG, netD, netD_2], num_channels)
        super().__init__(
            config,
            device,
            train_dataset,
            io_transform,
            validation_dataset,
        )
        batch_size = self.train_dataloader.batch_size

        self.fake_buffers_pool_1 = buffer_pool.BufferPool(batch_size * 10)
        self.fake_buffers_pool_2 = buffer_pool.BufferPool(batch_size * 10)
        self.lambda_feat_loss = 10.0

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def initialize_networks(self, nets: List[nn.Module], num_channels: int):
        self.netG = nets[0]
        self.netD = nets[1]
        self.netD_2 = nets[2]

        summary(self.netG, input_size=(num_channels, 512, 512))
        summary(self.netD, input_size=(num_channels + 3, 512, 512))
        summary(self.netD_2, input_size=(num_channels + 3, 512, 512))

    def initialize_optimizers(self, config: ConfigGAN):
        super().initialize_optimizers(config)

        # has the same props as optimizer_D
        self.optimizer_D_2 = optim.Adam(
            params=self.netD_2.parameters(),
            betas=(config.optim_D_beta_1, config.optim_D_beta_2),
            lr=config.lr_optim_D,
        )

    def intialize_schedulers(self, lambda_rule):
        super().intialize_schedulers(lambda_rule)

        self.scheduler_D_2 = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer_D_2, lr_lambda=lambda_rule
        )

    def initialize_losses(self) -> List[nn.Module]:
        gan_loss = GANLoss(tensor=torch.cuda.FloatTensor).to(self.device)
        feature_matching_loss = nn.L1Loss()
        vgg_loss = VGGLoss([0])

        return [gan_loss, feature_matching_loss, vgg_loss]

    def load_saved_model(self, model_path: str):
        raise Exception("Not inherited properly, base will cause problems!")

    def continue_training_loaded_model(self):
        raise Exception("Not inherited properly, base will cause problems!")

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

            input: Tensor = self.io_transform.transform_input(input)  # mask as DI
            albedo = self.io_transform.get_albedo()  # albedo in (0, 1]
            gt: Tensor = self.io_transform.transform_gt(gt)  # ray traced
            gt_mask: Tensor = self.io_transform.transform_gt_mask(
                gt
            )  # ray traced / albedo in log space (-1, 1)
            fake_mask: Tensor = self.io_transform.transform_output(
                output=self.netG(input)
            )  # mask in log space (-1, 1)
            fake_img: Tensor = (
                transform_mask_to_exp_space(fake_mask.clone()) * albedo
            ).nan_to_num()

            self.io_transform.clear()

            fake_cond_mask = torch.cat((input, fake_mask), 1)
            fake_cond_image = torch.cat((input, fake_img), 1)

            real_cond_mask = torch.cat((input, gt_mask), 1)
            real_cond_image = torch.cat((input, gt), 1)

            # D1 discriminates between generated mask and real mask = gt/albedo
            # D1 GAN losses
            pred_fake_mask_pool = self.discriminate(
                self.netD, fake_cond_mask, use_pool=True, pool=self.fake_buffers_pool_1
            )
            loss_D_1_fake = least_squares_loss_fn(pred_fake_mask_pool, False)

            pred_real_mask = self.discriminate(self.netD, real_cond_mask)
            loss_D_1_real = least_squares_loss_fn(pred_real_mask, True)

            loss_D_1 = (loss_D_1_real + loss_D_1_fake) * 0.5

            # D2 discriminates between generated image (gen_mask * albedo) and gt image
            # D2 GAN losses
            pred_fake_image_pool = self.discriminate(
                self.netD_2,
                fake_cond_image,
                use_pool=True,
                pool=self.fake_buffers_pool_2,
            )
            loss_D_2_fake = least_squares_loss_fn(pred_fake_image_pool, False)

            pred_real_image = self.discriminate(self.netD_2, real_cond_image)
            loss_D_2_real = least_squares_loss_fn(pred_real_image, True)

            loss_D_2 = (loss_D_2_real + loss_D_2_fake) * 0.5

            # G GAN loss
            fake_mask_pred = self.netD.forward(fake_cond_mask)
            loss_G_GAN_1 = least_squares_loss_fn(fake_mask_pred, True)
            fake_img_pred = self.netD_2.forward(fake_cond_image)
            loss_G_GAN_2 = least_squares_loss_fn(fake_img_pred, True)

            loss_G_GAN = loss_G_GAN_1 + loss_G_GAN_2

            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            feat_weights = 1.0
            D_weights = 0.333  # 3 discriminators

            # TODO - debug
            for i in range(3):  # 3 Discriminator scales for each discriminator
                for j in range(len(fake_img_pred[i]) - 1):
                    # Images
                    loss_G_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * feature_matching_loss_fn(
                            fake_img_pred[i][j], pred_real_image[i][j].detach()
                        )
                        * self.lambda_feat_loss
                    )
                    # Masks
                    loss_G_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * feature_matching_loss_fn(
                            fake_mask_pred[i][j], pred_real_mask[i][j].detach()
                        )
                        * self.lambda_feat_loss
                    )

            loss_G_VGG = vgg_loss_fn(fake_img, gt) * self.lambda_feat_loss

            loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG

            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            loss_D_1.backward()
            self.optimizer_D.step()

            self.optimizer_D_2.zero_grad()
            loss_D_2.backward()
            self.optimizer_D_2.step()

            losses_g.append(loss_G.item())
            losses_d.append((loss_D_1 + loss_D_2).item())

            if should_log:
                batch_end = timer()
                log_batch_stats_deepCG(
                    epoch=epoch_num,
                    batch_num=batch_num,
                    batch_train_time=batch_end - batch_load_start,
                    d_1_loss=loss_D_1.item(),
                    d_2_loss=loss_D_2.item(),
                    g_gan_loss=loss_G_GAN.item(),
                    g_feat_loss=loss_G_GAN_Feat.item(),
                    g_vgg_loss=loss_G_VGG.item(),
                    g_total_loss=loss_G.item(),
                )

            batch_load_start = timer()

        self.scheduler_D.step()
        self.scheduler_G.step()

        return (losses_d, losses_g)

    def discriminate(
        self,
        discriminator: nn.Module,
        fake_batch: Tensor,
        use_pool=False,
        pool: buffer_pool.BufferPool = None,
    ):
        fake_batch = fake_batch.detach()
        if use_pool:
            fake_query = pool.query(fake_batch)
            return discriminator.forward(fake_query)
        else:
            return discriminator.forward(fake_batch)