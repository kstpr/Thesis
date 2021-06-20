import wandb


def log_batch_stats(
    epoch: int,
    batch_num: int,
    batch_time: float,
    batch_train_time: float,
    total_loss: float,
    sdsim_loss: float,
    l1_loss: float = 0.0,
    l2_loss: float = 0.0,
    additional_loss: float = 0.0,
):
    print(
        "Batch {0} [{2:.4f}s /{1:.4f} s]. Total loss = {3:.4f}, SDSIM loss = {4:.4f}, L1 loss = {5:.4f}, L2 loss = {6:.4f}, Additional loss = {7:.4f}".format(
            batch_num, batch_time, batch_train_time, total_loss, sdsim_loss, l1_loss, l2_loss, additional_loss
        )
    )
    wandb.log(
        {
            "Epoch": epoch,
            "Total Loss": total_loss,
            "SDSIM Loss": sdsim_loss,
            "L1 Loss": l1_loss,
            "L2 Loss": l2_loss,
            "Additional Loss": additional_loss,
        }
    )

def log_batch_stats_gan(
    epoch: int,
    batch_num: int,
    batch_train_time: float,
    d_loss_real: float,
    d_loss_fake: float,
    d_total_loss: float,
    g_gan_loss: float,
    g_l1_loss: float,
    g_total_loss: float,
):
    print(
        "Batch {0} [{1:.4f} s]. L_D(real) = {2:.4f}, L_D(fake) = {3:.4f}, L_D = {4:.4f}, L_G-GAN = {5:.4f}, L_G-L1 = {6:.4f}, L_G = {7:.4f}".format(
            batch_num, batch_train_time, d_loss_real, d_loss_fake, d_total_loss, g_gan_loss, g_l1_loss, g_total_loss
        )
    )
    wandb.log(
        {
            "Epoch": epoch,
            "L_D(real)": d_loss_real,
            "L_D(fake)": d_loss_fake,
            "L_D": d_total_loss,
            "L_G-GAN": g_gan_loss,
            "L_G-L1": g_l1_loss,
            "L_G": g_total_loss
        }
    )


def log_batch_stats_pix2pixHD(
    epoch: int,
    batch_num: int,
    batch_train_time: float,
    d_loss_real: float,
    d_loss_fake: float,
    d_loss_total: float,
    g_gan_loss: float,
    g_feat_loss: float,
    g_vgg_loss: float,
    g_total_loss: float,
):
    print(
        "Batch {0} [{1:.4f} s]. L_D(real) = {2:.4f}, L_D(fake) = {3:.4f}, L_D = {4:.4f}, L_G-GAN = {5:.4f}, L_G-Feat = {6:.4f}, L_G_VGG = {7:.4f}, L_G_Total = {8:.4f}".format(
            batch_num, batch_train_time, d_loss_real, d_loss_fake, d_loss_total, g_gan_loss, g_feat_loss, g_vgg_loss, g_total_loss
        )
    )
    wandb.log(
        {
            "Epoch": epoch,
            "L_D(real)": d_loss_real,
            "L_D(fake)": d_loss_fake,
            "L_D": d_loss_total,
            "L_G-GAN": g_gan_loss,
            "L_G-Feat": g_feat_loss,
            "L_G-VGG": g_vgg_loss,
            "L_G": g_total_loss
        }
    )


def log_validation_stats(
    epoch: int,
    validation_time: float,
    total_loss: float,
    sdsim_loss: float,
    l1_loss: float = 0.0,
    l2_loss: float = 0.0,
    additional_loss: float = 0.0,
):
    print(
        "Validation took {0:.4f} s. Avg. losses: Total loss = {1:.4f}, SDSIM loss = {2:.4f}, L1 loss = {3:.4f}, L2 loss = {4:.4f}, Additional loss = {5:.4f}".format(
            validation_time, total_loss, sdsim_loss, l1_loss, l2_loss, additional_loss
        )
    )
    wandb.log(
        {
            "Epoch": epoch,
            "Total Val Loss": total_loss,
            "SDSIM Val Loss": sdsim_loss,
            "L1 Val Loss": l1_loss,
            "L2 Val Loss": l2_loss,
            "Additional Val Loss": additional_loss,
        }
    )


def log_validation_stats_gan(epoch: int, validation_time: float, avg_ssim: float):
    print("Validation took {0:.4f} s. Avg SSIM: {1:.4f}".format(validation_time, avg_ssim))
    wandb.log(
        {
            "Epoch": epoch,
            "Val Avg SSIM": avg_ssim,
        }
    )