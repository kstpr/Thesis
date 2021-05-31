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
    additional_loss: float = 0.0
):
    print(
        "Batch {0} [{2:.4f}s /{1:.4f} s][whole/train]. Total loss = {3:.4f}, SDSIM loss = {4:.4f}, L1 loss = {5:.4f}, L2 loss = {6:.4f}, Additional loss = {7:.4f}".format(
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
            "Additional Loss": additional_loss
        }
    )


def log_validation_stats(
    epoch: int,
    validation_time: float,
    total_loss: float,
    sdsim_loss: float,
    l1_loss: float = 0.0,
    l2_loss: float = 0.0,
    haar_loss: float = 0.0
):
    print(
        "Validation took {0:.4f} s. Avg. losses: Total loss = {1:.4f}, SDSIM loss = {2:.4f}, L1 loss = {3:.4f}, L2 loss = {4:.4f}, Additional loss = {5:.4f}".format(
            validation_time, total_loss, sdsim_loss, l1_loss, l2_loss, haar_loss
        )
    )
    wandb.log(
        {
            "Epoch": epoch,
            "Total Val Loss": total_loss,
            "SDSIM Val Loss": sdsim_loss,
            "L1 Val Loss": l1_loss,
            "L2 Val Loss": l2_loss,
            "Additional Val Loss": haar_loss
        }
    )