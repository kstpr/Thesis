import wandb


def log_validation_batch_stats(epoch, total_epochs, batch_num, num_batches, D_x, loss_D):
    if batch_num % 50 == 0:
        wandb.log({"IsValidation": True, "Epoch": epoch, "D Loss": loss_D, "D(x)": D_x})

        print(
            "Validation [%d/%d][%d/%d]\tLoss_D: %.4f\tD(x): %.4f"
            % (
                epoch,
                total_epochs,
                batch_num,
                num_batches,
                loss_D.item(),
                D_x,
            )
        )


def log_batch_stats(epoch, num_epochs, batch_num, num_batches, D_x, D_G_z1, loss_D, loss_G, D_G_z2):
    if batch_num % 50 == 0:
        wandb.log(
            {
                "Epoch": epoch,
                "D Loss": loss_D,
                "G Loss": loss_G,
                "D(x)": D_x,
                "D(G(z)) before D": D_G_z1,
                "D(G(z)) after D": D_G_z2,
            }
        )

        print(
            "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
            % (
                epoch,
                num_epochs,
                batch_num,
                num_batches,
                loss_D.item(),
                loss_G.item(),
                D_x,
                D_G_z1,
                D_G_z2,
            )
        )