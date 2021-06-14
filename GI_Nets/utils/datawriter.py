from utils.utils import results_to_latex_d
from configs.config import Config
import json
import dataclasses
from os.path import join


def serialize_and_save_config(config: Config) -> None:
    with open(join(config.dirs.experiment_results_root, "config.json"), "w") as text_file:
        text_file.write(json.dumps(dataclasses.asdict(config), indent=4))


def serialize_and_save_results(
    filepath: str,
    mae: float,
    mae_std: float,
    mse: float,
    mse_std: float,
    ssim: float,
    ssim_std: float,
    psnr: float,
    psnr_std: float,
    lpips: float,
    lpips_std,
    time: float,
    time_std: float,

):
    with open(filepath, "w") as text_file:
        results_dict = {
            "mae": "{0:.4f} +/- {1:.4f}".format(mae, mae_std),
            "mse": "{0:.4f} +/- {1:.4f}".format(mse, mse_std),
            "ssim": "{0:.4f} +/- {1:.4f}".format(ssim, ssim_std),
            "psnr": "{0:.4f} +/- {1:.4f}".format(psnr, psnr_std),
            "lpips": "{0:.4f} +/- {1:.4f}".format(lpips, lpips_std),
            "time": "{0:.4f} +/- {1:.4f}".format(time, time_std),
        }
        
        latex_ready = results_to_latex_d(results_dict)
        results_dict["latex"] = latex_ready

        text_file.write(json.dumps(results_dict, indent=4))

