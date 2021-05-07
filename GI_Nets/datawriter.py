from config import Config
import json
import dataclasses
from os.path import join

def serialize_and_save_config(config: Config) -> None:
    with open(join(config.dirs.experiment_results_root, "config.json"), "w") as text_file:
        text_file.write(json.dumps(dataclasses.asdict(config), indent=4))

def serialize_and_save_results(ssim: float, psnr: float, lpips: float):
    pass

# def serialize_results(self, time: float, fid: float) -> None:
#     with open(self.config.experiment_output_root + "results.json", "w") as text_file:
#         text_file.write(json.dumps({"fid": fid, "time": time}))