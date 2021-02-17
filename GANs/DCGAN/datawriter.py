import json
import dataclasses

from config import Config

class DataWriter:
    def __init__(self, config: Config) -> None:
        self.config = config

    def serialize_config(self) -> None:
        with open(self.config.experiment_output_root + "config.json", "w") as text_file:
            text_file.write(json.dumps(dataclasses.asdict(self.config), indent=4))

    def serialize_results(self, time: float, fid: float) -> None:
        with open(self.config.experiment_output_root + "results.json", "w") as text_file:
            text_file.write(json.dumps({"fid": fid, "time": time}))