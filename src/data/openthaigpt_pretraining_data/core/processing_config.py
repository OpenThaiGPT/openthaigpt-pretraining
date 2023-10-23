import os
import json
from hydra import compose, initialize
from typing import Dict, Any


def load_config(config_filename: str) -> Dict[str, Any]:

    config_dict = {}

    initialize("./config", caller_stack_depth=2)
    cfg = compose(config_filename)

    config_dict["processing_parameters"] = cfg.processing_parameters

    config_dict["output_dir"] = str(cfg.output.path)
    config_dict["scratch_location"] = (
        cfg.output.scratch_path if "scratch_path" in cfg.output else None
    )

    config_dict["input_based_path"] = str(cfg.input_dataset.path)

    info = json.load(open(f"{config_dict['input_based_path']}/info.json", "r"))
    config_dict["input_version"] = info["current_version"]
    config_dict["source"] = info["source"]

    print(f"Processing {config_dict['source']} dataset")

    if "version" in cfg.output:
        version = cfg.output.version
    else:
        if os.path.exists(f"{config_dict['output_dir']}/info.json"):
            info = json.load(open(f"{config_dict['output_dir']}/info.json", "r"))
            version = info["current_version"] + 1
        else:
            version = 1

    config_dict["version"] = version

    config_dict["note"] = cfg.note

    return config_dict
