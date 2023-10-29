import os
import git
import __main__
import json
from typing import Dict, Any


def create_info_file(config_dict: Dict[str, Any]) -> None:

    info_dict = {
        "source": config_dict["source"],
        "current_version": config_dict["version"],
    }
    json.dump(info_dict, open(f"{config_dict['output_dir']}/info.json", "w"))


def create_metadata_file(config_dict: Dict[str, Any], pipeline_name: str):

    metadata_dict = {
        "dataset_name": config_dict["source"],
        "dataset_version": config_dict["version"],
        "dataset_location": config_dict["output_dir"],
        "data_scratch_location": config_dict["scratch_location"],
        "input_name": config_dict["source"],
        "input_version": config_dict["input_version"],
        "processing_parameters": config_dict["processing_parameters"],
        "pipeline_name": pipeline_name,
        "note": config_dict["note"],
    }

    metadata_dict["pipeline_location"] = os.path.relpath(__main__.__file__)
    metadata_dict["pipeline_commit_hash"] = git.Repo(
        search_parent_directories=True
    ).head.object.hexsha

    metadata_dict["inputs"] = json.load(
        open(
            f"{config_dict['input_based_path']}/{metadata_dict['input_version']}/metadata.json",  # noqa: E501
            "r",
        )
    )

    json.dump(
        metadata_dict,
        open(
            f"{config_dict['output_dir']}/{metadata_dict['data_version']}/metadata.json",  # noqa: E501
            "w",
        ),
    )
