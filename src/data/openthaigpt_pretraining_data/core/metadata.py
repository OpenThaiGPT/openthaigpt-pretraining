import os
import git
import __main__
import json
from typing import Dict, Any


def create_info_file(info_dict: Dict[str, Any], output_dir: str) -> None:

    """Example info_dict : info = {"source": "mc4", "current_version": 1} (All fields are required.)"""  # noqa: E501
    json.dump(info_dict, open(f"{output_dir}/info.json", "w"))


def create_metadata_file(
    metadata_dict: Dict[str, Any], input_dir: str, output_dir: str
) -> None:

    """Example of metadata_dict (All fields are required.)
        metadata = {
        "dataset_name": "mc4",
        "data_version": 1,
        "dataset_location": "/project/lt200056-opgpth/mond_tmp_datasets/jsonl/mc4/raw",
        "data_scratch_location": "/lustrefs/flash/scratch/lt200056-opgpth/temp_dataset/mc4",
        "input_name": "mc4",
        "input_version": 1,
        "pipeline_name": "internet",
        "processing_parameters": {
            "do_perplexity": True,
            "batch_size": 1000,
            "sampled_back_ratio": 0.6,
        },
        "note": "",
    }
    """  # noqa: E501

    metadata_dict["pipeline_location"] = os.path.relpath(__main__.__file__)
    metadata_dict["pipeline_commit_hash"] = git.Repo(
        search_parent_directories=True
    ).head.object.hexsha
    metadata_dict["inputs"] = json.load(
        open(f"{input_dir}/{metadata_dict['input_version']}/metadata.json", "r")
    )

    json.dump(
        metadata_dict,
        open(f"{output_dir}/{metadata_dict['data_version']}/metadata.json", "w"),
    )
