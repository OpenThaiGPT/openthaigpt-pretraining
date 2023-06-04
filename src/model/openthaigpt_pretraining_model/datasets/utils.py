from datasets import Dataset, load_from_disk, concatenate_datasets
import json


def jsonl_to_dataset(jsonl_file_path):
    with open(jsonl_file_path, "r") as json_file:
        json_list = list(json_file)
    data = []
    for json_str in json_list:
        data.append(json.loads(json_str))
    return Dataset.from_dict(data)


def merge_datasets(dataset_paths):
    datasets = [load_from_disk(path) for path in dataset_paths]
    return concatenate_datasets(datasets)
