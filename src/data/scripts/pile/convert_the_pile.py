from datasets import Dataset
from argparse import ArgumentParser
import os
import tqdm
import json

DATASET_NAME = "pile"
NULL = "null"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train_jsonl_uncompressed_directory")
    parser.add_argument("output_hf_directory")
    parser.add_argument("--validation_size", default=0.001, type=float)

    args = parser.parse_args()

    data_list = {
        "text": [],
        "source": [],
        "source_id": [],
        "create_date": [],
        "update_date": [],
        "meta": [],
    }  # type: ignore
    for jsonl_file in tqdm.tqdm(os.listdir(args.train_jsonl_uncompressed_directory)):
        jsonl_file_path = os.path.join(
            args.train_jsonl_uncompressed_directory, jsonl_file
        )
        with open(jsonl_file_path, "r", encoding="utf-8") as file:
            i = 0
            file_name = os.path.basename(jsonl_file).split(".")[0]
            for line in file:
                data = json.loads(line)
                data_list["text"].append(data["text"])
                data_list["source"].append(DATASET_NAME)
                data_list["source_id"].append(f"{file_name}_{i}")
                data_list["create_date"].append(NULL)
                data_list["update_date"].append(NULL)
                data_list["meta"].append(data["meta"])
                i += 1

    dataset = Dataset.from_dict(data_list)
    split = dataset.train_test_split(test_size=args.validation_size)
    split["eval"] = split.pop("test")
    split.save_to_disk(args.output_hf_directory)
