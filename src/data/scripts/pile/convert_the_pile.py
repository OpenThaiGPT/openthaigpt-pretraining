from datasets import Dataset
from argparse import ArgumentParser

import zstandard as zstd
import os
import tqdm
import json

DATASET_NAME = "pile"
NULL = "null"


def data_generator(jsonl_compressed_directory):
    for jsonl_file in tqdm.tqdm(os.listdir(jsonl_compressed_directory)):
        jsonl_file_path = os.path.join(jsonl_compressed_directory, jsonl_file)
        i = 0
        file_info = os.path.basename(jsonl_file).split(".")
        # check zst compress
        if file_info[-1] != "zst":
            continue
        file_name = file_info[0]
        with zstd.open(open(jsonl_file_path, "rb"), "rt", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                yield {
                    "text": data["text"],
                    "source": DATASET_NAME,
                    "source_id": f"{file_name}_{i}",
                    "create_date": NULL,
                    "update_date": NULL,
                    "meta": data["meta"],
                }
                i += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("jsonl_compressed_directory")
    parser.add_argument("output_hf_directory")
    parser.add_argument("--validation_size", default=0.001, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    dataset = Dataset.from_generator(
        data_generator,
        gen_kwargs={"jsonl_compressed_directory": args.jsonl_compressed_directory},
    )
    split = dataset.train_test_split(test_size=args.validation_size, seed=args.seed)
    split["eval"] = split.pop("test")
    split.save_to_disk(args.output_hf_directory)
