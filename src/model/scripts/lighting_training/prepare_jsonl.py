import argparse
from openthaigpt_pretraining_model.datasets.utils import (
    jsonl_to_dataset,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="path/to/file",
    )
    parser.add_argument(
        "output_path",
        type=str,
        default="path/to/save",
    )
    args = parser.parse_args()
    print(args)
    file_path = args.file_path
    output_path = args.output_path
    dataset = jsonl_to_dataset(jsonl_file_path=file_path)

    dataset.save_to_disk(output_path)
