import argparse
from openthaigpt_pretraining_model.datasets.utils import (
    merge_datasets,
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
        default="hfsave",
    )
    args = parser.parse_args()
    print(args)
    file_path = args.file_path
    output_path = args.output_path
    dataset = merge_datasets(dataset_paths=file_path)
    """
    --dataset_paths path/to/dataset1 path/to/dataset2
    """
    dataset.save_to_disk(output_path)
