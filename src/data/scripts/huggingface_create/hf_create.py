import argparse
from datasets import Dataset, concatenate_datasets, DatasetDict
import glob
import json
import pandas as pd
from tqdm import tqdm


def load_data(filepaths):
    dataframes = []
    all_keys = set()
    for filepath in tqdm(filepaths):
        with open(filepath, "r") as f:
            lines = f.readlines()
            json_lines = [json.loads(line) for line in lines]

            # Find all unique keys in the JSON data
            for line in json_lines:
                all_keys.update(line.keys())

            data = []
            for line in json_lines:
                # Add missing keys with default value None
                normalized_line = {key: line.get(key, None) for key in all_keys}
                # Normalize values to strings
                normalized_line = {
                    key: str(value) for key, value in normalized_line.items()
                }
                data.append(normalized_line)

            dataframes.append(Dataset.from_pandas(pd.DataFrame(data)))
    return concatenate_datasets(dataframes)


def main(train_path, eval_path, output_path):
    # Get all file paths for train and eval
    train_filepaths = glob.glob(train_path)
    eval_filepaths = glob.glob(eval_path)

    # Load and concatenate datasets
    train_dataset = load_data(train_filepaths).shuffle(seed=42)
    eval_dataset = load_data(eval_filepaths).shuffle(seed=42)

    # Combine train and eval datasets into one dictionary
    datasets = DatasetDict(
        {
            "train": train_dataset,
            "eval": eval_dataset,
        }
    )

    datasets.save_to_disk(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Loader")
    parser.add_argument("--train_path", required=True, help="Path to train JSONL files")
    parser.add_argument("--eval_path", required=True, help="Path to eval JSONL files")
    parser.add_argument("--output_path", required=True, help="Path to save the dataset")

    args = parser.parse_args()

    main(args.train_path, args.eval_path, args.output_path)
