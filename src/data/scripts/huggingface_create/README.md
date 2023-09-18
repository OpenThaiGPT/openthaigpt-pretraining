# Hugging Face Dataset Creation Script
This script is designed to create datasets for use with the Hugging Face Transformers library. It takes JSONL files containing data and converts them into a format compatible with the Hugging Face Datasets library. The resulting dataset can then be easily loaded and used for training and evaluation in various NLP tasks.

## Usage
This script is intended to be used for creating custom datasets that can be utilized with Hugging Face Transformers. It takes as input a set of JSONL files containing data and produces a Hugging Face dataset in the desired format.

## Dependencies
Before using this script, make sure you have the following dependencies installed:

datasets
pandas
tqdm
You can typically install these dependencies using pip:

```bash
pip install datasets pandas tqdm
```

## How to Run
You can run this script from the command line with the following command:

```bash
python script.py --train_path path_to_train_data --eval_path path_to_eval_data --output_path output_dataset_directory
```

train_path: Path to the directory containing the training JSONL files.
eval_path: Path to the directory containing the evaluation JSONL files.
output_path: Path to the directory where the resulting dataset will be saved.

## Input Data Format
The input data should be in JSONL (JSON Lines) format, where each line represents a JSON object. Each JSON object represents a data instance with one or more fields.

The script assumes that all JSON objects within the JSONL files have the same set of keys. If some keys are missing in certain objects, they will be added with a default value of None.

## Output Data Format
The script converts the input JSONL data into a Hugging Face dataset format. The resulting dataset is saved as a directory, and it contains two splits:

train: The training split.
eval: The evaluation split.
Each split is stored as a set of Parquet files and can be loaded and used with Hugging Face Datasets library.
