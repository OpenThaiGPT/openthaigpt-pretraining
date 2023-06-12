from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from nlpo3 import segment, load_dict
import argparse
import json
import os

SPECIAL_TOKENS_FILE = f"{os.path.dirname(__file__)}/sp_token.json"
DICT_NAME = "dict"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path", type=str, help="path of tokenizer file", required=True
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50000, help="max vocab of tokenier"
    )
    parser.add_argument("--data_path", type=str, help="path of datasets", required=True)
    parser.add_argument(
        "--data_type",
        default=None,
        help="data type of dataset if None will be Huggingface format",
    )
    parser.add_argument(
        "--special_tokens_file",
        type=str,
        default=SPECIAL_TOKENS_FILE,
        help="Path to the JSON file containing special tokens",
    )
    parser.add_argument(
        "--dict_file",
        type=str,
        required=True,
        help="path to tokenization dictionary file",
    )
    parser.add_argument(
        "batch_size", type=int, default=1000, help="batch size for train tokenizer"
    )

    args = parser.parse_args()

    load_dict(args.dict_file, DICT_NAME)

    # load dataset
    if args.data_type is None:
        dataset = load_dataset(args.data_path, split="train", streaming=True)
    else:
        data_files = {
            "train": [
                f"{args.data_path}/{filename}"
                for filename in os.listdir(args.data_path)
            ]
        }
        dataset = load_dataset(
            args.data_type, data_files=data_files, split="train", streaming=True
        )

    # Instantiate tokenizer
    tokenizer = ByteLevelBPETokenizer()

    def th_tokenize(text):
        result = " ".join(segment(text, DICT_NAME))
        return result

    def batch_iterator(batch_size=args.batch_size):
        texts = []
        for i, data in enumerate(dataset):
            texts.append(data["text"])
            if (i + 1) % batch_size == 0:
                yield texts
                texts = []
        yield texts

    with open(args.special_tokens_file, "r") as file:
        special_tokens_data = json.load(file)
        special_tokens = special_tokens_data["special_tokens"]

    # Customized training
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    # Save files to disk
    tokenizer.save_model(args.output_path)
