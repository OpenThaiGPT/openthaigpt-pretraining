import argparse
from openthaigpt_pretraining_model.data_wrapper import (
    tokenize_function,
)
from openthaigpt_pretraining_model.utils import load_hydra_config
from openthaigpt_pretraining_model.datasets import get_dataset
from openthaigpt_pretraining_model.datasets.constants import SPLIT_TRAIN, SPLIT_VAL
from re import findall
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
)
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="decapoda-research/llama-7b-hf",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./tokendata",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25000,
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="src/model/configuration_example/config.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=f"{SPLIT_TRAIN} | {SPLIT_VAL}",
    )
    args = parser.parse_args()
    print(args)
    dataset_configuration = load_hydra_config(args.configuration).dataset
    dataset_configuration = dataset_configuration.get(args.split, None)
    if dataset_configuration is None:
        raise NotImplementedError(f"dataset don't have split {args.split}")
    dataset = get_dataset(dataset_configuration)
    if len(findall("llama", args.tokenizer)):
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        tokenize_function(tokenizer, args.max_tokens),
        desc="Tokenizing...",
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
    )

    save_path = os.path.join(
        args.save_path,
        args.split,
    )

    dataset.save_to_disk(save_path)
