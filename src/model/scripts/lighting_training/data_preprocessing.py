import argparse
from openthaigpt_pretraining_model.data_wrapper import (
    TokenizedDataset,
)
from openthaigpt_pretraining_model.utils import load_hydra_config
from openthaigpt_pretraining_model.datasets import get_dataset
from openthaigpt_pretraining_model.datasets.constants import SPLIT_TRAIN
from re import findall
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="decapoda-research/llama-7b-hf",
        help="train | val",
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
        default=SPLIT_TRAIN,
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

    dataset = TokenizedDataset(
        dataset=dataset,
        split=args.split,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        save_path=args.save_path,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
    )
    dataset.tokenize_data()
