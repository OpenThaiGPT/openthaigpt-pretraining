from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from transformers import Trainer
from datasets import load_from_disk
from torch.utils.data import IterableDataset


import os
import random


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    use_flash_attention_2: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: List[str] = field(
        default_factory=list, metadata={"help": "Path to the tokenized data."}
    )
    data_weights: List[float] = field(default_factory=list)
    train_split: Optional[str] = field(default="train")
    eval_split: Optional[str] = field(default="eval")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    checkpoint: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."  # noqa
        },
    )


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.tensor(input_ids)  # type: ignore
        return {
            "input_ids": input_ids,  # type: ignore
            "labels": input_ids,  # type: ignore
        }


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights

        n_datasets = len(datasets)

        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

        len_datasets = []
        for dataset in self._datasets:
            len_datasets.append(len(dataset))
        self.total_len = int(min(len_datasets) * sum(self._weights))

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return self.total_len


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)

        return next(dataset)


def load_dataset(paths, weights, split, seed=42):
    datasets = []
    for path in paths:
        path_to_split = os.path.join(path, split)
        dataset = load_from_disk(path_to_split)
        datasets.append(dataset)
    return CombinedDataset(datasets, seed, weights)


def make_supervised_data_module(data_args: DataArguments, seed=42) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = load_dataset(
        data_args.data_path, data_args.data_weights, data_args.train_split, seed
    )
    eval_dataset = load_dataset(
        data_args.data_path, data_args.data_weights, data_args.eval_split, seed
    )
    data_collator = DataCollatorForSupervisedDataset()
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # if tokenizer is not None and model.vocab_size != len(tokenizer):
    #     model.resize_token_embeddings(len(tokenizer))

    data_module = make_supervised_data_module(
        data_args=data_args, seed=training_args.data_seed
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train(training_args.checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
