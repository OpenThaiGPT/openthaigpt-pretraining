from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from transformers import Trainer
from datasets import load_from_disk

import os


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    use_flash_attention_2: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the tokenized data."}
    )
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


def load_dataset(path, split):
    path_to_split = os.path.join(path, split)
    return load_from_disk(path_to_split)


def make_supervised_data_module(data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = load_dataset(data_args.data_path, data_args.train_split)
    eval_dataset = load_dataset(data_args.data_path, data_args.eval_split)
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

    data_module = make_supervised_data_module(data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train(training_args.checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
