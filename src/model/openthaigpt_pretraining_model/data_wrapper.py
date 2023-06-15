import torch
from torch.utils.data import IterableDataset
from datasets import load_from_disk
from typing import Optional
import os


HF_TOKENIZER_INPUT_IDS_NAME = "input_ids"


class DatasetWrapper(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        max_tokens: int = 256,
        text_column_name: str = "text",
    ):
        """
        Args:
            tokenizer: hf tokenizer
            dataset: hf dataset which has [{text_column_name: "example"}, ...]
                structure.
            text_column_name: Column name which contain text data.
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.text_column_name = text_column_name

    def __iter__(self):
        buffer = []
        iter_dataset = self.dataset

        for sample in iter_dataset:
            buffer += self.tokenizer(sample[self.text_column_name])[
                HF_TOKENIZER_INPUT_IDS_NAME
            ]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


class TokenizedDataset:
    def __init__(
        self,
        dataset,
        split: str,
        tokenizer,
        max_tokens: int = 2048,
        save_path: str = "./",
        batch_size: int = 10000,
        num_proc: int = 16,
    ):
        """
        Args:
            tokenizer: hf tokenizer
            dataset: hf dataset which has [{text_column_name: "example"}, ...]
            chunk_size: size of each chunk that you want to split
            batch_size: higher for faster but care about OOM(out of memory)
            num_proc: int = 16: number of process, suggest to use equal number of cpus
        """
        self.tokenizer = tokenizer
        self.split = split
        self.max_tokens = max_tokens
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.dataset = dataset

    def tokenize_function(self, data):
        tokenized_text = self.tokenizer(
            data["text"],
            truncation=True,
            padding=True,
            max_length=self.max_tokens,
        )["input_ids"]
        return {HF_TOKENIZER_INPUT_IDS_NAME: tokenized_text}

    def tokenize_data(self):
        os.makedirs(self.save_path, exist_ok=True)
        tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
        )
        tokenized_dataset.save_to_disk(os.path.join(self.save_path, self.split))


def load_token_dataset(dataset_path: str, split: Optional[str] = None):
    if split is None:
        split = "train"
    file_path = os.path.join(
        dataset_path,
        split,
    )
    dataset = load_from_disk(file_path)

    return dataset.with_format("torch")
