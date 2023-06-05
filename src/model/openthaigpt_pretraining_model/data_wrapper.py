import torch
from torch.utils.data import IterableDataset
from datasets import Dataset, load_dataset, load_from_disk
import os
from re import findall
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
)
from ..datasets.constants import SPLIT_TRAIN, SPLIT_VAL

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
        mode: str,
        tokenizer: str,
        max_tokens: int = 2048,
        save_path: str = "./",
        chunk_size: int = 1024 * 1024,
        batch_size: int = 10000,
        num_proc: int = 16,
        dataset_name: str = "oscar",
        dataset_dir: str = "unshuffled_deduplicated_th",
    ):
        if len(findall("llama", tokenizer)):
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.mode = mode
        self.max_tokens = max_tokens
        self.save_path = save_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.num_proc = num_proc
        if mode == "val":
            self.data_set = load_dataset(
                dataset_name,
                dataset_dir,
                split=SPLIT_VAL,
            )
        elif mode == "train":
            self.data_set = load_dataset(
                dataset_name,
                dataset_dir,
                split=SPLIT_TRAIN,
            )
        else:
            raise NotImplementedError("only support Train,Val")
        self.num_shards = (len(self.data_set) + chunk_size - 1) // chunk_size

        self.tokenized_data = None

    def tokenize_data(self):
        def tokenize_function(examples):
            tokenized_text = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.max_tokens,
            )["input_ids"]
            return {HF_TOKENIZER_INPUT_IDS_NAME: tokenized_text}

        for i in tqdm(range(self.num_shards)):
            chunk = self.data_set.shard(self.num_shards, i)  # split chunk
            tokenized_dataset = chunk.map(
                tokenize_function,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
            )
            print(f"save {self.mode}_chunk_{i}")
            tokenized_dataset.save_to_disk(
                os.path.join(self.save_path, f"{self.mode}_chunk_{i}")
            )


class StreamingDatasetWrapper(IterableDataset):
    def __init__(
        self,
        model_or_path,
        mode,
        max_tokens,
        dataset_name,
        dataset_dir,
    ):
        self.mode = mode
        self.max_tokens = max_tokens

        if len(findall("llama", model_or_path)):
            self.tokenizer = LlamaTokenizer.from_pretrained(model_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_path)

        if mode == "val":
            self.data_set = load_dataset(
                dataset_name,
                dataset_dir,
                streaming=True,
                split=SPLIT_VAL,
            )
        elif mode == "train":
            self.data_set = load_dataset(
                dataset_name,
                dataset_dir,
                streaming=True,
                split=SPLIT_TRAIN,
            ).shuffle(buffer_size=10_000)
        else:
            raise NotImplementedError("only support Train,Val")

    def __iter__(self):
        buffer = []
        iter_dataset = self.data_set

        for sample in iter_dataset:
            buffer += self.tokenizer(sample["text"])[HF_TOKENIZER_INPUT_IDS_NAME]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


class TokenDatasetWrapper(Dataset):
    def __init__(
        self,
        mode: str,
        dataset_path: str,
    ):
        self.mode = mode
        self.file_paths = []
        self.chunk_lengths = []
        self.total_length = 0
        self.chunk = None
        self.chunk_start_index = 0
        self.chunk_end_index = 0

        chunk_count = 0
        file_path = os.path.join(
            dataset_path,
            f"{self.mode}_chunk_{chunk_count}",
        )
        while os.path.exists(file_path):
            dataset = load_from_disk(file_path)
            self.file_paths.append(file_path)
            self.chunk_lengths.append(len(dataset))
            self.total_length += len(dataset)
            chunk_count += 1
            file_path = os.path.join(
                dataset_path,
                f"{self.mode}_chunk_{chunk_count}",
            )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if not self.chunk_start_index <= idx < self.chunk_end_index:
            # Calculate the chunk index
            file_index = 0
            file_check = idx
            for i, chunk_length in enumerate(self.chunk_lengths):
                if file_check < chunk_length:
                    file_index = i
                    break
                file_check -= chunk_length

            self.chunk = load_from_disk(self.file_paths[file_index])
            self.chunk_start_index = sum(self.chunk_lengths[:file_index])
            self.chunk_end_index = (
                self.chunk_start_index + self.chunk_lengths[file_index]
            )
        return torch.tensor(
            self.chunk[idx - self.chunk_start_index][HF_TOKENIZER_INPUT_IDS_NAME]
        )
