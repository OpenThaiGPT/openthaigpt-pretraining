import torch
from torch.utils.data import IterableDataset
from torch.nn.functional import pad
from datasets import load_from_disk
import os
from typing import Optional


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
        self.dataset = dataset
        self.num_proc = num_proc

    def tokenize_function(self, data):
        outputs = self.tokenizer(data["text"])
        result_list = []
        # Iterate over each sublist and extend the result list with sublist elements
        for sublist in outputs[HF_TOKENIZER_INPUT_IDS_NAME]:
            result_list.extend(sublist)
            result_list.append(0)  # Insert 0 between sublist elements
        # desired_dim_2 = 4  # Desired size along the second dimension
        padding_value = self.tokenizer.eos_token_id  # Number to use for padding
        input_tensor = torch.Tensor(result_list).long()
        # Determine the size of the first dimension based on desired_dim_2
        desired_dim_1 = -(-input_tensor.size(0) // self.max_tokens)  # Round up division
        # Pad the input tensor if necessary
        padded_tensor = pad(
            input_tensor,
            (0, desired_dim_1 * self.max_tokens - input_tensor.size(0)),
            value=padding_value,
        )
        # Reshape the padded tensor
        reshaped_tensor = padded_tensor.reshape(desired_dim_1, self.max_tokens)
        return {HF_TOKENIZER_INPUT_IDS_NAME: reshaped_tensor}

    def tokenize_data(self):
        os.makedirs(self.save_path, exist_ok=True)
        dataset = self.dataset.map(
            self.tokenize_function,
            desc="Tokenizing...",
            num_proc=self.num_proc,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=self.dataset.column_names,
        )

        save_path = os.path.join(
            self.save_path,
            self.split,
        )

        dataset.save_to_disk(save_path)


def load_token_dataset(dataset_path: str, num_shards: int, split: Optional[str] = None):
    if split is None:
        split = "train"

    file_path = os.path.join(
        dataset_path,
        split,
    )
    dataset = load_from_disk(file_path)
    dataset = dataset.with_format("torch")

    return dataset.to_iterable_dataset(num_shards=num_shards)
