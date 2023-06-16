import torch
from torch.utils.data import IterableDataset
from torch.nn.functional import pad
from datasets import Dataset, load_from_disk
import os
from tqdm import tqdm
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
        chunk_size: int = 1024 * 1024,
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
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.dataset = dataset

        self.num_shards = (len(self.dataset) + chunk_size - 1) // chunk_size
        self.tokenized_data = None

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
        for i in tqdm(range(self.num_shards)):
            chunk = self.data_set.shard(self.num_shards, i)  # split chunk
            tokenized_dataset = chunk.map(
                self.tokenize_function,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.num_proc,
                remove_columns=self.dataset.column_names,
            )
            print(f"save {self.split}_chunk_{i}")
            tokenized_dataset.save_to_disk(
                os.path.join(self.save_path, f"{self.split}_chunk_{i}")
            )


class TokenDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: Optional[str] = None,
    ):
        if split is None:
            split = "train"
        self.split_ = split
        self.file_paths = []
        self.chunk_lengths = []
        self.total_length = 0
        self.chunk = None
        self.chunk_start_index = 0
        self.chunk_end_index = 0

        chunk_count = 0
        file_path = os.path.join(
            dataset_path,
            f"{self.split_}_chunk_{chunk_count}",
        )
        while os.path.exists(file_path):
            dataset = load_from_disk(file_path)
            self.file_paths.append(file_path)
            self.chunk_lengths.append(len(dataset))
            self.total_length += len(dataset)
            chunk_count += 1
            file_path = os.path.join(
                dataset_path,
                f"{self.split_}_chunk_{chunk_count}",
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
