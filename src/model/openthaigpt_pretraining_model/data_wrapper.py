import torch
from torch.utils.data import IterableDataset


class DatasetWrapper(IterableDataset):
    def __init__(self, tokenizer, dataset, max_tokens=256):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer = []
        iter_dataset = self.dataset

        for sample in iter_dataset:
            buffer += self.tokenizer(sample["text"])["input_ids"]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]
