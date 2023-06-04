import torch
from torch.utils.data import IterableDataset

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
