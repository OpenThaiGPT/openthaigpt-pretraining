import os
from typing import Optional
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset, Dataset, IterableDataset


# CONSTANTS
DOC_ID = "id"
DOC_TEXT = "text"


class SPMTokenizer:
    def __init__(self, tokenizer_model_path: str) -> None:
        # load the tokenizer model
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_model_path)

    @staticmethod
    def preprocess_text(text: str) -> str:
        # Custom
        return text

    # define a function to tokenize the text
    def tokenize_text(self, text: str) -> str:
        text = SPMTokenizer.preprocess_text(text)
        return self.tokenizer.encode(text, out_type=int)


def text_generator(data_path: str) -> dict:
    for filename in os.listdir(data_path):
        with open(f"{data_path}/{filename}", "r") as f:
            text = f.read()
        yield {"text": text}


def tokenize_dataset(
    tokenizer_model_path: str,
    output_path: str,
    num_docs: int = 5000,
    num_proc: Optional[int] = os.cpu_count(),
    batch_size: int = 1000,
    is_slurm: bool = False,
    load_dataset_path: str = "oscar",
    load_dataset_name: str = "unshuffled_deduplicated_th",
    load_dataset_local_path: Optional[str] = None,
) -> None:
    """
    Tokenizes a dataset using a given tokenizer model and saves the tokenized dataset to disk.

    Args:
        tokenizer_model_path (str): The path to the tokenizer model.
        output_path (str): The path to save the tokenized dataset.
        num_docs (int, optional): The number of documents to process. Defaults to 5000.
        num_proc (int, optional): The number of processes to use for multiprocessing. Defaults to the number of CPUs on the machine.
        batch_size (int, optional): The batch size to use when tokenizing the dataset. Defaults to 1000.
        is_slurm (bool, optional): Whether to use SLURM or not. Defaults to False.
        load_dataset_path (str, optional): The path to the dataset to load. Defaults to "oscar".
        load_dataset_name (str, optional): The name of the dataset to load. Defaults to "unshuffled_deduplicated_th".
        load_dataset_local_path (str, optional): The local path to the dataset to load. Defaults to None.

    Returns:
        None
    """

    if load_dataset_local_path is None:
        if not is_slurm:
            text_dataset = load_dataset(
                path=load_dataset_path,
                name=load_dataset_name,
                split="train",
                streaming=not is_slurm,
            )

            num_docs = len(text_dataset) if num_docs is None else num_docs
            new_dataset: dict = {DOC_ID: [], DOC_TEXT: []}
            for item in tqdm(text_dataset.shuffle().take(num_docs)):
                new_dataset[DOC_ID].append(item[DOC_ID])
                new_dataset[DOC_TEXT].append(item[DOC_TEXT])
            text_dataset = Dataset.from_dict(new_dataset)

        else:
            num_docs = "" if num_docs is None else num_docs
            text_dataset = load_dataset(
                path=load_dataset_path,
                name=load_dataset_name,
                split=f"train[:{num_docs}]",
            )

    else:
        # create a dataset from the generator
        text_dataset = Dataset.from_generator(
            text_generator, gen_kwargs={"data_path": load_dataset_local_path}
        )

    tokenizer = SPMTokenizer(tokenizer_model_path)

    # use the map method to tokenize the text and add it to a new column in the dataset
    tokenized_dataset = text_dataset.map(
        function=lambda example: {
            "text_tokens": tokenizer.tokenize_text(example["text"])
        },
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    tokenized_dataset.save_to_disk(output_path)
