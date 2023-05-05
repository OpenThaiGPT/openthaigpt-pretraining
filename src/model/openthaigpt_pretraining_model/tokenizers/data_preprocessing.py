import os
from typing import Optional, Union, Generator, Dict
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


# CONSTANTS
DOC_ID = "id"
DOC_TEXT = "text"
USER_DEFINED_SYMBOLS = ["<mask>"]


class BaseTokenizer:
    def __init__(self, tokenizer_model_path: str) -> None:
        self.tokenizer_model_path = tokenizer_model_path

    @staticmethod
    def preprocess_text(text: str) -> str:
        # Custom
        return text


class SPMTokenizerWrapper(BaseTokenizer):
    def __init__(self, tokenizer_model_path: str) -> None:
        super().__init__(tokenizer_model_path)

        # load the tokenizer model
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(self.tokenizer_model_path)
        self.tokenizer.set_user_defined_symbols(USER_DEFINED_SYMBOLS)

    # define a function to tokenize the text
    def tokenize_text(self, text: str) -> str:
        text = self.preprocess_text(text)
        return self.tokenizer.encode(text, out_type=int)


class FastTokenizerWrapper(BaseTokenizer):
    def __init__(self, tokenizer_model_path: str) -> None:
        super().__init__(tokenizer_model_path)

        # load the tokenizer model
        self.tokenizer = PreTrainedTokenizer.from_pretrained(self.tokenizer_model_path)
        self.tokenizer.add_tokens(USER_DEFINED_SYMBOLS)

    # define a function to tokenize the text
    def tokenize_text(self, text: str) -> str:
        text = self.preprocess_text(text)
        return self.tokenizer.encode(text)


def chunk_generator(file_path: str, chunk_size: int = 1024 * 1024 * 1024):
    """
    Generator that yields a file in fixed size chunks.
    """
    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def text_generator(
    data_path: str, chunk_size: int = 1024 * 1024 * 1024
) -> Generator[Dict[str, str], None, None]:
    """
    Generator that yields a text document as a dictionary with a "text" key.
    """
    for filename in os.listdir(data_path):
        file_path = f"{data_path}/{filename}"
        for file_chunk in chunk_generator(file_path, chunk_size):
            text = file_chunk.decode("utf-8")
            yield {"text": text}


def tokenize_dataset(
    tokenizer_model_path: str,
    output_path: str,
    tokenizer_model_type: str = "spm",
    num_docs: Optional[Union[str, int]] = 5000,
    num_proc: Optional[int] = os.cpu_count(),
    batch_size: int = 100_000,
    is_slurm: bool = False,
    load_dataset_path: str = "oscar",
    load_dataset_name: str = "unshuffled_deduplicated_th",
    load_dataset_local_path: Optional[str] = None,
    chunk_doc_size: int = 1024 * 1024 * 1024,
) -> None:
    """
    Tokenizes a dataset using a given tokenizer model and saves the tokenized dataset to disk.

    Args:
        tokenizer_model_path (str): The path to the tokenizer model.
        output_path (str): The path to save the tokenized dataset.
        tokenizer_model_type (str): type of tokenizer model (`spm`: sentencepiece, `fast`: `AutoTokenizer`).
        num_docs (int, optional): The number of documents to process. Defaults to 5000.
        num_proc (int, optional): The number of processes to use for multiprocessing. Defaults to the number of CPUs on the machine.
        batch_size (int, optional): The batch size to use when tokenizing the dataset. Defaults to 100_000.
        is_slurm (bool, optional): Whether to use SLURM or not. Defaults to False.
        load_dataset_path (str, optional): The path to the dataset to load. Defaults to "oscar".
        load_dataset_name (str, optional): The name of the dataset to load. Defaults to "unshuffled_deduplicated_th".
        load_dataset_local_path (str, optional): The local path to the dataset to load. Defaults to None.
        chunk_doc_size (int, optional): chunk size of each doc for generator

    Returns:
        None
    """  # noqa: E501

    tokenizer: Union[SPMTokenizerWrapper, FastTokenizerWrapper]
    if tokenizer_model_type == "spm":
        tokenizer = SPMTokenizerWrapper(tokenizer_model_path)
    elif tokenizer_model_type == "fast":
        tokenizer = FastTokenizerWrapper(tokenizer_model_path)
    else:
        raise ValueError("Invalid tokenizer_model_type")

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
            text_generator,
            gen_kwargs={
                "data_path": load_dataset_local_path,
                "chunk_size": chunk_doc_size,
            },
        )

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
