import os
from typing import Optional, Union
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset, Dataset

# CONSTANTS
PREPARE_DATASETS_KEY = "text_processed"
DOC_ID = "id"
DOC_TEXT = "text"
USER_DEFINED_SYMBOLS = ["<eos>"]


def prepare_datasets(texts: dict) -> dict:
    """
    Preprocesses a list of text documents and returns a dictionary with a single key 'PREPARE_DATASETS_KEY'
    that maps to the preprocessed texts.

    Args:
        texts (dict): A dictionary containing a key 'DOC_TEXT' that maps to a list of text documents.

    Returns:
        dict: A dictionary with a single key 'PREPARE_DATASETS_KEY' that maps to a list of preprocessed text documents.
    """  # noqa: E501
    preapared_texts = []
    for text in texts[DOC_TEXT]:  # for every doc
        # write custom preprocessing
        preapared_texts.append(text)

    return {PREPARE_DATASETS_KEY: preapared_texts}


class DataSetColumnIterator:
    def __init__(self, dataset, column_name: str):
        self.dataset = iter(dataset)
        self.column_name = column_name

    def __iter__(self):
        for item in self.dataset:
            try:
                yield item[self.column_name]
            except KeyError:
                raise ValueError(
                    f"Column '{self.column_name}' is not a valid index for the dataset"
                )


def train_tokenizer(
    output_path: str,
    vocab_size: int,
    num_docs: Optional[Union[str, int]] = None,
    num_proc: Optional[int] = os.cpu_count(),
    is_slurm: bool = False,
    load_dataset_path: str = "oscar",
    load_dataset_name: str = "unshuffled_deduplicated_th",
    load_dataset_local_path: Optional[str] = None,
    load_dataset_data_type: Optional[str] = None,
    large_corpus: bool = False,
) -> None:
    """
    Train a SentencePiece tokenizer on a large text dataset.

    Args:
        output_path (str): The path and prefix to use when saving the trained tokenizer.
        vocab_size (int): The size of the vocabulary to use when training the tokenizer.
        num_docs (int, optional): The number of documents to use from the input dataset.
        num_proc (int, optional): The number of CPU cores to use when training the tokenizer. Defaults to the number of available CPU cores.
        is_slurm (bool, optional): Whether the code is running on a Slurm cluster. Defaults to False.
        load_dataset_path (str, optional): The name of the Hugging Face dataset to load. Defaults to "oscar".
        load_dataset_name (str, optional): The name of the dataset split to use. Defaults to "unshuffled_deduplicated_th".
        load_dataset_local_path (str, optional): The path to a local directory containing the input data. If specified, the Hugging Face dataset is not used. Defaults to None.
        load_dataset_data_type (str, optional): The file type of the input data if using a local directory. Defaults to "csv".

    Returns:
        None
    """  # noqa: E501
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

        text_dataset = text_dataset.to_iterable_dataset()

    else:
        # Stream from local files
        if load_dataset_data_type is None:
            text_dataset = load_dataset(
                load_dataset_local_path, split="train", streaming=True
            )
        else:
            data_files = {
                "train": [
                    f"{load_dataset_local_path}/{filename}"
                    for filename in os.listdir(load_dataset_local_path)
                ]
            }
            text_dataset = load_dataset(
                load_dataset_data_type,
                data_files=data_files,
                split="train",
                streaming=True,
            )

    text_processed_dataset = text_dataset.map(
        function=prepare_datasets,
        batched=True,
    )

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(
            DataSetColumnIterator(text_processed_dataset, PREPARE_DATASETS_KEY)
        ),
        model_prefix=output_path,
        vocab_size=vocab_size,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        num_threads=num_proc,
        train_extremely_large_corpus=large_corpus,
    )
