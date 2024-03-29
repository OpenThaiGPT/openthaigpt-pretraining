import os
from typing import Optional, Union
from tqdm import tqdm
import sentencepiece as spm
from transformers import LlamaTokenizer
from datasets import load_dataset, load_from_disk, Dataset

# CONSTANTS
PREPARE_DATASETS_KEY = "text_processed"
DOC_ID = "id"
DOC_TEXT = "text"
EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"
UNK_TOKEN = "<unk>"

SPM_MODE = "spm"
BPE_MODE = "bpe"


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


def load_local_dataset(data_type, local_path):
    if data_type is None:
        text_dataset = load_from_disk(local_path)["train"]
    else:
        data_files = {
            "train": [f"{local_path}/{filename}" for filename in os.listdir(local_path)]
        }
        text_dataset = load_dataset(
            data_type,
            data_files=data_files,
            split="train",
            streaming=True,
        )

    return text_dataset


def train_tokenizer(
    output_path: str,
    vocab_size: int,
    num_docs: Optional[Union[str, int]] = None,
    num_proc: Optional[int] = os.cpu_count(),
    streaming: bool = True,
    load_dataset_path: str = "oscar",
    load_dataset_name: str = "unshuffled_deduplicated_th",
    load_dataset_local_path: Optional[str] = None,
    load_dataset_data_type: Optional[str] = None,
    large_corpus: bool = False,
    mode: str = SPM_MODE,
) -> None:
    """
    Train a SentencePiece tokenizer on a large text dataset.

    Args:
        output_path (str): The path and prefix to use when saving the trained tokenizer.
        vocab_size (int): The size of the vocabulary to use when training the tokenizer.
        num_docs (int, optional): The number of documents to use from the input dataset.
        num_proc (int, optional): The number of CPU cores to use when training the tokenizer. Defaults to the number of available CPU cores.
        streaming (bool, optional): Whether the code is running on a Slurm cluster. Defaults to False.
        load_dataset_path (str, optional): The name of the Hugging Face dataset to load. Defaults to "oscar".
        load_dataset_name (str, optional): The name of the dataset split to use. Defaults to "unshuffled_deduplicated_th".
        load_dataset_local_path (str, optional): The path to a local directory containing the input data. If specified, the Hugging Face dataset is not used. Defaults to None.
        load_dataset_data_type (str, optional): The file type of the input data if using a local directory. Defaults to "csv".

    Returns:
        None
    """  # noqa: E501

    if not (mode == SPM_MODE or mode == BPE_MODE):
        KeyError(f"mode mush be {SPM_MODE} or {BPE_MODE}")

    if load_dataset_local_path is None:
        if streaming:
            text_dataset = load_dataset(
                path=load_dataset_path,
                name=load_dataset_name,
                split="train",
                streaming=streaming,
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
        text_dataset = load_local_dataset(
            load_dataset_data_type, load_dataset_local_path
        )

    text_processed_dataset = text_dataset.map(
        function=prepare_datasets,
        batched=True,
    )

    os.makedirs(output_path, exist_ok=True)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(
            DataSetColumnIterator(text_processed_dataset, PREPARE_DATASETS_KEY)
        ),
        model_prefix=output_path + "/spm_tokenizer",
        vocab_size=vocab_size,
        user_defined_symbols=[],
        num_threads=num_proc,
        train_extremely_large_corpus=large_corpus,
        model_type=mode,
    )

    tokenizer = LlamaTokenizer(vocab_file=output_path + "/spm_tokenizer.model")

    tokenizer.eos_token = EOS_TOKEN
    tokenizer.bos_token = BOS_TOKEN
    tokenizer.unk_token = UNK_TOKEN

    tokenizer.save_pretrained(output_path)
