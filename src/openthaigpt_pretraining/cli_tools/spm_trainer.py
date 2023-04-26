import os
import argparse
import regex as re
from tqdm import tqdm

import sentencepiece as spm
import pandas as pd
from datasets import load_dataset, Dataset
from tokenizers import SentencePieceUnigramTokenizer

from openthaigpt_pretraining.utils.data_cleaning import clean_text, filter_out_line

LINES_DELIMETER = "\n"


def prepare_datasets(texts: str) -> dict:
    preapared_texts = []
    for text in texts['text']:  # for every doc
        for line in re.split(LINES_DELIMETER, text):  # for every paragraph
            if re.fullmatch(r"\s*", line):
                continue  # empty string or string with all space characters
            if filter_out_line(line):
                continue

            example = clean_text(line)
            preapared_texts.append(example)

    return {'text_processed': preapared_texts}


class DataSetColumnIterator:
    def __init__(self, dataset, column_name: str):
        self.dataset = iter(dataset)
        self.column_name = column_name

    def __iter__(self):
        for item in self.dataset:
            yield item[self.column_name]


def train_tokenizer(
        output_path: str,
        vocab_size: int,
        num_docs: int,
        is_slurm: bool,
        num_proc: int
):
    if not is_slurm:
        text_dataset = load_dataset(
            "oscar",
            "unshuffled_deduplicated_th",
            split="train",
            streaming=not is_slurm
        )
        new_dataset = {"id": [], "text": []}
        for item in tqdm(text_dataset.shuffle().take(num_docs)):
            new_dataset['id'].append(item['id'])
            new_dataset['text'].append(item['text'])
        text_dataset = Dataset.from_dict(new_dataset)

    else:
        text_dataset = load_dataset(
            "oscar",
            "unshuffled_deduplicated_th",
            split=f"train[:{num_docs}]",
        )

    text_processed_dataset = text_dataset.map(
        function=prepare_datasets,
        batched=True,
        remove_columns=['text', 'id'],  # this is must b/c we will return different number of rows
    )

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(DataSetColumnIterator(text_processed_dataset, 'text_processed')),
        model_prefix=output_path,
        vocab_size=vocab_size,
        user_defined_symbols=['<mask>'],
        num_threads=num_proc
    )
