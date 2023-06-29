import torch
import transformers
from setfit import get_templated_dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from datasets import load_dataset
import numpy as np
import pandas as pd
import re

import argparse
from timeit import default_timer as timer
from datetime import timedelta

import tqdm


def nop(it, *a, **k):
    return it


tqdm.tqdm = nop
import lmppl


def load_models():
    pp_model = lmppl.LM("nakcnx/TGPT-Neo-125M")

    fs_model1 = SetFitModel.from_pretrained(
        "nakcnx/setfit-paraphrase-multilingual-MiniLM-bad_topic",
        cache_dir="./Huggingface_model_cache/",
    )
    fs_model2 = SetFitModel.from_pretrained(
        "nakcnx/setfit-paraphrase-multilingual-mpnet-bad_topic",
        cache_dir="./Huggingface_model_cache/",
    )
    return pp_model, fs_model1, fs_model2


def load_cc100(nrow: int):
    ds = load_dataset("cc100", lang="th", streaming=True)
    ds = ds.shuffle()
    ds = ds["train"].take(nrow)
    return list(ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Inference Time on CC100")
    parser.add_argument(
        "--nrow", type=int, required=True, help="Number of Rows to Test"
    )
    args = parser.parse_args()

    pp_model, fs_model1, fs_model2 = load_models()
    ds_list = load_cc100(args.nrow)

    pp_threshold = 6
    good, bad = [], []
    start = timer()
    for txt in ds_list:
        bad.append(txt["text"]) if pp_model.get_perplexity(
            txt["text"]
        ) > pp_threshold else good.append(txt["text"])
    end = timer()
    print("Inference Time of Perplexity")
    print(timedelta(seconds=end - start))

    good2, bad2 = [], []
    start = timer()
    for txt in ds_list:
        r1 = fs_model1([txt["text"]])
        r2 = fs_model2([txt["text"]])
        good2.append(txt["text"]) if r1 == "good" and r2 == "good" else bad2.append(
            txt["text"]
        )
    end = timer()
    print("Inference Time of Few Shot Classifier")
    print(timedelta(seconds=end - start))
