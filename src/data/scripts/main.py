import argparse
from datasets import load_dataset, load_from_disk
import datetime
import time
import jsonlines
from openthaigpt_pretraining_data.mc4.preprocess import (
    clean_text as clean_mc4_text,
)
from openthaigpt_pretraining_data.oscar.preprocess import (
    clean_text as clean_oscar_text,
)

from openthaigpt_pretraining_data.core.perplexity import (
    classify_spam,
    sample_text_back,
)
import numpy as np
import scipy

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    help="""Name of an input file or directory (in case of oscar)
                (Default: "scripts/mc4_th_validation.json")""",
    default="scripts/mc4_th_validation.json",
)
parser.add_argument(
    "--source",
    help='mc4, cc100, oscar (Default: "mc4")',
    default="mc4",
)
parser.add_argument(
    "--num_proc",
    help="number of processor, you will use (Default: 2)",
    default=2,
)
parser.add_argument(
    "--batch_size",
    help="Chunk size of data (Data will be processed together) (Default: 1000)",
    default=1000,
)
parser.add_argument(
    "--output_file",
    help='Name of an output file (Default: "scripts/output.jsonl")',
    default="scripts/output.jsonl",
)
parser.add_argument(
    "--sampled_back_ratio",
    help="""Ratio of data classified as spam to sampled back to the dataset.
    (Default: 0.1)""",
    default=0.1,
)

args = parser.parse_args()


def clean_text(text):
    text = text.strip()
    text = clean_mc4_text(text)

    if text == "":
        return -1, 0, ""

    text = clean_oscar_text(text)

    if text == "":
        return -1, 0, ""

    prediction, log_pp_score = classify_spam(text)

    return prediction[0], log_pp_score, text


def process_chunk_data(chunk):
    n = len(chunk["text"])
    predictions = [-1] * n
    log_pp_scores = [0] * n
    updated_dates = ["None"] * n

    for i, text in enumerate(chunk["text"]):
        prediction, log_pp_score, new_text = clean_text(text)

        predictions[i] = prediction
        log_pp_scores[i] = log_pp_score

        if new_text != text:
            chunk["text"][i] = new_text
            updated_dates[i] = str(
                datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            )

    chunk["prediction"] = predictions
    chunk["log_pp_score"] = log_pp_scores
    chunk["updated_date"] = updated_dates

    non_spam_idx = [
        i for i, p in enumerate(chunk["prediction"]) if p == 0 and chunk["text"] != ""
    ]
    spam_idx = [i for i, p in enumerate(chunk["prediction"]) if p == 1]
    spam_idx = set(spam_idx)

    spam_log_pps = [
        log_pp for i, log_pp in enumerate(chunk["log_pp_score"]) if i in spam_idx
    ]
    log_pp_array = np.array(spam_log_pps)

    # sampled some data point classified as spam back
    probs = scipy.stats.norm.pdf(
        log_pp_array,
        loc=np.mean(log_pp_array),
        scale=np.std(log_pp_array),
    )

    sampled_back_idx = sample_text_back(
        probs,
        percentage=float(args.sampled_back_ratio),
    )

    selected_idx = set(non_spam_idx + sampled_back_idx)
    for field in chunk:
        chunk[field] = [val for i, val in enumerate(chunk[field]) if i in selected_idx]

    return chunk


def filter_field(data, source):
    data["source"] = source
    meta_dict = {"filename": args.input_file.split("/")[-1]}
    del data["log_pp_score"], data["prediction"]

    if source == "mc4":
        data["created_date"] = str(data["timestamp"])
        meta_dict["url"] = data["url"]
        del data["timestamp"], data["url"]

    elif source == "cc100":
        data["created_date"] = "2020-10-23T23:31:11.000Z"

    elif source == "oscar":
        data["created_date"] = "2023-06-08T14:30:28.000Z"
        data["source_id"] = data["id"]
        del data["id"]

    data["meta"] = str(meta_dict)
    if data["updated_date"] == "None":
        data["updated_date"] = data["created_date"]
    return data


NUM_PROC = int(args.num_proc)
DATASET_TO_FILETYPE = {"mc4": "json", "cc100": "txt"}

if __name__ == "__main__":
    with jsonlines.open(args.output_file, "w") as writer:
        start_time = time.perf_counter()

        if args.source != "oscar":
            dataset = load_dataset(
                DATASET_TO_FILETYPE[args.source],
                data_files=args.input_file,
                split="train",
            )
            dataset = dataset.add_column(
                "source_id", [i for i in range(len(dataset))]  # noqa: C416
            )
        else:
            dataset = load_from_disk(args.input_file)

        dataset = dataset.map(
            process_chunk_data,
            num_proc=NUM_PROC,
            batched=True,
            batch_size=args.batch_size,
        )

        for data in dataset:
            writer.write(filter_field(data, args.source))
