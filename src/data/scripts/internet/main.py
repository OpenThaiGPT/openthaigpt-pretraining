from datasets import load_dataset, load_from_disk
import datetime
import jsonlines
from openthaigpt_pretraining_data.internet.mc4.preprocess import (
    clean_text as clean_mc4_text,
)
from openthaigpt_pretraining_data.internet.oscar.preprocess import (
    clean_text as clean_oscar_text,
)

from openthaigpt_pretraining_data.internet.perplexity.perplexity import (
    classify_spam,
    sample_text_back,
)
import numpy as np
import scipy
import json
import os
import hydra

NUM_PROC = 128
DATASET_TO_FILETYPE = {"mc4": "json"}

do_perplexity = batch_size = sampled_back_ratio = None
output_dir = scratch_location = None
source = input_path = version = note = None


@hydra.main(config_path=".")
def load_config(cfg):

    global do_perplexity, batch_size, sampled_back_ratio
    global output_dir, scratch_location, source, input_path, version, note

    cfg = cfg.config

    do_perplexity = cfg.processing_parameters.do_perplexity
    batch_size = cfg.processing_parameters.batch_size
    sampled_back_ratio = cfg.processing_parameters.sampled_back_ratio

    output_dir = cfg.output.path
    scratch_location = cfg.output.scratch_path if "scratch_path" in cfg.output else None

    source = cfg.input_dataset.name
    print(f"Processing {source} dataset")
    input_path = cfg.input_dataset.path

    if "version" in cfg.output:
        version = cfg.output.version
    else:
        if os.path.exists(f"{output_dir}/info.json"):
            info = json.load(f"{output_dir}/info.json")
            version = info["current_version"] + 1
        else:
            version = 1

    note = cfg.note


load_config()


def clean_text(text):
    text = text.strip()
    text = clean_mc4_text(text)

    if text == "":
        return -1, 0, ""

    text = clean_oscar_text(text)

    if text == "":
        return -1, 0, ""

    if not do_perplexity:
        return 0, 0, text

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

    # filter blank
    blank_idx = set([i for i, t in enumerate(chunk["text"]) if t == ""])  # noqa: C403

    for field in chunk:
        chunk[field] = [val for i, val in enumerate(chunk[field]) if i not in blank_idx]

    non_spam_idx = [i for i, p in enumerate(chunk["prediction"]) if p == 0]

    sampled_back_idx = []

    if do_perplexity:
        spam_idx = [i for i, p in enumerate(chunk["prediction"]) if p == 1]
        spam_idx_set = set(spam_idx)

        spam_log_pps = [
            log_pp
            for i, log_pp in enumerate(chunk["log_pp_score"])
            if i in spam_idx_set
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
            percentage=float(sampled_back_ratio),
        )

        sampled_back_idx_set = set(sampled_back_idx)
        sampled_back_idx = [
            spam_idx[i] for i in range(len(spam_idx)) if i in sampled_back_idx_set
        ]  # Map Idx Back to the original index

    selected_idx = set(non_spam_idx + sampled_back_idx)
    for field in chunk:
        chunk[field] = [val for i, val in enumerate(chunk[field]) if i in selected_idx]
    return chunk


def filter_field(data, source):
    data["source"] = source
    meta_dict = {"filename": input_path.split("/")[-1]}
    del data["log_pp_score"], data["prediction"]

    if source == "mc4":
        data["created_date"] = str(data["timestamp"])
        meta_dict["url"] = data["url"]
        del data["timestamp"], data["url"]

    elif source == "cc100":
        data["created_date"] = "2020-10-23T23:31:11.000Z"

    elif "oscar" in source:
        data["created_date"] = "2023-06-08T14:30:28.000Z"
        data["source_id"] = data["id"]
        del data["id"]

    data["meta"] = str(meta_dict)
    if data["updated_date"] == "None":
        data["updated_date"] = data["created_date"]
    return data


if __name__ == "__main__":

    if not os.path.exists(f"{output_dir}/{version}/data/"):
        os.makedirs(f"{output_dir}/{version}/data/")

    if scratch_location:
        if not os.path.exists(f"{scratch_location}/{version}/data/"):
            os.makedirs(f"{scratch_location}/{version}/data/")
        scratch_writer = jsonlines.open(
            f"{scratch_location}/{version}/data/data.jsonl", "w"
        )

    with jsonlines.open(f"{output_dir}/{version}/data/data.jsonl", "w") as writer:

        print("Loading dataset")

        if source in DATASET_TO_FILETYPE.keys():
            dataset = load_dataset(
                DATASET_TO_FILETYPE[source],
                data_files=input_path,
            )

        else:
            dataset = load_from_disk(input_path)

        print(dataset)
        if "train" in dataset.column_names:
            dataset = dataset["train"]
        if "id" not in dataset.column_names and "source_id" not in dataset.column_names:
            dataset = dataset.add_column(
                "source_id", [i for i in range(len(dataset))]  # noqa: C416
            )

        print("Loaded dataset")

        dataset = dataset.map(
            process_chunk_data,
            num_proc=NUM_PROC,
            batched=True,
            batch_size=batch_size,
            # keep_in_memory=True,
            # Incase that I cannot write in public_datasets, so I write in this instead
            cache_file_name=f"hf_cache/{source}/processed.arrow",
        )

        for data in dataset:

            filtered_data = filter_field(data, source)
            writer.write(filtered_data)
            if scratch_location:
                scratch_writer.write(filtered_data)

        print("Finish processing")

        info = {"source": source, "current_version": version}
        json.dump(info, open(f"{output_dir}/info.json", "w"))

        metadata = {
            "dataset_name": source,
            "data_version": version,
            "data_scratch_location": scratch_location,
            "input_name": source,
            "input_version": version,
            "processing_parameters": {
                "do_perplexity": do_perplexity,
                "batch_size": batch_size,
                "sampled_back_ratio": sampled_back_ratio,
            },
            "note": note,
        }
        json.dump(metadata, open(f"{output_dir}/{version}/metadata.json", "w"))

        print("Finish Writing the dataset")
