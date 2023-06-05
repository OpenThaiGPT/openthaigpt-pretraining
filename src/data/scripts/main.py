from openthaigpt_pretraining_data.mc4.preprocess import (
    clean_dataset as clean_mc4_dataset,
)
from openthaigpt_pretraining_data.oscar.preprocess import (
    clean_dataset as clean_oscar_dataset,
)
from openthaigpt_pretraining_data.core.perplexity import remove_spam_from_dataset
import argparse
import os
import platform
from datetime import datetime


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == "Windows":
        return datetime.fromtimestamp(os.path.getctime(path_to_file)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    else:
        stat = os.stat(path_to_file)
        try:
            return datetime.fromtimestamp(stat.st_birthtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except AttributeError:
            return datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    help='Name of an input file (Default: "input.txt")',
    default="scripts/input.jsonl",
)
parser.add_argument(
    "--output_file",
    help='Name of an output file (Default: "output.txt")',
    default="scripts/output.jsonl",
)

args = parser.parse_args()

with open(args.input_file, "r", encoding="utf-8") as f:
    json_list = list(f)

dataset = []
for json_str in json_list:
    dataset.append(eval(json_str))

if "created_time" not in dataset[0]:
    created_datetime = creation_date(args.input_file)
    for i, data in enumerate(dataset):
        dataset[i]["created_time"] = created_datetime
        dataset[i]["updated_time"] = created_datetime

dataset = clean_mc4_dataset(dataset)
dataset = clean_oscar_dataset(dataset)
dataset = remove_spam_from_dataset(dataset)

for i, data in enumerate(dataset):
    dataset[i]["source"] = ""
    dataset[i]["source_id"] = str(i)

with open(args.output_file, "w", encoding="utf-8") as f:
    f.write("\n".join([str(data_point) for data_point in dataset]))
