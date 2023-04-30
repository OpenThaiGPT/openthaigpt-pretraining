from openthaigpt_pretraining_data.oscar.preprocess import (
    clean_dataset as clean_oscar_dataset,
)
from openthaigpt_pretraining_data.core.preprocess import clean_dataset as clean_dataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", help="Name of an input file", default="input.txt")
parser.add_argument(
    "--output_file", help="Name of an output file", default="output.txt"
)
parser.add_argument(
    "--engine",
    help="Engine of preprocessing (Default: core)",
    default="core",
    choices=["core", "oscar"],
)

args = parser.parse_args()

CLEAN_FUNCTION = {"oscar": clean_oscar_dataset, "core": clean_dataset}

with open(args.input_file, "r") as f:
    dataset = [line.strip() for line in f.readlines()]

output_dataset = CLEAN_FUNCTION[args.engine](dataset)

with open(args.output_file, "w") as f:
    f.write("\n".join(output_dataset))
