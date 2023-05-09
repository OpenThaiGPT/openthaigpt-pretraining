from openthaigpt_pretraining_data.mc4.preprocess import (
    clean_dataset as clean_mc4_dataset,
)

from openthaigpt_pretraining_data.oscar.preprocess import (
    clean_dataset as clean_oscar_dataset,
)
from openthaigpt_pretraining_data.core.preprocess import clean_dataset as clean_dataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    help='Name of an input file (Default: "input.txt")',
    default="input.txt",
)
parser.add_argument(
    "--output_file",
    help='Name of an output file (Default: "output.txt")',
    default="output.txt",
)
parser.add_argument(
    "--engine",
    help='Engine of preprocessing (Default: "core")',
    default="core",
    choices=["core", "oscar", "mc4"],
)

args = parser.parse_args()

CLEAN_FUNCTION = {"mc4": clean_mc4_dataset, "oscar": clean_oscar_dataset, "core": clean_dataset}

with open(args.input_file, "r", encoding="utf-8") as f:
    dataset = [line.strip() for line in f.readlines()]

output_dataset = CLEAN_FUNCTION[args.engine](dataset)

with open(args.output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(output_dataset))
