from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.merge import merge
from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.constants import (
    OUTPUT_HF_DIR,
)
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "thai_tokenizer_repo", default="flax-community/gpt2-base-thai", type=str
)

args = parser.parse_args()

thai_tokenizer_repo = args.thai_tokenizer_repo

tokenizer = merge(thai_tokenizer_repo)
tokenizer.save_pretrained(OUTPUT_HF_DIR)
