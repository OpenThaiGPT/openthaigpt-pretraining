from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.merge import merge
from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.constants import (
    OUTPUT_HF_DIR,
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer_1_dir", type=str, required=True, help="path to tokenizer to merge"
    )
    parser.add_argument(
        "--tokenizer_2_dir", type=str, required=True, help="path to tokenizer to merge"
    )
    parser.add_argument(
        "--merge_file_1",
        type=str,
        required=True,
        help="path to merge rule of tokenizer_1",
    )
    parser.add_argument(
        "--merge_file_2",
        type=str,
        required=True,
        help="path to merge rule of tokenizer_2",
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_HF_DIR, help="path of output tokenizer"
    )

    args = parser.parse_args()

    thai_tokenizer_repo = args.thai_tokenizer_repo

    tokenizer = merge(
        args.tokenizer_1_dir, args.tokenizer_2_dir, args.merge_file_1, args.merge_file_2
    )
    tokenizer.save_pretrained(args.output_dir)
