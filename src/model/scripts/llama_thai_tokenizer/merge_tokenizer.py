import argparse

from openthaigpt_pretraining_model.llama_thai_tokenizer.merge import merge
from openthaigpt_pretraining_model.llama_thai_tokenizer.constants import (
    OUTPUT_HF_DIR,
    LLAMA_TOKENIZER_DIR,
    THAI_SP_MODEL_DIR,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llama_path",
        type=str,
        default=LLAMA_TOKENIZER_DIR,
        help="path to llama tokenizer",
    )
    parser.add_argument(
        "--sp_path",
        type=str,
        default=THAI_SP_MODEL_DIR,
        help="path to tokenizer to merge",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_HF_DIR,
        help="path of output tokenizer",
    )

    args = parser.parse_args()

    tokenizer = merge(args.llama_path, args.sp_path)
    tokenizer.save_pretrained(args.output_path)
