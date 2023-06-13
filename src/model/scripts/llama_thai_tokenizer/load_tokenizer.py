from openthaigpt_pretraining_model.llama_thai_tokenizer.constants import (
    LLAMA_TOKENIZER_DIR,
)
from transformers import LlamaTokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default=LLAMA_TOKENIZER_DIR,
        help="llama model name from huggingface",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save llama tokenizer"
    )

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.output_path)
