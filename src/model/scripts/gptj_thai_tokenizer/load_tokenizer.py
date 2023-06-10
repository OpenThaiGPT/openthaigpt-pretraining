from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name from huggingface",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="path to save tokenizer"
    )
    parser.add_argument(
        "--merge_file_path", type=str, required=True, help="path to save merge rule"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.tokenizer_path)

    hf_hub_download(
        repo_id=args.model_name,
        filename="merges.txt",
        local_dir=args.merge_file_path,
    )
