from transformers import AutoModelForCausalLM
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
        "--output_path", type=str, required=True, help="path to save model in local"
    )

    args = parser.parse_args()

    tokenizer = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.output_path)
