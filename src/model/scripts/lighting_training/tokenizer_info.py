import argparse
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer", type=str, required=True, help="name or path to tokenizer"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"len: {len(tokenizer.get_vocab())}")
    if tokenizer.eos_token is not None:
        print(f"end of sentence tag {tokenizer.eos_token}: {tokenizer.eos_token_id}")
    else:
        print("no eos token")
    if tokenizer.unk_token is not None:
        print(f"unknow word tag {tokenizer.unk_token}: {tokenizer.unk_token_id}")
    else:
        print("no unk token")
