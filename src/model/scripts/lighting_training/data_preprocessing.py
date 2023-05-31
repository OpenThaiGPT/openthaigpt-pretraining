import argparse
from openthaigpt_pretraining_model.lightning.utils import (
    TokenizedDataset,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train | val",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="decapoda-research/llama-7b-hf",
        help="train | val",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./tokendata",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 1024,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25000,
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="oscar",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="unshuffled_deduplicated_th",
    )
    args = parser.parse_args()
    print(args)
    dataset = TokenizedDataset(
        mode=args.mode,
        model_or_path=args.model,
        max_tokens=args.max_tokens,
        save_path=args.save_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        use_cache=True,
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
    )
    dataset.tokenize_data()
