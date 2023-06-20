import argparse
from openthaigpt_pretraining_model.tokenizers.spm_trainer import (
    train_tokenizer,
    SPM_MODE,
    BPE_MODE,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path", type=str, help="path of tokenizer file", required=True
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50000, help="max vocab of tokenier"
    )
    parser.add_argument("--data_path", type=str, help="path of datasets", required=True)
    parser.add_argument(
        "--data_type",
        default=None,
        help="data type of dataset if None will be Huggingface format",
    )
    parser.add_argument(
        "--large_corpus", action="store_true", help="use large corpus in train"
    )
    parser.add_argument(
        "--mode", type=str, default=SPM_MODE, help=f"{SPM_MODE} | {BPE_MODE}"
    )

    args = parser.parse_args()

    train_tokenizer(
        args.output_path,
        args.vocab_size,
        load_dataset_local_path=args.data_path,
        load_dataset_data_type=args.data_type,
        large_corpus=args.large_corpus,
        mode=args.mode,
    )
