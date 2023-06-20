import argparse
from openthaigpt_pretraining_model.tokenizers.spm_trainer import (
    train_tokenizer,
)
from openthaigpt_pretraining_model.utils import load_hydra_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--configuration",
        type=str,
        default="../configuration_example/spm_training.yaml",
    )

    args = parser.parse_args()

    config = load_hydra_config(args.configuration)

    train_tokenizer(
        output_path=config.output_path,
        vocab_size=config.vocab_size,
        is_slurm=config.is_slurm,
        load_dataset_path=config.load_dataset_path,
        load_dataset_name=config.load_dataset_name,
        load_dataset_local_path=config.load_dataset_local_path,
        load_dataset_data_type=config.load_dataset_data_type,
        large_corpus=config.large_corpus,
        mode=config.mode,
    )
