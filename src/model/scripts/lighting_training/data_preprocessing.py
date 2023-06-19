import argparse
from openthaigpt_pretraining_model.data_wrapper import (
    tokenize_function,
)
from openthaigpt_pretraining_model.utils import load_hydra_config
from openthaigpt_pretraining_model.models import load_tokenizer
from openthaigpt_pretraining_model.datasets import get_dataset
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        type=str,
        default="../configuration_example/data_preprocess.yaml",
    )
    args = parser.parse_args()
    print(args)

    configuration = load_hydra_config(args.configuration)
    data_process_configuration = configuration.data_process

    dataset_configuration = configuration.dataset
    dataset_configuration = dataset_configuration.get(
        data_process_configuration.split, None
    )
    if dataset_configuration is None:
        raise NotImplementedError(
            f"dataset don't have split {data_process_configuration.split}"
        )
    dataset = get_dataset(dataset_configuration)

    tokenizer = load_tokenizer(configuration.model.tokenizer)

    dataset = dataset.map(
        tokenize_function(tokenizer, data_process_configuration.max_tokens),
        desc="Tokenizing...",
        num_proc=data_process_configuration.num_proc,
        batched=True,
        batch_size=data_process_configuration.batch_size,
        remove_columns=dataset.column_names,
    )

    save_path = os.path.join(
        data_process_configuration.save_path,
        data_process_configuration.split,
    )

    dataset.save_to_disk(save_path)
