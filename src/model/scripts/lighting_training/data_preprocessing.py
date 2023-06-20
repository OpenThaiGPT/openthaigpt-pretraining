from openthaigpt_pretraining_model.data_wrapper import (
    tokenize_function,
)
from openthaigpt_pretraining_model.models import load_tokenizer
from openthaigpt_pretraining_model.datasets import get_dataset
import os

import hydra


@hydra.main(
    version_base=None,
    config_path="../../configuration_example/",
    config_name="data_preprocess",
)
def main(cfg):
    data_process_configuration = cfg.data_process

    dataset_configuration = cfg.dataset
    dataset_configuration = cfg.get(data_process_configuration.split, None)
    if dataset_configuration is None:
        raise NotImplementedError(
            f"dataset don't have split {data_process_configuration.split}"
        )
    dataset = get_dataset(dataset_configuration)

    tokenizer = load_tokenizer(cfg.model.tokenizer)

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


if __name__ == "__main__":
    main()  # type: ignore
