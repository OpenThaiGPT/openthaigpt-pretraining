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
    dataset_split = data_process_configuration.get("split", None)
    if dataset_config is None:
        raise NotImplementedError(f"dataset don't have split {dataset_split}")
    if isinstance(dataset_split, str):
        dataset_split = [dataset_split]

    for split in dataset_split:
        dataset_config = cfg.dataset.get(split, None)

        dataset = get_dataset(dataset_config)

        tokenizer = load_tokenizer(cfg.model.tokenizer)

        dataset = dataset.map(
            tokenize_function(tokenizer, data_process_configuration.max_tokens),
            desc=f"Tokenizing {split} ...",
            num_proc=data_process_configuration.num_proc,
            batched=True,
            batch_size=data_process_configuration.batch_size,
            remove_columns=dataset.column_names,
        )

        save_path = os.path.join(
            data_process_configuration.save_path,
            split,
        )

        dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()  # type: ignore
