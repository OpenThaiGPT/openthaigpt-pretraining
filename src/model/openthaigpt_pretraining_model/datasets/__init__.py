from datasets import load_dataset, load_from_disk
from .constants import DATASET_ARGS


def get_dataset(config):
    dataset_args = DATASET_ARGS.get(config.dataset_name, None)  # type: ignore
    if isinstance(config.dataset_name, dict):
        dataset_args = config.dataset_name
    elif isinstance(config.dataset_name, str) and not dataset_args:
        dataset_args = {"path": config.dataset_name}
    else:
        raise NotImplementedError(f"No dataset name {config.dataset_name}")

    if config.from_disk:
        dataset = load_from_disk(
            config.dataset_name,
        )
        if config.split is not None:
            dataset = dataset[config.split]
    else:
        dataset = load_dataset(
            **dataset_args,
            split=config.split,
            streaming=config.streaming,
        )

    if config.shuffle:
        dataset = dataset.shuffle(buffer_size=config.buffer_size)
    return dataset
