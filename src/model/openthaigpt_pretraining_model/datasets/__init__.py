from datasets import load_dataset
from .constants import DATASET_ARGS, SPLIT_TRAIN, SPLIT_VAL


def get_datasets(dataset_name: str, buffer_size: int = 10_000):
    train_dataset = load_dataset(
        **DATASET_ARGS[dataset_name], split=SPLIT_TRAIN
    ).shuffle(buffer_size=buffer_size)
    val_dataset = load_dataset(
        **DATASET_ARGS[dataset_name],
        split=SPLIT_VAL,
    )
    return train_dataset, val_dataset
