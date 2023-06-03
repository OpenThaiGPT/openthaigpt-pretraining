from datasets import load_dataset
from .constants import C4_DATASET, MC4_DATASET, SPLIT_TRAIN, SPLIT_VAL


def get_datasets(dataset_name: str, buffer_size: int = 10_000):
    if dataset_name == C4_DATASET:
        train_dataset = load_dataset(
            dataset_name,
            "en",
            streaming=True,
            split=SPLIT_TRAIN,
        ).shuffle(buffer_size=buffer_size)
        val_dataset = load_dataset(
            dataset_name,
            "en",
            streaming=True,
            split=SPLIT_VAL,
        )
    elif dataset_name == MC4_DATASET:
        train_dataset = load_dataset(
            dataset_name,
            languages=["th"],
            streaming=True,
            split=SPLIT_TRAIN,
        ).shuffle(buffer_size=buffer_size)
        val_dataset = load_dataset(
            dataset_name,
            languages=["th"],
            streaming=True,
            split=SPLIT_VAL,
        )
    else:
        raise NotImplementedError()
    return train_dataset, val_dataset
