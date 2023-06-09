from datasets import load_dataset
from .constants import DATASET_ARGS
from typing import Union


def get_dataset(
    dataset_name: Union[str, dict],
    split: str = None,  # type: ignore
    shuffle: bool = False,
    buffer_size: int = 10_000,
    streaming: bool = False,
):
    """
    Args:
        dataset_name: dataset name in `DATASET_ARGS` or HF datasets
            `load_dataset` arguments in dictionary.
        split: `train` or `validation` default is `None`.
        shuffle: If `True`, it will be shuffle the dataset.
        buffer_size: Shuffle buffer size.
    """
    dataset_args = DATASET_ARGS.get(dataset_name, None)  # type: ignore
    if isinstance(dataset_name, dict):
        dataset_args = dataset_name
    elif isinstance(dataset_name, str) and not dataset_args:
        pass
    else:
        raise NotImplementedError(f"No dataset name {dataset_name}")
    dataset = load_dataset(
        **dataset_args,
        split=split,
        streaming=streaming,
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    return dataset
