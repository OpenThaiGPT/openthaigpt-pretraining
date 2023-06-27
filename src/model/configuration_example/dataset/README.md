# Config

## tokenized:
- path: path to dataset after preprocessed if null will use raw data
- train_split: split train of dataset
- eval_split: split validation of dataset

## train:
- dataset_name: name or path to dataset
- split: split train of dataset
- shuffle: if true dataset will shuffle
- buffer_size: buffer for shuffle
- streaming: load as streaming when true
- from_disk: use true when want to load from local

## eval:
- dataset_name: name or path to dataset
- split: split validation of dataset
- shuffle: if true dataset will shuffle
- buffer_size: buffer for shuffle
- streaming: load as streaming when true
- from_disk: use true when want to load from local