# OpenThaiGPT Dataset Pipeline Overview

Folder `openthaigpt_pretraining_data` contains the preprocess function, EDA notebook, and examples for each dataset

Folder `notebook` contains the EDA or other experiment notebook.

Folder `scripts` contains the main program which will call function in folder `openthaigpt_pretraining_data`

## Description

The OpenThaiGPT Overview dataset pipeline is meticulously designed to fully preprocess raw data, extract essential information, and prepare the dataset in a trainable format. Here's an extended breakdown of its key components:

## Workflow

1. Obtain Raw Data in various formats such as CSV, PDF, etc., and convert it into plain text format.
2. Transform plain text data from each raw data source into the JSONL format.
3. Consolidate all JSONL datasets into a single file. This pipeline can be found at: `src/data/scripts/merge_jsonl/` to merge JSONL file
4. Partition the JSONL dataset into training, validation, and test datasets. This pipeline can be found at: `src/data/scripts/split_data` to split the data into train/test/eval
5. Anonymize Sensitive Data, which may include personal and residential information, in compliance with PDPA regulations.
6. Remove duplicate entries from Huggingface datasets through a deduplication process utilizing the N-Gram MinHash approach combined with Locality-Sensitive Hashing. This pipeline can be found at: `src/data/scripts/deduplication` to remove duplication entries.
7. Implement Decontamination measures to prevent data leakage from the training dataset into the evaluation dataset, employing N-Gram MinHash and LSH techniques. This pipeline can be found at `src/data/scripts/decontamination`

