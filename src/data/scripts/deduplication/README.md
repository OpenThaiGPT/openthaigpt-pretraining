# OpenThaiGPT Deuplication Pipeline

## Description

This code help run deduplication pipeline for Huggingface dataset in this format

## What this code does

1. Load Huggingface dataset from the path and tokenize all the text in `text` column with newmm tokenizer using `nlpo3``library and compute N-Gram MinHash similar to this [paper](https://arxiv.org/abs/2107.06499).
2. Store MinHash result in Huggingface dataset on disk
3. Load MinHash dataset and original dataset. Use store MinHash of all documents into LSH index.
4. Query each documents against the LSH index and keep the neighbors that has similarity score more than thresold.
5. For each document, calculate approximate jaccard distance with all of its neighbors. If the jaccard score is more than the thresold mark this document as duplicated with its neighbors.
6. Save all marked duplicated documents in Huggingface dataset format.
7. Perform removal of all marked duplicated documents from the original dataset.
8. Save the deduplicated document in the new Huggingface dataset on disk.

## Usage

**Prerequisites:** Install openthai-gpt-data depedencies because running this code [link](/src/data/README.md)

Conda

```
python ./src/data/scripts/deduplication/deduplicate.py
```

Apptainer

```
apptainer run -B /lustrefs/flash/scratch --home /project/lt200056-opgpth/openthaigpt-refactor image_sandbox python ./src/data/scripts/deduplication/deduplicate.py
```

Note:

- We run it on ThaiSC Lanta's scratch disk to improve the I/O performance
- We tested it with Apptainer, but conda python should also work
- We run it on Memory node of Lanta but Compute node should also work without OOM
- Command `export HF_DATASETS_CACHE="/project/lt200056-opgpth/openthaigpt-refactor/.cache"` is needed to prevent Huggingface storing cache in home directory and empty disk storage quota.

## I/O

`Huggingface input dataset format`

```json
{
    "train": ["text", ...], // column names
    "validate": ["text", ...] // column names
    ...
}
```

`config/deduplicaiton.yaml`

```
More info on the file itself..
```

## Default Parameters

- DEFAULT_NUM_PERMUTATION = 128
- N_GRAM = 128
- deduplication thresold = 0.9
