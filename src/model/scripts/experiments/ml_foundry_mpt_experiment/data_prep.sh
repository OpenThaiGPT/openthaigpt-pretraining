#!/bin/bash
#SBATCH -p gpu              # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64          # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1 # Specify tasks per node
#SBATCH -t 1:00:00          # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200056         # Specify project name
#SBATCH -J data_prep        # Specify job name

ml purge
ml Apptainer
apptainer run --nv --home /lustrefs/disk/home/USERNAME/llm-foundry llmfoundry.sif python3 scripts/data_prep/convert_dataset_json.py \
  --path oscar100rows.jsonl.zst \
  --out_root my-copy-oscar --split train \
  --concat_tokens 2048 --tokenizer xlm-roberta-base --eos_text '<|endoftext|>'