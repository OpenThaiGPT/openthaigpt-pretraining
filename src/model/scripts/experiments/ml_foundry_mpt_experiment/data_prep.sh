#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 64 #number of CPUs
#SBATCH --ntasks-per-node=1 #number of GPUs
#SBATCH -t 1:00:00
#SBATCH -A lt200056
#SBATCH -J data_prep

ml purge
ml Apptainer
apptainer run --nv --home /lustrefs/disk/home/cutupon/llm-foundry llmfoundry3.sif python3 scripts/data_prep/convert_dataset_json.py \
  --path oscar100rows.jsonl.zst \
  --out_root my-copy-oscar --split train \
  --concat_tokens 2048 --tokenizer xlm-roberta-base --eos_text '<|endoftext|>'