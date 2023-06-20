#!/bin/bash
#SBATCH -p xxx
#SBATCH --gpus=0
#SBATCH -N 1 -c 128 
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A xxx
#SBATCH -J split-test

ml Miniconda3
conda deactivate
conda activate xxx

python split_jsonl.py /path/to/file.jsonl --test_size 0.01 --validation_size 0.001
