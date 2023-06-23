#!/bin/bash
#SBATCH -p compute
#SBATCH --gpus=0
#SBATCH -N 1 -c 32 
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200056
#SBATCH -J split-test

ml Miniconda3
conda deactivate
conda activate /project/lt200056-opgpth/decontamination/.conda/env

python src/data/scripts/split_data/split_jsonl.py /scratch/lt200056-opgpth/large_data/mC4/mc4_th_new.jsonl --test_size 0.001 --validation_size 0.001
python src/data/scripts/split_data/split_jsonl.py /scratch/lt200056-opgpth/large_data/CC100/cc100_th_new.jsonl --test_size 0.001 --validation_size 0.001
