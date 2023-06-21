#!/bin/bash
#SBATCH -p memory
#SBATCH --gpus=0
#SBATCH -N 1 -c 8 
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200056
#SBATCH -J split-test

ml Miniconda3
conda deactivate
conda activate /project/lt200056-opgpth/decontamination/.conda/env

python src/data/scripts/merge_jsonl/merge_jsonl.py /scratch/lt200056-opgpth/large_data/Oscar /scratch/lt200056-opgpth/large_data/Oscar/oscar_all.jsonl