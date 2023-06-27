#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=0
#SBATCH -N 1 -c 64 #number of CPUs
#SBATCH --ntasks-per-node=1 #number of GPUs
#SBATCH -t 12:00:00
#SBATCH -A # your project id
#SBATCH -J data_process

ml Miniconda3

conda deactivate
conda activate # your environment

python src/model/scripts/lighting_training/data_preprocessing.py