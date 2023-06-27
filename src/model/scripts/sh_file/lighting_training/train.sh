#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH -N 1 -c 64 #number of CPUs
#SBATCH --ntasks-per-node=1 #number of GPUs
#SBATCH -t 48:00:00
#SBATCH -A # your project id
#SBATCH -J train

ml Miniconda3

conda deactivate
conda activate # your environment

srun python src/model/scripts/lighting_training/train.py