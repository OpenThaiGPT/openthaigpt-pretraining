#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH -N 1 -c 16 #number of CPUs
#SBATCH --ntasks-per-node=4 #number of GPUs
#SBATCH -t 120:00:00
#SBATCH -A lt200056
#SBATCH -J causal_lm_pretraining
#SBATCH --signal=SIGUSR1@90

ml Miniconda3
conda deactivate
conda activate <env-name>

srun python pretraining.py
