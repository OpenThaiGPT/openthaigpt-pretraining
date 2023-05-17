#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=8
#SBATCH -N 2 -c 64 #number of CPUs
#SBATCH --ntasks-per-node=1 #number of GPUs
#SBATCH -t 01:00:00
#SBATCH -A lt200056
#SBATCH -J runpy

srun /bin/bash train_srun.sh