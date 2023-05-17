#!/bin/bash
#SBATCH -p gpu              # Specify partition [Compute/Memory/GPU]
#SBATCH --gpus=8            # Specify number of GPUs
#SBATCH -N 2 -c 64          # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1 # Specify tasks per node
#SBATCH -t 1:00:00          # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200056         # Specify project name
#SBATCH -J train            # Specify job name

srun /bin/bash train_srun.sh