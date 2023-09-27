#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH -t 00:60:00
#SBATCH -J <project_id>
#SBATCH -A 

module restore
module load Miniconda3
module load PrgEnv-gnu
module load cpe-cuda
module load cudatoolkit/22.7_11.7
module load craype-accel-nvidia80
module load aws-ofi-nccl

conda deactivate
conda activate <conda_prefix_path>

module list

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn 

# apptainer exec  --bind  /project/lt200056-opgpth/stanford_alpaca:/home/nbuppodo --writable pytorch_image_fixed pip install git+https://github.com/huggingface/accelerate

START=`date`
starttime=$(date +%s)


export WANDB_MODE="offline"

python zero_to_fp32.py \
    /project/lt200056-opgpth/boss/stanford_alpaca/output_deepspeed \
    /project/lt200056-opgpth/boss/stanford_alpaca/output_deepspeed/pytorch_model.bin \