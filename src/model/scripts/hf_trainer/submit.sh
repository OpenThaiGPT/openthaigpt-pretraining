#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		# Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 01:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A <project_id>                       # Specify project name
#SBATCH -J DDP-testNCCL                          # Specify job name

# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_PORT="$MASTER_PORT

# export WORLD_SIZE=8   # should be obtained from $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

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
#export FI_MR_CACHE_MONITOR=memhooks
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_NET=IB
#export NCCL_IB_HCA=mlx5
#export CXI_FORK_SAFE=1 
#export CXI_FORK_SAFE_HP=1 
#export FI_CXI_DISABLE_CQ_HUGETLB=1

#echo "using FI_MR_CACHE_MONITOR=memhooks"

START=`date`
starttime=$(date +%s)

export WANDB_MODE="offline"

torchrun --nproc_per_node=4 --master_port=3000 train.py \
    --model_name_or_path /project/lt200056-opgpth/boss/stanford_alpaca/llama_2_7b_fixed_resized \
    --tokenizer_name_or_path /project/lt200056-opgpth/boss/stanford_alpaca/llama_2_7b_fixed_resized \
    --data_path /scratch/lt200056-opgpth/hf_v5_token_llama_2_256 \
    --train_split train \
    --eval_split eval \
    --bf16 True \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \


END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END