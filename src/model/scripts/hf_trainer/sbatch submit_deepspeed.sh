#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH -t 00:60:00
#SBATCH -J test
#SBATCH -A <project_id>


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

torchrun --nproc_per_node=4 --master_port=3000 train.py \
    --model_name_or_path /project/lt200056-opgpth/boss/llama-2-7b-hf \
    --tokenizer_name_or_path /project/lt200056-opgpth/llama_2_tokenizer_merge \
    --data_path /scratch/lt200056-opgpth/hf_v5_token_llama_2_256 \
    --train_split train \
    --eval_split eval \
    --bf16 True \
    --output_dir output_deepspeed \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/llama_deepspeed.json" \
    --tf32 True \
    --gradient_checkpointing True

    # --deepspeed "./configs/default_offload_opt_param.json" \

END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END