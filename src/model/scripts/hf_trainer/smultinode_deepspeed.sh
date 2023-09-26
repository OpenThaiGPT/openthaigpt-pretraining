#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script



module restore
module load Miniconda3
module load PrgEnv-gnu
module load cpe-cuda
module load cudatoolkit/22.7_11.7
module load craype-accel-nvidia80
module load aws-ofi-nccl

conda deactivate
conda activate <conda_prefix_path>

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID



# source /fsx/dalle2/.dalle_env_38/bin/activate
# echo python3 version = `python3 --version`
# python -c "import torch"


accelerate launch --num_processes $(( 4 * $COUNT_NODE )) --num_machines $COUNT_NODE --multi_gpu --mixed_precision fp16 --machine_rank $SLURM_PROCID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT  train.py \
    --model_name_or_path /project/lt200056-opgpth/boss/llama-2-7b-hf \
    --tokenizer_name_or_path /project/lt200056-opgpth/llama_2_tokenizer_merge \
    --data_path /scratch/lt200056-opgpth/hf_v5_token_llama_2_256 \
    --train_split train \
    --eval_split eval \
    --bf16 True \
    --output_dir output_deepspeed \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
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

    # --fsdp "full_shard auto_wrap" \
    # --gradient_checkpointing True