DATA_DIR=/workspace/data
MODEL_DIR=/workspace/model

NAME=002-001-dpo-temp-0_3-v-all-ref

DATA_PATH="$DATA_DIR/training/temp-0_3-v-all-ref.jsonl"
EPOCH=5
LR=2e-5 # e-5 for full finetune, e-4 for lora
GRADIENT_ACCUMULATION_STEPS=16
MAX_LEN=4096
MAX_PROMPT_LEN=2048
MICRO_BSZ=8
VAL_SIZE=0.1
BASE_MODEL="$MODEL_DIR/llama2-7b-finetune-hf"
WANDB_NAME="wandb_name"

WARMUP_STEPS=0


OUTPUT_PATH="$MODEL_DIR/dpo/$NAME"

python finetune_dpo_full.py \
    --data_path $DATA_PATH \
    --epoch $EPOCH \
    --eval_steps 2 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --max_len $MAX_LEN \
    --max_prompt_len $MAX_PROMPT_LEN \
    --micro_bsz $MICRO_BSZ \
    --model_path $BASE_MODEL \
    --output_path $OUTPUT_PATH \
    --save_steps 2 \
    --val_size $VAL_SIZE \
    --wandb_name $WANDB_NAME