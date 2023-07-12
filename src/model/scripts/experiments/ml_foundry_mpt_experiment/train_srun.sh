master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

apptainer run --nv --home /home/USERNAME/llm-foundry/ llmfoundry.sif \
  composer --world_size 32 --node_rank $((SLURM_NODEID)) --master_addr $master_addr --master_port 7501 \
  scripts/train/train.py \
  scripts/train/yamls/mpt/125m.yaml \
  data_local=./my-copy-oscar \
  train_loader.dataset.split=train \
  train_loader.dataset.shuffle_seed=$((SLURM_NODEID)) \
  eval_loader.dataset.split=eval \
  eval_loader.dataset.shuffle_seed=$((SLURM_NODEID)) \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=mpt-125m \
  precision=amp_fp16 \
  global_train_batch_size=32 \
  model.attn_config.attn_impl=torch \
  tokenizer.name=xlm-roberta-base