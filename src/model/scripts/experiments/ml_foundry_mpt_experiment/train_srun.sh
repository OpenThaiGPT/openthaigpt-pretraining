ml purge
ml Apptainer

CACHE_DIR="/scratch/lt200056-opgpth/cache_$((SLURM_JOB_ID))_$((SLURM_NODEID))"
mkdir -p $CACHE_DIR
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

apptainer run --nv --home /lustrefs/disk/home/USERNAME/llm-foundry --bind /scratch/lt200056-opgpth/cache_$((SLURM_JOB_ID))_$((SLURM_NODEID)):/cache_$((SLURM_JOB_ID))_$((SLURM_NODEID)) llmfoundry2.sif \
  composer -n 4 --world_size 16 --node_rank $((SLURM_NODEID)) --base_rank  $((SLURM_NODEID * 4)) --master_addr $master_addr --master_port 7501 \
  scripts/train/train.py \
  scripts/train/yamls/mpt/125m.yaml \
  data_local=/cache_$((SLURM_JOB_ID))_$((SLURM_NODEID)) \
  data_remote=./my-copy-oscar \
  train_loader.dataset.split=train \
  eval_loader.dataset.split=eval \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=mpt-125m \
  precision=amp_fp16 \
  global_train_batch_size=16 \
  model.attn_config.attn_impl=torch \
  tokenizer.name=xlm-roberta-base