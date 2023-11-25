# ml foundry mpt experiment

## Prerequisites
Here's what you need to get started with our LLM stack:
* Use a Docker image with PyTorch 1.13+, e.g. [MosaicML's PyTorch base image](https://hub.docker.com/r/mosaicml/pytorch/tags)
   * Recommended tag: `mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04`
   * This image comes pre-configured with the following dependencies:
      * PyTorch Version: 1.13.1
      * CUDA Version: 11.7
      * Python Version: 3.10
      * Ubuntu Version: 20.04
      * FlashAttention kernels from [HazyResearch](https://github.com/HazyResearch/flash-attention)
* Use a system with NVIDIA GPUs

## Installation

To get started, clone this repo and install the requirements:

<!--pytest.mark.skip-->
```bash
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry

# Optional: we highly recommend creating and using a virtual environment
python -m venv llmfoundry-venv
source llmfoundry-venv/bin/activate

pip install -e ".[gpu]"  # or pip install -e . if no NVIDIA GPU
```

## Quickstart

Here is an end-to-end workflow for preparing a subset of the C4 dataset, training an MPT-125M model for 10 batches,
converting the model to HuggingFace format, evaluating the model on the Winograd challenge, and generating responses to prompts.

```bash
cd scripts

# Convert C4 dataset to StreamingDataset format
python data_prep/convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'

# Train an MPT-125m model for 10 batches
composer train/train.py \
  train/yamls/mpt/125m.yaml \
  data_local=my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=mpt-125m \
  precision=amp_fp16 \
  global_train_batch_size=1 \
  model.attn_config.attn_impl=torch \
  tokenizer.name=EleutherAI/gpt-neox-20b
``` 

note --out_root and save_folder must empty

## Dataset from json

```bash
# Convert C4 dataset to StreamingDataset format
python data_prep/convert_dataset_json.py \
  --path data_prep/example_data/arxiv.jsonl \
  --out_root my-copy-arxiv --split train \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
```

Where `--path` can be a single json file, or a folder containing json files, and split the intended split (hf defaults to train).

## MultiGPUs

```bash
composer --world_size 4 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 train/train.py \
  train/yamls/mpt/125m.yaml \
  data_local=my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=mpt-125m \
  precision=amp_fp16 \
  global_train_batch_size=1 \
  model.attn_config.attn_impl=torch \
  tokenizer.name=EleutherAI/gpt-neox-20b
```

# In Lanta

## Installation

clone tokenizer in this example use xlm-roberta-base

```bash
  git clone xlm-roberta-base
```

make container from [openthaigpt/openthaigpt-pretraining:llmfoundry](https://hub.docker.com/layers/openthaigpt/openthaigpt-pretraining/llmfoundry/images/sha256-3d56b25d90f75977cc84e34ef12a12f3a5c557c941644e8a0edb9b07bca2fb95?context=repo) 


```bash
  ml Apptainer
  apptrainer build llmfoundry.sif docker:openthaigpt/openthaigpt-pretraining:llmfoundry
```

## Convert dataset to StreamingDataset format

make data_prep.sh file for run by sbatch

```data_prep.sh
  #!/bin/bash
  #SBATCH -p gpu              # Specify partition [Compute/Memory/GPU]
  #SBATCH -N 1 -c 64          # Specify number of nodes and processors per task
  #SBATCH --ntasks-per-node=1 # Specify tasks per node
  #SBATCH -t 1:00:00          # Specify maximum time limit (hour: minute: second)
  #SBATCH -A lt200056         # Specify project name
  #SBATCH -J data_prep        # Specify job name

  ml purge
  ml Apptainer
  apptainer run --nv --home /home/USERNAME/llm-foundry llmfoundry.sif \
    python3 scripts/data_prep/convert_dataset_json.py \
    --path scripts/data_prep/example_data/arxiv.jsonl \
    --out_root my-copy-arxiv --split train \
    --concat_tokens 2048 --tokenizer xlm-roberta-base --eos_text '<|endoftext|>'
```

note you will change USERNAME to your username

run data_prep file

```bash
  sbatch data_prep.sh
```

## Training for multi node

make train_srun.sh file for run by srun

```train_srun.sh
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
```

note you will change USERNAME to your username

make train.sh file for run by sbatch

```train.sh
  #!/bin/bash
  #SBATCH -p gpu              # Specify partition [Compute/Memory/GPU]
  #SBATCH --gpus=32            # Specify number of GPUs
  #SBATCH -N 8 -c 64          # Specify number of nodes and processors per task
  #SBATCH --ntasks-per-node=1 # Specify tasks per node
  #SBATCH -t 1:00:00          # Specify maximum time limit (hour: minute: second)
  #SBATCH -A lt200056         # Specify project name
  #SBATCH -J train            # Specify job name

  ml purge
  ml Apptainer

  srun /bin/bash train_srun.sh
```

note when you change --gpus you will change --world_size and global_train_batch_size

run train file

```bash
  sbatch train.sh
```