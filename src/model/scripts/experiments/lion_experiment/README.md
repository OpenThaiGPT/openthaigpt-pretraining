# Optimizer Experimet

We should experiment which optimizer is best when training Thai GPT LLM

## Things to consider

1. Memory Usage of the Optimizer
2. Model convergence speed
3. Stability of the loss
4. **Scalibility when increasing model parameters count**

## Memory Experiment (#1)

We want to measure how optimizer choices impact hardware utilization of casual language model pretraining

### Experiment Models

- [gpt2-xl](https://huggingface.co/gpt2-xl)
- [cerebras/Cerebras-GPT-2.7B](https://huggingface.co/cerebras/Cerebras-GPT-2.7B)

### Optimizer Choices

- AdamW
- [Lion](https://twitter.com/ArYoMo/status/1633949392934772738)
- [8-bit Adam](https://www.kaggle.com/code/nbroad/8-bit-adam-optimization) (WIP)

### Results

#### GPT2-XL (1.5B)

| Model   | Batch Size | Sequence Length | Checkpoint | Flash Attention | Optimizer | VRAM used   | Iteration Speed |
| ------- | ---------- | --------------- | ---------- | --------------- | --------- | ----------- | --------------- |
| GPT2-XL | 4          | 256             |            |                 | AdamW     | 37.9 GB     | 4.99it/s        |
| GPT2-XL | 4          | 256             | &#10004;   |                 | AdamW     | 30.9 GB     | 3.76it/s        |
| GPT2-XL | 4          | 256             |            | &#10004;        | AdamW     | 36.6 GB     | 5.80it/s        |
| GPT2-XL | 4          | 256             |            |                 | Lion      | 33.3 GB     | 5.16it/s        |
| GPT2-XL | 4          | 256             | &#10004;   | &#10004;        | AdamW     | 28.8 GB     | 4.40it/s        |
| GPT2-XL | 4          | 256             | &#10004;   | &#10004;        | Lion      | **24.3 GB** | 4.57it/s        |

In this table, we have different configurations of the GPT2-XL model with varying options such as checkpoint, flash attention, and lion, along with their corresponding batch sizes, sequence lengths, and model to measure total sizes and iteration speeds.

#### cerebras/Cerebras-GPT-2.7B

| Model                      | Flash Attention | Activation Checkpointing | Optimizer | Max Batch Size |
| -------------------------- | --------------- | ------------------------ | --------- | -------------- |
| cerebras/Cerebras-GPT-2.7B | &#10004;        | &#10004;                 | AdamW     | OOM            |
| cerebras/Cerebras-GPT-2.7B | &#10004;        | &#10004;                 | Lion      | 32             |
| cerebras/Cerebras-GPT-2.7B | &#10004;        |                          | Lion      | 8              |
| cerebras/Cerebras-GPT-2.7B |                 | &#10004;                 | Lion      | 8              |
| cerebras/Cerebras-GPT-2.7B |                 |                          | Lion      | 2              |

In this table, we have different configurations of the cerebras/Cerebras-GPT-2.7B model with varying options such as flash attention, activation checkpointing, and optimizer (Lion or AdamW) to measure max batch size.

### Hardware

We run all experiments on Huaweii Cloud Elastic Cloud Server p3s.4xlarge.8 (A100 40GB \* 1, 16vCPUs, 128GB Ram) for 1.30 minutes each. All experimens are run in bf16 datatype.

## Convergence Experiment (#2) (WIP)

We want to measure how optimizer choices impact convergence when pretraining LLM

### Experiment Models

Models:

- [gpt2-xl](https://huggingface.co/gpt2-xl)
- [cerebras/Cerebras-GPT-2.7B](https://huggingface.co/cerebras/Cerebras-GPT-2.7B)

Tokenizers:

- Tokenizer: [gpt2-base-thai](https://huggingface.co/flax-community/gpt2-base-thai) (TBD)

### Optimizer Choices

- AdamW
- [Lion](https://twitter.com/ArYoMo/status/1633949392934772738)
- [8-bit Adam](https://www.kaggle.com/code/nbroad/8-bit-adam-optimization) (WIP)

## Datasets:

- [mC4, Thai subset](https://huggingface.co/datasets/mc4)

## Experiment steps

1. Intilize GPT2 model weight from scratch, and resize token embedding to align with selected tokenizer.
2. Train from scratch with 4 \* A100 GPUs for 1 hours with selected optimizer (TBD).
3. Change optimizer and do 1,2 again
4. Repeat 1-3 again but change the optimizer
5. Analyze perplexity of the optimizer choice

## Concerns

1. We are not sure which optimiers's hyperameters will affect convergence speed. Please refer to the well-experimented hyperparamters for now.
