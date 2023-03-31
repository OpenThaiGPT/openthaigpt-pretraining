# Optimizer Experimet

We should experiment which optimizer is best when training GPT LLM


## Things to consider
1. Memory Usage of the Optimizer
2. Model convergence speed 
3. Stability of the loss
4. **Scalibility when increasing model parameters count**

### Scalibility
We should measure how 1,2,3 are impacted by increasing model parameters count. If increasing model paramters diminish the saving by 1,2,3 then our proposed method should not be used in large scale pretraning.

## Optimizers
- AdamW
- (Lion)[https://twitter.com/ArYoMo/status/1633949392934772738]
- 


## Model
- Model: [gpt2](https://huggingface.co/gpt2)
    - [gpt2-base](https://huggingface.co/gpt2)
    - [gpt2-medium](https://huggingface.co/gpt2-medium)
    - [gpt2-large](https://huggingface.co/gpt2-large)
    - [gpt2-xl](https://huggingface.co/gpt2-xl)
- Tokenizer: [gpt2-base-thai](https://huggingface.co/flax-community/gpt2-base-thai)


## Datasets:
- [mC4, Thai subset](https://huggingface.co/datasets/mc4)

## Experiment steps
1. Intilize GPT2 model weight from scratch, and resize token embedding to align with gpt2-base-thai tokenizer.
2. Train from scratch with 4 * A100 GPUs for 1 hours with AdamW optimizer.
3. Change optimizer and do 1,2 again
4. Measure all loss on WanDB
5. Repeat 1-4 again but change to other size of GPT2

## Depedencies
- Start with Kaggle CPU only
- Always use Pytorch 2.0 compile and Flash Attention [[1]](https://www.reddit.com/r/MachineLearning/comments/11tmpc5/d_pytorch_20_native_flash_attention_32k_context/) [[2]](https://gist.github.com/NaxAlpha/1c36eaddd03ed102d24372493264694c). Make sure not to patch model embedding like the tutorial do. However we need to resize token embedding to align with gpt2-base-thai instead.
- When testing experiment, switch accelarator to `T4 x 2` to test multiGPU training
- Export enviroment.yml with command `conda env export > environment.yml`


## Concerns
1. We are not sure which optimiers's hyperameters will affect convergence speed. Please refer to the well-experimented hyperparamters for now.

## Script
```
python ....
```
