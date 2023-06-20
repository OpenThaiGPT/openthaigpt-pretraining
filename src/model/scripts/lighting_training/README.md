## preprocess dataset if don't want to use streaming dataset
```python src/model/scripts/lighting_training/data_preprocessing.py --mode train --model path/to/tokenizer --max_tokens 2048 --save_path ./tokendata```
argument of preprocessing
- mode: train | val
- model
- max_tokens
- save_path
- chunk_size
- batch_size
- num_proc
- dataset_name
- dataset_dir
## to test it with default argument
```python src/model/scripts/lighting_training/train.py --model_name llama```
## to test it with custom argument
```torchrun --standalone --nproc_per_node=num_gpu src/model/scripts/lighting_training/train.py --model_name llama --optimizer lion```
argument that can custom
- accelerator: dp | ddp | ddp_spawn | xla | deepspeed | fsdp
- strategy: Union[str, Strategy] = "auto"
- devices: number of gpus
- precision: 32-true | 32 | 16-mixed | bf16-mixed | 64-true
- seed
- batch_size
- num_workers
- streaming
- dataset_name_or_path
- dataset_dir
- grad
- context_length
- model_name= llama | llama_hf | gptj
- optimizer= adamw | lion
- weight_decay
- vocab_size
- lr
- attention origin | pytorch | xformers (llama_hf only support origin)
- checkpoint
- checkpoint_only_attention