## to test it with default argument
```torchrun --standalone --nproc_per_node=num_gpu src/model/scripts/lighting_training/train.py --model_name llama```
## to test it with custom argument
```torchrun --standalone --nproc_per_node=num_gpu src/model/scripts/lighting_training/train.py --model_name llama --optimizer lion```
argument that can custom
- accelerator: dp | ddp | ddp_spawn | xla | deepspeed | fsdp
- strategy: Union[str, Strategy] = "auto"
- devices: number of gpus
- precision: 32-true | 32 | 16-mixed | bf16-mixed | 64-true
- seed
- batch_size
- grad
- context_length
- model_name= llama | gptj
- optimizer= adamw | lion
- weight_decay
- vocab size
- lr
- xformers
- checkpoint
- checkpoint_only_attention