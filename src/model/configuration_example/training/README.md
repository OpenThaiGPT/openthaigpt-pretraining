# Config

- accelerator: device (cuda , cpu)
- strategy: mode of parallel (dp | ddp | ddp_spawn | xla | deepspeed | fsdp)
- stage: stage of deepspeed
- offload_optimizer: true when want to offload optimizer
- offload_parameters: true when want to offload parameters
- num_gpus: number of gpu
- precision: data type (32-true | 32 | 16-mixed | bf16-mixed | 64-true) 
- num_nodes: number of node
- seed: seed of this training (13, 21, 42, 87, 100)
- batch_size: batch size of model
- grad: gradient accumulation steps
- max_tokens: max sequence lenght
- num_shards: number chunk of dataset
- num_workers: number of worker in dataloader