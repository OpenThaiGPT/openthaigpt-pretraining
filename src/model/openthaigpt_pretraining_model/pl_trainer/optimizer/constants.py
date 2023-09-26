import torch.optim as optim
from .scheduler import (
    ConstantLRScheduler,
    CosineLRScheduler,
    WarmupAndDecayScheduler,
)

ADAM_OPTIMIZER = "adam"
ADAMW_OPTIMIZER = "adamw"
CONSTANT_SCHEDULER = "const_lr"
COSINE_SCHEDULER = "cosine_lr"
WARMUP_AND_DECAY_SCHEDULER = "warmup_and_decay_lr"


OPTIMIZERS = {
    ADAM_OPTIMIZER: optim.Adam,
    ADAMW_OPTIMIZER: optim.AdamW,
}

SCHEDULERS = {
    CONSTANT_SCHEDULER: ConstantLRScheduler,
    COSINE_SCHEDULER: CosineLRScheduler,
    WARMUP_AND_DECAY_SCHEDULER: WarmupAndDecayScheduler,
}
