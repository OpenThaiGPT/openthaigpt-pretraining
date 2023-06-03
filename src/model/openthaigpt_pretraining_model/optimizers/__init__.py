import deepspeed
from lion_pytorch import Lion
import torch.optim as optim
from bitsandbytes.optim import Adam8bit
from .constants import (
    ADAMW,
    LION,
    ADAM1BIT,
    ADAM8BIT,
)


def get_optimizer(
    model,
    optimizer: str = ADAMW,
    weight_decay: float = 1e-2,
    lr: float = 1e-4,
    batch_size: int = 1,
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
):
    if offload_optimizer or offload_parameters:
        if optimizer == ADAMW:
            opt = deepspeed.ops.adam.DeepSpeedCPUAdam(
                model.parameters(),
                lr=lr,
                bias_correction=True,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                amsgrad=False,
                adamw_mode=True,
                fp32_optimizer_states=True,
            )
        else:
            raise NotImplementedError("Optimizer does not support")
    else:
        if optimizer == LION:
            opt = Lion(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer == ADAMW:
            opt = optim.AdamW(  # type: ignore
                params=model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                fused=True,
            )
        elif optimizer == ADAM8BIT:
            opt = Adam8bit(
                params=model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
        elif optimizer == ADAM1BIT:
            config_params = {
                "train_batch_size": batch_size,
                "optimizer": {
                    "type": "OneBitAdam",
                    "params": {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "betas": (0.9, 0.95),
                    },
                },
            }
            model, opt, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config_params=config_params,
            )
        else:
            raise NotImplementedError("Optimizer does not support")
    return model, opt
