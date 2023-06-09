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
    optimizer_configuration,
    batch_size: int = 1,
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
):
    optimizer_name = optimizer_configuration.name
    if offload_optimizer or offload_parameters:
        if optimizer_name == ADAMW:
            opt = deepspeed.ops.adam.DeepSpeedCPUAdam(
                model.parameters(),
                bias_correction=True,
                amsgrad=False,
                adamw_mode=True,
                fp32_optimizer_states=True,
                **optimizer_name.hyps,
            )
        else:
            raise NotImplementedError("Optimizer does not support")
    else:
        if optimizer_name == LION:
            opt = Lion(
                model.parameters(),
                **optimizer_name.hyps,
            )
        elif optimizer_name == ADAMW:
            opt = optim.AdamW(  # type: ignore
                params=model.parameters(),
                **optimizer_configuration.hyps,
            )
        elif optimizer_name == ADAM8BIT:
            opt = Adam8bit(
                params=model.parameters(),
                **optimizer_name.hyps,
            )
        elif optimizer_name == ADAM1BIT:
            config_params = {
                "train_batch_size": batch_size,
                "optimizer": {
                    "type": "OneBitAdam",
                    "params": {
                        **optimizer_name.hyps,
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
