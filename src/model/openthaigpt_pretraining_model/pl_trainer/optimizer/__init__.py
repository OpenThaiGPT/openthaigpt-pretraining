from typing import Iterable
from .constants import OPTIMIZERS, SCHEDULERS


def get_optimizer(parameters: Iterable, optimizer_config):
    """
    Description: Get pytorch optimizers
    Args:
        parameters: Model parameters
        optimizer_config: Optimizer config including
            `name` and `hyps`
    Returns:
        optimizer: Pytorch optimizer
    """
    optimizer_module = OPTIMIZERS.get(optimizer_config.name, None)
    if optimizer_module is None:
        raise ValueError(f"No optimizer name `{optimizer_config.name}`")
    return optimizer_module(parameters, **optimizer_config.hyps)


def get_scheduler(
    optimizer,
    base_lr: float,
    total_steps: int,
    scheduler_config,
):
    """
    Description: Get pytorch scheduler
    Args:
        optimizer: Pytorch optimizer
        base_lr: Base learning rate
        total_steps: Total training steps
        scheduler_config: Scheduler config including
            `name` and `hyps`
    Returns:
        scheduler: Pytorch scheduler
    """
    scheduler_module = SCHEDULERS.get(scheduler_config.name, None)
    if scheduler_module is None:
        raise ValueError(f"No scheduler name `{scheduler_config.name}`")
    return scheduler_module(
        optimizer,
        base_lr=base_lr,
        total_steps=total_steps,
        **scheduler_config.hyps,
    )
