from .constants import OPTIMIZERS, SCHEDULERS


def get_optimizer(parameters, optimizer_config):
    optimizer_module = OPTIMIZERS.get(optimizer_config.name, None)
    if optimizer_module is None:
        raise ValueError(f"No optimizer name `{optimizer_config.name}`")
    return optimizer_module(parameters, **optimizer_config.hyps)


def get_scheduler(optimizer, base_lr, total_steps, scheduler_config):
    scheduler_module = SCHEDULERS.get(scheduler_config.name, None)
    if scheduler_module is None:
        raise ValueError(f"No scheduler name `{scheduler_config.name}`")
    return scheduler_module(
        optimizer,
        base_lr=base_lr,
        total_steps=total_steps,
        **scheduler_config.hyps,
    )
