# Implement from https://github.com/mlfoundations/open_clip/blob/main/src/training/scheduler.py
import numpy as np
import torch
import math

INITIAL_LR = "initial_lr"


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


class ConstantLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer,
        base_lr,
        warmup_steps,
        total_steps,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        init_lr = self.calculate_new_lr(0)
        for param_group in optimizer.param_groups:
            param_group[INITIAL_LR] = init_lr

        super(ConstantLRScheduler, self).__init__(optimizer, init_lr)

    def calculate_new_lr(self, step):
        if step < self.warmup_steps:
            lr = _warmup_lr(self.base_lr, self.warmup_steps, step)
        else:
            lr = self.base_lr
        return lr

    def get_lr(self):
        new_lr = self.calculate_new_lr(self.optimizer._step_count)
        return [new_lr for group in self.optimizer.param_groups]


class CosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer,
        base_lr,
        warmup_steps,
        total_steps,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        init_lr = self.calculate_new_lr(0)
        for param_group in optimizer.param_groups:
            param_group[INITIAL_LR] = init_lr

        super(CosineLRScheduler, self).__init__(optimizer, init_lr)

    def calculate_new_lr(self, step):
        if step < self.warmup_steps:
            lr = _warmup_lr(self.base_lr, self.warmup_steps, step)
        else:
            e = step - self.warmup_steps
            es = self.total_steps - self.warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
        return lr

    def get_lr(self):
        new_lr = self.calculate_new_lr(self.optimizer._step_count)
        return [new_lr for group in self.optimizer.param_groups]


class WarmupAndDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer,
        base_lr,
        min_lr,
        warmup_steps,
        decay_steps,
        total_steps,
    ) -> None:
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.total_steps = total_steps

        init_lr = self.calculate_new_lr(0)
        for param_group in optimizer.param_groups:
            param_group[INITIAL_LR] = init_lr

        super(WarmupAndDecayScheduler, self).__init__(optimizer, init_lr)

    def calculate_new_lr(self, step):
        # 1) linear warmup for warmup_steps
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        # 2) if step > decay_steps, return min learning rate
        if step > self.decay_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (
            self.decay_steps - self.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.base_lr - self.min_lr)

    def get_lr(self):
        new_lr = self.calculate_new_lr(self.optimizer._step_count)
        return [new_lr for group in self.optimizer.param_groups]
