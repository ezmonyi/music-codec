"""Minimal LR schedulers: WarmupLR and ConstantLR."""

from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """Warmup then decay: lr = base_lr * warmup_steps^0.5 * min(step^-0.5, step * warmup_steps^-1.5)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
        **kwargs,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return [lr * step_num**-0.5 for lr in self.base_lrs]
        return [
            lr
            * self.warmup_steps**0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]


class ConstantLR(_LRScheduler):
    """Constant learning rate."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__(optimizer)

    def get_lr(self):
        return self.base_lrs
