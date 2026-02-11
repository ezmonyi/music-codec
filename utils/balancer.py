"""
Loss balancer (EnCodec-style): combine losses with optional gradient rescaling.

Use to balance mel reconstruction and discriminator generator loss so that
discriminator scale fluctuations don't dominate. Flow / commitment / codebook
losses stay as a weighted sum and are not passed through the balancer.
"""

from collections import defaultdict
import typing as tp

import torch
from torch import autograd


def averager(beta: float = 1.0):
    """
    Exponential Moving Average callback for a dict of metrics.
    Returns a single function that can be called repeatedly to update the EMA.
    For beta=1 this is plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1.0) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}

    return _update


def average_metrics(metrics: tp.Dict[str, float], count: float = 1.0) -> tp.Dict[str, float]:
    """Optional DDP all-reduce; for single-process just return metrics (or divide by count)."""
    return {k: v / count for k, v in metrics.items()} if count != 1.0 else metrics


class Balancer:
    """Loss balancer: weight losses and optionally rescale gradients for stable multi-loss training.

    Expected usage:
        weights = {"mel_recon": 1.0, "disc_gen": 1.0}
        balancer = Balancer(weights, rescale_grads=True, ...)
        losses = {"mel_recon": mel_recon_loss, "disc_gen": g_loss}
        balancer.backward(losses, pred_mel)

    Args:
        weights: weight coefficient per loss key
        rescale_grads: if True, rescale each loss gradient by (ratio * total_norm / norm)
        total_norm: target norm when rescaling
        ema_decay: decay for EMA of gradient norms when rescale_grads
        per_batch_item: if True, compute norm per batch item then mean
        epsilon: numerical stability
        monitor: if True, store ratio per loss in .metrics
    """

    def __init__(
        self,
        weights: tp.Dict[str, float],
        rescale_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
        epsilon: float = 1e-12,
        monitor: bool = False,
    ):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        return self._metrics

    def backward(self, losses: tp.Dict[str, torch.Tensor], input_tensor: torch.Tensor):
        """Compute per-loss gradients w.r.t. input_tensor, combine (with optional rescaling), then backward."""
        norms = {}
        grads = {}
        for name, loss in losses.items():
            if not loss.requires_grad or loss.numel() != 1:
                continue
            grad, = autograd.grad(loss, [input_tensor], retain_graph=True)
            if self.per_batch_item and grad.dim() > 1:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean()
            else:
                norm = grad.norm()
            norms[name] = norm
            grads[name] = grad

        if not norms:
            return

        count = input_tensor.shape[0] if self.per_batch_item else 1.0
        avg_norms = average_metrics(self.averager(norms), count)
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor and total > 0:
            for k, v in avg_norms.items():
                self._metrics[f"ratio_{k}"] = v / total

        total_weights = sum(self.weights.get(k, 0.0) for k in avg_norms)
        if total_weights <= 0:
            return
        ratios = {k: self.weights.get(k, 0.0) / total_weights for k in avg_norms}

        out_grad = None
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                grad = grads[name] * scale
            else:
                grad = self.weights.get(name, 0.0) * grads[name]
            if out_grad is None:
                out_grad = grad
            else:
                out_grad = out_grad + grad

        if out_grad is not None:
            input_tensor.backward(out_grad)
