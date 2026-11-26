"""Device, AMP and validation lifecycle for window enhancement."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
import torch
from torch import nn
from src.utils.training import resolve_device
from .metrics import BatchMetrics, MetricAccumulator

class BaseTrainer:
    """Common window training state."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str | torch.device | None,
        use_amp: bool,
        gradient_clip_norm: float,
    ) -> None:
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.gradient_clip_norm = gradient_clip_norm
        self.amp_enabled = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)

    @torch.no_grad()
    def validate(self, batches: Iterable[Mapping[str, Any]]) -> BatchMetrics:
        self.model.eval()
        accumulator = MetricAccumulator()
        for batch in batches:
            lq, target = self._prepare_batch(batch)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                enhanced = self.model(lq)
                losses = self.criterion(enhanced, target)
            accumulator.update(self._metrics(lq, target, enhanced, losses))
        return accumulator.compute()

