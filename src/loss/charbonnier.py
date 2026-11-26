"""Pure pixel Charbonnier loss; this module has no feature backbone."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossComponents:
    """Loss values consumed by the existing enhancement trainer."""

    total: Tensor
    charbonnier: Tensor
    feature: Tensor


class MFQEv2CombinedLoss(nn.Module):
    """Compute Charbonnier in FP32, including inside an AMP context."""

    def __init__(self, charbonnier_epsilon: float = 1e-6) -> None:
        super().__init__()
        if charbonnier_epsilon <= 0:
            raise ValueError("Charbonnier epsilon must be positive")
        self.epsilon = charbonnier_epsilon

    def forward(self, prediction: Tensor, target: Tensor) -> LossComponents:
        """Return pixel loss and a zero feature scalar for metric compatibility."""
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")
        difference = prediction.float() - target.float()
        loss = (difference.square() + self.epsilon ** 2).sqrt().mean()
        return LossComponents(loss, loss, loss.new_zeros(()))
