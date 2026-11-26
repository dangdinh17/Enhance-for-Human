"""Training operations for a single blind model over mixed-QP windows."""

from __future__ import annotations

from typing import Any, Mapping
import warnings

import torch
from torch import Tensor, nn

from .base import BaseTrainer
from .metrics import BatchMetrics, batch_psnr


class WindowTrainer(BaseTrainer):
    """Reuse AMP/optimization while preserving the explicit temporal dimension."""

    def train_batch(self, batch: Mapping[str, Any]) -> BatchMetrics:
        """Retry AMP overflow on the same batch, counting only real updates.

        GradScaler must handle non-finite scaled gradients before clipping.
        Persistent overflow and any non-finite loss/FP32 gradient remain fatal.
        """
        self.model.train()
        lq, target = self._prepare_batch(batch)
        self.last_amp_retries = 0
        for attempt in range(9):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                enhanced = self.model(lq)
                losses = self.criterion(enhanced, target)
            if not bool(torch.isfinite(losses.total)):
                raise FloatingPointError("Non-finite New QE training loss")
            self.scaler.scale(losses.total).backward()
            self.scaler.unscale_(self.optimizer)
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if not parameters:
                raise RuntimeError("New QE backward produced no parameter gradients")
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach(), 2) for p in parameters
            ]), 2)
            if bool(torch.isfinite(norm)):
                nn.utils.clip_grad_norm_(
                    parameters, self.gradient_clip_norm, error_if_nonfinite=True,
                )
                self.last_gradient_norm = float(norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.last_amp_retries = attempt
                return self._metrics(lq, target, enhanced, losses)
            bad = [name for name, p in self.model.named_parameters()
                   if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            if not self.amp_enabled or not bad or attempt == 8:
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite New QE gradients after {attempt} AMP retries; "
                    f"scale={self.scaler.get_scale()}, parameters={bad[:12]}"
                )
            old_scale = self.scaler.get_scale()
            # unscale_ recorded non-finite gradients: step skips Adam entirely,
            # and update backs off the scale and resets its growth tracker.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            self.optimizer.zero_grad(set_to_none=True)
            if not new_scale < old_scale:
                raise RuntimeError("AMP overflow did not reduce the loss scale")
            warnings.warn(
                f"AMP overflow: retrying same batch {attempt + 1}/8; "
                f"scale {old_scale:g} -> {new_scale:g}; parameters={bad[:12]}",
                RuntimeWarning, stacklevel=2,
            )
        raise AssertionError("Unreachable AMP retry state")

    def _prepare_batch(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        lq, target = batch["lq"], batch["target"]
        if not isinstance(lq, Tensor) or not isinstance(target, Tensor):
            raise TypeError("Window lq/target must be tensors")
        if lq.ndim != 5 or lq.shape[1:3] != (7, 1):
            raise ValueError(f"Expected [B,7,1,H,W], got {tuple(lq.shape)}")
        return (lq.to(self.device, non_blocking=True),
                target.to(self.device, non_blocking=True))

    @staticmethod
    def _metrics(lq: Tensor, target: Tensor, enhanced: Tensor,
                 losses: Any) -> BatchMetrics:
        return BatchMetrics(
            total_loss=float(losses.total.detach()),
            charbonnier_loss=float(losses.charbonnier.detach()),
            feature_loss=float(losses.feature.detach()),
            compressed_psnr=float(batch_psnr(target, lq[:, 3])),
            enhanced_psnr=float(batch_psnr(target, enhanced)),
            samples=int(target.shape[0]),
        )
