"""Measured loss and PSNR accumulators."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from src.utils.training import AverageMeter

@dataclass(frozen=True)
class BatchMetrics:
    """Sample-averaged loss and PSNR metrics."""

    total_loss: float
    charbonnier_loss: float
    feature_loss: float
    compressed_psnr: float
    enhanced_psnr: float
    samples: int

    @property
    def delta_psnr(self) -> float:
        """Return enhanced minus compressed PSNR."""
        return self.enhanced_psnr - self.compressed_psnr

    def to_dict(self) -> dict[str, float | int]:
        """Return a flat JSON/CSV-compatible mapping."""
        return {
            "total_loss": self.total_loss,
            "charbonnier_loss": self.charbonnier_loss,
            "feature_loss": self.feature_loss,
            "compressed_psnr": self.compressed_psnr,
            "enhanced_psnr": self.enhanced_psnr,
            "delta_psnr": self.delta_psnr,
            "samples": self.samples,
        }


class MetricAccumulator:
    """Accumulate the metrics returned by multiple mini-batches."""

    def __init__(self) -> None:
        self.total_loss = AverageMeter()
        self.charbonnier_loss = AverageMeter()
        self.feature_loss = AverageMeter()
        self.compressed_psnr = AverageMeter()
        self.enhanced_psnr = AverageMeter()

    def update(self, metrics: BatchMetrics) -> None:
        """Add one batch using its sample count as weight."""
        count = metrics.samples
        self.total_loss.update(metrics.total_loss, count)
        self.charbonnier_loss.update(metrics.charbonnier_loss, count)
        self.feature_loss.update(metrics.feature_loss, count)
        self.compressed_psnr.update(metrics.compressed_psnr, count)
        self.enhanced_psnr.update(metrics.enhanced_psnr, count)

    def compute(self) -> BatchMetrics:
        """Return averages for all accumulated samples."""
        if self.total_loss.count == 0:
            raise ValueError("No metrics have been accumulated")
        return BatchMetrics(
            total_loss=self.total_loss.average,
            charbonnier_loss=self.charbonnier_loss.average,
            feature_loss=self.feature_loss.average,
            compressed_psnr=self.compressed_psnr.average,
            enhanced_psnr=self.enhanced_psnr.average,
            samples=self.total_loss.count,
        )


def batch_psnr(reference: Tensor, test: Tensor) -> Tensor:
    """Return mean per-image PSNR for normalized BCHW tensors."""
    if reference.shape != test.shape or reference.ndim != 4:
        raise ValueError("PSNR inputs must be identically shaped BCHW tensors")
    difference = reference.float() - test.float().clamp(0.0, 1.0)
    mse = difference.square().flatten(1).mean(dim=1)
    return (-10.0 * torch.log10(mse.clamp_min(1.0e-12))).mean()

