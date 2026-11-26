"""Residual feature alignment and spatial/channel fusion before MFQEv2 C10."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import DeformConv2d


class ResidualDeformableGatedFusion(nn.Module):
    """Align compensated PQFs to the current feature, preserving 288 channels.

    The three kernel groups correspond to MFQE's 3/5/7 feature branches.
    Initialization reproduces concatenation, including its feature amplitude.
    """

    def __init__(self, offset_bound: float = 4.0) -> None:
        super().__init__()
        if offset_bound <= 0:
            raise ValueError("offset_bound must be positive")
        self.offset_bound = float(offset_bound)
        self.projection = nn.Conv2d(96, 32, 1)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1), nn.PReLU(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.PReLU(64),
            nn.Conv2d(64, 81, 3, padding=1),
        )
        self.deform = DeformConv2d(96, 96, 3, padding=1, groups=3)
        self.gate = nn.Sequential(
            nn.Conv2d(288, 32, 1), nn.PReLU(32), nn.Conv2d(32, 96, 1),
        )
        self.mix = nn.Conv2d(288, 288, 1)
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.deform.weight.zero_()
            self.deform.bias.zero_()
            for channel in range(96):
                self.deform.weight[channel, channel % 32, 1, 1] = 2.0
            self.mix.weight.copy_(torch.eye(288).view(288, 288, 1, 1))
            self.mix.bias.zero_()

    def align(self, neighbor: Tensor, current: Tensor) -> Tensor:
        """Apply bounded residual offsets in the already-warped feature space."""
        left, center = self.projection(neighbor), self.projection(current)
        packed = torch.cat((left, center, (left - center).abs()), dim=1)
        prediction = self.offset_predictor(packed)
        offset = self.offset_bound * prediction[:, :54].tanh()
        mask = prediction[:, 54:].sigmoid()
        return self.deform(neighbor, offset, mask)

    def forward(self, previous: Tensor, current: Tensor, following: Tensor) -> Tensor:
        """Return learned, aligned and gated features of shape B x 288 x H x W."""
        if previous.shape != current.shape or following.shape != current.shape:
            raise ValueError("Fusion input shapes must match")
        if current.ndim != 4 or current.shape[1] != 96:
            raise ValueError("Fusion inputs must be BCHW tensors with 96 channels")
        # Torchvision DCN and attention normalization run in FP32 under AMP.
        with torch.autocast(device_type=current.device.type, enabled=False):
            previous, current, following = (
                value.float() for value in (previous, current, following)
            )
            features = (self.align(previous, current), current,
                        self.align(following, current))
            logits = torch.stack([
                self.gate(torch.cat((value, current, (value - current).abs()), dim=1))
                for value in features
            ], dim=1)
            weights = logits.softmax(dim=1)
            weighted = torch.cat([
                3.0 * weights[:, index] * value
                for index, value in enumerate(features)
            ], dim=1)
            return self.mix(weighted)
