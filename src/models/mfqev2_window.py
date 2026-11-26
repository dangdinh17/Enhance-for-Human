"""Blind multi-QP enhancement with soft key selection and temporal aggregation.

This is a repository-specific architecture inspired by temporal selection and
motion-aware aggregation, not a reproduction of RVRT or FMA-Net.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qe_alignment_fusion import ResidualDeformableGatedFusion


class SharedWindowEncoder(nn.Module):
    """Preserve MFQE's three spatial scales using shared half-resolution filters."""

    def __init__(self) -> None:
        super().__init__()
        self.scales = nn.ModuleList(
            nn.Sequential(nn.Conv2d(1, 32, k, stride=2, padding=k // 2), nn.PReLU(32))
            for k in (3, 5, 7)
        )

    def forward(self, frames: Tensor) -> Tensor:
        batch, temporal, _, height, width = frames.shape
        flattened = frames.reshape(batch * temporal, 1, height, width)
        encoded = torch.cat([layer(flattened) for layer in self.scales], dim=1)
        return encoded.reshape(batch, temporal, 96, *encoded.shape[-2:])


class SoftKeyFeatureSelection(nn.Module):
    """Predict global frame/channel relevance without comparing unaligned pixels."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.projection = nn.Conv2d(96, 32, 1)
        self.score = nn.Sequential(nn.Linear(97, 64), nn.PReLU(64), nn.Linear(64, 96))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        batch, temporal, channels, height, width = features.shape
        if not self.enabled:
            return features.new_zeros(batch, temporal, channels)
        descriptors = self.projection(features.reshape(-1, channels, height, width))
        descriptors = descriptors.mean(dim=(-2, -1)).reshape(batch, temporal, 32)
        return self.from_descriptors(descriptors)

    def from_descriptors(self, descriptors: Tensor) -> Tensor:
        """Score cached whole-frame descriptors consistently across feature tiles."""
        batch, temporal, _ = descriptors.shape
        if not self.enabled:
            return descriptors.new_zeros(batch, temporal, 96)
        center = descriptors[:, temporal // 2:temporal // 2 + 1].expand_as(descriptors)
        distances = torch.linspace(-1, 1, temporal, device=descriptors.device,
                                   dtype=descriptors.dtype).view(1, temporal, 1)
        distances = distances.expand(batch, -1, -1)
        pairs = torch.cat((descriptors, center,
                           (descriptors - center).abs(), distances), dim=-1)
        return self.score(pairs.reshape(-1, 97)).reshape(batch, temporal, 96)


class SamplingOffsets(nn.Module):
    """Lightweight shared offset estimator for four learned sampling points."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, groups=96), nn.PReLU(96),
            nn.Conv2d(96, 8, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, reference: Tensor, neighbor: Tensor) -> Tensor:
        return self.layers(torch.cat((neighbor, reference,
                                      (neighbor - reference).abs()), dim=1))


class MotionAwareTemporalAggregation(nn.Module):
    """Aggregate a temporal side in current coordinates, including a self fallback.

    Global key-selection priors combine with local query/key similarity. Each
    neighbor contributes four sample points; a -log(4) term avoids giving it four
    times the prior probability of the single, unwarped current fallback.
    """

    def __init__(self, coarse_bound: float = 8.0, fine_bound: float = 4.0) -> None:
        super().__init__()
        if coarse_bound <= 0 or fine_bound <= 0:
            raise ValueError("Sampling bounds must be positive")
        self.coarse_bound = float(coarse_bound)
        self.fine_bound = float(fine_bound)
        self.projection = nn.Conv2d(96, 32, 1)
        self.coarse_offsets = SamplingOffsets()
        self.fine_offsets = SamplingOffsets()
        self.register_buffer("base_points", torch.tensor(
            [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]).view(1, 4, 2, 1, 1))

    @staticmethod
    def sample(value: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor]:
        """Bilinearly sample B,C,H,W using B,M,2,H,W offsets ordered (dy, dx)."""
        batch, channels, height, width = value.shape
        points = offsets.shape[1]
        yy, xx = torch.meshgrid(torch.arange(height, device=value.device, dtype=value.dtype),
                                torch.arange(width, device=value.device, dtype=value.dtype),
                                indexing="ij")
        sy = yy[None, None] + offsets[:, :, 0]
        sx = xx[None, None] + offsets[:, :, 1]
        valid = (sy >= 0) & (sy <= height - 1) & (sx >= 0) & (sx <= width - 1)
        grid = torch.stack((2 * (sx + 0.5) / width - 1,
                            2 * (sy + 0.5) / height - 1), dim=-1)
        sampled = F.grid_sample(value, grid.reshape(batch, points * height, width, 2),
                                mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.reshape(batch, channels, points, height, width)
        return sampled.permute(0, 2, 1, 3, 4), valid

    def forward(self, features: Tensor, scores: Tensor, indices: list[int]) -> tuple[Tensor, Tensor]:
        batch, temporal, channels, height, width = features.shape
        center_index = temporal // 2
        reference = features[:, center_index]
        projected = self.projection(features.reshape(-1, channels, height, width))
        projected = projected.reshape(batch, temporal, 32, height, width)
        query = projected[:, center_index]
        coarse_query = F.avg_pool2d(query, 2)
        candidates = [*indices, center_index]
        priors = scores[:, candidates].log_softmax(dim=1)
        values, logits = [], []
        for order, index in enumerate(indices):
            key = projected[:, index]
            coarse_key = F.avg_pool2d(key, 2)
            coarse = self.coarse_bound * self.coarse_offsets(coarse_query, coarse_key).tanh()
            coarse = coarse.reshape(batch, 4, 2, *coarse.shape[-2:]) + self.base_points
            coarse = F.interpolate(coarse.flatten(1, 2), size=(height, width),
                                   mode="bilinear", align_corners=False).reshape(batch, 4, 2, height, width) * 2
            preliminary, _ = self.sample(key, coarse)
            fine = self.fine_bound * self.fine_offsets(query, preliminary.mean(dim=1)).tanh()
            offsets = coarse + fine.reshape(batch, 4, 2, height, width)
            packed = torch.cat((features[:, index], key), dim=1)
            sampled, valid = self.sample(packed, offsets)
            similarity = (query[:, None] * sampled[:, :, 96:]).sum(dim=2) / math.sqrt(32)
            logit = similarity[:, :, None] + priors[:, order, None, :, None, None] - math.log(4)
            logits.append(logit.masked_fill(~valid[:, :, None], -1.0e4))
            values.append(sampled[:, :, :96])
        self_similarity = query.square().sum(dim=1, keepdim=True) / math.sqrt(32)
        logits.append(self_similarity[:, :, None] + priors[:, -1, None, :, None, None])
        values.append(reference[:, None])
        attention = torch.cat(logits, dim=1).softmax(dim=1)
        output = (attention * torch.cat(values, dim=1)).sum(dim=1)
        return output, attention[:, -1].mean().detach()


class WindowResidualBlock(nn.Module):
    """A small reconstruction block with no normalization or QP routing."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.PReLU(32),
                                    nn.Conv2d(32, 32, 3, padding=1))

    def forward(self, value: Tensor) -> Tensor:
        return value + 0.1 * self.layers(value)


class MFQEv2Window(nn.Module):
    """One blind model for adjacent frames across all compression levels."""

    def __init__(self, radius: int = 3, selection: bool = True,
                 fusion: str = "alignment_gated", reconstruction_blocks: int = 4,
                 coarse_bound: float = 8.0, fine_bound: float = 4.0,
                 fusion_offset_bound: float = 4.0) -> None:
        super().__init__()
        if radius < 1 or reconstruction_blocks < 1:
            raise ValueError("radius and reconstruction_blocks must be positive")
        if fusion not in {"alignment_gated", "concat"}:
            raise ValueError("fusion must be alignment_gated or concat")
        self.network_options = dict(radius=radius, selection=selection, fusion=fusion,
                                    reconstruction_blocks=reconstruction_blocks,
                                    coarse_bound=coarse_bound, fine_bound=fine_bound,
                                    fusion_offset_bound=fusion_offset_bound)
        self.radius = radius
        self.encoder = SharedWindowEncoder()
        self.selector = SoftKeyFeatureSelection(selection)
        self.aggregation = MotionAwareTemporalAggregation(coarse_bound, fine_bound)
        # Fusion controls must start with identical encoder/aggregation/head
        # weights under the same seed, despite different fusion parameter counts.
        with torch.random.fork_rng(devices=[]):
            self.fusion = (ResidualDeformableGatedFusion(fusion_offset_bound)
                           if fusion == "alignment_gated" else nn.Conv2d(288, 288, 1))
        if fusion == "concat":
            with torch.no_grad():
                self.fusion.weight.copy_(torch.eye(288).view(288, 288, 1, 1))
                self.fusion.bias.zero_()
        self.reconstruction = nn.Sequential(
            nn.Conv2d(288, 32, 3, padding=1), nn.PReLU(32),
            *[WindowResidualBlock() for _ in range(reconstruction_blocks)],
        )
        self.output = nn.Conv2d(32, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostics: dict[str, Tensor] = {}

    def encode(self, window: Tensor) -> Tensor:
        """Encode adjacent frames once; usable by cached full-video inference."""
        return self.encoder(window)

    def enhance_features(self, features: Tensor, scores: Tensor | None = None) -> Tensor:
        """Return an upsampled image residual from cached feature tiles."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            scores = self.selector(features) if scores is None else scores.float()
            center = features.shape[1] // 2
            past, past_fallback = self.aggregation(features, scores, list(range(center)))
            future, future_fallback = self.aggregation(
                features, scores, list(range(center + 1, features.shape[1])))
            current = features[:, center]
            if isinstance(self.fusion, ResidualDeformableGatedFusion):
                merged = self.fusion(past, current, future)
            else:
                merged = self.fusion(torch.cat((past, current, future), dim=1))
            self.last_diagnostics = {
                "past_self_weight": past_fallback, "future_self_weight": future_fallback,
                "past_selection": scores[:, :center + 1].softmax(1).mean((0, 2)).detach(),
                "future_selection": scores[:, center:].softmax(1).mean((0, 2)).detach(),
            }
        return F.pixel_shuffle(self.output(self.reconstruction(merged)), 2)

    def forward(self, window: Tensor) -> Tensor:
        if window.ndim != 5 or window.shape[1:3] != (2 * self.radius + 1, 1):
            raise ValueError(f"Expected [B,{2 * self.radius + 1},1,H,W], got {tuple(window.shape)}")
        height, width = window.shape[-2:]
        if min(height, width) < 4:
            raise ValueError("Frame dimensions must be at least four pixels")
        current = window[:, self.radius]
        pad_h, pad_w = (-height) % 4, (-width) % 4
        if pad_h or pad_w:
            flat = window.flatten(0, 1)
            flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="reflect")
            window = flat.reshape(*window.shape[:3], *flat.shape[-2:])
        residual = self.enhance_features(self.encode(window))
        return current + residual[..., :height, :width]


def load_mfqev2_window(path: str | Path, device: str | torch.device = "cpu") -> MFQEv2Window:
    """Load one model without requiring QP or peak/non-peak metadata."""
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "MFQEv2Window":
        raise ValueError("Checkpoint is not an MFQEv2Window model")
    model = MFQEv2Window(**checkpoint["network_options"])
    model.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model_state_dict")), strict=True)
    return model.to(device).eval()
