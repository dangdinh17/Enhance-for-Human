"""Bounded-memory sliding-window inference with shared encoder feature caching."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.models.mfqev2_window import MFQEv2Window


class CachedWindowEnhancer:
    """Cache seven CPU feature maps; process spatial tiles one at a time on GPU.

    Whole-frame selection descriptors are reused for every tile. ``read_luma``
    returns uint8 HxW arrays. Frame indices are clamped at sequence boundaries.
    Tile core/halo are measured in input pixels; halo must be divisible by four
    so both aggregation pyramid levels retain their sampling phase.
    """

    def __init__(self, model: MFQEv2Window, read_luma: Callable[[int], np.ndarray],
                 frames: int, tile_core: int = 128, halo: int = 96,
                 use_amp: bool = True) -> None:
        if frames < 1 or tile_core < 4 or tile_core % 4 or halo < 0 or halo % 4:
            raise ValueError("Invalid frame count or tile geometry")
        self.model, self.read_luma, self.frames = model.eval(), read_luma, frames
        self.device = next(model.parameters()).device
        self.tile_core, self.halo = tile_core, halo
        self.use_amp = use_amp and self.device.type == "cuda"
        self.cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = OrderedDict()

    @torch.inference_mode()
    def _get(self, index: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        frame = self.read_luma(index)
        if frame.ndim != 2 or frame.dtype != np.uint8:
            raise ValueError("read_luma must return uint8 HxW")
        value = torch.from_numpy(frame.copy()).to(self.device).float()[None, None] / 255
        h, w = frame.shape
        if min(h, w) < 4:
            raise ValueError("Frame dimensions must be at least four")
        value = F.pad(value, (0, (-w) % 4, 0, (-h) % 4), mode="reflect")
        with torch.autocast(self.device.type, enabled=self.use_amp):
            feature = self.model.encode(value[:, None])[:, 0]
        with torch.autocast(self.device.type, enabled=False):
            # Projection is affine, so project the mean instead of the full map.
            descriptor = F.conv2d(feature.float().mean((-2, -1), keepdim=True),
                                  self.model.selector.projection.weight,
                                  self.model.selector.projection.bias).flatten(1)
        storage = feature.half() if self.use_amp else feature.float()
        record = (storage.cpu(), descriptor, frame)
        self.cache[index] = record
        return record

    @torch.inference_mode()
    def __call__(self, index: int) -> np.ndarray:
        if not 0 <= index < self.frames:
            raise IndexError(index)
        radius = self.model.radius
        indices = [min(max(i, 0), self.frames - 1)
                   for i in range(index - radius, index + radius + 1)]
        # Evict before loading the next window, keeping memory bounded to 2N+1.
        for old in list(self.cache):
            if old not in indices:
                del self.cache[old]
        records = [self._get(i) for i in indices]
        descriptors = torch.stack([record[1] for record in records], dim=1)
        scores = self.model.selector.from_descriptors(descriptors.float())
        current = records[radius][2]
        h, w = current.shape
        ph, pw = records[0][0].shape[-2:]
        output = np.empty((h, w), dtype=np.float32)
        core, halo = self.tile_core // 2, self.halo // 2
        for top in range(0, ph, core):
            for left in range(0, pw, core):
                bottom, right = min(top + core, ph), min(left + core, pw)
                y0, x0 = max(0, top - halo), max(0, left - halo)
                y1, x1 = min(ph, bottom + halo), min(pw, right + halo)
                features = torch.stack([r[0][..., y0:y1, x0:x1] for r in records], dim=1)
                features = features.to(self.device)
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    residual = self.model.enhance_features(features, scores)
                y, x = top * 2, left * 2
                b, r = min(bottom * 2, h), min(right * 2, w)
                tile = residual[0, 0, (top - y0) * 2:(top - y0) * 2 + b - y,
                                (left - x0) * 2:(left - x0) * 2 + r - x]
                output[y:b, x:r] = current[y:b, x:r] / 255.0 + tile.float().cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("Non-finite window enhancement output")
        return np.rint(output.clip(0, 1) * 255).astype(np.uint8)
