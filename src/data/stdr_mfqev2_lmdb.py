"""Seven-frame LMDB dataset used to train STDR on MFQEv2."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal, Optional, cast

import cv2
import lmdb
import numpy as np
import torch
from lmdb import Environment
from torch import Tensor
from torch.utils.data import Dataset


Split = Literal["train", "validation"]


def load_stdr_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate one STDR seven-frame manifest."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_stdr_manifest(manifest)
    return manifest


def validate_stdr_manifest(manifest: dict[str, Any]) -> None:
    """Validate STDR storage metadata and temporal sample relationships."""
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported STDR manifest schema_version")
    if manifest.get("storage") != "lmdb":
        raise ValueError("STDR manifest storage must be lmdb")
    if manifest.get("format") != "stdr_mfqev2_7frame":
        raise ValueError("Unsupported STDR manifest format")
    for field in ("gt_lmdb", "lq_lmdb"):
        if not isinstance(manifest.get(field), str):
            raise ValueError(f"STDR manifest must define {field}")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("STDR manifest must contain splits")
    for split in ("train", "validation"):
        entries = splits.get(split)
        if not isinstance(entries, list):
            raise ValueError(f"STDR split {split} must be a list")
        for entry in entries:
            keys = entry.get("lq_keys")
            if not isinstance(keys, list) or len(keys) != 7:
                raise ValueError("Every STDR sample must contain seven LQ keys")
            if not isinstance(entry.get("target_key"), str):
                raise ValueError("Every STDR sample must contain a target key")
            indices = entry.get("lq_frame_indices")
            if not isinstance(indices, list) or len(indices) != 7:
                raise ValueError("Every STDR sample must contain seven indices")
            target = int(entry["target_frame_index"])
            if int(indices[3]) != target:
                raise ValueError("The fourth STDR input must be the target frame")
            if split == "validation" and (
                target < 0
                or target >= 30
                or any(int(value) < 0 or int(value) >= 30 for value in indices)
            ):
                raise ValueError("Validation references must stay in frames 0-29")


class STDRMFQEv2LMDBDataset(Dataset):
    """Read synchronized seven-frame luma samples from lossless LMDBs."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: Split,
        crop_size: int,
        training: bool,
        use_flip: bool = True,
        use_rotation: bool = True,
        temporal_reverse: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown STDR split: {split}")
        if crop_size <= 0 or crop_size % 8:
            raise ValueError("STDR crop_size must be positive and divisible by 8")
        self.manifest_path = Path(manifest_path)
        manifest = load_stdr_manifest(self.manifest_path)
        self.qp = int(manifest["qp"])
        self.entries = list(manifest["splits"][split])
        if not self.entries:
            raise ValueError(f"Empty STDR split: {split}")
        self.crop_size = crop_size
        self.training = training
        self.use_flip = use_flip
        self.use_rotation = use_rotation
        self.temporal_reverse = temporal_reverse
        self.gt_root = self._resolve(str(manifest["gt_lmdb"]))
        self.lq_root = self._resolve(str(manifest["lq_lmdb"]))
        self._gt_environment: Environment | None = None
        self._lq_environment: Environment | None = None

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.manifest_path.parent / path).resolve()

    def __len__(self) -> int:
        return len(self.entries)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_gt_environment"] = None
        state["_lq_environment"] = None
        return state

    def _environment(self, client: Literal["gt", "lq"]) -> Environment:
        attribute = f"_{client}_environment"
        environment = cast(Optional[Environment], getattr(self, attribute))
        if environment is None:
            root = self.gt_root if client == "gt" else self.lq_root
            if not root.is_dir():
                raise FileNotFoundError(root)
            environment = lmdb.open(
                str(root),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
            setattr(self, attribute, environment)
        return environment

    def _decode(self, client: Literal["gt", "lq"], key: str) -> np.ndarray:
        with self._environment(client).begin(write=False) as transaction:
            payload = transaction.get(key.encode("ascii"))
        if payload is None:
            raise KeyError(f"Missing {client.upper()} key: {key}")
        image = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            raise ValueError(f"Invalid PNG payload: {client}:{key}")
        return image

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        entry = self.entries[index]
        lq = [self._decode("lq", str(key)) for key in entry["lq_keys"]]
        target = self._decode("gt", str(entry["target_key"]))
        images = [*lq, target]
        if any(image.shape != target.shape for image in images):
            raise ValueError(f"Shape mismatch for {entry['target_key']}")
        images = self._crop(images)
        if self.training:
            images = self._augment(images)
        tensors = [
            torch.from_numpy(np.ascontiguousarray(image))
            .unsqueeze(0)
            .float()
            .div_(255.0)
            for image in images
        ]
        return {
            "lq": torch.stack(tensors[:7], dim=0),
            "target": tensors[7],
            "sequence": str(entry["sequence"]),
            "frame_index": int(entry["target_frame_index"]),
            "qp": self.qp,
        }

    def _crop(self, images: list[np.ndarray]) -> list[np.ndarray]:
        height, width = images[0].shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"Frame {width}x{height} is smaller than crop {self.crop_size}"
            )
        if self.training:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        return [
            image[top : top + self.crop_size, left : left + self.crop_size]
            for image in images
        ]

    def _augment(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if self.use_flip and random.random() < 0.5:
            images = [np.fliplr(image) for image in images]
        if self.use_flip and random.random() < 0.5:
            images = [np.flipud(image) for image in images]
        if self.use_rotation:
            rotations = random.randint(0, 3)
            if rotations:
                images = [np.rot90(image, rotations) for image in images]
        if self.temporal_reverse and random.random() < 0.5:
            images[:7] = list(reversed(images[:7]))
        return images

    def close(self) -> None:
        """Close process-local LMDB handles."""
        for attribute in ("_gt_environment", "_lq_environment"):
            environment = cast(Optional[Environment], getattr(self, attribute))
            if environment is not None:
                environment.close()
                setattr(self, attribute, None)
