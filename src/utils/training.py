"""Small, reusable helpers for reproducible PyTorch training."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


class AverageMeter:
    """Track a sample-weighted running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated values."""
        self.total = 0.0
        self.count = 0

    @property
    def average(self) -> float:
        """Return the current average, or zero before the first update."""
        return self.total / self.count if self.count else 0.0

    def update(self, value: float, sample_count: int = 1) -> None:
        """Add a scalar measured over ``sample_count`` samples."""
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        self.total += float(value) * sample_count
        self.count += sample_count


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python and PyTorch random number generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_data_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_random_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    """Capture process and optional DataLoader generator random states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state["data_loader_generator"] = generator.get_state()
    return state


def restore_random_state(
    state: dict[str, Any] | None,
    generator: torch.Generator | None = None,
) -> None:
    """Restore a state produced by :func:`capture_random_state`."""
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A training checkpoint may have been loaded with ``map_location=cuda``.
    # PyTorch's CPU and Generator RNG setters only accept CPU ByteTensors.
    torch.set_rng_state(state["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            value.detach().to(device="cpu", dtype=torch.uint8)
            for value in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    if generator is not None and "data_loader_generator" in state:
        generator.set_state(
            state["data_loader_generator"]
            .detach()
            .to(device="cpu", dtype=torch.uint8)
        )


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or select CUDA when it is available."""
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Atomically save a checkpoint, avoiding partially written files."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    try:
        torch.save(state, temporary_name)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
