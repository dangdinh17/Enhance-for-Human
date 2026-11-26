"""Validation checkpoint ranking, independent of the training objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CheckpointSelection:
    """Select a checkpoint using a configured, finite validation metric."""

    metric: str = "total_loss"
    mode: str = "min"

    def __post_init__(self) -> None:
        expected = {"total_loss": "min", "enhanced_psnr": "max"}
        if expected.get(self.metric) != self.mode:
            raise ValueError(f"Unsupported checkpoint selection: {self.metric}/{self.mode}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CheckpointSelection:
        """Keep loss selection for legacy configs lacking an explicit policy."""
        return cls(**config["train"].get("checkpoint_selection", {}))

    @property
    def initial_value(self) -> float:
        """Return a sentinel that admits the first finite validation score."""
        return -math.inf if self.mode == "max" else math.inf

    def score(self, validation: Mapping[str, Any]) -> float:
        """Average the configured metric with equal weight for each QP."""
        values = [float(row[self.metric]) for row in validation["per_qp"].values()]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Validation {self.metric} must be nonempty and finite")
        return sum(values) / len(values)

    def improves(self, candidate: float, best: float) -> bool:
        """Keep the earlier checkpoint on a tie; reject invalid candidates."""
        if not math.isfinite(candidate):
            raise ValueError(f"Non-finite validation {self.metric}")
        return candidate > best if self.mode == "max" else candidate < best

    def metadata(self, qps: list[int], value: float) -> dict[str, Any]:
        """Describe the policy without serializing infinite sentinels to JSON."""
        return {"metric": self.metric, "mode": self.mode, "qps": qps,
                "value": value if math.isfinite(value) else None}

    def check_resume(self, saved: Mapping[str, Any], qps: list[int]) -> None:
        """Prevent a saved loss score from being interpreted as a PSNR score."""
        if any(saved.get(key) != value for key, value in
               (("metric", self.metric), ("mode", self.mode), ("qps", qps))):
            raise ValueError("Resume checkpoint selection policy mismatch")
