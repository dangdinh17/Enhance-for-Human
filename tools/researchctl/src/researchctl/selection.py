"""Deterministic metric-based trial ranking."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Mapping, Sequence

from .config import ConfigError


def _oriented(value: float, direction: str) -> float:
    if not math.isfinite(value):
        raise ConfigError("Selection metrics must be finite")
    if direction == "minimize":
        return value
    if direction == "maximize":
        return -value
    raise ConfigError(f"Invalid metric direction: {direction}")


def _runtime(row: Mapping[str, Any]) -> float:
    if not row.get("started_at") or not row.get("ended_at"):
        return float("inf")
    start = datetime.fromisoformat(str(row["started_at"]))
    end = datetime.fromisoformat(str(row["ended_at"]))
    return max(0.0, (end - start).total_seconds())


def rank_rows(
    rows: Sequence[Mapping[str, Any]], metric_contract: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Rank by primary, available secondary metrics, runtime, then trial ID."""
    specifications = [metric_contract["primary"], *metric_contract.get("secondary", [])]

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = json.loads(str(row["metrics_json"]))
        values: list[Any] = []
        for index, specification in enumerate(specifications):
            name = str(specification["name"])
            if name not in metrics:
                if index == 0:
                    raise ConfigError(
                        f"Completed tuning node has no primary metric: {name}"
                    )
                values.append(float("inf"))
                continue
            values.append(
                _oriented(float(metrics[name]), str(specification["direction"]))
            )
        values.extend((_runtime(row), str(row["trial_id"])))
        return tuple(values)

    return sorted(rows, key=key)
