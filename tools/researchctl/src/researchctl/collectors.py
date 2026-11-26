"""Result collectors for existing ML command-line programs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, get_dot_path, interpolate


def collect(
    collectors: list[Mapping[str, Any]], context: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Collect normalized metrics and artifact paths."""
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for specification in collectors:
        collector_type = str(specification.get("type"))
        path = Path(interpolate(str(specification["path"]), context))
        if not path.is_file():
            raise ConfigError(f"Collector input does not exist: {path}")
        if collector_type == "result_json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") != "complete":
                raise ConfigError(f"Adapter result is not complete: {path}")
            metrics.update(value.get("metrics", {}))
            for name, artifact in value.get("artifacts", {}).items():
                if not Path(artifact).is_file():
                    raise ConfigError(f"Adapter artifact missing: {artifact}")
                artifacts[str(name)] = str(artifact)
        elif collector_type in {"json", "done_json"}:
            value = json.loads(path.read_text(encoding="utf-8"))
            if collector_type == "done_json" and value.get("status") != "complete":
                raise ConfigError(f"Completion file is not complete: {path}")
            for name, source in specification.get("metrics", {}).items():
                metrics[str(name)] = get_dot_path(value, str(source))
        elif collector_type == "csv_best":
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ConfigError(f"CSV collector has no rows: {path}")
            column = str(specification["select_column"])
            mode = str(specification.get("mode", "min"))
            if mode not in {"min", "max"}:
                raise ConfigError(f"Invalid CSV collector mode: {mode}")
            row = (min if mode == "min" else max)(
                rows, key=lambda item: float(item[column])
            )
            for name, source in specification.get("metrics", {}).items():
                metrics[str(name)] = float(row[str(source)])
        else:
            raise ConfigError(f"Unsupported collector type: {collector_type}")
        artifacts[str(specification.get("name", path.stem))] = str(path)
    return metrics, artifacts
