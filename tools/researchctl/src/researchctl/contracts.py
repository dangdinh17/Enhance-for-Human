"""Verifiable output receipts and evidence checks for scientific results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .collectors import collect
from .config import ConfigError, hash_file, interpolate, redact, redact_text


def check_outputs(
    task: Mapping[str, Any],
    context: Mapping[str, Any],
    metric_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Require declared outputs and validate normalized adapter evidence."""
    paths = [
        Path(interpolate(str(item), context))
        for item in task.get("outputs", {}).get("required", [])
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ConfigError(f"Required outputs are missing: {missing}")
    metrics, artifacts = collect(list(task.get("collectors", [])), context)
    for index, path in enumerate(paths):
        artifacts[f"required:{index}"] = str(path)
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ConfigError(f"Metric {name} is not finite")
    specification = task.get("result_contract")
    if specification:
        path = Path(interpolate(str(specification["path"]), context))
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            raise ConfigError(f"Adapter did not complete: {path}")
        if result.get("dataset_fingerprint") != context["runtime.dataset_hash"]:
            raise ConfigError("Adapter dataset fingerprint does not match campaign")
        if result.get("scope") != specification["scope"]:
            raise ConfigError("Adapter result scope differs from the locked task")
        artifacts["adapter_result"] = str(path)
        if specification.get("primary_metric", False) and metric_contract:
            name = metric_contract["primary"]["name"]
            if name not in metrics or not math.isfinite(float(metrics[name])):
                raise ConfigError(f"Missing/non-finite primary metric: {name}")
        if specification["scope"] in {"full_video", "evaluation"} and metric_contract:
            validation = metric_contract.get("validation", {})
            if validation.get("require_baseline", False):
                baseline = result.get("baseline", {})
                if not baseline.get("artifacts") or not baseline.get("metrics"):
                    raise ConfigError(
                        "Full evaluation requires measured baseline evidence"
                    )
                for value in baseline["metrics"].values():
                    if not math.isfinite(float(value)):
                        raise ConfigError("Baseline metrics must be finite")
                for name in baseline["artifacts"]:
                    baseline_path = Path(name)
                    if not baseline_path.is_file():
                        raise ConfigError(f"Baseline artifact missing: {name}")
                    artifacts[f"baseline:{name}"] = str(baseline_path)
    return metrics, artifacts


def artifact_receipt(artifacts: Mapping[str, str]) -> dict[str, str]:
    """Hash actual files, including checkpoints, before caching a completion."""
    return {
        str(Path(path).resolve()): hash_file(Path(path))
        for path in set(artifacts.values())
        if Path(path).is_file()
    }


def valid_receipt(path: Path, input_hash: str) -> bool:
    """A completed journal row alone is never sufficient for cache reuse."""
    try:
        result = json.loads(path.read_text())
        receipt = result["artifact_hashes"]
        return (
            result["status"] == "complete"
            and result["input_hash"] == input_hash
            and bool(receipt)
            and all(
                Path(name).is_file() and hash_file(Path(name)) == digest
                for name, digest in receipt.items()
            )
        )
    except (OSError, KeyError, ValueError, TypeError):
        return False


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically persist redacted controller metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        redact_text(json.dumps(redact(value), indent=2, sort_keys=True)) + "\n"
    )
    temporary.replace(path)
