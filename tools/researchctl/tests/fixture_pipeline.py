"""Real CPU linear regression used solely to verify controller integration."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import yaml


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "evaluate", "aggregate"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    directory = Path(config.get("run_dir", args.output.parent))
    checkpoint = directory / "checkpoint.json"
    x = [-1.0, -0.5, 0.0, 0.5, 1.0]
    y = [2 * value + 0.5 for value in x]
    if args.phase == "train":
        stopped = []
        signal.signal(signal.SIGTERM, lambda *unused: stopped.append(True))
        state = (
            json.loads(checkpoint.read_text())
            if checkpoint.exists()
            else {"weight": 0.0, "bias": 0.0, "step": 0}
        )
        for step in range(state["step"], config.get("steps", 20)):
            residuals = [
                state["weight"] * value + state["bias"] - target
                for value, target in zip(x, y)
            ]
            state["weight"] -= (
                0.2 * sum(error * value for error, value in zip(residuals, x)) / len(x)
            )
            state["bias"] -= 0.2 * sum(residuals) / len(x)
            state["step"] = step + 1
            save(checkpoint, state)
            time.sleep(float(config.get("delay", 0)))
            if stopped:
                raise SystemExit(143)
        mse = sum(
            (state["weight"] * value + state["bias"] - target) ** 2
            for value, target in zip(x, y)
        ) / len(x)
        metrics = {"validation.mse": mse}
        save(directory / "DONE.json", {"status": "complete", "steps": state["step"]})
        artifacts = {
            "checkpoint": str(checkpoint),
            "done": str(directory / "DONE.json"),
        }
        scope = "training"
        extra = {}
    elif args.phase == "evaluate":
        state = json.loads(checkpoint.read_text())
        inputs = [0.25, 0.75, 1.25]
        mse = sum(
            (state["weight"] * value + state["bias"] - (2 * value + 0.5)) ** 2
            for value in inputs
        ) / len(inputs)
        baseline = sum((2 * value + 0.5) ** 2 for value in inputs) / len(inputs)
        baseline_file = args.output.parent / "baseline.json"
        save(baseline_file, {"mse": baseline})
        metrics, artifacts, scope = (
            {"evaluation.mse": mse},
            {"checkpoint": str(checkpoint)},
            "evaluation",
        )
        extra = {
            "baseline": {
                "metrics": {"mse": baseline},
                "artifacts": [str(baseline_file)],
            }
        }
    else:
        inputs = json.loads(args.inputs.read_text())
        results = [
            json.loads(Path(node["result"]).read_text())
            for node in inputs["nodes"]
            if node["purpose"] == "evaluation"
        ]
        if not results:
            raise ValueError("No evaluation inputs")
        metrics = {
            "aggregate.mse": sum(row["metrics"]["evaluation.mse"] for row in results)
            / len(results)
        }
        artifacts, scope, extra = {}, "aggregate", {}
    save(
        args.output,
        {
            "status": "complete",
            "scope": scope,
            "metrics": metrics,
            "artifacts": artifacts,
            "dataset_fingerprint": os.environ["RESEARCHCTL_DATASET_HASH"],
            **extra,
        },
    )


if __name__ == "__main__":
    main()
