#!/usr/bin/env python3
"""Typed task entrypoints executed only through the researchctl contracts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.human_data import (CACHE, READY, inventory, manifest,
                                 verify_sources, verify_training_inputs)
from src.utils.experiment import atomic_write_json, load_yaml_mapping
from src.utils.hashing import file_sha256


def result(output: Path, scope: str, artifacts: list[Path],
           metrics: dict[str, Any] | None = None,
           baseline: dict[str, Any] | None = None) -> None:
    """Publish normalized receipts; smoke output is explicitly diagnostic."""
    atomic_write_json(output, {"status": "complete", "scope": scope,
                               "dataset_fingerprint": os.environ["RESEARCHCTL_DATASET_HASH"],
                               "metrics": metrics or {},
                               "artifacts": {f"artifact_{i}": str(p.resolve()) for i, p in enumerate(artifacts)},
                               **({"baseline": baseline} if baseline else {})})


def train(config_path: Path, output: Path, smoke: bool) -> None:
    """Invoke the trainer, verify checkpoints and exercise smoke resume."""
    from src.inference.evaluation import checked_checkpoint, evaluate_qp
    config = load_yaml_mapping(config_path)
    command = [sys.executable, "-B", "scripts/train_mfqev2_window.py", "--config", str(config_path), "--device", "cuda:0"]
    if smoke:
        subprocess.run(command + ["--stop-after", "1"], check=True)
    subprocess.run(command, check=True)
    directory = ROOT / config["run_dir"]
    checkpoint = checked_checkpoint(directory, config["qp"], config["train"]["iterations"])
    saved = torch.load(directory / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    if saved["sampler_state"]["next_iteration"] != config["train"]["iterations"]:
        raise ValueError("Sampler resume receipt mismatch")
    adam_steps = {int(state["step"]) for state in saved["optimizer_state_dict"]["state"].values()}
    if adam_steps != {config["train"]["iterations"]}:
        raise ValueError("Optimizer successful-update count mismatch")
    if saved["provenance"]["loss_mode"] != "charbonnier_only":
        raise ValueError("Unexpected loss provenance")
    best = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = best["validation"]["per_qp"][str(config["qp"])]
    artifacts = [directory / "STARTED.json", directory / "DONE.json", checkpoint, directory / "checkpoints/last.pt", directory / "validation.csv", directory / "train.csv", directory / "status.json"]
    if smoke:
        artifacts += evaluate_qp(config, output.parent / "inference_smoke", True)
    result(output, "smoke" if smoke else "training", artifacts,
           {"validation.total_loss": metrics["total_loss"], "validation.psnr_y": metrics["enhanced_psnr"]})


def main() -> None:
    """Dispatch one researchctl task without performing scheduling itself."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["runtime", "sources", "data_smoke", "data_full", "dataset", "train_smoke", "train", "evaluate", "aggregate"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if args.action == "runtime":
        from verify_training_runtime import verify
        receipt = verify(output.parent)
        result(output, "smoke", [receipt, output.parent / "runtime_state.pt"])
    elif args.action == "sources":
        checks = verify_sources()
        evidence = output.parent / "source_checks.json"
        atomic_write_json(evidence, checks)
        result(output, "preparation", [evidence])
    elif args.action in {"data_smoke", "data_full"}:
        from prepare_mfqev2 import prepare
        smoke = args.action == "data_smoke"
        receipt = prepare(output.parent / "bounded_data" if smoke else CACHE, smoke)
        result(output, "smoke" if smoke else "preparation", [receipt])
    elif args.action == "dataset":
        verify_sources()
        started = []
        for qp in (22, 27, 32, 37):
            config = load_yaml_mapping(ROOT / f"configs/newqe/QP{qp}.yml")
            verify_training_inputs(config)
            directory = ROOT / config["run_dir"]
            if (directory / "checkpoints").exists():
                raise ValueError(f"Fresh campaign refuses preexisting main checkpoints: {directory}")
            started.append({"qp": qp, "started_iteration": 0, "seed": config["seed"], "run_dir": config["run_dir"]})
        evidence = output.parent / "scratch_guard.json"
        atomic_write_json(evidence, {"status": "complete", "runs": started, "dataset_sha256": file_sha256(READY)})
        result(output, "preparation", [evidence, READY])
    elif args.action in {"train_smoke", "train"}:
        train(args.config, output, args.action == "train_smoke")
    elif args.action == "evaluate":
        from src.inference.evaluation import evaluate_qp
        paths = evaluate_qp(load_yaml_mapping(args.config))
        records = [json.loads(path.read_text()) for path in paths]
        baseline = {"metrics": {"psnr_y": sum(r["vvc_psnr_y"] for r in records) / len(records)},
                    "artifacts": [str(p) for p in paths]}
        result(output, "full_video", paths, baseline=baseline)
    else:
        from src.inference.report import aggregate
        receipt = aggregate()
        result(output, "aggregate", [receipt, receipt.parent / "per_sequence_qp.csv", receipt.parent / "macro18.csv", receipt.parent / "bd_rate_per_sequence.csv", receipt.parent / "report.md"])


if __name__ == "__main__":
    main()
