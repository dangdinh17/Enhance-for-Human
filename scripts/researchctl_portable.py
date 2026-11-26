#!/usr/bin/env python3
"""Train/validation-only researchctl tasks for the portable Kaggle bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.human_data import verify_training_inputs
from src.inference.evaluation import checked_checkpoint
from src.train.checkpoint_selection import CheckpointSelection
from src.utils.experiment import atomic_write_json, load_yaml_mapping


def publish(output: Path, scope: str, paths: list[Path],
            metrics: dict[str, float] | None = None) -> None:
    """Publish content-bound task evidence to the researchctl collector."""
    atomic_write_json(output, {
        "status": "complete", "scope": scope,
        "dataset_fingerprint": os.environ["RESEARCHCTL_DATASET_HASH"],
        "metrics": metrics or {},
        "artifacts": {f"artifact_{i}": str(p.resolve()) for i, p in enumerate(paths)},
    })


def train(config_path: Path, output: Path, device: str, smoke: bool) -> None:
    """Exercise actual optimizer/resume in smoke; keep main validation complete."""
    config = load_yaml_mapping(config_path)
    if not smoke and (
        config["train"].get("validation_max_samples") is not None
        or config["train"]["iterations"] != 300000
    ):
        raise ValueError("Main requires 300k updates and full validation")
    command = [sys.executable, "-B", "scripts/train_mfqev2_window.py",
               "--config", str(config_path), "--device", device]
    if smoke:
        subprocess.run(command + ["--stop-after", "1"], check=True)
    subprocess.run(command, check=True)
    directory = ROOT / config["run_dir"]
    steps = config["train"]["iterations"]
    best_path = checked_checkpoint(directory, config["qp"], steps)
    last = torch.load(directory / "checkpoints/last.pt",
                      map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    selection = CheckpointSelection.from_config(config)
    selection.check_resume(best["checkpoint_selection"], [config["qp"]])
    selection.check_resume(last["checkpoint_selection"], [config["qp"]])
    if last["sampler_state"]["next_iteration"] != steps:
        raise ValueError("Sampler resume did not reach the expected update")
    if {int(s["step"]) for s in last["optimizer_state_dict"]["state"].values()} != {steps}:
        raise ValueError("Adam successful-update count differs from iteration")
    if last["provenance"]["loss_mode"] != "charbonnier_only":
        raise ValueError("Unexpected feature-loss provenance")
    started = json.loads((directory / "STARTED.json").read_text())
    if started["started_iteration"] != 0:
        raise ValueError("Run did not begin from scratch")
    row = best["validation"]["per_qp"][str(config["qp"])]
    paths = [directory / name for name in (
        "STARTED.json", "DONE.json", "status.json", "train.csv", "validation.csv",
        "checkpoints/best.pth", "checkpoints/last.pt")]
    publish(output, "smoke" if smoke else "training", paths, {
        "validation.psnr_y": row["enhanced_psnr"],
        "validation.total_loss": row["total_loss"]})


def main() -> None:
    """Dispatch tasks without an independent queue or training launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "smoke", "train"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "verify":
        for qp in (22, 27, 32, 37):
            config = load_yaml_mapping(ROOT / f"configs/kaggle/QP{qp}.yml")
            verify_training_inputs(config)
        publish(args.output, "preparation", [ROOT / "data/bundle.json"])
    else:
        if args.config is None:
            parser.error("--config is required for smoke/train")
        train(args.config, args.output, args.device, args.action == "smoke")


if __name__ == "__main__":
    main()
