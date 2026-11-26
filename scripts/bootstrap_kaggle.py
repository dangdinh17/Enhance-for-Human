#!/usr/bin/env python3
"""Verify and attach the Kaggle train/validation dataset to this checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.training_bundle import DATASET_ID, bootstrap


def discover_dataset_root(dataset_root: Path | None = None,
                          input_root: Path = Path("/kaggle/input")) -> Path:
    """Find the actual mounted bundle by identity, across Kaggle layouts."""
    if dataset_root is not None:
        receipt = dataset_root / "bundle.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"Missing {receipt}; --dataset-root must name the bundle folder")
        return dataset_root.resolve()
    candidates = set()
    for receipt in sorted(input_root.rglob("bundle.json")):
        try:
            value = json.loads(receipt.read_text())
        except (OSError, ValueError) as error:
            print(f"Skipping unreadable bundle metadata {receipt}: {error}", file=sys.stderr)
            continue
        if isinstance(value, dict) and value.get("dataset_id") == DATASET_ID:
            candidates.add(receipt.parent.resolve())
    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise ValueError(f"Multiple matching datasets: {paths}. Select one with --dataset-root.")
    raise FileNotFoundError(
        f"Dataset {DATASET_ID} is not attached under {input_root}. "
        "Use Kaggle Add Input to attach https://www.kaggle.com/datasets/"
        f"{DATASET_ID}, then rerun bootstrap. For another location, pass --dataset-root."
    )


def main() -> None:
    """Create symlinks and a local verification receipt, never copy LMDB data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        help="Bundle folder; by default discover the dataset under /kaggle/input")
    args = parser.parse_args()
    try:
        dataset_root = discover_dataset_root(args.dataset_root)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"dataset_root": str(dataset_root)}), flush=True)
    receipt = bootstrap(dataset_root, ROOT)
    print(json.dumps({"status": "complete", "verification": str(receipt)}))


if __name__ == "__main__":
    main()
