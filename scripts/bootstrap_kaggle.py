#!/usr/bin/env python3
"""Verify and attach the Kaggle train/validation dataset to this checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.training_bundle import bootstrap


def main() -> None:
    """Create symlinks and a local verification receipt, never copy LMDB data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/kaggle/input/enhance-for-human-mfqe-vvc-trainval-qp22-37"))
    args = parser.parse_args()
    receipt = bootstrap(args.dataset_root, ROOT)
    print(json.dumps({"status": "complete", "verification": str(receipt)}))


if __name__ == "__main__":
    main()
