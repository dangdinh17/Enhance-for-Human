#!/usr/bin/env python3
"""Export verified training data for Kaggle, without encoding or training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.training_bundle import DATASET_ID, export_bundle


def main() -> None:
    """Export a bundle and publication metadata from an existing Human project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = export_bundle(args.source_project, args.output)
    description = (
        "MFQEv2-derived VVC luma training/validation bundle for Enhance for Human. "
        "QP22/27/32/37, 5407 seven-frame training windows and 540 validation "
        "samples per QP. 108 training sequences and first 30 frames of 18 test "
        "sequences for validation. Full test18 is NOT included. Frames are "
        "lossless 8-bit grayscale PNG inside LMDB. VVC randomaccess_faster, "
        "50 fps. Original video rights remain with their owners; no new "
        "permissive license is granted to original video material. Source: "
        "https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset . "
        "Please cite the MFQEv2 paper. Code and usage: "
        "https://github.com/dangdinh17/Enhance-for-Human . "
        "Mount this dataset and run scripts/bootstrap_kaggle.py; LMDB files "
        "are read directly from Kaggle Input without copying or extraction. "
        "See bundle.json for SHA-256 checksums and exact split provenance."
    )
    metadata = {"id": DATASET_ID, "title": "Enhance for Human MFQE VVC TrainVal",
                "licenses": [{"name": "other"}], "description": description,
                "subtitle": "Lossless Y LMDBs for QP22/27/32/37 seven-frame training"}
    (args.output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# Enhance for Human training dataset\n\n" + description + "\n",
        encoding="utf-8")
    print(json.dumps({"status": "complete", "receipt": str(receipt)}), flush=True)


if __name__ == "__main__":
    main()
