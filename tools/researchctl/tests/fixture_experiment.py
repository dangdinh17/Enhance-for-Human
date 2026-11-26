"""Small real command used to exercise researchctl end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_directory = Path(config["paths"]["run_root"]) / config["experiment"]["name"]
    run_directory.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"]["random_seed"])
    value = sum((seed + index) ** 2 for index in range(4)) / 4.0
    (run_directory / "checkpoint.bin").write_bytes(str(value).encode("ascii"))
    (run_directory / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "iteration": 1,
                "checkpoint_selection": {
                    "metric": "loss",
                    "mode": "min",
                    "best_value": value,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
