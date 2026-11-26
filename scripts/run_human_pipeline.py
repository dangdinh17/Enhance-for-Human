#!/usr/bin/env python3
"""Delegate the approved two-campaign workflow to installed researchctl.

This wrapper has no worker queue and never invokes an encoder or trainer.
researchctl alone enforces task isolation, resources, receipts and approvals.
"""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = "human_mfqev2_qp22_data_v1"
TRAIN = "newqe_charb_300k_qp22_37_v1"


def main() -> None:
    """Finish data, produce training evidence, approve, then launch training."""
    executable = "/home/lab704/.local/bin/researchctl"
    stop = []
    active = []

    def handler(signum: int, frame: Any) -> None:
        del frame
        stop.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, handler)
    steps = [
        ("production", ["run", "--campaign", DATA, "--until", "main"]),
        ("production", ["status"]),
        ("smoke", ["validate"]),
        ("smoke", ["plan", "--campaign", TRAIN]),
        ("smoke", ["preflight"]),
        ("smoke", ["run", "--campaign", TRAIN, "--until", "smoke"]),
        ("production", ["validate"]),
        ("production", ["plan", "--campaign", TRAIN]),
        ("production", ["preflight"]),
        ("production", ["run", "--campaign", TRAIN, "--until", "smoke"]),
        ("production", ["approve", "--campaign", TRAIN, "--stage", "main"]),
        ("production", ["launch", "--campaign", TRAIN]),
    ]
    for profile, arguments in steps:
        if stop:
            raise SystemExit(128 + stop[-1])
        campaign = DATA if DATA in arguments else TRAIN
        command = [executable, "--project-root", str(ROOT), "--profile", profile] + arguments
        print("Delegating: " + " ".join(command), flush=True)
        child = subprocess.Popen(command, cwd=ROOT)
        active[:] = [child]
        # Signal handler only sets a flag; no nested wait or campaign transition.
        while child.poll() is None:
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if stop:
                    subprocess.run([executable, "--project-root", str(ROOT), "--profile", profile,
                                    "cancel", "--campaign", campaign], check=False)
                    child.terminate()
                    child.wait(timeout=90)
        if stop:
            raise SystemExit(128 + stop[-1])
        if child.returncode:
            raise SystemExit(child.returncode)
    print("Training/test/report delegated to the persistent researchctl controller.", flush=True)


if __name__ == "__main__":
    main()
