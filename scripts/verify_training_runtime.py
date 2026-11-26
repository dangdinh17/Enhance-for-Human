#!/usr/bin/env python3
"""Bounded GPU/worker/checkpoint verification on real shared QP27 windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.human_data import manifest
from src.data.window_mixed import (EpochWindowBatches, MixedQPWindows,
                                  close_window_environments)
from src.loss.charbonnier import MFQEv2CombinedLoss
from src.models.mfqev2_window import MFQEv2Window
from src.train.window_trainer import WindowTrainer
from src.utils.experiment import atomic_write_json
from src.utils.training import (capture_random_state, restore_random_state,
                                seed_everything)
from train_mfqev2_window import worker_init


def verify(directory: Path) -> Path:
    """Check the next Adam update matches after a full state reload."""
    seed_everything(34)
    torch.set_num_threads(4)
    dataset = MixedQPWindows([manifest(27)[0]], 128)
    loader = DataLoader(dataset, batch_sampler=EpochWindowBatches(5407, 8, 34, 0, 2),
                        num_workers=4, multiprocessing_context="spawn",
                        persistent_workers=True, worker_init_fn=worker_init,
                        generator=torch.Generator().manual_seed(34))
    iterator = iter(loader)
    try:
        batches = [next(iterator), next(iterator)]
        if any(sorted(set(b["qp"].tolist())) != [27] for b in batches):
            raise ValueError("Unexpected QP in runtime probe")
        model = MFQEv2Window().cuda()
        trainer = WindowTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-4),
                                MFQEv2CombinedLoss(), "cuda:0", True, 1.)
        trainer.train_batch(batches[0])
        state_path = directory / "runtime_state.pt"
        torch.save({"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                    "scaler": trainer.scaler.state_dict(), "rng": capture_random_state()}, state_path)
        trainer.train_batch(batches[1])
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.scaler.load_state_dict(saved["scaler"])
        restore_random_state(saved["rng"])
        trainer.train_batch(batches[1])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key], rtol=1e-5, atol=1e-7)
        if any("jde" in name.lower() for name in sys.modules):
            raise ValueError("A JDE module was imported")
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("Nonfinite runtime weights")
        result = directory / "runtime_checks.json"
        atomic_write_json(result, {"status": "complete", "scope": "smoke", "workers": 4,
                                   "qp": 27, "parameters": sum(p.numel() for p in model.parameters()),
                                   "gradient_norm": trainer.last_gradient_norm,
                                   "amp_scale": trainer.scaler.get_scale(),
                                   "reload_next_update_matches": True, "jde_imported": False})
        return result
    finally:
        iterator._shutdown_workers()
        close_window_environments()


def main() -> None:
    """Run only when scheduled as a bounded researchctl GPU task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output_root)


if __name__ == "__main__":
    main()
