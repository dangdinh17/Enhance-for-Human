"""Full-sequence luma evaluation with content-bound cached measurements."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.data.human_data import (ROOT, YUVReader, inventory, metadata_path,
                                 source_path)
from src.inference.window import CachedWindowEnhancer
from src.models.mfqev2_window import load_mfqev2_window
from src.utils.experiment import atomic_write_json
from src.utils.hashing import file_identity, file_sha256, object_sha256

RESULTS = ROOT / "outputs/research/newqe_charb_300k_qp22_37_v1/test18"
LEGACY = Path("/work/lab704/Enhance_for_Machine/runs/research/mfqev2_window_ampfix_scratch_v1")


def checked_checkpoint(directory: Path, qp: int, updates: int) -> Path:
    """Require a complete training receipt and the exact selected best weights."""
    if directory.parent == LEGACY:
        if file_sha256(directory / "DONE.json") != file_sha256(ROOT / f"data/reference_receipts/QP{qp}.json"):
            raise ValueError("Legacy training receipt changed after campaign configuration")
    done = json.loads((directory / "DONE.json").read_text())
    best, last = directory / "checkpoints/best.pth", directory / "checkpoints/last.pt"
    if done["iteration"] != updates or done["training_qps"] != [qp]:
        raise ValueError("Training is incomplete or belongs to a different QP")
    if done["checkpoint_sha256"] != file_sha256(best) or done["last_checkpoint_sha256"] != file_sha256(last):
        raise ValueError("Training receipt/checkpoint SHA mismatch")
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    if checkpoint["iteration"] != done["best_iteration"] or checkpoint["training_qps"] != [qp]:
        raise ValueError("Best-checkpoint selection mismatch")
    return best


def quality(gt: np.ndarray, image: np.ndarray) -> tuple[float, float, float]:
    """Measure PSNR and fixed Gaussian-window SSIM on the full uint8 Y plane."""
    mse = float(np.square(gt.astype(np.float64) - image.astype(np.float64)).mean())
    psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-12))
    ssim = float(structural_similarity(gt, image, data_range=255,
                                      gaussian_weights=True, sigma=1.5,
                                      win_size=11, use_sample_covariance=False))
    return mse, psnr, ssim


def evaluate_sequence(video: dict[str, Any], qp: int, checkpoint: Path,
                      output: Path, method: str, budget: int,
                      indices: list[int] | None = None) -> Path:
    """Evaluate one exact sequence/QP pair, reusing only matching receipts."""
    files = {"gt": file_identity(source_path(video)),
             "lq": file_identity(source_path(video, qp)),
             "metadata": file_identity(metadata_path(video, qp), True),
             "checkpoint": file_identity(checkpoint, True)}
    identity = {"files": files, "qp": qp, "method": method, "training_updates": budget,
                "tile_core": 256, "halo": 96, "amp": True,
                "scope": "smoke" if indices is not None else "full_video",
                "frame_indices": indices, "frames": video["frames"],
                "sources": {name: file_sha256(ROOT / name) for name in (
                    "src/inference/evaluation.py", "src/inference/window.py",
                    "src/models/mfqev2_window.py", "src/models/qe_alignment_fusion.py")},
                "ssim": "Gaussian11_sigma1.5_population_full_Y_8bit"}
    fingerprint = object_sha256(identity)
    if output.exists():
        cached = json.loads(output.read_text())
        if cached.get("fingerprint") == fingerprint and cached.get("status") == "complete":
            return output
    device = "cuda:0"
    model = load_mfqev2_window(checkpoint, device)
    torch.cuda.reset_peak_memory_stats()
    selected = list(range(video["frames"])) if indices is None else indices
    totals = np.zeros((2, 3), dtype=np.float64)
    inference_seconds = 0.0
    with YUVReader(source_path(video), video["width"], video["height"], video["frames"]) as gt_reader, YUVReader(source_path(video, qp), video["width"], video["height"], video["frames"]) as lq_reader:
        enhancer = CachedWindowEnhancer(model, lq_reader.read, video["frames"], 256, 96, True)
        for count, index in enumerate(selected, 1):
            gt, lq = gt_reader.read(index), lq_reader.read(index)
            torch.cuda.synchronize()
            start = time.perf_counter()
            enhanced = enhancer(index)
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            totals[0] += quality(gt, lq)
            totals[1] += quality(gt, enhanced)
            if count % 100 == 0:
                print(json.dumps({"method": method, "qp": qp, "sequence": video["stem"], "frames": count, "total": len(selected)}), flush=True)
    meta = json.loads(metadata_path(video, qp).read_text())
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    del enhancer, model
    torch.cuda.empty_cache()
    totals /= len(selected)
    result = {"status": "complete", "fingerprint": fingerprint, "identity": identity,
              "scope": identity["scope"], "method": method, "sequence": video["stem"],
              "qp": qp, "frames": len(selected), "training_updates": budget,
              "bitrate_kbps": meta["bitrate_kbps"], "psnr_y": float(totals[1, 1]),
              "ssim_y": float(totals[1, 2]), "vvc_psnr_y": float(totals[0, 1]),
              "vvc_ssim_y": float(totals[0, 2]), "delta_psnr_y": float(totals[1, 1] - totals[0, 1]),
              "global_psnr_y": 10 * math.log10(255 ** 2 / max(totals[1, 0], 1e-12)),
              "fps": len(selected) / inference_seconds,
              "inference_seconds": inference_seconds, "peak_vram_mib": peak}
    atomic_write_json(output, result)
    return output


def evaluate_qp(config: dict[str, Any], directory: Path = RESULTS,
                smoke: bool = False) -> list[Path]:
    """Evaluate current best and, where available, the old JDE-loss model."""
    qp = config["qp"]
    current = checked_checkpoint(ROOT / config["run_dir"], qp, config["train"]["iterations"])
    videos = [v for v in inventory()["videos"] if v["split"] == "test"]
    if smoke:
        small = min(videos, key=lambda v: v["width"] * v["height"])
        large = max(videos, key=lambda v: v["width"] * v["height"])
        videos = [small, large]
    paths = []
    methods = [("NewQE_Charbonnier_300k", current, config["train"]["iterations"])]
    if not smoke and qp != 22:
        methods.append(("NewQE_JDE_100k", checked_checkpoint(LEGACY / f"QP{qp}", qp, 100000), 100000))
    for method, checkpoint, budget in methods:
        for video in videos:
            indices = [0, video["frames"] // 2, video["frames"] - 1] if smoke else None
            paths.append(evaluate_sequence(video, qp, checkpoint,
                         directory / method / f"QP{qp}" / f"{video['stem']}.json",
                         method, budget, indices))
    return paths
