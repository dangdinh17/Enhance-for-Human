"""Verified 18-sequence rate-distortion tables and slide-format reporting."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.data.human_data import inventory
from src.inference.evaluation import RESULTS
from src.utils.experiment import atomic_write_json
from src.utils.hashing import file_sha256


def bd_rate(anchor: list[tuple[float, float]], enhanced: list[tuple[float, float]]) -> float:
    """Integrate PCHIP log-bitrate over the shared PSNR interval."""
    curves = []
    for pairs in (anchor, enhanced):
        ordered = sorted(pairs, key=lambda row: row[1])
        rate, psnr = np.array(ordered, dtype=np.float64).T
        if len(rate) != 4 or np.any(rate <= 0) or np.any(np.diff(psnr) <= 0) or np.any(np.diff(rate) <= 0):
            raise ValueError("BD-rate needs four distinct monotonic RD points")
        curves.append((psnr, np.log(rate)))
    lo, hi = max(c[0][0] for c in curves), min(c[0][-1] for c in curves)
    if hi <= lo:
        raise ValueError("RD curves have no common quality interval")
    area = [PchipInterpolator(x, y).integrate(lo, hi) / (hi - lo) for x, y in curves]
    return (math.exp(area[1] - area[0]) - 1) * 100


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write uniform rows as an atomic CSV artifact."""
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(directory: Path = RESULTS) -> Path:
    """Require all 72 new and 54 reference pairs before publishing results."""
    sequences = {v["stem"]: v["frames"] for v in inventory()["videos"] if v["split"] == "test"}
    rows, receipts = [], []
    for method, qps in (("NewQE_Charbonnier_300k", (22, 27, 32, 37)), ("NewQE_JDE_100k", (27, 32, 37))):
        for qp in qps:
            for sequence, frames in sequences.items():
                path = directory / method / f"QP{qp}" / f"{sequence}.json"
                record = json.loads(path.read_text())
                if record["status"] != "complete" or record["scope"] != "full_video" or record["frames"] != frames or record["qp"] != qp or record["sequence"] != sequence:
                    raise ValueError(f"Incomplete/mismatched evaluation: {path}")
                receipts.append({"path": str(path), "sha256": file_sha256(path)})
                rows.append({k: record[k] for k in ("method", "sequence", "qp", "frames", "training_updates", "bitrate_kbps", "psnr_y", "vvc_psnr_y", "delta_psnr_y", "ssim_y", "vvc_ssim_y", "fps", "peak_vram_mib")})
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "per_sequence_qp.csv", rows)
    macro = []
    for method in {r["method"] for r in rows}:
        for qp in sorted({r["qp"] for r in rows if r["method"] == method}):
            group = [r for r in rows if r["method"] == method and r["qp"] == qp]
            macro.append({"method": method, "qp": qp, "sequences": len(group),
                          **{k: float(np.mean([r[k] for r in group])) for k in ("psnr_y", "vvc_psnr_y", "delta_psnr_y", "ssim_y", "fps", "peak_vram_mib")}})
    write_csv(directory / "macro18.csv", macro)
    rates = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sequence in sequences:
        group = [r for r in rows if r["method"] == "NewQE_Charbonnier_300k" and r["sequence"] == sequence]
        baseline = [(r["bitrate_kbps"], r["vvc_psnr_y"]) for r in group]
        improved = [(r["bitrate_kbps"], r["psnr_y"]) for r in group]
        rates.append({"sequence": sequence, "bd_rate_psnr_percent": bd_rate(baseline, improved)})
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, pairs in (("VVC", baseline), ("New QE Charbonnier 300k", improved)):
            pairs = sorted(pairs)
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "o-", label=name)
        jde = sorted((r["bitrate_kbps"], r["psnr_y"]) for r in rows if r["sequence"] == sequence and r["method"] == "NewQE_JDE_100k")
        ax.plot([p[0] for p in jde], [p[1] for p in jde], "s--", label="New QE JDE 100k (3 QPs)")
        ax.set(xlabel="Bitrate (kbps)", ylabel="Mean frame PSNR-Y (dB)", title=sequence)
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"RD_{sequence}.png", dpi=150)
        plt.close(fig)
    write_csv(directory / "bd_rate_per_sequence.csv", rates)
    body = ['---\nmarp: true\nsize: 16:9\n---\n\n# Enhance for Human — New QE\n',
            'Charbonnier-only: 300,000 successful updates/model; JDE reference: 100,000.\n',
            '> Validation selects checkpoints using the first 30 frames of each test_18 sequence. Full-test results include those frames and are not an independent hold-out estimate.\n',
            '\n---\n\n# Protocol\n\n7 adjacent Y frames → enhanced current frame. Best by validation Charbonnier. QP22/27/32/37; tiled inference core 256, halo 96, AMP, uint8 Y.\n',
            '\nPSNR is mean per-frame PSNR. SSIM uses an 11×11 Gaussian window, sigma 1.5, population covariance. FPS includes temporal-feature reads and GPU synchronization; excludes metric computation and model load. VRAM is peak allocated torch memory.\n',
            '\nBD-rate uses four-point PCHIP log-rate integration over shared PSNR. Old JDE has only three common QPs; no four-point JDE BD-rate is reported.\n',
            '\n---\n\n# Mean across 18 sequences\n\n| Method | QP | PSNR-Y | ΔPSNR | SSIM-Y | FPS |\n|---|---:|---:|---:|---:|---:|\n']
    body += [f"| {r['method']} | {r['qp']} | {r['psnr_y']:.4f} | {r['delta_psnr_y']:+.4f} | {r['ssim_y']:.5f} | {r['fps']:.3f} |\n" for r in sorted(macro, key=lambda r: (r['method'], r['qp']))]
    body += ['\n---\n\n# BD-rate versus VVC\n\n| Sequence | BD-rate PSNR (%) |\n|---|---:|\n']
    body += [f"| {r['sequence']} | {r['bd_rate_psnr_percent']:+.3f} |\n" for r in rates]
    body += [f"\nMacro-18 BD-rate: {np.mean([r['bd_rate_psnr_percent'] for r in rates]):+.3f}%.\n"]
    (directory / "report.md").write_text("".join(body))
    completion = directory / "DONE.json"
    atomic_write_json(completion, {"status": "complete", "new_pairs": 72, "reference_pairs": 54,
                                  "frames_per_qp": 7980, "receipts": receipts,
                                  "report_sha256": file_sha256(directory / "report.md")})
    return completion
