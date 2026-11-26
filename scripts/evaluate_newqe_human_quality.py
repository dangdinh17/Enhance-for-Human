#!/usr/bin/env python3
"""Evaluate the four trained Human New QE models and summarize luma quality."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.human_data import inventory, metadata_path, source_path
from src.inference.evaluation import RESULTS, checked_checkpoint, evaluate_sequence
from src.utils.experiment import atomic_write_json, load_yaml_mapping
from src.utils.hashing import file_identity, file_sha256, object_sha256

METHOD = "NewQE_Charbonnier_300k"
QPS = (22, 27, 32, 37)
CAMPAIGN = "newqe_charb_human_quality_test18_v1"
REPORTS = ROOT / "outputs/research" / CAMPAIGN
TRAINING = ROOT / "runs/research/newqe_charb_300k_qp22_37_v1"
SOURCES = ("src/inference/evaluation.py", "src/inference/window.py",
           "src/models/mfqev2_window.py", "src/models/qe_alignment_fusion.py")
METRICS = ("vvc_psnr_y", "psnr_y", "delta_psnr_y", "vvc_ssim_y", "ssim_y")


def videos() -> list[dict[str, Any]]:
    """Require the agreed complete test split."""
    selected = [v for v in inventory()["videos"] if v["split"] == "test"]
    if len(selected) != 18 or sum(v["frames"] for v in selected) != 7980:
        raise ValueError("Expected exactly 18 test sequences and 7,980 frames")
    if len({v["stem"] for v in selected}) != 18:
        raise ValueError("Duplicate sequence names in the test split")
    return selected


def verify_record(record: dict[str, Any], video: dict[str, Any], qp: int,
                  checkpoint: Path) -> None:
    """Bind a full-video result to its checkpoint, inputs and metric code."""
    expected = {"status": "complete", "scope": "full_video", "method": METHOD,
                "qp": qp, "sequence": video["stem"], "frames": video["frames"],
                "training_updates": 300000}
    if any(record.get(k) != value for k, value in expected.items()):
        raise ValueError(f"Incomplete/mismatched result: QP{qp}/{video['stem']}")
    identity = record["identity"]
    expected_files = {"gt": file_identity(source_path(video)),
                      "lq": file_identity(source_path(video, qp)),
                      "metadata": file_identity(metadata_path(video, qp), True),
                      "checkpoint": file_identity(checkpoint, True)}
    if (record["fingerprint"] != object_sha256(identity)
            or identity.get("files") != expected_files
            or identity.get("sources") != {name: file_sha256(ROOT / name) for name in SOURCES}
            or identity.get("scope") != "full_video"
            or identity.get("frame_indices") is not None
            or identity.get("method") != METHOD or identity.get("qp") != qp
            or identity.get("frames") != video["frames"]
            or identity.get("training_updates") != 300000
            or identity.get("tile_core") != 256 or identity.get("halo") != 96
            or identity.get("amp") is not True
            or identity.get("ssim") != "Gaussian11_sigma1.5_population_full_Y_8bit"):
        raise ValueError("Cached-result provenance does not match the current evaluation")
    if not all(math.isfinite(float(record[k])) for k in METRICS):
        raise ValueError("Non-finite quality metric")
    if not math.isclose(record["delta_psnr_y"], record["psnr_y"] - record["vvc_psnr_y"], abs_tol=1e-9):
        raise ValueError("Incorrect PSNR difference")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average sequence means equally; retain the compressed-video baseline."""
    seen = set()
    for row in rows:
        key = (row["qp"], row["sequence"])
        if key in seen or row["method"] != METHOD or row["scope"] != "full_video":
            raise ValueError("Duplicate, smoke, or different-model result in summary")
        if not all(math.isfinite(float(row[k])) for k in METRICS):
            raise ValueError("Non-finite quality metric in summary")
        seen.add(key)
    summaries = []
    for qp in sorted({r["qp"] for r in rows}):
        group = [r for r in rows if r["qp"] == qp]
        summaries.append({"qp": qp, "sequences": len(group),
                          "frames": sum(r["frames"] for r in group),
                          **{key: statistics.mean(r[key] for r in group) for key in METRICS},
                          "delta_ssim_y": statistics.mean(r["ssim_y"] - r["vvc_ssim_y"] for r in group)})
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write a nonempty quality table."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def publish(output: Path, scope: str, paths: list[Path],
            metrics: dict[str, Any] | None = None,
            baseline: dict[str, Any] | None = None) -> None:
    """Return normalized researchctl artifacts without starting other jobs."""
    atomic_write_json(output, {"status": "complete", "scope": scope,
        "dataset_fingerprint": os.environ["RESEARCHCTL_DATASET_HASH"],
        "metrics": metrics or {},
        "artifacts": {f"artifact_{i}": str(p.resolve()) for i, p in enumerate(paths)},
        **({"baseline": baseline} if baseline else {})})


def verify_inputs(output: Path) -> None:
    """Verify frozen checkpoints and preserve a verified partial readout."""
    rows, checkpoints, counts = [], [], {}
    selected = videos()
    for qp in QPS:
        checkpoint = checked_checkpoint(TRAINING / f"QP{qp}", qp, 300000)
        checkpoints.append({"qp": qp, "path": str(checkpoint), "sha256": file_sha256(checkpoint)})
        group = []
        for video in selected:
            path = RESULTS / METHOD / f"QP{qp}" / f"{video['stem']}.json"
            if path.exists():
                record = json.loads(path.read_text())
                verify_record(record, video, qp, checkpoint)
                group.append(record)
        counts[str(qp)] = len(group)
        if len(group) == 18:
            rows.extend(group)
    evidence = output.parent / "verified_inputs.json"
    atomic_write_json(evidence, {"checkpoints": checkpoints, "completed_new_model_pairs": counts,
                                "training": "read_only_existing_300k_checkpoints"})
    partial = output.parent / "completed_qps_mean_quality.csv"
    write_csv(partial, summarize(rows))
    publish(output, "preparation", [evidence] + ([partial] if rows else []))


def evaluate(config: dict[str, Any], output: Path, smoke: bool) -> None:
    """Evaluate only the requested New QE weights; reuse matching full results."""
    qp = int(config["qp"])
    if qp not in QPS:
        raise ValueError("Unsupported QP")
    checkpoint = checked_checkpoint(ROOT / config["checkpoint_run_dir"], qp, 300000)
    selected = videos()
    if smoke:
        selected = [min(selected, key=lambda v: v["width"] * v["height"]),
                    max(selected, key=lambda v: v["width"] * v["height"])]
    paths, rows = [], []
    cache = output.parent / "bounded_smoke" if smoke else RESULTS
    for video in selected:
        indices = [0, video["frames"] // 2, video["frames"] - 1] if smoke else None
        path = evaluate_sequence(video, qp, checkpoint,
                                cache / METHOD / f"QP{qp}" / f"{video['stem']}.json",
                                METHOD, 300000, indices)
        record = json.loads(path.read_text())
        if not smoke:
            verify_record(record, video, qp, checkpoint)
        paths.append(path)
        rows.append(record)
    metrics = {"quality.delta_psnr_y": statistics.mean(r["delta_psnr_y"] for r in rows),
               "quality.ssim_y": statistics.mean(r["ssim_y"] for r in rows)}
    publish(output, "smoke" if smoke else "full_video", paths, metrics,
            {"metrics": {"psnr_y": statistics.mean(r["vvc_psnr_y"] for r in rows)},
             "artifacts": [str(p) for p in paths]})


def aggregate(output: Path) -> None:
    """Require all 72 current-model pairs before producing the Human report."""
    rows, receipts = [], []
    for qp in QPS:
        checkpoint = checked_checkpoint(TRAINING / f"QP{qp}", qp, 300000)
        for video in videos():
            path = RESULTS / METHOD / f"QP{qp}" / f"{video['stem']}.json"
            row = json.loads(path.read_text())
            verify_record(row, video, qp, checkpoint)
            rows.append(row)
            receipts.append({"path": str(path), "sha256": file_sha256(path)})
    summary = summarize(rows)
    if len(rows) != 72 or any(r["sequences"] != 18 or r["frames"] != 7980 for r in summary):
        raise ValueError("Incomplete four-QP Human evaluation")
    REPORTS.mkdir(parents=True, exist_ok=True)
    columns = ("qp", "sequence", "frames", *METRICS)
    detailed = [{**{k: row[k] for k in columns},
                 "delta_ssim_y": row["ssim_y"] - row["vvc_ssim_y"]} for row in rows]
    per_sequence = REPORTS / "per_sequence_qp.csv"
    macro = REPORTS / "mean_quality_by_qp.csv"
    write_csv(per_sequence, detailed)
    write_csv(macro, summary)
    body = ["---\nmarp: true\nsize: 16:9\n---\n\n# Enhance for Human — New QE\n\n",
            "Charbonnier-only; 300,000 updates/model; QP22/27/32/37.\n\n",
            "---\n\n# Evaluation protocol\n\n",
            "18 sequences and 7,980 frames/QP. Best checkpoint by validation Charbonnier. ",
            "Seven adjacent Y frames; tiled inference core 256, halo 96, AMP.\n\n",
            "PSNR-Y and SSIM-Y are averaged across frames within each sequence, ",
            "then equally across the 18 sequences. Deltas use the decoded VVC frame as baseline.\n\n",
            "SSIM: Gaussian 11×11 window, sigma 1.5, population covariance, uint8 Y. ",
            "Checkpoint validation uses the first 30 frames of these test sequences; full test includes them.\n\n",
            "---\n\n# Mean quality across 18 sequences\n\n",
            "| QP | VVC PSNR-Y | New QE PSNR-Y | ΔPSNR-Y (dB) | VVC SSIM-Y | New QE SSIM-Y | ΔSSIM-Y |\n",
            "|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in summary:
        body.append(f"| {r['qp']} | {r['vvc_psnr_y']:.4f} | {r['psnr_y']:.4f} | {r['delta_psnr_y']:+.4f} | {r['vvc_ssim_y']:.6f} | {r['ssim_y']:.6f} | {r['delta_ssim_y']:+.6f} |\n")
    report = REPORTS / "report.md"
    report.write_text("".join(body), encoding="utf-8")
    plan = REPORTS / "implementation_plan.md"
    plan.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    artifacts = [per_sequence, macro, report, plan]
    done = REPORTS / "DONE.json"
    atomic_write_json(done, {"status": "complete", "method": METHOD, "pairs": 72,
                            "frames_per_qp": 7980, "averaging": "equal_sequence_mean_of_frame_means",
                            "receipts": receipts,
                            "artifacts": {str(p): file_sha256(p) for p in artifacts}})
    publish(output, "aggregate", [done, *artifacts])


def main() -> None:
    """Dispatch only inference and reporting through researchctl."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "smoke", "evaluate", "aggregate"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "verify":
        verify_inputs(args.output)
    elif args.action == "aggregate":
        aggregate(args.output)
    else:
        if args.config is None:
            parser.error("--config is required for smoke/evaluate")
        evaluate(load_yaml_mapping(args.config), args.output, args.action == "smoke")


if __name__ == "__main__":
    main()
