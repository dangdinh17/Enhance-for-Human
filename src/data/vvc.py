#!/usr/bin/env python3
"""Encode the MFQEv2 train/test videos with VVenC at multiple QPs.

The output layout mirrors the NAFOSTED project::

    data/MFQEv2/
    |-- train/{original,QP27,QP32,QP37,QP42}/
    |-- test/{original,QP27,QP32,QP37,QP42}/
    |-- bitstreams/{train,test}/QP*/
    `-- logs/{train,test}/QP*/

Raw reconstructions are written atomically and validated by their expected
YUV420 8-bit byte size. Completed jobs are skipped on subsequent runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "data" / "MFQEv2"
DEFAULT_ENCODER = (
    ROOT / "third_party" / "vvenc" / "bin" / "release-static" / "vvencFFapp"
)
DEFAULT_CONFIG = ROOT / "third_party" / "vvenc" / "cfg" / "randomaccess_faster.cfg"
DEFAULT_QPS = (27, 32, 37, 42)
DEFAULT_FRAME_RATE = 50
MIN_FREE_SPACE_GIB = 150
VIDEO_RE = re.compile(r"^(?P<name>.+)_(?P<width>\d+)x(?P<height>\d+)_(?P<frames>\d+)\.yuv$")
POC_RE = re.compile(r"POC\s+(\d+).*?\[Y\s+([0-9.]+)\s+dB")


@dataclass(frozen=True)
class VideoSpec:
    """Metadata parsed from one MFQEv2 raw-video filename."""

    split: str
    path: Path
    stem: str
    width: int
    height: int
    frames: int

    @property
    def expected_bytes(self) -> int:
        """Return the exact byte count for an 8-bit YUV420 video."""

        return self.width * self.height * 3 // 2 * self.frames


def parse_video(path: Path, split: str) -> VideoSpec:
    """Parse and validate a raw MFQEv2 video."""

    match = VIDEO_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unsupported MFQEv2 filename: {path.name}")
    spec = VideoSpec(
        split=split,
        path=path,
        stem=path.stem,
        width=int(match.group("width")),
        height=int(match.group("height")),
        frames=int(match.group("frames")),
    )
    actual_bytes = path.stat().st_size
    if actual_bytes != spec.expected_bytes:
        raise ValueError(
            f"Wrong YUV size for {path}: {actual_bytes} != {spec.expected_bytes}"
        )
    return spec


def discover_videos(dataset_root: Path, selected_split: str | None) -> list[VideoSpec]:
    """Discover all validated train and test videos in deterministic order."""

    source_dirs = {
        "train": dataset_root / "train_108" / "raw",
        "test": dataset_root / "test_18" / "raw",
    }
    videos: list[VideoSpec] = []
    for split, source_dir in source_dirs.items():
        if selected_split is not None and split != selected_split:
            continue
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing MFQEv2 source directory: {source_dir}")
        videos.extend(parse_video(path, split) for path in sorted(source_dir.glob("*.yuv")))
    return videos


def ensure_layout(dataset_root: Path, qps: Sequence[int]) -> None:
    """Create the NAFOSTED-style output tree and original-data links."""

    source_names = {"train": "train_108", "test": "test_18"}
    for split, source_name in source_names.items():
        split_root = dataset_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        original = split_root / "original"
        expected_target = Path("..") / source_name / "raw"
        if original.is_symlink():
            if Path(os.readlink(original)) != expected_target:
                raise RuntimeError(f"Unexpected symlink target for {original}")
        elif original.exists():
            raise FileExistsError(f"Cannot create original-data link; path exists: {original}")
        else:
            original.symlink_to(expected_target, target_is_directory=True)
        for qp in qps:
            (split_root / f"QP{qp}").mkdir(parents=True, exist_ok=True)
            (dataset_root / "bitstreams" / split / f"QP{qp}").mkdir(
                parents=True, exist_ok=True
            )
            (dataset_root / "logs" / split / f"QP{qp}").mkdir(
                parents=True, exist_ok=True
            )


def output_paths(dataset_root: Path, video: VideoSpec, qp: int) -> dict[str, Path]:
    """Return final and temporary paths for one encoding task."""

    recon = dataset_root / video.split / f"QP{qp}" / f"{video.stem}.yuv"
    bitstream = (
        dataset_root / "bitstreams" / video.split / f"QP{qp}" / f"{video.stem}.266"
    )
    log_path = dataset_root / "logs" / video.split / f"QP{qp}" / f"{video.stem}.log"
    metadata = log_path.with_suffix(".json")
    return {
        "recon": recon,
        "bitstream": bitstream,
        "log": log_path,
        "metadata": metadata,
        "recon_tmp": recon.with_suffix(".yuv.part"),
        "bitstream_tmp": bitstream.with_suffix(".266.part"),
    }


def is_complete(paths: dict[str, Path], video: VideoSpec, keep_recon: bool) -> bool:
    """Return whether one output set is complete and validated."""

    bitstream_ok = paths["bitstream"].is_file() and paths["bitstream"].stat().st_size > 0
    metadata_ok = paths["metadata"].is_file()
    recon_ok = not keep_recon or (
        paths["recon"].is_file()
        and paths["recon"].stat().st_size == video.expected_bytes
    )
    return bitstream_ok and metadata_ok and recon_ok


def check_disk_space(
    dataset_root: Path,
    videos: Sequence[VideoSpec],
    qps: Sequence[int],
    keep_recon: bool,
) -> None:
    """Fail before encoding if known reconstruction outputs cannot fit."""

    if not keep_recon:
        return
    required = 0
    for video in videos:
        for qp in qps:
            paths = output_paths(dataset_root, video, qp)
            if not (
                paths["recon"].is_file()
                and paths["recon"].stat().st_size == video.expected_bytes
            ):
                required += video.expected_bytes
    free = shutil.disk_usage(dataset_root).free
    reserve = MIN_FREE_SPACE_GIB * 1024**3
    if free < required + reserve:
        raise RuntimeError(
            "Insufficient disk space for VVC reconstructions: "
            f"need {required / 1024**3:.1f} GiB plus {MIN_FREE_SPACE_GIB} GiB reserve, "
            f"but only {free / 1024**3:.1f} GiB is free"
        )


def encode_one(
    encoder: Path,
    config: Path,
    dataset_root: Path,
    video: VideoSpec,
    qp: int,
    threads: int,
    frame_rate: int,
    keep_recon: bool,
    force: bool,
) -> dict[str, object]:
    """Encode one video/QP pair with atomic output promotion."""

    paths = output_paths(dataset_root, video, qp)
    if not force and is_complete(paths, video, keep_recon):
        return {"split": video.split, "sequence": video.stem, "qp": qp, "status": "cached"}

    for key in ("recon", "bitstream", "log", "metadata"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    for key in ("bitstream_tmp", "recon_tmp"):
        paths[key].unlink(missing_ok=True)

    command = [
        str(encoder),
        "-c",
        str(config),
        "-i",
        str(video.path),
        "-s",
        f"{video.width}x{video.height}",
        "--InputBitDepth",
        "8",
        "-fr",
        str(frame_rate),
        "-f",
        str(video.frames),
        "-q",
        str(qp),
        "-b",
        str(paths["bitstream_tmp"]),
        "-t",
        str(threads),
        "--MTProfile",
        "1",
        "--PerceptQPA",
        "0",
        "-v",
        "5",
    ]
    if keep_recon:
        command.extend(["-o", str(paths["recon_tmp"]), "--OutputBitDepth", "8"])

    started_at = time.time()
    with paths["log"].open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"VVenC failed for {video.stem} QP{qp} with exit code {result.returncode}; "
            f"see {paths['log']}"
        )
    if not paths["bitstream_tmp"].is_file() or paths["bitstream_tmp"].stat().st_size == 0:
        raise RuntimeError(f"VVenC produced no bitstream for {video.stem} QP{qp}")
    if keep_recon and paths["recon_tmp"].stat().st_size != video.expected_bytes:
        raise RuntimeError(
            f"Wrong reconstruction size for {video.stem} QP{qp}: "
            f"{paths['recon_tmp'].stat().st_size} != {video.expected_bytes}"
        )

    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    psnr_by_poc = {int(poc): float(value) for poc, value in POC_RE.findall(log_text)}
    if len(psnr_by_poc) != video.frames:
        raise RuntimeError(
            f"Parsed {len(psnr_by_poc)}/{video.frames} POC PSNR values for "
            f"{video.stem} QP{qp}"
        )

    os.replace(paths["bitstream_tmp"], paths["bitstream"])
    if keep_recon:
        os.replace(paths["recon_tmp"], paths["recon"])
    elapsed = time.time() - started_at
    bitstream_bytes = paths["bitstream"].stat().st_size
    metadata = {
        **asdict(video),
        "path": str(video.path),
        "codec": "VVC",
        "encoder": "VVenC 1.14.0",
        "configuration": config.stem,
        "qp": qp,
        "frame_rate": frame_rate,
        "input_bitdepth": 8,
        "reconstruction_bitdepth": 8 if keep_recon else None,
        "bitstream_bytes": bitstream_bytes,
        "bitrate_kbps": bitstream_bytes * 8.0 * frame_rate / video.frames / 1000.0,
        "encode_seconds": elapsed,
        "psnr_y_by_poc": [psnr_by_poc[poc] for poc in range(video.frames)],
    }
    from src.utils.experiment import atomic_write_json
    atomic_write_json(paths["metadata"], metadata)
    return {
        "split": video.split,
        "sequence": video.stem,
        "qp": qp,
        "status": "done",
        "seconds": round(elapsed, 3),
        "bitstream_bytes": bitstream_bytes,
    }

