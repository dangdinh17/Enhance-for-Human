#!/usr/bin/env python3
"""Prepare QP22 video reconstructions and a verified lossless luma LMDB."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.human_data import (CACHE, READY, build_lmdb, inventory, manifest,
                                 metadata_path, source_path, verify_lmdb,
                                 verify_sources)
from src.data.vvc import VideoSpec, check_disk_space, encode_one, output_paths
from src.utils.experiment import atomic_write_json
from src.utils.hashing import file_identity, file_sha256


def prepare(directory: Path, smoke: bool = False) -> Path:
    """Run bounded source probes or complete all 126 QP22 encodes."""
    source_checks = verify_sources()
    inv = inventory()
    videos = inv["videos"]
    specs = [VideoSpec(v["split"], source_path(v), v["stem"], v["width"], v["height"], v["frames"]) for v in videos]
    _, value = manifest(22)
    if smoke:
        selected = sorted([s for s in specs if s.split == "train"], key=lambda s: s.expected_bytes)[:2]
        selected += sorted([s for s in specs if s.split == "test"], key=lambda s: s.expected_bytes)[:1]
        specs = [replace(s, frames=min(s.frames, 56 if s.split == "train" else 32)) for s in selected]
        names = {s.stem: s.frames for s in specs}
        value = {**value, "splits": {
            split: [row for row in rows if row["sequence"] in names and max(row["lq_frame_indices"]) < names[row["sequence"]]][:20]
            for split, rows in value["splits"].items()}}
        videos = [{**v, "frames": names[v["stem"]]} for v in videos if v["stem"] in names]
    directory.mkdir(parents=True, exist_ok=True)
    check_disk_space(directory, specs, [22], True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(encode_one, Path(inv["encoder"]["path"]), ROOT / "configs/codec/randomaccess_faster.cfg", directory, spec, 22, 8, 50, True, False) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    identities = []
    for spec in specs:
        paths = output_paths(directory, spec, 22)
        meta = json.loads(paths["metadata"].read_text())
        if meta["qp"] != 22 or meta["frames"] != spec.frames or len(meta["psnr_y_by_poc"]) != spec.frames:
            raise ValueError("QP22 encoder receipt mismatch")
        if paths["recon"].stat().st_size != spec.expected_bytes:
            raise ValueError("Incomplete reconstruction")
        identities.append({"sequence": spec.stem, "split": spec.split,
                           **{k: file_identity(paths[k], k in {"metadata", "bitstream"}) for k in ("recon", "metadata", "bitstream")}})
    build_lmdb(directory / "lq_QP22.lmdb", value, videos,
               {s.stem: output_paths(directory, s, 22)["recon"] for s in specs})
    if smoke:
        # Reopen actual encoded PNGs and compare every bounded smoke frame.
        import cv2
        import lmdb
        import numpy as np
        from src.data.human_data import YUVReader, key_records
        env = lmdb.open(str(directory / "lq_QP22.lmdb"), readonly=True, lock=False)
        by_name = {s.stem: s for s in specs}
        try:
            with env.begin() as txn:
                for key, (row, index) in key_records(value, "lq").items():
                    spec = by_name[row["sequence"]]
                    with YUVReader(output_paths(directory, spec, 22)["recon"], spec.width, spec.height, spec.frames) as reader:
                        decoded = cv2.imdecode(np.frombuffer(txn.get(key.encode()), np.uint8), cv2.IMREAD_UNCHANGED)
                        if not np.array_equal(decoded, reader.read(index)):
                            raise ValueError("Bounded LMDB round-trip mismatch")
        finally:
            env.close()
        receipt = {"status": "complete", "scope": "smoke", "videos": len(specs),
                   "encoded_frames": sum(s.frames for s in specs), "source_checks": source_checks,
                   "encoder_outputs": identities}
    else:
        checks = [verify_lmdb(22, "gt")]
        checks.extend(verify_lmdb(q, "lq") for q in (22, 27, 32, 37))
        receipt = {"status": "complete", "scope": "preparation", "source_checks": source_checks,
                   "inventory_sha256": file_sha256(ROOT / "data/source_inventory.json"),
                   "manifest_sha256": {str(q): file_sha256(manifest(q)[0]) for q in (22, 27, 32, 37)},
                   "videos": 126, "test_frames_per_qp": 7980,
                   "encoder_outputs": identities, "lmdb": checks,
                   "validation_note": "First 30 frames of each test_18 sequence are used for checkpoint selection."}
    path = directory / ("SMOKE.json" if smoke else "READY.json")
    atomic_write_json(path, receipt)
    return path


def main() -> None:
    """Parse the bounded/full data preparation invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=CACHE)
    args = parser.parse_args()
    if args.smoke and args.output_root.resolve() == CACHE.resolve():
        parser.error("Smoke must use a separate output root")
    prepare(args.output_root, args.smoke)


if __name__ == "__main__":
    main()
