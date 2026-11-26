"""Read-only MFQEv2 sources, explicit manifests and streaming luma LMDBs."""
from __future__ import annotations

import json
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import lmdb
import numpy as np

from src.data.stdr_mfqev2_lmdb import load_stdr_manifest
from src.utils.experiment import atomic_write_json
from src.utils.hashing import file_identity, file_sha256

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs/research/datasets/mfqev2_qp22_v1"
READY = CACHE / "READY.json"
RESERVE = 150 * 1024 ** 3


class YUVReader:
    """Seek individual Y planes in an 8-bit planar YUV420 file."""

    def __init__(self, path: Path, width: int, height: int, frames: int) -> None:
        self.path, self.width, self.height, self.frames = path, width, height, frames
        self.stride = width * height * 3 // 2
        if width % 2 or height % 2 or path.stat().st_size != self.stride * frames:
            raise ValueError(f"Invalid 8-bit YUV420 geometry/size: {path}")
        self.handle = path.open("rb")

    def read(self, index: int) -> np.ndarray:
        """Read one owned uint8 HxW array."""
        if not 0 <= index < self.frames:
            raise IndexError(index)
        self.handle.seek(index * self.stride)
        value = self.handle.read(self.width * self.height)
        if len(value) != self.width * self.height:
            raise EOFError(self.path)
        return np.frombuffer(value, dtype=np.uint8).reshape(self.height, self.width).copy()

    def close(self) -> None:
        """Close the underlying source file."""
        self.handle.close()

    def __enter__(self) -> "YUVReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def inventory() -> dict[str, Any]:
    """Read the immutable project-owned source inventory."""
    return json.loads((ROOT / "data/source_inventory.json").read_text())


def manifest(qp: int) -> tuple[Path, dict[str, Any]]:
    """Load one project-owned manifest and reject duplicate temporal samples."""
    path = ROOT / f"data/manifests/QP{qp}.json"
    contract = json.loads((ROOT / "data/dataset_contract.json").read_text())
    if file_sha256(path) != contract["manifest_sha256"][str(qp)]:
        raise ValueError("Manifest differs from the approved dataset contract")
    value = load_stdr_manifest(path)
    if value["qp"] != qp or value["counts"] != {"train": 5407, "validation": 540}:
        raise ValueError(f"Unexpected manifest counts/QP: {path}")
    groups = []
    for split, expected in (("train", 108), ("validation", 18)):
        rows = value["splits"][split]
        identities = [(r["sequence"], tuple(r["lq_keys"]), r["target_key"]) for r in rows]
        if len(set(identities)) != len(rows) or len(rows) != value["counts"][split]:
            raise ValueError(f"Duplicate samples or incorrect count: {split}")
        sequences = {r["sequence"] for r in rows}
        if len(sequences) != expected:
            raise ValueError(f"Wrong sequence count: {split}")
        groups.append(sequences)
    if groups[0] & groups[1]:
        raise ValueError("Training/validation sequence overlap")
    return path, value


def source_path(video: dict[str, Any], qp: int | None = None) -> Path:
    """Resolve the full filename; no prefix or sorted-list matching."""
    if qp is None:
        return Path(video["gt"]["path"])
    if qp == 22:
        return CACHE / video["split"] / "QP22" / f"{video['stem']}.yuv"
    return Path(video["compressed"][str(qp)]["recon"]["path"])


def metadata_path(video: dict[str, Any], qp: int) -> Path:
    """Return the corresponding encoder receipt."""
    if qp == 22:
        return CACHE / "logs" / video["split"] / "QP22" / f"{video['stem']}.json"
    return Path(video["compressed"][str(qp)]["metadata"]["path"])


def verify_sources() -> dict[str, Any]:
    """Check immutable shared identities and matching VVC protocol metadata."""
    inv = inventory()
    for name in ("encoder", "preset"):
        record = inv[name]
        if file_identity(Path(record["path"]), True) != record:
            raise ValueError(f"Changed source {name}")
    counts = defaultdict(int)
    for video in inv["videos"]:
        counts[video["split"]] += 1
        expected = video["width"] * video["height"] * 3 // 2 * video["frames"]
        for record in [video["gt"]] + [video["compressed"][str(q)]["recon"] for q in (27, 32, 37)]:
            if file_identity(Path(record["path"])) != record or record["bytes"] != expected:
                raise ValueError(f"Changed or invalid YUV: {record['path']}")
        for qp in (27, 32, 37):
            records = video["compressed"][str(qp)]
            for key in ("metadata", "bitstream"):
                item = records[key]
                if file_identity(Path(item["path"]), key == "metadata") != item:
                    raise ValueError(f"Changed encoder output: {item['path']}")
            meta = json.loads(Path(records["metadata"]["path"]).read_text())
            expected_fields = {"qp": qp, "width": video["width"], "height": video["height"],
                               "frames": video["frames"], "input_bitdepth": 8,
                               "reconstruction_bitdepth": 8, "frame_rate": 50,
                               "configuration": "randomaccess_faster"}
            if any(meta.get(k) != v for k, v in expected_fields.items()):
                raise ValueError(f"Encoding protocol mismatch: {video['stem']}, QP{qp}")
            if len(meta["psnr_y_by_poc"]) != video["frames"]:
                raise ValueError("Incomplete POC metadata")
    if dict(counts) != {"train": 108, "test": 18}:
        raise ValueError(f"Wrong split inventory: {dict(counts)}")
    for record in inv["shared_lmdb_files"]:
        if file_identity(Path(record["path"]), "sha256" in record) != record:
            raise ValueError(f"Changed shared LMDB: {record['path']}")
    for qp in (22, 27, 32, 37):
        manifest(qp)
    required = sum(v["gt"]["bytes"] for v in inv["videos"]
                   if not source_path(v, 22).is_file()) + 30 * 1024 ** 3
    free = shutil.disk_usage(ROOT).free
    if free < RESERVE + required:
        raise RuntimeError(f"Need {required / 1024**3:.1f} GiB + 150 GiB reserve; free {free/1024**3:.1f}")
    return {"train_videos": counts["train"], "test_videos": counts["test"],
            "estimated_additional_gib": required / 1024 ** 3, "free_gib": free / 1024 ** 3}


def key_records(value: dict[str, Any], kind: str) -> dict[str, tuple[dict[str, Any], int]]:
    """Collect each unique GT or LQ key once."""
    keys = {}
    for rows in value["splits"].values():
        for row in rows:
            pairs = [(row["target_key"], row["target_frame_index"])] if kind == "gt" else zip(row["lq_keys"], row["lq_frame_indices"])
            for key, index in pairs:
                keys[key] = (row, index)
    return keys


def verify_lmdb(qp: int, kind: str, compare_pixels: bool = True) -> dict[str, Any]:
    """Verify every key/PNG header and source pixels at sequence boundaries."""
    path, value = manifest(qp)
    db = (path.parent / value[f"{kind}_lmdb"]).resolve()
    receipt = json.loads((db / "_SUCCESS.json").read_text())
    keys = key_records(value, kind)
    if receipt["entries"] != len(keys):
        raise ValueError(f"LMDB completion count mismatch: {db}")
    videos = {(v["split"], v["stem"]): v for v in inventory()["videos"]}
    selected = {}
    for key, (row, index) in sorted(keys.items()):
        group = (key.split("/")[0], row["sequence"])
        selected.setdefault(group, [key, key])[1] = key
    probes = {k for pair in selected.values() for k in pair} if compare_pixels else set()
    env = lmdb.open(str(db), readonly=True, lock=False, readahead=False, max_readers=512)
    try:
        if env.stat()["entries"] != len(keys):
            raise ValueError(f"Unexpected LMDB keys: {db}")
        with env.begin(buffers=True) as txn:
            for key, (row, index) in keys.items():
                raw = txn.get(key.encode())
                if raw is None:
                    raise KeyError(f"Missing {key} in {db}")
                header = bytes(raw[:29])
                if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (row["width"], row["height"]) or header[24:26] != b"\x08\x00":
                    raise ValueError(f"Invalid grayscale 8-bit PNG geometry: {key}")
                if key in probes:
                    split = "train" if key.startswith("train/") else "test"
                    video = videos[(split, row["sequence"])]
                    with YUVReader(source_path(video, None if kind == "gt" else qp), video["width"], video["height"], video["frames"]) as reader:
                        pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                        if not np.array_equal(pixels, reader.read(index)):
                            raise ValueError(f"Source/LMDB pixel mismatch: {key}")
    finally:
        env.close()
    return {"qp": qp, "kind": kind, "entries": len(keys), "pixel_probes": len(probes),
            "completion_sha256": file_sha256(db / "_SUCCESS.json"),
            "data_identity": file_identity(db / "data.mdb")}


def build_lmdb(output: Path, value: dict[str, Any], videos: list[dict[str, Any]],
               paths: dict[str, Path]) -> dict[str, Any]:
    """Stream unique Y frames to a resumable temporary LMDB, then promote."""
    expected = key_records(value, "lq")
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        receipt = json.loads((output / "_SUCCESS.json").read_text())
        if receipt["entries"] != len(expected):
            raise ValueError("Existing LMDB count mismatch")
        return receipt
    partial.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(partial), map_size=96 * 1024 ** 3, max_readers=512)
    grouped = defaultdict(list)
    for key, (row, index) in expected.items():
        grouped[row["sequence"]].append((key, index))
    by_name = {v["stem"]: v for v in videos}
    count = 0
    try:
        for stem, entries in sorted(grouped.items()):
            video = by_name[stem]
            with YUVReader(paths[stem], video["width"], video["height"], video["frames"]) as reader:
                for start in range(0, len(entries), 128):
                    if shutil.disk_usage(partial).free < RESERVE:
                        raise RuntimeError("Reached 150 GiB free-space reserve")
                    with env.begin(write=True) as txn:
                        for key, index in entries[start:start + 128]:
                            old = txn.get(key.encode())
                            if old is None:
                                image = reader.read(index)
                                ok, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                                if not ok:
                                    raise RuntimeError("PNG encode failed")
                                txn.put(key.encode(), png.tobytes(), overwrite=False)
                            count += 1
                    if count % 1000 < 128:
                        print(json.dumps({"lmdb_frames": count, "total": len(expected)}), flush=True)
        env.sync(True)
        if env.stat()["entries"] != len(expected):
            raise ValueError("Unexpected final LMDB count")
    finally:
        env.close()
    (partial / "meta_info.txt").write_text("".join(f"{key} ({row['height']}, {row['width']}, 1)\n" for key, (row, _) in sorted(expected.items())))
    receipt = {"schema_version": 1, "encoding": "lossless grayscale PNG compression=1",
               "entries": len(expected), "database_bytes": (partial / "data.mdb").stat().st_size}
    atomic_write_json(partial / "_SUCCESS.json", receipt)
    partial.rename(output)
    return receipt


def verify_training_inputs(config: dict[str, Any]) -> None:
    """Bind each run to completed, verified data owned by this project."""
    if config["dataset"].get("bundle_receipt"):
        from src.data.training_bundle import verify_bundle_inputs
        verify_bundle_inputs(ROOT, config)
        manifest(config["qp"])
        return
    completion = ROOT / config["dataset"]["completion"]
    ready = json.loads(completion.read_text())
    if ready.get("status") != "complete":
        raise ValueError("Dataset is incomplete")
    if ready["inventory_sha256"] != file_sha256(ROOT / "data/source_inventory.json"):
        raise ValueError("Dataset source inventory changed")
    qp = config["qp"]
    path, _ = manifest(qp)
    if ready["manifest_sha256"][str(qp)] != file_sha256(path):
        raise ValueError("Dataset manifest changed")
    for record in ready["lmdb"]:
        if record["qp"] == qp or record["kind"] == "gt":
            if file_identity(Path(record["data_identity"]["path"])) != record["data_identity"]:
                raise ValueError("LMDB changed after verification")
