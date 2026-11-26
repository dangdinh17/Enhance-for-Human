"""Portable, content-verified MFQEv2 train/validation bundles."""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import lmdb

from src.data.stdr_mfqev2_lmdb import load_stdr_manifest
from src.utils.experiment import atomic_write_json
from src.utils.hashing import file_identity, file_sha256

QPS = (22, 27, 32, 37)
DATASET_ID = "anginhok/enhance-for-human-mfqe-vvc-trainval-qp22-37"


def bundle_file(root: Path, name: str) -> Path:
    """Resolve a flat bundle filename without accepting traversal."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"Invalid flat bundle filename: {name}")
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def inspect_keys(database: Path, value: dict[str, Any],
                 kind: str) -> dict[str, int]:
    """Verify the exact key set and boundary PNG headers for every sequence."""
    expected = {}
    boundaries = {}
    for rows in value["splits"].values():
        for row in rows:
            keys = [row["target_key"]] if kind == "gt" else row["lq_keys"]
            for key in keys:
                expected[key] = (row["width"], row["height"])
                group = (key.split("/")[0], row["sequence"])
                if group not in boundaries:
                    boundaries[group] = [key, key]
                boundaries[group][1] = key
    env = lmdb.open(str(database), readonly=True, lock=False, readahead=False,
                    meminit=False, subdir=database.is_dir(), max_readers=512)
    try:
        if env.stat()["entries"] != len(expected):
            raise ValueError(f"Unexpected key count in {database}")
        with env.begin(buffers=True) as txn:
            actual = {bytes(key).decode() for key in txn.cursor().iternext(values=False)}
            if actual != set(expected):
                raise ValueError(f"Unexpected key set in {database}")
            probes = {key for pair in boundaries.values() for key in pair}
            for key in probes:
                geometry = expected[key]
                raw = txn.get(key.encode())
                if raw is None:
                    raise ValueError(f"Missing key: {key}")
                header = bytes(raw[:29])
                if (len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n"
                        or struct.unpack(">II", header[16:24]) != geometry
                        or header[24:26] != b"\x08\x00"):
                    raise ValueError(f"Invalid PNG geometry/bit depth: {key}")
    finally:
        env.close()
    return {"entries": len(expected), "verified_keys": len(actual),
            "verified_png_headers": len(probes)}


def export_bundle(source: Path, output: Path) -> Path:
    """Export immutable data using hardlinks, with streaming SHA-256 checks."""
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, manifests = {}, {}
    contract = source / "data/dataset_contract.json"
    contract_value = json.loads(contract.read_text())
    shutil.copy2(contract, output / "dataset_contract.json")
    for qp in QPS:
        path = source / f"data/manifests/QP{qp}.json"
        value = load_stdr_manifest(path)
        if (value["counts"] != {"train": 5407, "validation": 540}
                or value["qp"] != qp
                or file_sha256(path) != contract_value["manifest_sha256"][str(qp)]):
            raise ValueError(f"Unexpected source manifest: {qp}")
        shutil.copy2(path, output / path.name)
        manifests[str(qp)] = {"filename": path.name, "sha256": file_sha256(path)}
        for kind in ("gt", "lq"):
            key = "gt" if kind == "gt" else f"lq_QP{qp}"
            if key in records:
                continue
            database = (path.parent / value[f"{kind}_lmdb"]).resolve()
            original = database / "data.mdb"
            target = output / f"{key}.mdb"
            if not target.exists():
                os.link(original, target)
            elif not os.path.samefile(original, target):
                raise ValueError(f"Refusing unrelated existing file: {target}")
            initial = file_identity(original)
            checks = inspect_keys(database, value, kind)
            print(json.dumps({"hashing": key, "bytes": initial["bytes"]}), flush=True)
            digest = file_sha256(original)
            if file_identity(original) != initial:
                raise ValueError("Source database changed while exporting")
            records[key] = {"filename": target.name, "sha256": digest,
                            "bytes": initial["bytes"], "kind": kind,
                            "qp": None if kind == "gt" else qp, **checks}
            print(json.dumps({"verified": key, **records[key]}), flush=True)
    receipt = {
        "schema_version": 1, "status": "complete", "scope": "train_validation",
        "dataset_id": DATASET_ID, "qps": list(QPS), "databases": records,
        "manifests": manifests, "contract_sha256": file_sha256(contract),
        "train_windows_per_qp": 5407, "validation_samples_per_qp": 540,
        "validation_overlap": "First 30 frames of each test18 sequence",
        "full_test18_included": False,
        "source_ready_sha256": file_sha256(
            source / "outputs/research/datasets/mfqev2_qp22_v1/READY.json"),
        "codec": {"name": "VVC", "configuration": "randomaccess_faster",
                  "frame_rate": 50, "bit_depth": 8},
    }
    result = output / "bundle.json"
    atomic_write_json(result, receipt)
    return result


def _link(source: Path, target: Path) -> None:
    """Make an idempotent symlink, refusing unrelated existing content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():
            if (target.is_file() and source.suffix == ".json"
                    and file_sha256(target) == file_sha256(source)):
                return
            raise ValueError(f"Refusing existing unrelated path: {target}")
    else:
        target.symlink_to(source.resolve())


def _metadata(source: Path, target: Path) -> None:
    """Keep small manifests inside the project for researchctl provenance."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != file_sha256(source):
            raise ValueError(f"Refusing unrelated metadata: {target}")
        if not target.is_symlink():
            return
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def bootstrap(bundle_root: Path, project: Path) -> Path:
    """Verify content once and attach read-only data without duplicating it."""
    bundle_root, project = bundle_root.resolve(), project.resolve()
    receipt_path = bundle_file(bundle_root, "bundle.json")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "complete" or receipt.get("qps") != list(QPS)
            or receipt.get("scope") != "train_validation"
            or set(receipt.get("databases", {}))
            != {"gt", "lq_QP22", "lq_QP27", "lq_QP32", "lq_QP37"}):
        raise ValueError("Incomplete or unsupported training bundle")
    contract = bundle_file(bundle_root, "dataset_contract.json")
    if file_sha256(contract) != receipt["contract_sha256"]:
        raise ValueError("Dataset contract checksum mismatch")
    target_contract = project / "data/dataset_contract.json"
    if target_contract.exists():
        if file_sha256(target_contract) != receipt["contract_sha256"]:
            raise ValueError("Project and bundle dataset contracts differ")
    else:
        _metadata(contract, target_contract)
    cache_path = project / "data/bundle_verified.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    same_bundle = cached.get("bundle_sha256") == file_sha256(receipt_path)
    verified = {}
    for name, record in receipt["databases"].items():
        source = bundle_file(bundle_root, record["filename"])
        identity = file_identity(source)
        if identity["bytes"] != record["bytes"]:
            raise ValueError(f"Database size mismatch: {name}")
        previous = cached.get("databases", {}).get(name, {})
        if not same_bundle or previous.get("identity") != identity:
            print(json.dumps({"verifying_sha256": name}), flush=True)
            if file_sha256(source) != record["sha256"]:
                raise ValueError(f"Database checksum mismatch: {name}")
            if file_identity(source) != identity:
                raise ValueError("Database changed during verification")
        verified[name] = {"identity": identity, "sha256": record["sha256"]}
        if name == "lq_QP22":
            target = project / "outputs/research/datasets/mfqev2_qp22_v1/lq_QP22.lmdb"
        else:
            db_name = "gt.lmdb" if name == "gt" else f"{name}.lmdb"
            target = project / "data/shared/prepared/stdr_lmdb" / db_name
        _link(source, target / "data.mdb")
    for qp in QPS:
        record = receipt["manifests"][str(qp)]
        source = bundle_file(bundle_root, record["filename"])
        if file_sha256(source) != record["sha256"]:
            raise ValueError(f"Manifest checksum mismatch: QP{qp}")
        load_stdr_manifest(source)
        _metadata(source, project / f"data/manifests/QP{qp}.json")
    _metadata(receipt_path, project / "data/bundle.json")
    verified_receipt = {
        "status": "complete", "bundle_sha256": file_sha256(receipt_path),
        "databases": verified}
    if cached != verified_receipt:
        atomic_write_json(cache_path, verified_receipt)
    return cache_path


def verify_bundle_inputs(project: Path, config: dict[str, Any]) -> None:
    """Check verified content identities without relying on source-machine paths."""
    receipt_path = project / config["dataset"]["bundle_receipt"]
    receipt = json.loads(receipt_path.read_text())
    cache = json.loads((project / "data/bundle_verified.json").read_text())
    if (receipt.get("status") != "complete" or cache.get("status") != "complete"
            or cache["bundle_sha256"] != file_sha256(receipt_path)):
        raise ValueError("Run bootstrap to verify the dataset bundle")
    qp = config["qp"]
    if qp not in QPS:
        raise ValueError(f"Unsupported QP: {qp}")
    path = project / config["dataset"]["manifests"][0]
    value = load_stdr_manifest(path)
    expected = receipt["manifests"][str(qp)]["sha256"]
    if file_sha256(path) != expected or value["qp"] != qp:
        raise ValueError("Bundle manifest mismatch")
    for kind, name in (("gt", "gt"), ("lq", f"lq_QP{qp}")):
        database = (path.parent / value[f"{kind}_lmdb"]).resolve()
        actual = file_identity(database / "data.mdb")
        if (cache["databases"][name]["identity"] != actual
                or cache["databases"][name]["sha256"]
                != receipt["databases"][name]["sha256"]):
            raise ValueError(f"Verified database changed: {name}")
