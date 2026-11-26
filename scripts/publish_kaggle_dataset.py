#!/usr/bin/env python3
"""Publish the explicitly requested public bundle and verify remote file sizes."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from requests.exceptions import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.experiment import atomic_write_json
from src.utils.hashing import file_sha256


def main() -> None:
    """Use Kaggle's resumable uploader; never silently overwrite a dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    folder = args.folder.resolve()
    metadata = json.loads((folder / "dataset-metadata.json").read_text())
    bundle = json.loads((folder / "bundle.json").read_text())
    dataset = metadata["id"]
    if dataset != bundle["dataset_id"] or bundle["status"] != "complete":
        raise ValueError("Dataset publication identity mismatch")
    expected = {p.name: p.stat().st_size for p in folder.iterdir()
                if p.is_file() and p.name != "dataset-metadata.json"}
    api = KaggleApi()
    api.authenticate()
    if dataset.split("/")[0] != api.get_config_value("username"):
        raise ValueError("Authenticated Kaggle owner does not match dataset")
    owned = []
    for page in range(1, 1001):
        rows = api.dataset_list(mine=True, page=page) or []
        owned.extend(rows)
        if len(rows) < 20:
            break
    else:
        raise RuntimeError("Could not enumerate owned datasets safely")
    owned_refs = {record.ref for record in owned}
    preflight = {
        "dataset": dataset, "public": True, "upload_bytes": sum(expected.values()),
        "largest_file_bytes": max(expected.values()),
        "owned_datasets": len(owned),
        "owned_bytes": sum(d.total_bytes for d in owned),
        "storage_quota": "Not exposed by Kaggle SDK; enforced by server on upload/create",
        "bundle_sha256": file_sha256(folder / "bundle.json"),
    }
    atomic_write_json(args.receipt.with_name("upload_preflight.json"), preflight)
    print(json.dumps({"preflight": preflight}), flush=True)
    exists = True
    try:
        api.dataset_status(dataset)
    except HTTPError as error:
        # Kaggle returns 403 as well as 404 for a dataset that does not exist.
        # An owned dataset with denied status must never be recreated.
        if (error.response is None
                or error.response.status_code not in {403, 404}
                or dataset in owned_refs):
            raise
        exists = False
    if not exists:
        if args.verify_only:
            raise ValueError("Dataset does not exist yet")
        response = api.dataset_create_new(str(folder), public=True, quiet=False,
                                          convert_to_csv=False, dir_mode="skip")
        if response.error:
            raise RuntimeError(response.error)
        atomic_write_json(args.receipt, {
            "status": "processing", "dataset": dataset, "url": response.url,
            "bundle_sha256": preflight["bundle_sha256"]})
    for attempt in range(240):
        state = json.loads(api.dataset_status(dataset, format="json"))
        print(json.dumps({"remote": state, "attempt": attempt}), flush=True)
        if state["status"] in {"ready", "complete"}:
            break
        if state["status"] in {"error", "failed"}:
            raise RuntimeError(f"Kaggle processing failed: {state['status']}")
        time.sleep(30)
    else:
        raise TimeoutError("Dataset processing exceeded two hours; rerun --verify-only")
    remote, page = {}, None
    while True:
        response = api.dataset_list_files(dataset, page_token=page, page_size=200)
        if response.error_message:
            raise RuntimeError(response.error_message)
        for record in response.dataset_files:
            remote[record.name] = record.total_bytes
        page = response.next_page_token
        if not page:
            break
    if remote != expected:
        raise ValueError(f"Remote filenames/sizes differ: expected={expected}, actual={remote}")
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = dataset.split("/")
    with api.build_kaggle_client() as client:
        published = client.datasets.dataset_api_client.get_dataset(request)
    if published.is_private:
        raise ValueError("Dataset exists but is not public")
    with tempfile.TemporaryDirectory(prefix="human-kaggle-verify-") as temporary:
        api.dataset_download_file(dataset, "bundle.json", path=temporary,
                                  force=True, quiet=True)
        downloaded = Path(temporary) / "bundle.json"
        if not downloaded.exists():
            archive = Path(temporary) / "bundle.json.zip"
            with zipfile.ZipFile(archive) as zipped:
                downloaded.write_bytes(zipped.read("bundle.json"))
        if file_sha256(downloaded) != preflight["bundle_sha256"]:
            raise ValueError("Published bundle receipt checksum differs")
    atomic_write_json(args.receipt, {
        "status": "complete", "dataset": dataset,
        "url": f"https://www.kaggle.com/datasets/{dataset}",
        "version": state["current_version_number"], "public_verified": True,
        "bundle_sha256": preflight["bundle_sha256"], "verified_files": remote,
        "remote_verification": "Ready, public, exact names/sizes, downloaded bundle.json SHA-256",
        "content_verification": "Local SHA-256 in bundle.json; bootstrap verifies downloaded content",
    })
    print(json.dumps({"status": "complete", "receipt": str(args.receipt)}), flush=True)


if __name__ == "__main__":
    main()
