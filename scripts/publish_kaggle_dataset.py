#!/usr/bin/env python3
"""Publish the explicitly requested public bundle and verify remote file sizes."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from requests.exceptions import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.experiment import atomic_write_json
from src.utils.hashing import file_sha256


class ParallelDatasetApi(KaggleApi):
    """Upload independent flat files with bounded, resumable SDK workers."""

    def __init__(self, workers: int) -> None:
        super().__init__()
        self.workers = workers
        self.log_lock = threading.Lock()

    def upload_files(self, request: Any, resources: Any, folder: str,
                     blob_type: Any, upload_context: Any,
                     quiet: bool = False, dir_mode: str = "skip",
                     ignore_patterns: list[str] | None = None) -> None:
        """Retain SDK tokens and context cleanup; only file transport is parallel."""
        if self.workers == 1:
            return super().upload_files(request, resources, folder, blob_type,
                                        upload_context, quiet, dir_mode,
                                        ignore_patterns=ignore_patterns)
        if ignore_patterns:
            raise ValueError("Bundle publication requires every declared file")
        if dir_mode != "skip":
            raise ValueError("Parallel dataset publication requires flat files")
        skipped = {
            self.DATASET_METADATA_FILE, self.OLD_DATASET_METADATA_FILE,
            *self.DATASET_COVER_IMAGE_FILES, self.KERNEL_METADATA_FILE,
            self.MODEL_METADATA_FILE, self.MODEL_INSTANCE_METADATA_FILE,
        }
        paths = sorted(
            (path for path in Path(folder).iterdir()
             if path.is_file() and path.name not in skipped),
            key=lambda path: (-path.stat().st_size, path.name))

        def upload(path: Path) -> tuple[str, Any]:
            with self.log_lock:
                print(json.dumps({"upload_started": path.name,
                                  "bytes": path.stat().st_size}), flush=True)
            uploaded = self._upload_file(
                path.name, str(path), blob_type, upload_context, True, resources)
            if uploaded is None:
                raise RuntimeError(f"File upload failed: {path.name}")
            with self.log_lock:
                print(json.dumps({"upload_completed": path.name,
                                  "bytes": path.stat().st_size}), flush=True)
            return path.name, self._new_file(uploaded)

        files = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(upload, path) for path in paths]
            for future in as_completed(futures):
                name, uploaded = future.result()
                files[name] = uploaded
        request.files.extend(files[name] for name in sorted(files))


def get_published_dataset(api: KaggleApi, dataset: str) -> Any:
    """Read dataset metadata without the separately restricted status API."""
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = dataset.split("/")
    with api.build_kaggle_client() as client:
        return client.datasets.dataset_api_client.get_dataset(request)


def get_published_files(api: KaggleApi, dataset: str) -> dict[str, int]:
    """Enumerate every published file, including paginated bundles."""
    remote, page = {}, None
    while True:
        response = api.dataset_list_files(dataset, page_token=page, page_size=200)
        if response.error_message:
            raise RuntimeError(response.error_message)
        for record in response.dataset_files:
            remote[record.name] = record.total_bytes
        page = response.next_page_token
        if not page:
            return remote


def main() -> None:
    """Use Kaggle's resumable uploader; never silently overwrite a dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    folder = args.folder.resolve()
    metadata = json.loads((folder / "dataset-metadata.json").read_text())
    bundle = json.loads((folder / "bundle.json").read_text())
    dataset = metadata["id"]
    if dataset != bundle["dataset_id"] or bundle["status"] != "complete":
        raise ValueError("Dataset publication identity mismatch")
    expected = {p.name: p.stat().st_size for p in folder.iterdir()
                if p.is_file() and p.name != "dataset-metadata.json"}
    api = ParallelDatasetApi(args.workers)
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
        "upload_workers": args.workers,
    }
    atomic_write_json(args.receipt.with_name("upload_preflight.json"), preflight)
    print(json.dumps({"preflight": preflight}), flush=True)
    exists = True
    try:
        get_published_dataset(api, dataset)
    except HTTPError as error:
        # Never recreate an owned dataset if metadata is temporarily restricted.
        if (error.response is None
                or error.response.status_code not in {403, 404}):
            raise
        exists = dataset in owned_refs
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
        # GetDatasetStatus can return 403 for a successfully published dataset.
        # Require available metadata and every expected file instead; the
        # receipt download below independently verifies actual serving.
        try:
            published = get_published_dataset(api, dataset)
            remote = get_published_files(api, dataset)
        except HTTPError as error:
            if (error.response is None
                    or error.response.status_code not in {403, 404}):
                raise
            print(json.dumps({"remote": "processing", "attempt": attempt,
                              "http_status": error.response.status_code}), flush=True)
            time.sleep(30)
            continue
        print(json.dumps({"remote_files": len(remote), "attempt": attempt,
                          "version": published.current_version_number}), flush=True)
        if remote == expected and published.current_version_number > 0:
            break
        if args.verify_only:
            raise ValueError(f"Remote filenames/sizes differ: expected={expected}, actual={remote}")
        time.sleep(30)
    else:
        raise TimeoutError("Dataset processing exceeded two hours; rerun --verify-only")
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
        "version": published.current_version_number, "public_verified": True,
        "bundle_sha256": preflight["bundle_sha256"], "verified_files": remote,
        "remote_verification": "Public version available, exact names/sizes, downloaded bundle.json SHA-256",
        "content_verification": "Local SHA-256 in bundle.json; bootstrap verifies downloaded content",
    })
    print(json.dumps({"status": "complete", "receipt": str(args.receipt)}), flush=True)


if __name__ == "__main__":
    main()
