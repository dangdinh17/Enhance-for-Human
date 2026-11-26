"""Streaming hashes and durable file identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    """Hash canonical JSON metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path: Path, content: bool = False) -> dict[str, Any]:
    """Record resolved path, byte size and nanosecond modification time."""
    path = Path(path).resolve()
    info = path.stat()
    result = {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    if content:
        result["sha256"] = file_sha256(path)
    return result
