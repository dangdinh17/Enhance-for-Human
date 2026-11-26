"""User-private credential storage; secrets only reach tasks via environment."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from .config import ConfigError


def credential_path() -> Path:
    """Return the per-account store outside project configuration."""
    return Path.home() / ".config" / "researchctl" / "credentials.json"


def _check_directory(path: Path) -> None:
    """Reject shared, foreign-owned or redirected credential directories."""
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ConfigError("Credential directory must be user-owned with mode 0700")


def read_credentials() -> dict[str, str]:
    """Read a bounded, owner-only regular file without following symlinks."""
    path = credential_path()
    if not path.parent.exists() and not path.parent.is_symlink():
        return {}
    _check_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {}
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > 4096
        ):
            raise ConfigError("Credential file must be user-owned with mode 0600")
        try:
            value = json.loads(stream.read(4097))
        except (ValueError, UnicodeError) as error:
            raise ConfigError("Invalid private credential store") from error
    if not isinstance(value, dict) or set(value) - {"COMET_API_KEY"}:
        raise ConfigError("Unsupported private credential fields")
    for key in value.values():
        _validate_key(key)
    return value


def _validate_key(key: str) -> None:
    """Reject blank/multiline input without including it in diagnostics."""
    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= 512
        or any(not 33 <= ord(character) <= 126 for character in key)
    ):
        raise ConfigError("API key must be a nonempty single printable token")


def save_comet_key(key: str) -> Path:
    """Atomically persist a key with private directory and file permissions."""
    _validate_key(key)
    path = credential_path()
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _check_directory(path.parent)
    existing = read_credentials()
    existing["COMET_API_KEY"] = key
    descriptor, temporary = tempfile.mkstemp(prefix=".credentials-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(existing, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_credentials() -> None:
    """Load account defaults while preserving explicitly provided environment."""
    if "COMET_API_KEY" not in os.environ:
        for name, value in read_credentials().items():
            os.environ.setdefault(name, value)
