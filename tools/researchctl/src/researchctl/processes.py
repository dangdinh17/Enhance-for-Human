"""Process identities protect resume/cancel from stale or reused PIDs."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any, Mapping


def start_ticks(pid: int) -> str | None:
    """Return a Linux process start identity, excluding zombie processes."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def node_alive(row: Mapping[str, Any]) -> bool:
    """Require both PID and creation time, not merely the existence of /proc."""
    pid = row.get("pid")
    return bool(
        pid
        and row.get("process_start_ticks")
        and start_ticks(int(pid)) == row["process_start_ticks"]
    )


def signal_node(row: Mapping[str, Any], signum: int = signal.SIGTERM) -> bool:
    """Signal only the verified task's dedicated process group."""
    if not node_alive(row):
        return False
    pid = int(row["pid"])
    try:
        if os.getpgid(pid) != row.get("pgid") or row["pgid"] != pid:
            return False
        os.killpg(pid, signum)
        return True
    except ProcessLookupError:
        return False
