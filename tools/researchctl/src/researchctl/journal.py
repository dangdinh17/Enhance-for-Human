"""SQLite journal and JSONL audit export."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import NODE_STATES, redact, redact_text


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class Journal:
    """Durable campaign state with appendable audit export."""

    def __init__(self, path: Path, readonly: bool = False) -> None:
        self.path = path
        if not readonly:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = (
            sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            if readonly
            else sqlite3.connect(path)
        )
        self.connection.row_factory = sqlite3.Row
        if not readonly:
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        if not readonly:
            self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                campaign_hash TEXT NOT NULL,
                trial_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                artifacts_json TEXT NOT NULL DEFAULT '{}',
                pid INTEGER,
                pgid INTEGER,
                exit_code INTEGER,
                started_at TEXT,
                ended_at TEXT,
                failure_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS nodes_campaign
                ON nodes(campaign_id, campaign_hash, stage, state);
            CREATE TABLE IF NOT EXISTS approvals (
                campaign_id TEXT NOT NULL,
                campaign_hash TEXT NOT NULL,
                stage TEXT NOT NULL,
                selected_trials_json TEXT NOT NULL,
                identity_json TEXT NOT NULL,
                approved_by TEXT NOT NULL,
                approved_at TEXT NOT NULL,
                PRIMARY KEY(campaign_id, campaign_hash, stage)
            );
            """
        )
        self.connection.commit()
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(nodes)")
        }
        if "process_start_ticks" not in columns:
            self.connection.execute(
                "ALTER TABLE nodes ADD COLUMN process_start_ticks TEXT"
            )
            self.connection.commit()

    def close(self) -> None:
        """Close the journal database."""
        self.connection.close()

    def ensure_node(self, payload: Mapping[str, Any]) -> None:
        """Create a pending node without overwriting existing state."""
        self.connection.execute(
            """
            INSERT OR IGNORE INTO nodes(
                node_id, campaign_id, campaign_hash, trial_id, stage,
                state, payload_json
            ) VALUES (?, ?, ?, ?, ?, 'PENDING', ?)
            """,
            (
                payload["node_id"],
                payload["campaign_id"],
                payload["campaign_hash"],
                payload["trial_id"],
                payload["stage"],
                redact_text(json.dumps(redact(payload), sort_keys=True)),
            ),
        )
        self.connection.commit()

    def transition(self, node_id: str, state: str, **fields: Any) -> None:
        """Update node state and selected runtime fields."""
        if state not in NODE_STATES:
            raise ValueError(f"Invalid node state: {state}")
        previous = self.node(node_id)
        if previous is None:
            raise KeyError(f"Unknown node: {node_id}")
        transitions = {
            "PENDING": {"PENDING", "RUNNING", "FAILED", "CANCELLED"},
            "RUNNING": {
                "RUNNING",
                "COMPLETE",
                "FAILED",
                "TIMED_OUT",
                "CANCELLED",
                "QUOTA_EXCEEDED",
            },
        }
        allowed_states = transitions.get(
            previous["state"], {previous["state"], "PENDING", "RUNNING"}
        )
        if state not in allowed_states:
            raise ValueError(f"Invalid transition: {previous['state']} -> {state}")
        allowed = {
            "metrics",
            "artifacts",
            "pid",
            "pgid",
            "exit_code",
            "failure_reason",
            "process_start_ticks",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported journal fields: {sorted(unknown)}")
        assignments = ["state = ?"]
        values: list[Any] = [state]
        if state == "PENDING":
            assignments.extend(["started_at = NULL", "ended_at = NULL"])
        if state == "RUNNING":
            assignments.append("started_at = ?")
            values.append(utc_now())
            assignments.append("ended_at = NULL")
        if state in {"COMPLETE", "FAILED", "TIMED_OUT", "CANCELLED", "QUOTA_EXCEEDED"}:
            assignments.append("ended_at = ?")
            values.append(utc_now())
        for key, value in fields.items():
            column = f"{key}_json" if key in {"metrics", "artifacts"} else key
            assignments.append(f"{column} = ?")
            if key in {"metrics", "artifacts"}:
                value = redact_text(json.dumps(redact(value), sort_keys=True))
            elif isinstance(value, str):
                value = redact_text(value)
            values.append(value)
        values.append(node_id)
        cursor = self.connection.execute(
            f"UPDATE nodes SET {', '.join(assignments)} WHERE node_id = ?", values
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown node: {node_id}")
        self.connection.commit()

    def node(self, node_id: str) -> dict[str, Any] | None:
        """Return one node as a mapping."""
        row = self.connection.execute(
            "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return dict(row) if row else None

    def nodes(
        self,
        campaign_id: str | None = None,
        campaign_hash: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query nodes using optional filters."""
        clauses, values = [], []
        for column, value in (
            ("campaign_id", campaign_id),
            ("campaign_hash", campaign_hash),
            ("stage", stage),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM nodes{where} ORDER BY stage, node_id", values
        ).fetchall()
        return [dict(row) for row in rows]

    def approve(
        self,
        campaign_id: str,
        campaign_hash: str,
        stage: str,
        selected_trials: Iterable[str],
        identity: Mapping[str, Any],
        approved_by: str,
    ) -> None:
        """Record or replace a hash-bound human approval."""
        self.connection.execute(
            """
            INSERT OR REPLACE INTO approvals(
                campaign_id, campaign_hash, stage, selected_trials_json,
                identity_json, approved_by, approved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                campaign_hash,
                stage,
                json.dumps(sorted(selected_trials)),
                redact_text(json.dumps(redact(identity), sort_keys=True)),
                approved_by,
                utc_now(),
            ),
        )
        self.connection.commit()

    def approval(
        self, campaign_id: str, campaign_hash: str, stage: str
    ) -> dict[str, Any] | None:
        """Return a valid approval for the exact campaign identity."""
        row = self.connection.execute(
            """
            SELECT * FROM approvals
            WHERE campaign_id = ? AND campaign_hash = ? AND stage = ?
            """,
            (campaign_id, campaign_hash, stage),
        ).fetchone()
        return dict(row) if row else None

    def export_jsonl(self, path: Path) -> None:
        """Atomically export nodes and approvals for auditability."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for node in self.nodes():
                handle.write(
                    json.dumps({"type": "node", **node}, sort_keys=True) + "\n"
                )
            rows = self.connection.execute(
                "SELECT * FROM approvals ORDER BY campaign_id, stage"
            ).fetchall()
            for row in rows:
                handle.write(
                    json.dumps({"type": "approval", **dict(row)}, sort_keys=True) + "\n"
                )
        temporary.replace(path)
