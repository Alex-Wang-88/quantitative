from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .domain import jsonable, utc_now


class Ledger:
    """Small append-only SQLite ledger for paper state and audit events."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path or os.environ.get("QUANTPAPER_DB_PATH", "data/paper.db")
        self.path = Path(configured_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    name TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    entity_id TEXT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_type_time
                    ON events(event_type, created_at DESC);
                CREATE TABLE IF NOT EXISTS reports (
                    report_date TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    generated_at TEXT NOT NULL
                );
                """
            )

    def save_snapshot(self, name: str, payload: Any) -> None:
        serialized = json.dumps(jsonable(payload), ensure_ascii=False)
        timestamp = utc_now().isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO snapshots(name, payload, updated_at) VALUES(?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (name, serialized, timestamp),
            )

    def load_snapshot(self, name: str) -> Any | None:
        row = self._connection.execute(
            "SELECT payload FROM snapshots WHERE name = ?", (name,)
        ).fetchone()
        return None if row is None else json.loads(row["payload"])

    def append_event(self, event_type: str, payload: Any, entity_id: str | None = None) -> str:
        event_id = str(uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                (
                    "INSERT INTO events(id, event_type, entity_id, payload, created_at) "
                    "VALUES(?, ?, ?, ?, ?)"
                ),
                (
                    event_id,
                    event_type,
                    entity_id,
                    json.dumps(jsonable(payload), ensure_ascii=False),
                    utc_now().isoformat(),
                ),
            )
        return event_id

    def recent_events(self, event_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if event_type:
            rows = self._connection.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY created_at DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "entity_id": row["entity_id"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_report(
        self, report_date: str, report: Any, generated_at: datetime | None = None
    ) -> None:
        generated_at = generated_at or utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO reports(report_date, payload, generated_at) VALUES(?, ?, ?)
                ON CONFLICT(report_date) DO UPDATE SET
                    payload=excluded.payload, generated_at=excluded.generated_at
                """,
                (
                    report_date,
                    json.dumps(jsonable(report), ensure_ascii=False),
                    generated_at.isoformat(),
                ),
            )

    def list_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            (
                "SELECT report_date, payload, generated_at FROM reports "
                "ORDER BY report_date DESC LIMIT ?"
            ),
            (limit,),
        ).fetchall()
        return [
            {
                "report_date": row["report_date"],
                "generated_at": row["generated_at"],
                **json.loads(row["payload"]),
            }
            for row in rows
        ]

    def close(self) -> None:
        self._connection.close()
