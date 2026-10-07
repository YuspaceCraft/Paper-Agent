"""Durable idempotency state for side-effecting tool calls.

The gateway needs stronger semantics than an in-process result cache:

* a successful result must be replayable after a process restart;
* a second concurrent call with the same key must not execute the adapter;
* if the process dies after the adapter started but before the result was
  recorded, the outcome is unknowable.  Re-running automatically could duplicate
  the side effect, so the key is marked ``indeterminate`` and requires explicit
  recovery instead.

The SQLite implementation uses short-lived connections and ``BEGIN IMMEDIATE``
transactions, so it is safe across threads and processes without owning a
long-lived event-loop-bound connection.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol


IdempotencyStatus = Literal[
    "running", "succeeded", "failed", "indeterminate"
]


@dataclass(frozen=True)
class IdempotencyRecord:
    key: str
    tool_name: str
    tool_version: str
    status: IdempotencyStatus
    envelope: str = ""
    error_code: str = ""


@dataclass(frozen=True)
class Reservation:
    acquired: bool
    record: IdempotencyRecord | None = None


class IdempotencyStore(Protocol):
    async def reserve(self, key: str, *, tool_name: str,
                      tool_version: str) -> Reservation: ...

    async def complete(self, key: str, envelope: str) -> None: ...

    async def fail(self, key: str, error_code: str) -> None: ...

    async def reject(self, key: str, error_code: str) -> None: ...


class InMemoryIdempotencyStore:
    """Process-local store for tests and explicitly scoped runtimes.

    ``reserve`` contains no await point, so a single event loop cannot
    interleave two reservations for the same key.
    """

    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}

    async def reserve(self, key: str, *, tool_name: str,
                      tool_version: str) -> Reservation:
        existing = self._records.get(key)
        if existing is not None:
            return Reservation(acquired=False, record=existing)
        record = IdempotencyRecord(
            key=key, tool_name=tool_name, tool_version=tool_version,
            status="running",
        )
        self._records[key] = record
        return Reservation(acquired=True, record=record)

    async def complete(self, key: str, envelope: str) -> None:
        old = self._records.get(key)
        if old is None:
            return
        self._records[key] = IdempotencyRecord(
            key=key, tool_name=old.tool_name, tool_version=old.tool_version,
            status="succeeded", envelope=str(envelope),
        )

    async def fail(self, key: str, error_code: str) -> None:
        old = self._records.get(key)
        if old is None or old.status != "running":
            return
        self._records[key] = IdempotencyRecord(
            key=key, tool_name=old.tool_name, tool_version=old.tool_version,
            status="indeterminate", error_code=str(error_code),
        )

    async def reject(self, key: str, error_code: str) -> None:
        """Record a definitive failed attempt that did not apply a side effect."""
        old = self._records.get(key)
        if old is None or old.status != "running":
            return
        self._records[key] = IdempotencyRecord(
            key=key, tool_name=old.tool_name, tool_version=old.tool_version,
            status="failed", error_code=str(error_code),
        )


class SQLiteIdempotencyStore:
    """Crash-safe store backed by a standalone SQLite file."""

    def __init__(
        self,
        path: str | Path,
        *,
        stale_after_seconds: float | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.stale_after_seconds = max(
            0.001,
            float(
                stale_after_seconds
                if stale_after_seconds is not None
                else os.getenv("AGENT_IDEMPOTENCY_STALE_SECONDS", "900")
            ),
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_idempotency (
                key TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                tool_version TEXT NOT NULL,
                status TEXT NOT NULL,
                envelope TEXT NOT NULL DEFAULT '',
                error_code TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        return conn

    @staticmethod
    def _record(row: tuple) -> IdempotencyRecord:
        return IdempotencyRecord(
            key=str(row[0]),
            tool_name=str(row[1]),
            tool_version=str(row[2]),
            status=str(row[3]),  # validated when read below
            envelope=str(row[4] or ""),
            error_code=str(row[5] or ""),
        )

    def _reserve_sync(self, key: str, tool_name: str,
                      tool_version: str) -> Reservation:
        now = self._now()
        with closing(self._connect()) as conn:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT key, tool_name, tool_version, status, envelope, "
                    "error_code, updated_at FROM tool_idempotency WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    status = str(row[3])
                    updated_at = str(row[6] or "")
                    if status == "running" and updated_at:
                        try:
                            updated = datetime.fromisoformat(updated_at)
                            if updated.tzinfo is None:
                                updated = updated.replace(tzinfo=timezone.utc)
                            age = (
                                datetime.now(timezone.utc) - updated
                            ).total_seconds()
                        except ValueError:
                            age = self.stale_after_seconds + 1
                        if age > self.stale_after_seconds:
                            status = "indeterminate"
                            conn.execute(
                                "UPDATE tool_idempotency SET status = ?, "
                                "error_code = 'STALE_RUNNING', updated_at = ? "
                                "WHERE key = ?",
                                (status, now, key),
                            )
                    if status not in (
                        "running", "succeeded", "failed", "indeterminate",
                    ):
                        status = "indeterminate"
                        conn.execute(
                            "UPDATE tool_idempotency SET status = ?, updated_at = ? "
                            "WHERE key = ?",
                            (status, now, key),
                        )
                    return Reservation(acquired=False, record=self._record(
                        (row[0], row[1], row[2], status, row[4], row[5])
                    ))
                conn.execute(
                    "INSERT INTO tool_idempotency "
                    "(key, tool_name, tool_version, status, envelope, error_code, "
                    " created_at, updated_at) VALUES (?, ?, ?, 'running', '', '', ?, ?)",
                    (key, tool_name, tool_version, now, now),
                )
                return Reservation(
                    acquired=True,
                    record=IdempotencyRecord(
                        key=key, tool_name=tool_name, tool_version=tool_version,
                        status="running",
                    ),
                )

    def _update_sync(self, key: str, *, status: IdempotencyStatus,
                     envelope: str = "", error_code: str = "") -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    "UPDATE tool_idempotency SET status = ?, envelope = ?, "
                    "error_code = ?, updated_at = ? WHERE key = ? AND status = 'running'",
                    (status, envelope, error_code, self._now(), key),
                )

    async def reserve(self, key: str, *, tool_name: str,
                      tool_version: str) -> Reservation:
        return await asyncio.to_thread(
            self._reserve_sync, key, tool_name, tool_version
        )

    async def complete(self, key: str, envelope: str) -> None:
        await asyncio.to_thread(
            self._update_sync, key, status="succeeded", envelope=str(envelope)
        )

    async def fail(self, key: str, error_code: str) -> None:
        await asyncio.to_thread(
            self._update_sync, key, status="indeterminate",
            error_code=str(error_code),
        )

    async def reject(self, key: str, error_code: str) -> None:
        await asyncio.to_thread(
            self._update_sync, key, status="failed",
            error_code=str(error_code),
        )


def idempotency_payload(record: IdempotencyRecord | None) -> dict:
    """Small, JSON-safe audit view with no raw arguments or result bodies."""
    if record is None:
        return {}
    return {
        "key": record.key,
        "tool": record.tool_name,
        "tool_version": record.tool_version,
        "status": record.status,
        "error_code": record.error_code,
    }
