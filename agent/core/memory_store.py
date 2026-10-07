"""Persistent typed long-term memory with policy-controlled injection.

``profile.json`` remains the legacy preference adapter.  New memory writes use
this store so each item carries provenance, confidence, TTL, consent and a
revision, matching ADR-0005.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory_policy import MemoryPolicy, MemoryRecord, MemoryType, _parse_ts


_lock = threading.RLock()
MAX_CONTENT_CHARS = 1000


def memory_path() -> Path:
    override = os.getenv("AGENT_MEMORY_PATH", "")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parent.parent.parent
    return root / ".demo" / "memory" / "records.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_id(record_type: MemoryType, content: str) -> str:
    digest = hashlib.sha256(
        f"{record_type.value}|{content.strip()}".encode("utf-8")
    ).hexdigest()[:12]
    return f"memory:{record_type.value}:{digest}"


def _normalize_content(content: Any) -> str:
    normalized = " ".join(str(content or "").split()).strip()
    return normalized[:MAX_CONTENT_CHARS]


def _infer_semantic_key(memory_type: MemoryType, content: str) -> str:
    match = re.match(r"^([^:：=]{2,80})\s*[:：=]\s*.+$", content)
    if not match:
        return ""
    field = " ".join(match.group(1).casefold().split())
    return f"{memory_type.value}:{field}"


def _source_rank(source_ref: str) -> int:
    source = str(source_ref or "").casefold()
    if source in {"user", "manual", "api:user"} or "user" in source:
        return 2
    if source in {"conversation", "chat"} or "thread" in source:
        return 1
    return 0


def _default_payload() -> dict:
    return {"records": [], "disabled_ids": [], "disabled_types": []}


def _load_payload() -> dict:
    path = memory_path()
    if not path.exists():
        return _default_payload()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _default_payload()
    if not isinstance(raw, dict):
        return _default_payload()
    payload = {**_default_payload(), **raw}
    payload["records"] = [
        item for item in payload.get("records", []) if isinstance(item, dict)
    ]
    payload["disabled_ids"] = [
        str(item) for item in payload.get("disabled_ids", [])
    ]
    payload["disabled_types"] = [
        str(item) for item in payload.get("disabled_types", [])
    ]
    return payload


def _save_payload(payload: dict) -> None:
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def list_records() -> list[MemoryRecord]:
    with _lock:
        payload = _load_payload()
        records: list[MemoryRecord] = []
        for raw in payload["records"]:
            try:
                records.append(MemoryRecord.model_validate(raw))
            except Exception:
                continue
        return records


def upsert_record(
    *,
    content: str,
    memory_type: MemoryType = MemoryType.PREFERENCE,
    source_ref: str = "user",
    confidence: float = 0.9,
    expires_at: str | None = None,
    consent: bool = True,
    semantic_key: str = "",
    status: str = "active",
) -> MemoryRecord:
    content = _normalize_content(content)
    if not content:
        raise ValueError("memory content is required")
    if expires_at and _parse_ts(expires_at) is None:
        raise ValueError("expires_at must be an ISO-8601 timestamp")
    record = MemoryRecord(
        memory_id=_record_id(memory_type, content),
        type=memory_type,
        content=content,
        source_ref=str(source_ref or "")[:200],
        confidence=max(0.0, min(1.0, float(confidence))),
        created_at=_now(),
        expires_at=expires_at,
        consent=bool(consent),
        revision="1",
        semantic_key=(
            str(semantic_key or "").strip()
            or _infer_semantic_key(memory_type, content)
        ),
        status=str(status or "active"),
    )
    with _lock:
        payload = _load_payload()
        existing = {
            str(item.get("memory_id")): item for item in payload["records"]
        }
        previous = existing.get(record.memory_id)
        if previous:
            record.created_at = str(previous.get("created_at") or record.created_at)
            try:
                record.revision = str(int(previous.get("revision") or 1) + 1)
            except ValueError:
                record.revision = "2"
            record.consent = bool(previous.get("consent", record.consent))
            record.status = (
                "active"
                if str(status or "active") == "active"
                else str(previous.get("status") or status or "candidate")
            )
            record.conflicts_with = []
            record.supersedes = list(previous.get("supersedes") or [])
        elif record.semantic_key:
            related = [
                item for item in payload["records"]
                if str(item.get("memory_id")) != record.memory_id
                and str(item.get("type")) == record.type.value
                and str(item.get("semantic_key") or "") == record.semantic_key
                and str(item.get("status") or "active") in {"active", "conflict"}
            ]
            if related:
                previous_record = max(
                    related,
                    key=lambda item: (
                        _source_rank(str(item.get("source_ref") or "")),
                        float(item.get("confidence") or 0.0),
                        str(item.get("created_at") or ""),
                    ),
                )
                previous_confidence = float(
                    previous_record.get("confidence") or 0.0
                )
                previous_rank = _source_rank(
                    str(previous_record.get("source_ref") or "")
                )
                new_rank = _source_rank(record.source_ref)
                new_wins = (
                    new_rank > previous_rank
                    or (
                        new_rank == previous_rank
                        and record.confidence >= previous_confidence + 0.1
                    )
                )
                old_wins = (
                    previous_rank > new_rank
                    or (
                        previous_rank == new_rank
                        and previous_confidence >= record.confidence + 0.1
                    )
                )
                if new_wins or old_wins:
                    old_id = str(previous_record.get("memory_id") or "")
                    old_supersedes = list(
                        previous_record.get("supersedes") or []
                    )
                    if new_wins:
                        previous_record["status"] = "superseded"
                        previous_record["supersedes"] = []
                        previous_record["conflicts_with"] = []
                        record.status = "active"
                        record.supersedes = [old_id, *old_supersedes]
                    else:
                        record.status = "superseded"
                        record.supersedes = [old_id, *old_supersedes]
                        previous_record["status"] = "active"
                        previous_record["conflicts_with"] = []
                else:
                    record.status = "conflict"
                    record.conflicts_with = [
                        str(item.get("memory_id") or "") for item in related
                    ]
                    for item in related:
                        item["status"] = "conflict"
                        conflicts = list(item.get("conflicts_with") or [])
                        if record.memory_id not in conflicts:
                            conflicts.append(record.memory_id)
                        item["conflicts_with"] = conflicts
        existing[record.memory_id] = record.model_dump(mode="json")
        payload["records"] = list(existing.values())
        _save_payload(payload)
    return record


def delete_record(memory_id: str) -> bool:
    with _lock:
        payload = _load_payload()
        before = len(payload["records"])
        payload["records"] = [
            item for item in payload["records"]
            if str(item.get("memory_id")) != str(memory_id)
        ]
        payload["disabled_ids"] = [
            item for item in payload["disabled_ids"] if item != str(memory_id)
        ]
        changed = len(payload["records"]) != before
        if changed:
            _save_payload(payload)
        return changed


def approve_record(memory_id: str) -> MemoryRecord | None:
    """Promote a postmortem candidate into policy-eligible long-term memory."""
    with _lock:
        payload = _load_payload()
        for raw in payload["records"]:
            if str(raw.get("memory_id") or "") != str(memory_id):
                continue
            raw["status"] = "active"
            raw["consent"] = True
            raw["confidence"] = max(0.7, float(raw.get("confidence") or 0.0))
            raw["supersedes"] = list(raw.get("supersedes") or [])
            raw["conflicts_with"] = []
            _save_payload(payload)
            try:
                return MemoryRecord.model_validate(raw)
            except Exception:
                return None
        return None


def get_policy() -> MemoryPolicy:
    with _lock:
        payload = _load_payload()
        disabled_types: set[MemoryType] = set()
        for raw in payload["disabled_types"]:
            try:
                disabled_types.add(MemoryType(raw))
            except ValueError:
                continue
        return MemoryPolicy(
            disabled_ids=set(payload["disabled_ids"]),
            disabled_types=disabled_types,
        )


def update_policy(
    *,
    disabled_ids: list[str] | None = None,
    disabled_types: list[str] | None = None,
) -> MemoryPolicy:
    with _lock:
        payload = _load_payload()
        if disabled_ids is not None:
            payload["disabled_ids"] = list(dict.fromkeys(map(str, disabled_ids)))
        if disabled_types is not None:
            valid = {item.value for item in MemoryType}
            payload["disabled_types"] = list(dict.fromkeys(
                item for item in map(str, disabled_types) if item in valid
            ))
        _save_payload(payload)
    return get_policy()


_CAPTURE_PATTERNS = (
    re.compile(r"(?:请)?记住[:：]?\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:以后|今后)(?:请)?\s*(.+)", re.IGNORECASE),
    re.compile(r"\bremember(?: that)?\s+(.+)", re.IGNORECASE),
)


def capture_explicit_memory(text: str, *, source_ref: str = "conversation") -> MemoryRecord | None:
    """Capture a direct user memory request; never infer from ordinary queries."""
    clean = " ".join(str(text or "").split()).strip()
    if not clean or len(clean) > 500:
        return None
    for pattern in _CAPTURE_PATTERNS:
        match = pattern.search(clean)
        if not match:
            continue
        content = match.group(1).strip(" ，。；;")
        if len(content) < 2:
            return None
        return upsert_record(
            content=content,
            memory_type=MemoryType.PREFERENCE,
            source_ref=source_ref,
            confidence=0.95,
        )
    return None
