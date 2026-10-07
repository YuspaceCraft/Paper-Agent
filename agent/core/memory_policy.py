"""Typed memory records and the policy that decides what may be reused.

Four lifecycles live side by side (platform design §6.3): session working
memory (LangGraph checkpoints), conversation summaries, user profile and
domain/task facts. Long-lived records all share one shape so that confidence,
TTL and consent are enforced in one place instead of at each injection point.

Retrieved full text must never be written into the profile: a record's
``source_ref`` points at where it came from, ``content`` is the preference or
fact itself.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    LONG_TERM_FACT = "long_term_fact"
    TASK = "task"
    SUMMARY = "summary"
    WORKSPACE = "workspace"
    POSTMORTEM = "postmortem"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class MemoryRecord(BaseModel):
    """One reusable memory item with its provenance and lifecycle."""

    memory_id: str
    type: MemoryType = MemoryType.PREFERENCE
    content: str
    source_ref: str = ""
    confidence: float = 1.0
    created_at: str = ""
    expires_at: str | None = None
    consent: bool = True
    revision: str = "1"
    semantic_key: str = ""
    status: str = "active"
    supersedes: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)

    def is_expired(self, now: datetime | None = None) -> bool:
        expires = _parse_ts(self.expires_at)
        if expires is None:
            return False
        return expires <= (now or _now())

    def render(self) -> str:
        """Prompt-facing line; content is the only payload."""
        return f"- ({self.type.value}) {self.content}"

    def trace_view(self) -> dict:
        """Metadata-only view: the trace never carries memory content."""
        return {
            "memory_id": self.memory_id,
            "type": self.type.value,
            "confidence": round(self.confidence, 3),
            "source_ref": self.source_ref,
            "revision": self.revision,
            "semantic_key": self.semantic_key,
            "status": self.status,
            "supersedes": list(self.supersedes),
            "conflicts_with": list(self.conflicts_with),
        }


class MemoryPolicy(BaseModel):
    """Confidence / TTL / consent gate applied before any injection."""

    min_confidence: float = 0.3
    max_age_days: int | None = 365
    require_consent: bool = True
    disabled_types: set[MemoryType] = Field(default_factory=set)
    disabled_ids: set[str] = Field(default_factory=set)

    def evaluate(self, record: MemoryRecord, *, now: datetime | None = None
                 ) -> tuple[bool, str]:
        if record.memory_id in self.disabled_ids:
            return False, "disabled by the user"
        if record.status == "superseded":
            return False, "superseded by a newer memory"
        if record.status == "conflict":
            return False, "conflicting memory requires user resolution"
        if record.status == "candidate":
            return False, "postmortem candidate awaiting approval"
        if record.type in self.disabled_types:
            return False, f"type '{record.type.value}' is disabled"
        if self.require_consent and not record.consent:
            return False, "no consent recorded"
        if record.confidence < self.min_confidence:
            return False, f"confidence {record.confidence:.2f} below threshold"
        if record.is_expired(now):
            return False, "expired"
        if self.max_age_days is not None:
            created = _parse_ts(record.created_at)
            if created is not None:
                age_days = ((now or _now()) - created).days
                if age_days > self.max_age_days:
                    return False, f"older than {self.max_age_days} days"
        return True, "usable"

    def filter(self, records: Iterable[MemoryRecord], *, now: datetime | None = None
               ) -> tuple[list[MemoryRecord], list[dict]]:
        """Split records into (kept, dropped-audit-entries-without-content)."""
        kept: list[MemoryRecord] = []
        dropped: list[dict] = []
        for record in records or ():
            ok, reason = self.evaluate(record, now=now)
            if ok:
                kept.append(record)
            else:
                dropped.append({**record.trace_view(), "kept": False,
                                "reason": reason})
        return kept, dropped


def _record_id(field: str, value: Any) -> str:
    digest = hashlib.sha256(f"{field}={value}".encode("utf-8")).hexdigest()[:10]
    return f"profile:{field}:{digest}"


def records_from_profile(profile: dict | None, *,
                         created_at: str = "") -> list[MemoryRecord]:
    """Adapt the existing ``profile.json`` into typed records.

    This is a read adapter: the profile file keeps its current shape, and the
    context pack gets confidence/TTL/consent handling for free.
    """
    if not isinstance(profile, dict) or not profile:
        return []
    created = created_at or _now().isoformat(timespec="seconds")
    type_by_field: dict[str, MemoryType] = {
        "preferred_language": MemoryType.PREFERENCE,
        "frequent_topics": MemoryType.PREFERENCE,
        "writing_style": MemoryType.PREFERENCE,
        "known_papers": MemoryType.LONG_TERM_FACT,
    }
    records: list[MemoryRecord] = []
    for field, value in profile.items():
        if value in (None, "", [], {}):
            continue
        memory_type = type_by_field.get(field, MemoryType.PROFILE)
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = f"{field.replace('_', ' ')}: {item}"
            records.append(MemoryRecord(
                memory_id=_record_id(field, item),
                type=memory_type,
                content=text,
                source_ref="profile.json",
                created_at=created,
                semantic_key=(
                    f"{memory_type.value}:"
                    + " ".join(field.replace("_", " ").casefold().split())
                ),
            ))
    return records
