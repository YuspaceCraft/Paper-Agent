"""Candidate postmortem records for failed or incomplete turns."""

from __future__ import annotations

from typing import Any

from .memory_policy import MemoryRecord, MemoryType
from .memory_store import upsert_record


def should_record_postmortem(
    *,
    status: str,
    error: str | None,
    verification: dict | None,
) -> bool:
    if str(status) not in {"ok", "completed"}:
        return True
    if isinstance(verification, dict):
        if str(verification.get("status") or "") in {
            "partial", "failed", "no_evidence",
        }:
            return True
        if verification.get("outstanding"):
            return True
    return bool(error)


def record_turn_postmortem(
    *,
    query: str,
    status: str,
    error: str | None = None,
    verification: dict | None = None,
    trace_id: str = "",
) -> MemoryRecord | None:
    """Store a non-injectable candidate lesson for later user approval."""
    if not should_record_postmortem(
        status=status, error=error, verification=verification,
    ):
        return None
    query_text = " ".join(str(query or "").split()).strip()[:300]
    outcomes: list[str] = []
    if str(status) != "ok":
        outcomes.append(f"turn status: {status}")
    if error:
        outcomes.append(f"error: {str(error)[:120]}")
    if isinstance(verification, dict):
        verification_status = str(verification.get("status") or "")
        if verification_status:
            outcomes.append(f"verification: {verification_status}")
        outstanding = verification.get("outstanding") or []
        if outstanding:
            codes = [
                str(item.get("id") or item.get("reason") or "")
                for item in outstanding[:5]
                if isinstance(item, dict)
            ]
            outcomes.append("outstanding: " + ", ".join(filter(None, codes)))
    content = (
        f"Review this prior task before a similar request. Request: {query_text}. "
        + "; ".join(outcomes)
    ).strip()
    from ..safety import sanitize_output

    return upsert_record(
        content=sanitize_output(content),
        memory_type=MemoryType.POSTMORTEM,
        source_ref=f"trace:{trace_id}" if trace_id else "turn_postmortem",
        confidence=0.6,
        consent=False,
        status="candidate",
    )
