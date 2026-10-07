"""Long-term memory API — inspect, add, delete and disable typed records."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.core.memory_policy import MemoryType
from agent.core.memory_store import (
    approve_record,
    delete_record,
    get_policy,
    list_records,
    memory_path,
    update_policy,
    upsert_record,
)


router = APIRouter(prefix="/api/memory", tags=["memory"])


def _record_payload(record, policy=None) -> dict:
    policy = policy or get_policy()
    return {
        **record.model_dump(mode="json"),
        "enabled": record.status == "active"
        and record.memory_id not in policy.disabled_ids
        and record.type not in policy.disabled_types,
    }


def _payload() -> dict:
    policy = get_policy()
    return {
        "path": str(memory_path()),
        "records": [
            _record_payload(record, policy) for record in list_records()
        ],
        "policy": {
            "min_confidence": policy.min_confidence,
            "max_age_days": policy.max_age_days,
            "require_consent": policy.require_consent,
            "disabled_ids": sorted(policy.disabled_ids),
            "disabled_types": sorted(item.value for item in policy.disabled_types),
        },
    }


@router.get("")
async def get_memory():
    return _payload()


class CreateMemoryBody(BaseModel):
    content: str = Field(..., min_length=2, max_length=1000)
    type: str = "preference"
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source_ref: str = "user"
    expires_at: str | None = None
    consent: bool = True


@router.post("")
async def create_memory(body: CreateMemoryBody):
    try:
        memory_type = MemoryType(body.type)
    except ValueError:
        raise HTTPException(400, f"unknown memory type: {body.type}")
    try:
        record = upsert_record(
            content=body.content,
            memory_type=memory_type,
            source_ref=body.source_ref,
            confidence=body.confidence,
            expires_at=body.expires_at,
            consent=body.consent,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"outcome": "succeeded", "record": _record_payload(record)}


@router.delete("/{memory_id}")
async def remove_memory(memory_id: str):
    if not delete_record(memory_id):
        raise HTTPException(404, f"memory not found: {memory_id}")
    return {"outcome": "succeeded", "deleted": memory_id}


@router.post("/{memory_id}/approve")
async def approve_memory(memory_id: str):
    record = approve_record(memory_id)
    if record is None:
        raise HTTPException(404, f"memory not found: {memory_id}")
    return {"outcome": "succeeded", "record": _record_payload(record)}


class PolicyBody(BaseModel):
    disabled_ids: list[str] | None = None
    disabled_types: list[str] | None = None


@router.put("/policy")
async def set_policy(body: PolicyBody):
    policy = update_policy(
        disabled_ids=body.disabled_ids,
        disabled_types=body.disabled_types,
    )
    return {
        "outcome": "succeeded",
        "policy": {
            "disabled_ids": sorted(policy.disabled_ids),
            "disabled_types": sorted(item.value for item in policy.disabled_types),
        },
    }
