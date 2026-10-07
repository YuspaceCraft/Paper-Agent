"""Opt-in operational canaries for model and moderation providers."""

from __future__ import annotations

import time
from typing import Any

from .content_safety import classify_content_safety, remote_moderation_enabled


async def run_model_canary(model: Any | None = None) -> dict:
    """Probe the configured model chain without returning prompts or secrets."""
    started = time.monotonic()
    if model is None:
        from ..nodes import _get_model

        model = _get_model({"configurable": {}}, task="agent")
    if hasattr(model, "probe"):
        result = await model.probe()
        return {
            "check": "model",
            "status": result.get("status") or "degraded",
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "candidates": result.get("candidates") or [],
            "error": result.get("error") or "",
        }
    try:
        from langchain_core.messages import HumanMessage

        await model.ainvoke([HumanMessage(content="Reply with OK.")])
    except Exception as exc:  # noqa: BLE001
        return {
            "check": "model",
            "status": "degraded",
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "error": type(exc).__name__,
        }
    return {
        "check": "model",
        "status": "ready",
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "error": "",
    }


async def run_moderation_canary() -> dict:
    """Verify that a configured remote moderation endpoint accepts benign text."""
    if not remote_moderation_enabled():
        return {
            "check": "moderation",
            "status": "skipped",
            "reason": "AGENT_MODERATION_ENDPOINT is not configured",
        }
    started = time.monotonic()
    decision = await classify_content_safety(
        "The sky is blue and water is wet.",
    )
    if decision.blocked:
        return {
            "check": "moderation",
            "status": "degraded",
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "error": decision.category or decision.reason,
            "source": decision.source,
        }
    return {
        "check": "moderation",
        "status": "ready",
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "source": decision.source,
    }


async def run_provider_canary(
    *,
    model: Any | None = None,
    include_model: bool = True,
    include_moderation: bool = True,
) -> dict:
    checks: list[dict] = []
    if include_model:
        checks.append(await run_model_canary(model))
    if include_moderation:
        checks.append(await run_moderation_canary())
    status = (
        "ready"
        if checks and all(item.get("status") in {"ready", "skipped"} for item in checks)
        else "degraded"
    )
    return {"status": status, "checks": checks}
