"""Structured completion and continuation state for one agent turn."""

from __future__ import annotations

from typing import Any

from .confidence_policy import calibrate_final_confidence


def _outstanding_items(state: dict[str, Any]) -> list[dict]:
    verification = state.get("verification") or {}
    outstanding = verification.get("outstanding")
    if isinstance(outstanding, list):
        return [item for item in outstanding if isinstance(item, dict)]
    plan = state.get("plan") or []
    results = {
        str(item.get("step_id") or ""): item
        for item in (state.get("subagent_results") or [])
        if isinstance(item, dict)
    }
    return [
        {
            "id": str(step.get("id") or ""),
            "description": str(step.get("description") or ""),
            "reason": "step has no successful result",
        }
        for step in plan
        if isinstance(step, dict)
        and str(step.get("id") or "") not in results
    ]


def build_completion_report(
    state: dict[str, Any] | None,
    *,
    answer: str,
    error: str | None = None,
    output_validation: dict | None = None,
) -> dict:
    """Return stable status/confidence/continuation metadata for an API turn."""
    state = state if isinstance(state, dict) else {}
    verification = state.get("verification") or {}
    verification_status = str(verification.get("status") or "")
    outstanding = _outstanding_items(state)
    plan = state.get("plan") or []
    plan_cost = state.get("plan_cost") or {}

    if error == "approval_required":
        status = "interrupted"
    elif error:
        status = "failed"
    elif plan_cost.get("exceeded") and not answer:
        status = "partial"
    elif verification_status == "satisfied" and not outstanding:
        status = "completed"
    elif verification_status in {"partial", "failed", "no_evidence"}:
        status = "partial" if verification_status != "failed" else "failed"
    elif outstanding:
        status = "partial"
    elif str(answer or "").strip():
        status = "completed"
    else:
        status = "failed"

    can_resume = bool(
        status in {"partial", "interrupted"}
        and (outstanding or status == "interrupted")
    )
    next_actions: list[str] = []
    if status == "interrupted":
        next_actions.append("resolve_pending_approval")
    if outstanding:
        next_actions.append("continue_outstanding_steps")
    if plan_cost.get("exceeded"):
        next_actions.append("increase_budget_or_reduce_scope")
    if status == "failed":
        next_actions.append("inspect_trace_or_retry")

    confidence = calibrate_final_confidence(
        state,
        answer=answer,
        output_validation=output_validation,
        completion_status=status,
    )
    return {
        "status": status,
        "done": int(
            (verification.get("done") if isinstance(verification, dict) else 0)
            or state.get("plan_progress")
            or 0
        ),
        "total": int(
            (verification.get("total") if isinstance(verification, dict) else 0)
            or len(plan)
            or 0
        ),
        "outstanding": outstanding,
        "can_resume": can_resume,
        "next_actions": next_actions,
        "plan_cost_exceeded": bool(plan_cost.get("exceeded")),
        "confidence": confidence.trace_view(),
    }
