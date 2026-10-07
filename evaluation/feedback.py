"""LangSmith feedback write-back for selected evaluation traces."""

from __future__ import annotations

from typing import Any


async def submit_feedback(
    *,
    trace_id: str,
    key: str,
    score: float | int | bool | None = None,
    comment: str = "",
    run_id: str = "",
    project_name: str = "",
    client: Any = None,
) -> dict:
    """Attach one feedback entry to a run's LangSmith root run.

    The caller may pass the root run id directly. Otherwise resolve it from the
    evaluation trace archive mapping created during the run.
    """
    if not trace_id:
        raise ValueError("trace_id is required")
    if not key:
        raise ValueError("feedback key is required")

    resolved = run_id
    if not resolved:
        from agent.core.trace_export import lookup_langsmith_run_id

        resolved = await lookup_langsmith_run_id(trace_id)
    if not resolved:
        raise ValueError(f"LangSmith run id not found for trace: {trace_id}")

    if client is None:
        import langsmith
        client = langsmith.Client()
    client.create_feedback(
        run_id=resolved,
        key=key,
        score=score,
        comment=comment or None,
    )
    return {
        "outcome": "succeeded",
        "trace_id": trace_id,
        "run_id": resolved,
        "key": key,
        "score": score,
        "project": project_name,
    }
