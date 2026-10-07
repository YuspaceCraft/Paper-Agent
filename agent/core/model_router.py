"""Task-level model routing frozen into each turn's configuration snapshot."""

from __future__ import annotations

import json
import os
from typing import Mapping

MODEL_TASKS = (
    "router",
    "summary",
    "planner",
    "agent",
    "verify",
    "synthesizer",
    "chat",
    "subagent",
    "notifier",
)

_SMALL_MODEL_TASKS = {"router", "summary", "notifier"}


def resolve_model_routes(
    main_model: str,
    *,
    small_model: str = "",
    explicit: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return one concrete model per task.

    The safe default keeps every task on the main model. Setting
    ``AGENT_MODEL_SMALL`` moves routing/summary/notification to a cheaper model.
    ``AGENT_MODEL_ROUTES`` can override individual tasks with a JSON object.
    """
    routes = {task: main_model for task in MODEL_TASKS}
    if small_model:
        for task in _SMALL_MODEL_TASKS:
            routes[task] = small_model

    raw_env = explicit
    if raw_env is None:
        raw_env = {}
        value = os.getenv("AGENT_MODEL_ROUTES", "").strip()
        if value:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    raw_env = parsed
            except ValueError:
                raw_env = {}
    for task, model in (raw_env or {}).items():
        if task in routes and str(model or "").strip():
            routes[task] = str(model).strip()
    return routes
