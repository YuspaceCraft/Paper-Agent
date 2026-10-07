"""Per-turn ``ExecutionContext`` construction and propagation.

Every node, model call and tool invocation reads the frozen turn metadata from
here instead of re-reading env vars or YAML mid-turn. The context also carries
the live LangGraph ``RunnableConfig`` so child runs nest under the current
graph run (a bare call creates a detached LangSmith root run — see ADR-0002).

Propagation uses a ``contextvars`` variable rather than graph state: the
context holds callbacks and must never be checkpointed.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import uuid
from typing import Any, Iterable

from .contracts import Budget, ExecutionContext, Permission
from .configuration import build_configuration_snapshot

_current: contextvars.ContextVar[ExecutionContext | None] = contextvars.ContextVar(
    "agent_execution_context", default=None
)


def set_current_execution_context(ctx: ExecutionContext) -> contextvars.Token:
    return _current.set(ctx)


def reset_current_execution_context(token: contextvars.Token) -> None:
    _current.reset(token)


def get_current_execution_context() -> ExecutionContext | None:
    return _current.get()


def graph_version_from_nodes(names: Iterable[str]) -> str:
    """Stable fingerprint of the compiled graph layout.

    Traces and reports reference it so a behaviour change caused by a node or
    edge edit can be separated from a model or prompt change.
    """
    try:
        joined = "|".join(sorted(str(n) for n in names))
    except Exception:  # noqa: BLE001 — never let bookkeeping break a turn
        return "unknown"
    if not joined:
        return "unknown"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def build_budget() -> Budget:
    """Freeze the effective execution envelope for one turn."""
    from ..config import get_limits

    limits = get_limits()

    def _int(env: str, default: int) -> int:
        try:
            return int(os.getenv(env) or default)
        except (TypeError, ValueError):
            return default

    def _float(env: str, default: float) -> float:
        try:
            return float(os.getenv(env) or default)
        except (TypeError, ValueError):
            return default

    return Budget(
        max_steps=_int("AGENT_MAX_STEPS", int(limits.max_steps)),
        max_turns=_int("AGENT_MAX_TURNS", int(limits.max_turns)),
        plan_step_max_steps=_int(
            "AGENT_PLAN_STEP_MAX_STEPS",
            int(limits.plan_step_max_steps),
        ),
        turn_timeout_seconds=_float("AGENT_TURN_TIMEOUT", 900.0),
        token_budget=_int("AGENT_TOKEN_BUDGET", 60000),
        context_window_tokens=_int("AGENT_CONTEXT_WINDOW", 32768),
        max_output_tokens=_int("AGENT_MAX_OUTPUT_TOKENS", 4096),
        token_safety_ratio=_float("AGENT_TOKEN_SAFETY_RATIO", 0.90),
    )


def build_execution_context(
    *,
    thread_id: str,
    runnable_config: Any = None,
    model: str | None = None,
    request_id: str | None = None,
    actor_id: str = "local",
    roles: Iterable[str] | None = None,
    domain: str = "",
    graph_version: str = "",
    eval_dataset_id: str = "",
    trace_id: str | None = None,
    execution_id: str | None = None,
    require_initialized_tools: bool = False,
) -> ExecutionContext:
    """Build the immutable metadata bundle for a single agent turn."""
    from .prompt_registry import (
        active_prompt_snapshot,
    )
    from .policy import permissions_for_roles
    from ..tools import get_tool_registry, tools_initialized

    frozen_roles = set(roles or {"user"})
    frozen_request_id = request_id or uuid.uuid4().hex
    trace = trace_id or uuid.uuid4().hex
    prompt_snapshot = active_prompt_snapshot(thread_id)
    if require_initialized_tools and not tools_initialized():
        raise RuntimeError(
            "tool registry must be initialized before building ExecutionContext"
        )
    try:
        tool_versions = {
            spec.name: spec.version for spec in get_tool_registry().all()
        }
    except Exception:  # noqa: BLE001 — registry may be empty before startup
        if require_initialized_tools:
            raise
        tool_versions = {}

    ctx = ExecutionContext(
        request_id=frozen_request_id,
        execution_id=execution_id or frozen_request_id,
        trace_id=trace,
        thread_id=thread_id,
        actor_id=actor_id,
        roles=frozen_roles,
        permissions=permissions_for_roles(frozen_roles),
        config=build_configuration_snapshot(
            model=model, unit_id=thread_id,
            require_initialized_tools=require_initialized_tools,
        ),
        prompt_bindings=prompt_snapshot["bindings"],
        prompt_evaluation_suites=prompt_snapshot["evaluation_suites"],
        prompt_templates=prompt_snapshot["templates"],
        budget=build_budget(),
        domain=domain,
        graph_version=graph_version,
        model_route=model or os.getenv("LLM_MODEL", "qwen-plus"),
        tool_versions=tool_versions,
        eval_dataset_id=eval_dataset_id,
    )
    if runnable_config is not None:
        ctx.bind_runnable_config(runnable_config)
    return ctx


def turn_metadata(ctx: ExecutionContext, **extra: Any) -> dict[str, Any]:
    """Trace metadata for the root run of a turn."""
    metadata = ctx.trace_metadata()
    metadata.update(extra)
    return metadata
