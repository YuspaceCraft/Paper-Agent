"""
stream.py — plan-mode execution event channel.

executor_node / plan_node push structured progress events (plan / tool_start /
tool_end) into a per-request asyncio.Queue held in a contextvar. The SSE
endpoint (web/api/routers/agent.py) installs the queue via set_event_queue()
and drains it concurrently with the message stream.

Why not LangGraph get_stream_writer(): it relies on contextvar propagation
that is broken in async nodes on Python < 3.11 (raises "outside of a runnable
context"). A plain contextvar holding a *reference* to a shared queue does
propagate (contextvars copy the reference on task creation; all tasks mutate
the same queue object).
"""

from __future__ import annotations

import contextvars

_event_q: contextvars.ContextVar = contextvars.ContextVar(
    "agent_event_q", default=None
)


def emit(event: dict) -> None:
    """Push a structured event to the current request's queue (no-op outside
    a stream request, e.g. non-streaming /chat or agent.graph.run())."""
    q = _event_q.get()
    if q is not None:
        q.put_nowait(event)


def set_event_queue(q) -> contextvars.Token:
    """Install the queue for the current request; returns a reset token."""
    return _event_q.set(q)


def reset_event_queue(token: contextvars.Token) -> None:
    _event_q.reset(token)


# ---- subagent scope (for hierarchical tool events) ----
# While a subagent runs, leaf tool calls are tagged with {agent, id} so the
# frontend can nest them under the *specific* parent step (by id, not name —
# the same subagent may be called more than once per turn). Only one level
# deep (subagents never spawn subagents).

_scope: contextvars.ContextVar = contextvars.ContextVar(
    "agent_scope", default=None
)


def set_scope(agent: str, run_id: str) -> contextvars.Token:
    """Mark the current task as running inside `agent` (a subagent), with the
    subagent boundary's own event id `run_id` used as the leaf tools' parent."""
    return _scope.set({"agent": agent, "id": run_id})


def reset_scope(token: contextvars.Token) -> None:
    _scope.reset(token)


def current_scope() -> dict | None:
    """{agent, id} if running inside a subagent, else None (main level)."""
    return _scope.get()
