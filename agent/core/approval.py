"""Graph-native approval bridge for side-effecting tool calls.

The policy layer decides *whether* a call needs approval.  This module handles
the LangGraph mechanics: a node binds its live ``RunnableConfig`` for the
duration of a tool call, and ``ToolGateway`` uses that config to call
``langgraph.types.interrupt`` from inside the active graph task.

Outside a graph (unit tests, direct providers), ``request_tool_approval``
returns ``NO_APPROVAL_CONTEXT`` so callers can keep the existing fail-closed
``APPROVAL_REQUIRED`` envelope instead of trying to interrupt a non-graph stack.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator


NO_APPROVAL_CONTEXT = object()

_tool_config: ContextVar[Any] = ContextVar(
    "agent_tool_approval_config", default=None,
)
_preapproved_keys: ContextVar[frozenset[str]] = ContextVar(
    "agent_preapproved_tool_keys", default=frozenset(),
)


@contextmanager
def tool_approval_scope(config: Any) -> Iterator[None]:
    """Bind the current graph node config while a tool is executing."""
    token: Token = _tool_config.set(config)
    try:
        yield
    finally:
        _tool_config.reset(token)


def request_tool_approval(payload: dict) -> Any:
    """Pause the graph and return the approval value supplied on resume.

    LangGraph's ``interrupt`` reads its runnable config from a context variable.
    Current LangGraph/LangChain combinations do not always propagate that
    variable into async node executors, so re-bind the node config locally.
    """
    config = _tool_config.get()
    if config is None:
        return NO_APPROVAL_CONTEXT

    from langchain_core.runnables.config import var_child_runnable_config
    from langgraph.types import interrupt

    token = var_child_runnable_config.set(config)
    try:
        return interrupt(payload)
    finally:
        var_child_runnable_config.reset(token)


def approval_granted(value: Any) -> bool:
    """Accept the explicit resume shapes used by the API and tests."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return value.get("approved") is True
    return False


@contextmanager
def preapproved_tool_scope(keys) -> Iterator[None]:
    """Mark exact side-effecting calls already approved by a batch preflight."""
    token: Token = _preapproved_keys.set(frozenset(str(key) for key in keys if key))
    try:
        yield
    finally:
        _preapproved_keys.reset(token)


def consume_preapproved(key: str) -> bool:
    """Return and consume one preapproved key for the current execution."""
    current = _preapproved_keys.get()
    if not key or key not in current:
        return False
    _preapproved_keys.set(current - {key})
    return True


def interrupt_values(source: Any) -> list[Any]:
    """Normalise ``ainvoke`` results and ``StateSnapshot`` to interrupt values."""
    if source is None:
        return []
    if isinstance(source, dict):
        items = source.get("__interrupt__") or ()
    else:
        items = getattr(source, "interrupts", ()) or ()
    values: list[Any] = []
    for item in items:
        values.append(getattr(item, "value", item))
    return values


def pending_tool_approval(source: Any) -> dict | None:
    """Return the first pending tool-approval payload, if any."""
    for value in interrupt_values(source):
        if isinstance(value, dict) and value.get("kind") == "tool_approval":
            return value
    return None
