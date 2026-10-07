"""Hard model-input budgeting with tool-call/result pair preservation."""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_core.messages import SystemMessage


@dataclass
class ContextFit:
    messages: list[Any]
    input_tokens: int
    budget_tokens: int
    safety_ratio: float
    original_tokens: int
    dropped_messages: int = 0
    truncated_messages: list[str] = field(default_factory=list)
    system_truncated: bool = False

    @property
    def changed(self) -> bool:
        return bool(
            self.dropped_messages
            or self.truncated_messages
            or self.system_truncated
        )

    def trace_view(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "budget_tokens": self.budget_tokens,
            "original_tokens": self.original_tokens,
            "safety_ratio": self.safety_ratio,
            "dropped_messages": self.dropped_messages,
            "truncated_messages": list(self.truncated_messages),
            "system_truncated": self.system_truncated,
        }


_last_fit: contextvars.ContextVar[ContextFit | None] = contextvars.ContextVar(
    "last_context_fit", default=None,
)


def _estimate(text: str) -> int:
    from ..memory import _estimate_tokens

    return _estimate_tokens(text or "")


def _truncate(text: str, max_tokens: int) -> str:
    from ..memory import _truncate_to_tokens

    return _truncate_to_tokens(text, max_tokens)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(content)


def _message_tokens(message: Any) -> int:
    total = _estimate(_message_text(message))
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        try:
            total += _estimate(json.dumps(
                tool_calls, ensure_ascii=False, default=str,
            ))
        except Exception:  # noqa: BLE001
            total += _estimate(str(tool_calls))
    return total


def _copy_with_content(message: Any, content: str) -> Any:
    try:
        return message.model_copy(update={"content": content})
    except Exception:  # noqa: BLE001
        return message


def _message_units(messages: list[Any]) -> list[list[Any]]:
    """Group an assistant tool call with its immediately following results."""
    units: list[list[Any]] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        tool_calls = getattr(message, "tool_calls", None) or []
        if getattr(message, "type", "") == "ai" and tool_calls:
            ids = {
                str(tc.get("id") or "")
                for tc in tool_calls
                if isinstance(tc, dict)
            } | {
                str(getattr(tc, "id", "") or "")
                for tc in tool_calls
                if not isinstance(tc, dict)
            }
            ids.discard("")
            j = i + 1
            while j < len(messages):
                if getattr(messages[j], "type", "") != "tool":
                    break
                call_id = str(getattr(messages[j], "tool_call_id", "") or "")
                if ids and call_id not in ids:
                    break
                j += 1
            if j > i + 1:
                units.append(messages[i:j])
                i = j
                continue
        units.append([message])
        i += 1
    return units


def _unit_tokens(unit: Iterable[Any]) -> int:
    return sum(_message_tokens(message) for message in unit)


def _shrink_unit(unit: list[Any], budget: int) -> list[Any]:
    """Compact the longest payload until one unit fits without splitting pairs."""
    if budget <= 0:
        return []
    out = list(unit)
    for _ in range(12):
        current = _unit_tokens(out)
        if current <= budget:
            return out
        idx = max(range(len(out)), key=lambda n: _message_tokens(out[n]))
        message = out[idx]
        target = max(16, _message_tokens(message) - (current - budget) - 8)
        text = _message_text(message)
        if getattr(message, "type", "") == "tool":
            try:
                from ..tool_contract import truncate_tool_result

                compact = truncate_tool_result(text, max(256, target * 4))
            except Exception:  # noqa: BLE001
                compact = _truncate(text, target)
        else:
            compact = _truncate(text, target)
        if compact == text:
            break
        out[idx] = _copy_with_content(message, compact)
    return out if _unit_tokens(out) <= budget else []


def fit_messages(
    messages: list[Any],
    *,
    max_tokens: int,
    safety_ratio: float = 1.0,
) -> ContextFit:
    """Return messages guaranteed to fit the effective input budget.

    System prompts are retained first. Conversation is assembled from newest
    complete message units to oldest; an assistant tool call and its tool
    results are never split.
    """
    original_tokens = sum(_message_tokens(message) for message in messages)
    ratio = min(1.0, max(0.1, float(safety_ratio or 1.0)))
    budget = max(1, int(max(1, max_tokens) * ratio))
    if original_tokens <= budget:
        return ContextFit(
            messages=list(messages), input_tokens=original_tokens,
            budget_tokens=budget, safety_ratio=ratio,
            original_tokens=original_tokens,
        )

    system = [
        message for message in messages
        if getattr(message, "type", "") == "system"
    ]
    conversation = [
        message for message in messages
        if getattr(message, "type", "") != "system"
    ]
    system_text = "\n\n".join(_message_text(message) for message in system)
    system_tokens = _estimate(system_text)
    system_truncated = False
    if system_tokens > max(1, int(budget * 0.65)):
        system_target = max(1, int(budget * 0.50))
        system_text = _truncate(system_text, system_target)
        system_tokens = _estimate(system_text)
        system_truncated = True
    available = max(0, budget - system_tokens)

    units = _message_units(conversation)
    selected: list[list[Any]] = []
    spent = 0
    dropped = 0
    truncated: list[str] = []
    for unit in reversed(units):
        cost = _unit_tokens(unit)
        remaining = available - spent
        if cost <= remaining:
            selected.append(unit)
            spent += cost
            continue
        if not selected:
            compact = _shrink_unit(unit, remaining)
            if compact:
                selected.append(compact)
                spent += _unit_tokens(compact)
                truncated.append(
                    str(getattr(compact[-1], "type", "") or "message")
                )
                continue
        dropped += len(unit)

    ordered = [
        message
        for unit in reversed(selected)
        for message in unit
    ]
    fitted: list[Any] = []
    if system_text:
        fitted.append(SystemMessage(content=system_text))
    fitted.extend(ordered)
    if not ordered and conversation:
        last = conversation[-1]
        compact = _truncate(_message_text(last), max(1, available))
        if compact:
            fitted.append(_copy_with_content(last, compact))
            truncated.append(str(getattr(last, "type", "") or "message"))
            dropped = max(0, dropped - 1)
    input_tokens = sum(_message_tokens(message) for message in fitted)
    lead = ContextFit(
        messages=fitted,
        input_tokens=input_tokens,
        budget_tokens=budget,
        safety_ratio=ratio,
        original_tokens=original_tokens,
        dropped_messages=dropped,
        truncated_messages=truncated,
        system_truncated=system_truncated,
    )
    _last_fit.set(lead)
    return lead


def fit_for_context(messages: list[Any], ctx: Any = None) -> ContextFit:
    """Fit against the turn's frozen model window and output reserve."""
    if ctx is None:
        from .execution_context import get_current_execution_context

        ctx = get_current_execution_context()
    if ctx is None:
        return ContextFit(
            messages=list(messages),
            input_tokens=sum(_message_tokens(message) for message in messages),
            budget_tokens=0,
            safety_ratio=1.0,
            original_tokens=sum(_message_tokens(message) for message in messages),
        )
    budget = ctx.budget.usable_context_tokens()
    fit = fit_messages(
        list(messages),
        max_tokens=budget,
        safety_ratio=ctx.budget.token_safety_ratio,
    )
    _last_fit.set(fit)
    return fit


def get_last_context_fit() -> ContextFit | None:
    return _last_fit.get()
