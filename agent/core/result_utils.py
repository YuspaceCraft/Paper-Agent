"""Shared helpers for final-answer extraction and result classification."""

from __future__ import annotations

import json
from typing import Any


_PLACEHOLDER_ANSWERS = {
    "no answer produced.",
    "(no answer produced.)",
    "子代理未生成最终结果。",
}


def is_placeholder_answer(content: Any) -> bool:
    text = str(content or "").strip().casefold()
    return text in _PLACEHOLDER_ANSWERS


def current_turn_messages(messages: list) -> list:
    """Return messages produced after the latest user message."""
    last_human = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", "") == "human":
            last_human = index
    return messages[last_human + 1:] if last_human >= 0 else list(messages)


def extract_final_answer(messages: list) -> str:
    """Return the last complete AI answer from the current turn only."""
    for message in reversed(current_turn_messages(list(messages or []))):
        if getattr(message, "type", "") != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = str(getattr(message, "content", "") or "").strip()
        if text and not is_placeholder_answer(text):
            return text
    return ""


def structured_error_payload(content: Any) -> dict | None:
    """Normalize a failure envelope or structured error answer."""
    from agent.tool_contract import parse_tool_result

    text = str(content or "")
    parsed = parse_tool_result(text)
    if parsed.is_envelope and parsed.outcome in {
        "failed", "timed_out", "cancelled",
    }:
        return {
            "error": parsed.code or parsed.error_type or "tool_failed",
            "detail": parsed.error or "",
            "error_type": parsed.error_type,
        }
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("error") or payload.get("code"):
        return payload
    return None


def tool_ui_status(outcome: str) -> str:
    """Map operation outcomes to the status vocabulary consumed by the UI."""
    return {
        "succeeded": "success",
        "partial": "partial",
        "failed": "error",
        "timed_out": "error",
        "cancelled": "error",
        "interrupted": "interrupted",
        "skipped": "skipped",
    }.get(str(outcome or ""), "error")
