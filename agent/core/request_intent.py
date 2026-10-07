"""Small, deterministic request-intent helpers shared by plan and graph nodes."""

from __future__ import annotations


_VERBATIM_HINTS = (
    "原文",
    "逐字",
    "完整内容",
    "原样",
    "verbatim",
    "exact text",
    "original text",
    "original chapter",
    "original section",
    "source text",
    "full text",
    "do not summarize",
    "without summarizing",
)


def is_verbatim_request(text: str) -> bool:
    """True when the user explicitly wants source text rather than a summary."""
    lowered = str(text or "").casefold()
    return any(hint in lowered for hint in _VERBATIM_HINTS)
