"""Deterministic user-input policy.

The prompt trust boundary explains that user and tool content are data, but it
does not prevent a malicious request from reaching the model.  This module is
the deterministic gate used by graph and API entrypoints before a turn starts.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


DEFAULT_MAX_INPUT_CHARS = 20000


@dataclass(frozen=True)
class InputPolicyDecision:
    """Safe, serializable result of a pre-execution input check."""

    allowed: bool
    code: str
    message: str
    sanitized_text: str
    matched_rules: tuple[str, ...] = field(default_factory=tuple)
    original_chars: int = 0


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# These patterns intentionally target instructions that attempt to change the
# agent's authority model, reveal hidden prompts/secrets, or disable controls.
# Ordinary security discussion is still subject to the normal model flow unless
# it directly asks for an instruction override or secret disclosure.
_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ROLE_OVERRIDE",
        re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)?"
            r"(?:previous|prior|above)\s+(?:instruction|prompt|rule)s?"
            r"|忽略|无视|忘掉.{0,12}(?:之前|以上|所有|先前).{0,8}"
            r"(?:指令|提示|规则|要求)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "SYSTEM_PROMPT_EXFIL",
        re.compile(
            r"(?:reveal|show|print|display|leak|dump|repeat)\s+"
            r"(?:your|the\s+)?(?:system|developer|hidden)\s+"
            r"(?:prompt|message|instructions?)"
            r"|(?:输出|展示|打印|泄露|复述).{0,8}"
            r"(?:系统|开发者|隐藏).{0,8}(?:提示词|消息|指令)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "SECRET_EXFIL",
        re.compile(
            r"(?:reveal|show|print|display|leak|dump|send|give)\s+"
            r"(?:me\s+)?(?:the\s+)?(?:api[\s_-]?key|secret|password|token)"
            r"|(?:告诉我|输出|展示|打印|泄露|发送).{0,8}"
            r"(?:api[\s_-]?key|密钥|口令|密码|令牌)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "SAFETY_BYPASS",
        re.compile(
            r"(?:bypass|disable|turn\s+off|override)\s+"
            r"(?:all\s+)?(?:safety|security|permission|approval|guardrail)s?"
            r"|(?:绕过|关闭|禁用|跳过).{0,8}"
            r"(?:安全|权限|审批|审核|防护)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "JAILBREAK_PERSONA",
        re.compile(
            r"\b(?:jailbreak|DAN\s+mode|developer\s+mode)\b"
            r"|越狱模式|开发者模式.{0,8}(?:绕过|无视|关闭)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


def input_guard_enabled() -> bool:
    raw = (os.getenv("AGENT_INPUT_GUARD", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _max_input_chars() -> int:
    try:
        return max(
            1000,
            int(os.getenv("AGENT_INPUT_MAX_CHARS", str(DEFAULT_MAX_INPUT_CHARS))),
        )
    except (TypeError, ValueError):
        return DEFAULT_MAX_INPUT_CHARS


def guard_user_input(text: str, *, max_chars: int | None = None) -> InputPolicyDecision:
    """Normalize and evaluate one user message before any model/tool call."""
    raw = str(text or "")
    normalized = _CONTROL_RE.sub("", raw).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    original_chars = len(normalized)

    if not normalized:
        return InputPolicyDecision(
            allowed=False,
            code="INPUT_EMPTY",
            message="请输入有效的问题或任务。",
            sanitized_text="",
            original_chars=original_chars,
        )

    limit = _max_input_chars() if max_chars is None else max(1, int(max_chars))
    if original_chars > limit:
        return InputPolicyDecision(
            allowed=False,
            code="INPUT_TOO_LARGE",
            message=f"输入超过安全长度限制（{limit} 字符），请缩小范围后重试。",
            sanitized_text=normalized[:limit],
            original_chars=original_chars,
        )

    if not input_guard_enabled():
        return InputPolicyDecision(
            allowed=True,
            code="INPUT_ALLOWED",
            message="",
            sanitized_text=normalized,
            original_chars=original_chars,
        )

    matched = tuple(
        code for code, pattern in _INJECTION_RULES if pattern.search(normalized)
    )
    if matched:
        return InputPolicyDecision(
            allowed=False,
            code="INPUT_BLOCKED",
            message="安全策略已阻止本次请求。请改为描述可执行的正常任务。",
            sanitized_text=normalized,
            matched_rules=matched,
            original_chars=original_chars,
        )

    from .content_safety import assess_content_safety

    safety = assess_content_safety(normalized)
    if safety.blocked:
        return InputPolicyDecision(
            allowed=False,
            code="INPUT_BLOCKED",
            message=safety.safe_response,
            sanitized_text=normalized,
            matched_rules=safety.matched_rules,
            original_chars=original_chars,
        )

    return InputPolicyDecision(
        allowed=True,
        code="INPUT_ALLOWED",
        message="",
        sanitized_text=normalized,
        original_chars=original_chars,
    )


def scan_untrusted_content(text: str) -> tuple[str, ...]:
    """Return injection rule codes found in tool/retrieval content.

    External content is not blocked outright because it may be legitimate
    research material.  Callers should attach the result as provenance metadata
    and explicitly remind the model that the content remains DATA.
    """
    value = str(text or "")
    return tuple(
        code for code, pattern in _INJECTION_RULES if pattern.search(value)
    )
