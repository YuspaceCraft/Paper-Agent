"""Local and optional remote content-safety classification."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


SAFE_POLICY_RESPONSE = (
    "无法提供该请求所要求的可操作高风险内容。"
    "我可以改为提供安全概述、风险识别、预防措施或合规替代方案。"
)


@dataclass(frozen=True)
class ContentSafetyDecision:
    blocked: bool
    category: str = ""
    severity: str = "none"
    source: str = "local"
    reason: str = ""
    matched_rules: tuple[str, ...] = field(default_factory=tuple)
    safe_response: str = SAFE_POLICY_RESPONSE

    def trace_view(self) -> dict:
        return {
            "blocked": self.blocked,
            "category": self.category,
            "severity": self.severity,
            "source": self.source,
            "reason": self.reason,
            "matched_rules": list(self.matched_rules),
        }


_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "SELF_HARM",
        "critical",
        re.compile(
            r"(?:how|ways?|instructions?|steps?|method)\s+to\s+"
            r"(?:kill\s+myself|commit\s+suicide|self[-\s]?harm)"
            r"|(?:怎么|如何|方法|步骤).{0,12}"
            r"(?:自杀|轻生|伤害自己|无痛死亡)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "VIOLENCE_WEAPONS",
        "critical",
        re.compile(
            r"(?:build|make|assemble|synthesi[sz]e|detonate)\s+"
            r"(?:a\s+)?(?:bomb|explosive|weapon|nerve\s+agent|poison)"
            r"|(?:制作|制造|组装|合成|引爆).{0,12}"
            r"(?:炸弹|爆炸物|武器|神经毒剂|毒药)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "MALWARE",
        "critical",
        re.compile(
            r"(?:write|create|develop|generate)\s+"
            r"(?:ransomware|malware|keylogger|rootkit|credential\s+stealer)"
            r"|(?:编写|开发|生成).{0,12}"
            r"(?:勒索软件|恶意软件|键盘记录器|凭据窃取器)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "SEXUAL_EXPLOITATION",
        "critical",
        re.compile(
            r"(?:sexual|explicit|pornographic).{0,24}"
            r"(?:minor|child|underage|school\s+student)"
            r"|(?:色情|性内容).{0,16}(?:未成年|儿童|小学生|中学生)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "PRIVACY_ABUSE",
        "high",
        re.compile(
            r"(?:find|reveal|expose|dox)\s+(?:someone(?:'s)?|a\s+person(?:'s)?|"
            r"this\s+person(?:'s)?)\s+(?:home\s+address|phone\s+number|"
            r"social\s+security|identity\s+number)"
            r"|(?:查找|曝光|人肉).{0,12}(?:他人|某人|这个人).{0,12}"
            r"(?:住址|电话|身份证|社保号)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "CREDENTIAL_THEFT",
        "high",
        re.compile(
            r"(?:steal|harvest|capture)\s+(?:passwords?|api\s+keys?|tokens?|"
            r"session\s+cookies?)"
            r"|(?:窃取|批量采集).{0,12}(?:密码|密钥|令牌|会话凭据)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


def assess_content_safety(text: str) -> ContentSafetyDecision:
    """Classify actionable harmful content using deterministic local rules."""
    value = str(text or "")
    matched = tuple(
        code for code, _severity, pattern in _RULES if pattern.search(value)
    )
    if not matched:
        return ContentSafetyDecision(blocked=False)
    severity_by_code = {
        code: severity for code, severity, _pattern in _RULES
    }
    severity = max(
        (severity_by_code[code] for code in matched),
        key=lambda item: {"high": 1, "critical": 2}.get(item, 0),
    )
    return ContentSafetyDecision(
        blocked=True,
        category=matched[0],
        severity=severity,
        reason="actionable harmful-content pattern detected",
        matched_rules=matched,
    )


def remote_moderation_enabled() -> bool:
    return bool((os.getenv("AGENT_MODERATION_ENDPOINT", "") or "").strip())


def _remote_category(payload: dict) -> tuple[str, str]:
    try:
        result = (payload.get("results") or [])[0]
    except (AttributeError, IndexError, TypeError):
        return "REMOTE_UNKNOWN", "high"
    categories = result.get("categories") or {}
    scores = result.get("category_scores") or {}
    flagged = [
        str(name) for name, value in categories.items() if bool(value)
    ]
    if flagged:
        category = max(
            flagged,
            key=lambda name: float(scores.get(name) or 0.0),
        )
        return category.upper(), "high"
    return "REMOTE_FLAGGED", "high"


async def classify_content_safety(
    text: str,
    *,
    timeout_seconds: float | None = None,
) -> ContentSafetyDecision:
    """Run the local classifier, then optional remote moderation if configured."""
    local = assess_content_safety(text)
    if local.blocked or not remote_moderation_enabled():
        return local

    import httpx

    endpoint = str(os.getenv("AGENT_MODERATION_ENDPOINT") or "").strip()
    model = str(os.getenv("AGENT_MODERATION_MODEL") or "omni-moderation-latest")
    key_env = str(os.getenv("AGENT_MODERATION_API_KEY_ENV") or "").strip()
    api_key = os.getenv(key_env, "") if key_env else ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(
            timeout=float(
                timeout_seconds
                if timeout_seconds is not None
                else os.getenv("AGENT_MODERATION_TIMEOUT", "8")
            )
        ) as client:
            response = await client.post(
                endpoint,
                headers=headers,
                content=json.dumps(
                    {"model": model, "input": str(text or "")},
                    ensure_ascii=False,
                ),
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001
        fail_open = os.getenv(
            "AGENT_MODERATION_FAIL_MODE", "closed"
        ).strip().lower() in {"open", "allow"}
        if fail_open:
            return ContentSafetyDecision(
                blocked=False,
                source="remote",
                reason=f"remote moderation unavailable: {type(exc).__name__}",
            )
        return ContentSafetyDecision(
            blocked=True,
            category="MODERATION_UNAVAILABLE",
            severity="high",
            source="remote",
            reason=f"remote moderation unavailable: {type(exc).__name__}",
        )

    if not isinstance(payload, dict):
        return ContentSafetyDecision(
            blocked=True,
            category="MODERATION_PROTOCOL_INVALID",
            severity="high",
            source="remote",
            reason="remote moderation returned a non-object payload",
        )
    result = (payload.get("results") or [{}])[0]
    if not isinstance(result, dict):
        return ContentSafetyDecision(
            blocked=True,
            category="MODERATION_PROTOCOL_INVALID",
            severity="high",
            source="remote",
            reason="remote moderation result is not an object",
        )
    if not bool(result.get("flagged")):
        return ContentSafetyDecision(blocked=False, source="remote")
    category, severity = _remote_category(payload)
    return ContentSafetyDecision(
        blocked=True,
        category=category,
        severity=severity,
        source="remote",
        reason="remote moderation flagged the content",
        matched_rules=(category,),
    )
