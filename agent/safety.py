"""
safety.py — 规则安全层：PII 脱敏 + 权限门。

纯代码，无新依赖。两个职责：
1. PII 脱敏（输出审核）：邮箱/手机号/身份证/银行卡正则替换。
2. 权限门（工具调用前置）：读 ToolDef.annotations 的 readOnlyHint 判断
   destructive 工具，按 role 决定是否放行。

过滤器链：输出过滤器依次应用（当前仅 PII 脱敏），后续加规则不改调用点。
"""

from __future__ import annotations

import os
import re


# ---- PII 脱敏 ----

# 顺序敏感：长格式先匹配，避免被短格式吞掉。
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'), '[邮箱]'),
    (re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)'), '[手机号]'),
    (re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)'), '[身份证]'),
    (re.compile(r'(?<!\d)\d{16,19}(?!\d)'), '[银行卡]'),
]


def mask_pii(text: str) -> str:
    """Mask common PII (email/phone/ID/bank card) in a string.

    Best-effort regex. False positives over-mask (safe direction); values
    split across streaming token chunks may slip through — see router.
    """
    if not text:
        return text
    for pat, repl in _PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


# ---- 权限门 ----

def get_role() -> str:
    """当前用户角色。单用户场景默认 admin（可触发 destructive 工具）。"""
    return os.getenv("AGENT_USER_ROLE", "admin")


# role → 能力。destructive 工具（download_paper/ingest_paper）默认需授权。
# admin 放行，user 拒绝。后续 HITL（Phase 6）把「拒绝」升级为「审批后放行」。
ROLE_CAPABILITIES: dict[str, dict[str, bool]] = {
    "admin": {"destructive": True},
    "user": {"destructive": False},
}


def is_destructive(annotations: dict) -> bool:
    """判断工具是否为破坏性操作。

    builtin ToolDef 无 destructiveHint 字段（计划设想有误）——实际用
    readOnlyHint=False 标记 download_paper/ingest_paper 这类写操作。
    MCP 透传的 destructiveHint=True 也走这里统一判定。
    """
    if annotations.get("destructiveHint") is True:
        return True
    return annotations.get("readOnlyHint", True) is False


def tool_allowed(annotations: dict, role: str | None = None) -> bool:
    """权限门：非破坏性工具放行；破坏性工具按 role 能力表判定。"""
    if not is_destructive(annotations):
        return True
    role = role or get_role()
    return ROLE_CAPABILITIES.get(role, {}).get("destructive", False)


# ---- 输出过滤器链 ----

_OUTPUT_FILTERS = [mask_pii]


def sanitize_output(text: str) -> str:
    """依次应用输出过滤器。当前仅 PII 脱敏；加规则在此追加，不改调用点。"""
    for fn in _OUTPUT_FILTERS:
        text = fn(text)
    return text


# ---- self-check ----

if __name__ == "__main__":
    raw = "联系我 alice@example.com 电话 13800138000 身份证 110101199001011234 卡 6222021234567890"
    masked = mask_pii(raw)
    assert "alice@example.com" not in masked, "email must be masked"
    assert "13800138000" not in masked, "phone must be masked"
    assert "110101199001011234" not in masked, "ID must be masked"
    assert "6222021234567890" not in masked, "bank card must be masked"

    # 权限门：destructive 工具默认拒绝非 admin 角色
    destructive = {"readOnlyHint": False}
    readonly = {"readOnlyHint": True}
    assert tool_allowed(destructive, "admin") is True
    assert tool_allowed(destructive, "user") is False
    assert tool_allowed(readonly, "user") is True
    print("Phase 2 safety self-check OK")
