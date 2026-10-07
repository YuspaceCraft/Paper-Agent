"""Capability boundary checks and handoff decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CapabilityDecision:
    allowed: bool
    capability: str
    code: str = "CAPABILITY_ALLOWED"
    message: str = ""
    matched_rules: tuple[str, ...] = field(default_factory=tuple)


_UNSUPPORTED_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "FINANCIAL_TRANSFER",
        re.compile(
            r"(?:transfer|send|move|wire)\s+(?:money|funds|crypto|payment)"
            r"|(?:pay|purchase|buy)\s+(?:for me|on my behalf)"
            r"|(?:转账|汇款|付款|代付|代购|购买).{0,20}"
            r"(?:银行|账户|钱包|加密货币|订单)?",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "ACCOUNT_CONTROL",
        re.compile(
            r"(?:log\s*in|sign\s*in|take\s+over|access)\s+"
            r"(?:my|someone(?:'s)?|another)\s+account"
            r"|(?:登录|接管|访问).{0,12}(?:我的|他人的|别人的).{0,8}"
            r"(?:账号|账户)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "PHYSICAL_WORLD_ACTION",
        re.compile(
            r"(?:book|order|reserve|unlock|control)\s+"
            r"(?:me\s+)?(?:a\s+)?(?:flight|hotel|car|door|device)"
            r"|(?:预订|下单|控制|开锁|操作).{0,12}"
            r"(?:航班|酒店|车辆|门锁|设备)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "MEDICAL_OR_LEGAL_DECISION",
        re.compile(
            r"(?:diagnose|prescribe|represent)\s+(?:me|my|this)"
            r"|(?:诊断|开药|处方|代理诉讼).{0,12}(?:我|本人|这个病例)?",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

_CAPABILITY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "paper_download_ingest",
        re.compile(
            r"(?:download|save|import|ingest).{0,12}(?:paper|pdf|arxiv)"
            r"|(?:下载|保存|导入|入库).{0,12}(?:论文|pdf|文献)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "creation_write",
        re.compile(
            r"(?:write|draft|polish|rewrite|outline).{0,16}"
            r"(?:paper|article|report|survey|section)"
            r"|(?:写|撰写|起草|润色|改写|大纲).{0,16}"
            r"(?:论文|文章|报告|综述|章节)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "coding_experiment",
        re.compile(
            r"(?:run|reproduce|debug|tune|implement).{0,16}"
            r"(?:experiment|code|model|training)"
            r"|(?:跑|复现|调试|调参|实现|优化).{0,16}"
            r"(?:实验|代码|模型|训练)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "paper_research",
        re.compile(
            r"(?:paper|literature|research|compare|summarize|read)"
            r"|(?:论文|文献|研究|对比|比较|总结|阅读)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


def assess_capability(text: str) -> CapabilityDecision:
    """Determine whether the agent can execute the request itself."""
    value = str(text or "").strip()
    unsupported = tuple(
        code for code, pattern in _UNSUPPORTED_RULES if pattern.search(value)
    )
    if unsupported:
        return CapabilityDecision(
            allowed=False,
            capability="unsupported",
            code="HANDOFF_REQUIRED",
            message=(
                "该请求需要执行现实世界或受保护账户操作，当前 Agent 无法完成。"
                "我可以继续协助检索资料、整理方案或生成执行清单。"
            ),
            matched_rules=unsupported,
        )

    for capability, pattern in _CAPABILITY_RULES:
        if pattern.search(value):
            return CapabilityDecision(True, capability)
    return CapabilityDecision(True, "general_chat")
