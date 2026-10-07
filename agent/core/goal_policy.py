"""Lightweight goal contract and drift detection for tool-using turns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class GoalContract:
    original_goal: str
    domain: str
    allowed_actions: frozenset[str]
    entities: tuple[str, ...] = field(default_factory=tuple)

    def trace_view(self) -> dict:
        return {
            "domain": self.domain,
            "allowed_actions": sorted(self.allowed_actions),
            "entities": list(self.entities),
        }


@dataclass(frozen=True)
class GoalDrift:
    score: float
    severity: str
    reasons: tuple[str, ...]
    drifted_tools: tuple[str, ...]

    @property
    def should_remind(self) -> bool:
        return self.score >= 0.5

    def trace_view(self) -> dict:
        return {
            "score": round(self.score, 3),
            "severity": self.severity,
            "reasons": list(self.reasons),
            "drifted_tools": list(self.drifted_tools),
        }


_WRITE_RE = re.compile(
    r"(?:write|save|create|edit|delete|upload|ingest|download|import|"
    r"写|保存|创建|修改|删除|上传|入库|下载|导入)",
    re.IGNORECASE,
)
_EXECUTE_RE = re.compile(
    r"(?:run|execute|reproduce|debug|train|tune|deploy|commit|"
    r"跑|运行|执行|复现|调试|训练|调参|部署|提交)",
    re.IGNORECASE,
)


def _tool_action(name: str) -> str:
    token = str(name or "").casefold()
    if any(part in token for part in (
        "run_experiment", "execute", "git_commit", "shell", "train",
        "deploy",
    )):
        return "execute"
    if any(part in token for part in (
        "write", "delete", "upload", "ingest", "download", "doc_create",
        "doc_set", "index",
    )):
        return "mutate"
    if any(part in token for part in (
        "search", "fetch", "read", "list", "check", "status", "metrics",
        "context", "artifact_read",
    )):
        return "read"
    return "unknown"


def build_goal_contract(
    query: str,
    *,
    domain: str = "paper",
    entities: Iterable[str] | None = None,
) -> GoalContract:
    text = str(query or "")
    actions = {"read"}
    if _WRITE_RE.search(text):
        actions.add("mutate")
    if _EXECUTE_RE.search(text):
        actions.add("execute")
    if str(domain) == "coding":
        actions.update({"read", "execute"})
    return GoalContract(
        original_goal=text.strip(),
        domain=str(domain or "paper"),
        allowed_actions=frozenset(actions),
        entities=tuple(
            str(item) for item in (entities or ()) if str(item or "").strip()
        ),
    )


def evaluate_goal_drift(
    contract: GoalContract | dict | None,
    tool_calls: Iterable[Any],
) -> GoalDrift:
    """Flag side effects or execution that the original request did not authorize."""
    if contract is None:
        return GoalDrift(0.0, "none", (), ())
    if isinstance(contract, dict):
        contract = GoalContract(
            original_goal=str(contract.get("original_goal") or ""),
            domain=str(contract.get("domain") or "paper"),
            allowed_actions=frozenset(contract.get("allowed_actions") or ["read"]),
            entities=tuple(contract.get("entities") or ()),
        )

    allowed = set(contract.allowed_actions)
    drifted: list[str] = []
    reasons: list[str] = []
    for call in tool_calls or ():
        name = (
            str(call.get("name") or "")
            if isinstance(call, dict)
            else str(getattr(call, "name", "") or "")
        )
        action = _tool_action(name)
        if action in {"mutate", "execute"} and action not in allowed:
            drifted.append(name)
            reasons.append(
                f"{name} performs {action} work outside the user's requested scope"
            )

    if drifted:
        return GoalDrift(1.0, "high", tuple(dict.fromkeys(reasons)),
                         tuple(dict.fromkeys(drifted)))
    return GoalDrift(0.0, "none", (), ())


def goal_drift_feedback(drift: GoalDrift | dict | None) -> str:
    if drift is None:
        return ""
    if isinstance(drift, dict):
        reasons = [str(item) for item in drift.get("reasons") or []]
        severity = str(drift.get("severity") or "none")
    else:
        reasons = list(drift.reasons)
        severity = drift.severity
    if severity != "high" or not reasons:
        return ""
    return (
        "Goal drift detected. Return to the original user objective and do not "
        "perform unrequested side effects. Drift details: "
        + "; ".join(reasons)
    )
