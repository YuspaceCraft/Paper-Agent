"""Final answer confidence calibration from runtime evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FinalConfidence:
    score: float
    level: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    uncertainty_note: str = ""

    def trace_view(self) -> dict:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "reasons": list(self.reasons),
            "uncertainty_note": self.uncertainty_note,
        }


def calibrate_final_confidence(
    state: dict[str, Any],
    *,
    answer: str,
    output_validation: dict | None = None,
    completion_status: str = "completed",
) -> FinalConfidence:
    """Combine intent, verification, evidence and output quality signals."""
    score = 0.45
    reasons: list[str] = []

    intent_confidence = float(state.get("confidence") or 0.0)
    if intent_confidence:
        score += (intent_confidence - 0.5) * 0.20
        reasons.append(f"intent_confidence={intent_confidence:.2f}")

    if str(answer or "").strip():
        score += 0.08
        reasons.append("non_empty_answer")
    else:
        score -= 0.30
        reasons.append("empty_answer")

    verification = state.get("verification") or {}
    verification_status = str(verification.get("status") or "")
    if verification_status == "satisfied":
        score += 0.25
        reasons.append("verification=satisfied")
    elif verification_status == "partial":
        score -= 0.12
        reasons.append("verification=partial")
    elif verification_status in {"failed", "no_evidence"}:
        score -= 0.30
        reasons.append(f"verification={verification_status}")

    results = state.get("subagent_results") or []
    evidence = [
        item for item in results
        if isinstance(item, dict)
        and str(item.get("outcome") or "") in {"succeeded", "partial"}
        and str(item.get("output") or "").strip()
    ]
    if evidence:
        score += min(0.15, 0.05 * len(evidence))
        reasons.append(f"evidence_steps={len(evidence)}")

    failures = [
        item for item in results
        if isinstance(item, dict)
        and str(item.get("outcome") or "") not in {"succeeded", "skipped"}
    ]
    if failures:
        score -= min(0.25, 0.07 * len(failures))
        reasons.append(f"failed_steps={len(failures)}")

    drift = state.get("goal_drift") or {}
    if str(drift.get("severity") or "") == "high":
        score -= 0.20
        reasons.append("goal_drift=high")

    if output_validation is not None:
        if output_validation.get("passed") is True:
            score += 0.08
            reasons.append("output_validation=passed")
        elif output_validation.get("passed") is False:
            score -= 0.20
            reasons.append("output_validation=failed")

    if completion_status == "completed":
        score += 0.08
        reasons.append("task_status=completed")
    elif completion_status == "partial":
        score -= 0.12
        reasons.append("task_status=partial")
    elif completion_status in {"failed", "interrupted"}:
        score -= 0.25
        reasons.append(f"task_status={completion_status}")

    score = round(max(0.05, min(0.95, score)), 3)
    level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    note = (
        "结果存在未完成项或证据不足，请将回答视为部分结论。"
        if level == "low"
        else (
            "仍有少量不确定性，关键结论建议结合引用证据复核。"
            if level == "medium" else ""
        )
    )
    return FinalConfidence(score, level, tuple(reasons), note)
