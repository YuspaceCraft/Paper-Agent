"""Explicit speed/quality/cost optimization profiles for one turn."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


OptimizationProfile = Literal["fast", "balanced", "quality", "cost_saver"]


@dataclass(frozen=True)
class OptimizationDecision:
    profile: OptimizationProfile
    reason: str

    def trace_view(self) -> dict:
        return {"profile": self.profile, "reason": self.reason}


_QUALITY_RE = re.compile(
    r"(?:高质量|尽量准确|详尽|全面|深入|严谨|关键结论|full\s+analysis|"
    r"comprehensive|thorough|high\s+quality|accurate|rigorous)",
    re.IGNORECASE,
)
_FAST_RE = re.compile(
    r"(?:快速|尽快|简要|简短|一句话|先给结论|quick|brief|short answer|"
    r"asap|fast)",
    re.IGNORECASE,
)
_COST_RE = re.compile(
    r"(?:省成本|低成本|越便宜越好|token\s*少|minimal\s+cost|cheap|"
    r"budget[-\s]?friendly|low\s+cost)",
    re.IGNORECASE,
)


def infer_optimization_profile(query: str) -> OptimizationDecision:
    """Infer the user's explicit preference, defaulting to balanced."""
    text = str(query or "")
    if _QUALITY_RE.search(text):
        return OptimizationDecision("quality", "explicit quality requirement")
    if _FAST_RE.search(text):
        return OptimizationDecision("fast", "explicit speed requirement")
    if _COST_RE.search(text):
        return OptimizationDecision("cost_saver", "explicit cost requirement")
    return OptimizationDecision("balanced", "no explicit preference")


def optimization_prompt(profile: str) -> str:
    """Render deterministic planning/synthesis guidance for a profile."""
    normalized = str(profile or "balanced")
    if normalized == "fast":
        return (
            "Optimize for speed. Prefer the minimum sufficient evidence, fewer "
            "steps, preview scope, and an early concise answer."
        )
    if normalized == "quality":
        return (
            "Optimize for correctness and completeness. Prefer sufficient "
            "evidence, explicit verification, and clear citation of gaps."
        )
    if normalized == "cost_saver":
        return (
            "Optimize for lower token and tool cost. Avoid duplicate calls, "
            "prefer cached/local evidence, and trim non-essential steps."
        )
    return (
        "Balance speed, correctness, and cost. Add depth only when evidence "
        "gaps materially affect the answer."
    )
