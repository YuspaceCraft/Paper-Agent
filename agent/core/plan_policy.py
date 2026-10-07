"""Pre-execution validation and cost control for plan-mode DAGs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .cost_history import history_key


@dataclass(frozen=True)
class PlanIssue:
    code: str
    message: str
    step_ids: tuple[str, ...] = ()
    severity: str = "error"

    def trace_view(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "step_ids": list(self.step_ids),
            "severity": self.severity,
        }


@dataclass
class PlanValidation:
    steps: list[dict]
    issues: list[PlanIssue] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(
            issue.code == "PLAN_MISSING_ID" for issue in self.issues
        )

    @property
    def had_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def trace_view(self) -> dict:
        return {
            "valid": self.valid,
            "had_errors": self.had_errors,
            "issues": [issue.trace_view() for issue in self.issues],
            "repairs": list(self.repairs),
        }


@dataclass(frozen=True)
class PlanCostEstimate:
    estimated_tokens: int
    estimated_seconds: float
    token_budget: int
    time_budget_seconds: float
    token_exceeded: bool
    time_exceeded: bool
    trimmed_steps: tuple[str, ...] = ()
    calibrated_steps: tuple[str, ...] = ()

    @property
    def exceeded(self) -> bool:
        return self.token_exceeded or self.time_exceeded

    def trace_view(self) -> dict:
        return {
            "estimated_tokens": self.estimated_tokens,
            "estimated_seconds": round(self.estimated_seconds, 2),
            "token_budget": self.token_budget,
            "time_budget_seconds": round(self.time_budget_seconds, 2),
            "token_exceeded": self.token_exceeded,
            "time_exceeded": self.time_exceeded,
            "trimmed_steps": list(self.trimmed_steps),
            "calibrated_steps": list(self.calibrated_steps),
        }


_SCOPE_TOKEN_FACTOR = {
    "preview": 1.0,
    "excerpt": 1.5,
    "section": 2.5,
    "full": 4.0,
}
_SCOPE_SECONDS = {
    "preview": 8.0,
    "excerpt": 15.0,
    "section": 30.0,
    "full": 60.0,
}
_AGENT_TARGETS = {"auto", "arxiv", "creator", "coder"}


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _cycle_nodes(steps: list[dict]) -> list[str]:
    dependencies = {
        str(step.get("id") or ""): [
            str(dep) for dep in (step.get("depends_on") or []) if dep
        ]
        for step in steps
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    in_cycle: set[str] = set()

    def visit(node: str, path: list[str]) -> None:
        if node in visiting:
            if node in path:
                in_cycle.update(path[path.index(node):])
            return
        if node in visited:
            return
        visiting.add(node)
        path.append(node)
        for dep in dependencies.get(node, ()):
            visit(dep, path)
        path.pop()
        visiting.remove(node)
        visited.add(node)

    for node in dependencies:
        visit(node, [])
    return sorted(in_cycle)


def validate_and_repair_plan(steps: list[dict]) -> PlanValidation:
    """Validate IDs/dependencies and remove graph defects deterministically.

    Missing descriptions are still rejected by the planner parser.  This
    function focuses on graph-level defects that can cause hangs or invalid
    scheduling after a syntactically valid plan has already been produced.
    """
    original = [dict(step) for step in (steps or []) if isinstance(step, dict)]
    issues: list[PlanIssue] = []
    repairs: list[dict] = []
    seen_ids: set[str] = set()
    unique: list[dict] = []
    duplicate_ids: list[str] = []

    for step in original:
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            issues.append(PlanIssue(
                "PLAN_MISSING_ID", "plan step is missing an id",
            ))
            continue
        if step_id in seen_ids:
            duplicate_ids.append(step_id)
            repairs.append({
                "code": "PLAN_DUPLICATE_ID_DROPPED",
                "step_id": step_id,
            })
            continue
        seen_ids.add(step_id)
        clean = dict(step)
        clean["id"] = step_id
        unique.append(clean)

    if duplicate_ids:
        issues.append(PlanIssue(
            "PLAN_DUPLICATE_ID",
            "duplicate step ids were removed",
            tuple(sorted(set(duplicate_ids))),
            severity="warning",
        ))

    known = {str(step.get("id") or "") for step in unique}
    for step in unique:
        step_id = str(step.get("id") or "")
        clean_deps: list[str] = []
        for raw_dep in step.get("depends_on") or []:
            dep = str(raw_dep or "").strip()
            if not dep:
                continue
            if dep == step_id:
                issues.append(PlanIssue(
                    "PLAN_SELF_DEPENDENCY",
                    f"step {step_id} depends on itself",
                    (step_id,),
                ))
                repairs.append({
                    "code": "PLAN_SELF_DEPENDENCY_REMOVED",
                    "step_id": step_id,
                })
                continue
            if dep not in known:
                issues.append(PlanIssue(
                    "PLAN_UNKNOWN_DEPENDENCY",
                    f"step {step_id} depends on unknown step {dep}",
                    (step_id, dep),
                ))
                repairs.append({
                    "code": "PLAN_UNKNOWN_DEPENDENCY_REMOVED",
                    "step_id": step_id,
                    "dependency": dep,
                })
                continue
            if dep not in clean_deps:
                clean_deps.append(dep)
        step["depends_on"] = clean_deps

    cycles = _cycle_nodes(unique)
    if cycles:
        issues.append(PlanIssue(
            "PLAN_CYCLE",
            "dependency cycle detected",
            tuple(cycles),
        ))
        cycle_set = set(cycles)
        for step in unique:
            if step.get("id") not in cycle_set:
                continue
            removed = [
                dep for dep in (step.get("depends_on") or [])
                if dep in cycle_set
            ]
            if removed:
                step["depends_on"] = [
                    dep for dep in (step.get("depends_on") or [])
                    if dep not in cycle_set
                ]
                repairs.append({
                    "code": "PLAN_CYCLE_EDGE_REMOVED",
                    "step_id": step.get("id"),
                    "dependencies": removed,
                })

    signatures: dict[str, str] = {}
    exact_duplicates: list[str] = []
    for step in unique:
        signature = _canonical({
            "target": step.get("target"),
            "args": step.get("args") or {},
            "description": step.get("description"),
        })
        first = signatures.get(signature)
        if first is not None:
            exact_duplicates.append(str(step.get("id") or ""))
        else:
            signatures[signature] = str(step.get("id") or "")
    if exact_duplicates:
        issues.append(PlanIssue(
            "PLAN_DUPLICATE_STEP",
            "semantically duplicate steps detected",
            tuple(exact_duplicates),
            severity="warning",
        ))

    # A repaired graph must not contain a leftover dangling dependency.
    known = {str(step.get("id") or "") for step in unique}
    for step in unique:
        step["depends_on"] = [
            dep for dep in (step.get("depends_on") or []) if dep in known
        ]

    return PlanValidation(unique, issues, repairs)


def _estimate_step_cost(
    step: dict,
    historical: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, float, bool]:
    scope = str(step.get("required_scope") or "preview")
    factor = _SCOPE_TOKEN_FACTOR.get(scope, 1.0)
    target = str(step.get("target") or "auto")
    base = 1600 if target in _AGENT_TARGETS else 900
    if str(step.get("delivery") or "answer") == "artifact":
        base += 400
    tokens = int(base * factor)
    seconds = _SCOPE_SECONDS.get(scope, 15.0)
    if target in {"creator", "coder"}:
        seconds *= 1.5
    calibrated = False
    stats = (historical or {}).get(history_key(step)) or {}
    if int(stats.get("samples") or 0) >= 3:
        historical_tokens = max(1.0, float(stats.get("tokens_p50") or 0.0))
        historical_seconds = max(0.0, float(stats.get("seconds_p50") or 0.0))
        tokens = int((tokens + historical_tokens) / 2)
        seconds = (seconds + historical_seconds) / 2
        calibrated = True
    return tokens, seconds, calibrated


def estimate_and_trim_plan(
    steps: list[dict],
    *,
    token_budget: int,
    time_budget_seconds: float,
    historical: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict], PlanCostEstimate]:
    """Estimate plan cost and trim optional steps newest-first."""
    remaining = [dict(step) for step in (steps or [])]
    trimmed: list[str] = []
    calibrated: list[str] = []
    token_budget = max(0, int(token_budget))
    time_budget_seconds = max(0.0, float(time_budget_seconds))

    def estimate(items: list[dict]) -> tuple[int, float]:
        token_total = 0
        seconds_total = 0.0
        for step in items:
            tokens, seconds, is_calibrated = _estimate_step_cost(
                step, historical,
            )
            token_total += tokens
            seconds_total += seconds
            step_id = str(step.get("id") or "")
            if is_calibrated and step_id not in calibrated:
                calibrated.append(step_id)
        return token_total, seconds_total

    tokens, seconds = estimate(remaining)
    while remaining and (tokens > token_budget or seconds > time_budget_seconds):
        optional_index = None
        for index in range(len(remaining) - 1, -1, -1):
            step = remaining[index]
            if str(step.get("priority") or "required") != "optional":
                continue
            step_id = str(step.get("id") or "")
            has_dependents = any(
                step_id in {
                    str(dep) for dep in (candidate.get("depends_on") or [])
                }
                for candidate in remaining
                if candidate is not step
            )
            if not has_dependents:
                optional_index = index
                break
        if optional_index is None:
            break
        removed = remaining.pop(optional_index)
        trimmed.append(str(removed.get("id") or ""))
        tokens, seconds = estimate(remaining)

    return remaining, PlanCostEstimate(
        estimated_tokens=tokens,
        estimated_seconds=seconds,
        token_budget=token_budget,
        time_budget_seconds=time_budget_seconds,
        token_exceeded=tokens > token_budget,
        time_exceeded=seconds > time_budget_seconds,
        trimmed_steps=tuple(trimmed),
        calibrated_steps=tuple(calibrated),
    )
