"""Machine-checkable task contracts for the v3 benchmark.

The generic task metric answers "did the agent finish cleanly?".  A task
contract answers the more specific question required by retrieval, download,
read/precision and ingest benchmarks: did it call the right tools, produce the
expected grounded values, write the expected artifacts and stay inside budget?

Some asynchronous ingest postconditions cannot be proved from a single trace.
Those remain explicit ``postconditions`` on the manifest row and are evaluated
by the isolated ingest harness, not silently treated as passed here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evaluation.config import PROJECT_ROOT


def _tool_calls(events: list[dict]) -> list[dict]:
    return [
        event for event in events
        if event.get("event_type") == "tool_call"
    ]


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in (path or "").split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else None
        else:
            return None
    return current


def _normalized(value: Any) -> str:
    return str(value or "").replace("\\", "/").casefold()


def _answer_text(value: Any) -> str:
    return re.sub(r"[\s,，]+", "", str(value or "").casefold())


def _arg_matches(actual: Any, check: dict) -> bool:
    rendered = _normalized(actual)
    if "equals" in check:
        expected = _normalized(check.get("equals"))
        return rendered == expected or rendered.lstrip("./") == expected.lstrip("./")
    if "contains" in check:
        expected = _normalized(check.get("contains")).lstrip("./")
        return expected in rendered.lstrip("./")
    if "one_of" in check:
        return rendered in {
            _normalized(item) for item in (check.get("one_of") or [])
        }
    if check.get("exists") is True:
        return actual not in (None, "")
    return True


def _artifact_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def evaluate_task_contract(
    qa: dict,
    events: list[dict],
    *,
    observed: dict | None = None,
    query_started_at: float | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, Any]:
    """Evaluate deterministic expectations from one task manifest row."""
    expected = qa.get("expected") or {}
    budgets = qa.get("budgets") or {}
    observed = observed or {}
    root = Path(project_root)
    calls = _tool_calls(events)
    tool_names = {str(event.get("tool") or "") for event in calls}
    results: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append({
            "name": name,
            "passed": bool(passed),
            "detail": detail,
        })

    for tool_name in expected.get("required_tools") or []:
        add(
            f"required_tool:{tool_name}",
            tool_name in tool_names,
            f"calls={sorted(tool_names)}",
        )

    for tool_name in expected.get("forbidden_tools") or []:
        add(
            f"forbidden_tool:{tool_name}",
            tool_name not in tool_names,
            f"calls={sorted(tool_names)}",
        )

    for index, check in enumerate(expected.get("tool_args") or [], start=1):
        tool_name = str(check.get("tool") or "")
        arg_path = str(check.get("arg") or "")
        matching = [
            event for event in calls
            if str(event.get("tool") or "") == tool_name
            and _arg_matches(
                _get_path((event.get("payload") or {}).get("args") or {}, arg_path),
                check,
            )
        ]
        add(
            f"tool_arg:{index}:{tool_name}.{arg_path}",
            bool(matching),
            f"matches={len(matching)}",
        )

    answer = str(observed.get("answer") or "")
    for term in expected.get("answer_contains") or []:
        add(f"answer_contains:{term}", _answer_text(term) in _answer_text(answer), "")
    for term in expected.get("answer_not_contains") or []:
        add(f"answer_not_contains:{term}",
            _answer_text(term) not in _answer_text(answer), "")

    for index, artifact in enumerate(expected.get("artifacts") or [], start=1):
        path = _artifact_path(str(artifact.get("path") or ""), root)
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        passed = exists and size >= int(artifact.get("min_size_bytes") or 0)
        detail = f"path={path}, exists={exists}, size={size}"
        if passed and artifact.get("modified_after_query_start"):
            if query_started_at is None:
                passed = False
                detail += ", mtime=unknown"
            else:
                mtime = path.stat().st_mtime
                passed = mtime >= float(query_started_at) - 1.0
                detail += f", mtime={mtime}, started={query_started_at}"
        add(f"artifact:{index}", passed, detail)

    duration_s = observed.get("duration_s")
    max_duration = budgets.get("timeout_s")
    if duration_s is not None and max_duration is not None:
        add(
            "budget:duration_s",
            float(duration_s) <= float(max_duration),
            f"{duration_s}>{max_duration}",
        )

    tool_calls = len(calls)
    max_tool_calls = budgets.get("max_tool_calls")
    if max_tool_calls is not None:
        add(
            "budget:tool_calls",
            tool_calls <= int(max_tool_calls),
            f"{tool_calls}>{max_tool_calls}",
        )

    step_count = observed.get("step_count")
    max_steps = budgets.get("max_steps")
    if step_count is not None and max_steps is not None:
        add(
            "budget:step_count",
            int(step_count) <= int(max_steps),
            f"{step_count}>{max_steps}",
        )

    tokens_total = observed.get("tokens_total")
    max_tokens = budgets.get("max_tokens")
    if tokens_total is not None and max_tokens is not None:
        add(
            "budget:tokens_total",
            int(tokens_total) <= int(max_tokens),
            f"{tokens_total}>{max_tokens}",
        )

    passed_count = sum(1 for item in results if item["passed"])
    total = len(results)
    failures = [item for item in results if not item["passed"]]
    return {
        "contract_enabled": True,
        "contract_success": not failures,
        "contract_checks_passed": passed_count,
        "contract_checks_total": total,
        "contract_success_rate": round(passed_count / total, 4) if total else 1.0,
        "contract_failures": failures,
        "contract_checks": results,
        "postcondition_checks": len(qa.get("postconditions") or []),
    }
