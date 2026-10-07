"""
metrics/task.py — 任务执行质量：执行成功率 + 中断恢复率。

- task_success：任务真的达成了才算成功
    success = 有最终回答 ∧ 无 run_error（超时/异常）∧ 无工具调用失败
              ∧（plan 模式）verification.status == satisfied
    status  = ok / satisfied（达成）· degraded（有回答但工具失败或验证未过）
              · failed（无回答或中途中断）
  （2026-09-15 修订：历史上只看「有没有 final_answer」，工具全挂时的兜底回答也
   算成功 → task_success_rate 恒为 1，失败任务也进不了 badcase。）
- recovery_test：故障注入（AGENT_FAULT_TOOL=<tool>）使指定工具抛异常打断整轮
  → 记录中断 – 清故障 – 同 thread_id 同 query 续跑 → recovered = 续跑产出
  final_answer 且关键副作用工具（download/ingest 等）未在两次运行中成功重复。

定义（评测报告标注）：
  task_success_rate = 达成条数 / 评测条数
  recovery_rate     = recovered / injected
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Awaitable, Callable

from .tools import call_error

_SIDE_EFFECT_TOOLS = {
    "download_paper", "ingest_paper", "doc_write_section", "doc_create",
    "run_experiment", "git_commit", "write_file", "study_add_hypothesis",
    "set_experiment_project",
}


def task_success(events: list[dict], *, mode: str = "",
                 run_error: str | None = None) -> dict[str, Any]:
    """单 turn 的执行成功判定（从 trace 事件推导，不依赖运行态）。

    run_error: runner 侧捕获的执行异常（turn_timeout / 异常类型+消息）。非空即
    失败——否则「超时/异常之前留下的半截回答」会被算成任务达成。
    """
    ans_ev = next(
        (e for e in reversed(events) if e.get("event_type") == "final_answer"),
        None,
    )
    answer = ((ans_ev.get("payload") or {}).get("answer", "")
              if ans_ev is not None else "")
    has_answer = bool(answer and answer.strip())

    verify_ev = next(
        (e for e in reversed(events) if e.get("event_type") == "plan_verify"),
        None,
    )
    vstatus = ""
    if verify_ev:
        vstatus = (verify_ev.get("payload") or {}).get("verification", {}).get("status", "")

    tool_calls = [e for e in events if e.get("event_type") == "tool_call"]
    # 工具失败口径与 metrics/tools.py 一致（传输层 ∧ 信封 ∧ 文案三层）
    tool_fail = sum(1 for e in tool_calls if call_error(e))
    calls = len(tool_calls)

    mode = mode or ((ans_ev.get("payload") or {}).get("mode", "")
                if ans_ev is not None else "")
    error = str(run_error or "")
    if mode == "plan":
        verified = vstatus == "satisfied"
        success = bool(has_answer) and verified and not error and tool_fail == 0
        reason = "" if success else (
            error or f"verification={vstatus or 'n/a'}")
        status = ("satisfied" if success
                  else ("failed" if not has_answer else "degraded"))
    else:
        success = bool(has_answer) and not error and tool_fail == 0
        reason = "" if success else (
            error or (f"{tool_fail} tool call(s) failed" if tool_fail
                      else "no final answer"))
        status = ("ok" if success
                  else ("failed" if not has_answer else "degraded"))

    if not success and (error or not has_answer):
        severity = 2
    elif tool_fail:
        severity = 1 if tool_fail <= 2 else 2
    else:
        severity = 0

    return {
        "success": success,
        "status": status,
        "task_error": reason,
        "has_answer": has_answer,
        "answer_length": len(answer),
        "verification_status": vstatus or "n/a",
        "tool_calls": calls,
        "tool_failures": tool_fail,
        "severity": severity,
    }


async def recovery_test(
    query: str,
    *,
    thread_id: str,
    run_fn: Callable[[str, str], Awaitable[dict]],
    fault_tool: str = "download_paper",
) -> dict[str, Any]:
    """单条中断恢复测试：注入故障 → 中断 → 续跑 → 判定恢复。

    run_fn(query, thread_id) 必须完整走 agent.graph.run 生产路径。
    使用 AGENT_FAULT_TOOL env 注入（评测专用，不加 env 不改变生产行为）。
    """
    os.environ["AGENT_FAULT_TOOL"] = fault_tool
    os.environ.setdefault("AGENT_FAULT_MODE", "raise")
    interrupted = False
    faulted_events: list[dict] = []
    try:
        try:
            await run_fn(query, thread_id)
        except Exception:  # noqa: BLE001 — 预期中断
            interrupted = True
        from evaluation.trace_store import get_trace_store
        faulted_events = await get_trace_store().get_thread(thread_id)
    finally:
        os.environ.pop("AGENT_FAULT_TOOL", None)
        os.environ.pop("AGENT_FAULT_MODE", None)

    # 续跑（同 thread_id 同 query，实际是对话继续 + 复用已 checkpoint 状态）
    recovered = False
    duplicated: list[str] = []
    second_events: list[dict] = []
    try:
        result = await run_fn(query, thread_id)
        second_events = await get_trace_store().get_thread(thread_id)
        recovered = bool(
            any(e.get("event_type") == "final_answer"
                and (e.get("payload") or {}).get("answer", "").strip()
                for e in second_events)
        )
    except Exception as exc:  # noqa: BLE001
        recovered = False

    if faulted_events and second_events:
        first_ok = {e.get("tool") for e in faulted_events
                    if e.get("event_type") == "tool_call" and not call_error(e)
                    and e.get("tool") in _SIDE_EFFECT_TOOLS}
        second_ok = {e.get("tool") for e in second_events
                     if e.get("event_type") == "tool_call" and not call_error(e)
                     and e.get("tool") in _SIDE_EFFECT_TOOLS}
        duplicated = sorted(first_ok & second_ok)

    return {
        "thread_id": thread_id,
        "interrupted": interrupted,
        "fault_tool": fault_tool,
        "recovered": bool(recovered),
        "duplicated_side_effects": duplicated,
        "events_first_turn": len(faulted_events),
        "events_second_turn": len(second_events),
    }


def aggregate_recovery(results: list[dict]) -> dict[str, Any]:
    injected = len(results)
    recovered = sum(1 for r in results if r.get("recovered"))
    duplicated = sum(1 for r in results if r.get("duplicated_side_effects"))
    return {
        "injected": injected,
        "recovered": recovered,
        "recovery_rate": round(recovered / injected, 4) if injected else 0.0,
        "with_duplicate_side_effects": duplicated,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * percentile + 0.999) - 1))
    return round(ordered[index], 2)


def _aggregate_task_group(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    success = sum(1 for row in rows if row.get("success"))
    contract_rows = [
        row for row in rows if row.get("contract_enabled", False)
    ]
    contract_success = sum(
        1 for row in contract_rows if row.get("contract_success"))
    durations = [
        float(row["duration_s"])
        for row in rows
        if row.get("duration_s") is not None
    ]
    tokens = [
        int(row["tokens_total"])
        for row in rows
        if row.get("tokens_total") is not None
    ]
    return {
        "count": len(rows),
        "success": success,
        "task_success": success,
        "task_success_rate": round(success / len(rows), 4),
        "degraded": sum(1 for row in rows if row.get("status") == "degraded"),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "tool_calls": sum(int(row.get("tool_calls") or 0) for row in rows),
        "tool_failures": sum(int(row.get("tool_failures") or 0) for row in rows),
        "step_count": sum(int(row.get("step_count") or 0) for row in rows),
        "contract_checks": len(contract_rows),
        "contract_success": contract_success,
        "contract_success_rate": (
            round(contract_success / len(contract_rows), 4)
            if contract_rows else None
        ),
        "duration_p50_s": _percentile(durations, 0.50),
        "duration_p95_s": _percentile(durations, 0.95),
        "tokens_total": sum(tokens),
        "tokens_p50": _percentile(tokens, 0.50),
    }


def aggregate_task_metrics(
    query_metrics: list[dict],
    *,
    dimensions: tuple[str, ...] = (
        "task_type", "difficulty_level", "task_length", "generation_mode",
    ),
) -> dict[str, Any]:
    """Aggregate task success, contract, time and token metrics.

    The legacy report only broke retrieval metrics down by dimensions.  The v3
    benchmark also needs task-level performance by family, difficulty and task
    length, otherwise download/ingest failures disappear behind one global
    success rate.
    """
    base = _aggregate_task_group(query_metrics)
    if not query_metrics:
        base.update({
            "failed_queries": [],
            "failures": [],
            "recovery_rate": None,
            "dimension": [],
        })
        return base

    base["failed_queries"] = [
        row.get("query_id") for row in query_metrics if not row.get("success")
    ]
    base["failures"] = [{
        "query_id": row.get("query_id"),
        "status": row.get("status"),
        "category": row.get("category"),
        "reason": row.get("task_error") or row.get("run_error") or "",
        "contract_failures": [
            item.get("name") for item in row.get("contract_failures") or []
        ],
        "tool_failures": row.get("tool_failures") or 0,
    } for row in query_metrics if not row.get("success")]
    base["recovery_rate"] = None

    breakdown: list[dict[str, Any]] = []
    for dimension in dimensions:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in query_metrics:
            value = str(row.get(dimension) or "unknown")
            groups[value].append(row)
        for value, rows in sorted(groups.items()):
            breakdown.append({
                "dimension": dimension,
                "value": value,
                **_aggregate_task_group(rows),
            })
    base["dimension"] = breakdown
    return base
