"""
metrics/tools.py — 工具调用质量：成功率 / 耗时分布 / badcase 归因。

输入：trace 事件列表（tool_call 事件，含 tool/outcome/duration_ms/error/args/
result_summary/parent_id）。三视图归因：错误类型 / 参数模式 / 耗时异常。

口径（2026-09-15 修订，修「工具全挂成功率仍为 1」）：
- success_rate = 成功调用次数 / 调用总次数。每次调用只应有一行 trace
  （dispatcher 的 log_event("tool_call") 是唯一写入口）。
- 「成功」= operation.outcome=succeeded：
    * timed_out/cancelled/failed：超时、取消、熔断或业务失败；
    * partial：有结果但未完整完成，不计入 full success；
    * 纯文本工具（read_file / arxiv MCP 等）返回 "Timeout: ..." / "Error: ..."
      这类降级文案时按失败计。
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

# 纯文本结果的失败文案前缀 → error_type（非 envelope 工具的唯一可用信号）
_TEXT_ERROR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timeout", ("timeout", "timed out", "超时")),
    ("error", ("error", "traceback", "失败", "异常", "exception")),
)


def text_error(text: str) -> str:
    """纯文本结果是否是失败文案 → error_type（不是则返回 ""）。"""
    head = " ".join(str(text or "").strip().split())[:120].lower()
    if not head:
        return ""
    for error_type, prefixes in _TEXT_ERROR_RULES:
        if head.startswith(prefixes):
            return error_type
    return ""


def call_error(event: dict) -> str:
    """一次工具调用的失败原因；成功返回 ""（唯一判定入口）。

    runner/task/flow 都从这里取口径，避免各算各的「成功」。
    """
    payload = event.get("payload") or {}
    parsed = payload.get("parsed") or {}
    operation = payload.get("operation") or {}

    # Failure dominates contradictory writes. A stale top-level outcome must not
    # hide a failed operation/envelope written by the gateway.
    outcomes = [
        str(value).strip()
        for value in (
            event.get("outcome"),
            operation.get("outcome"),
            parsed.get("outcome"),
        )
        if value not in (None, "")
    ]
    failed_outcome = next(
        (value for value in outcomes if value != "succeeded"), "")
    if failed_outcome:
        operation_error = operation.get("error")
        if isinstance(operation_error, dict):
            operation_error = (
                operation_error.get("code")
                or operation_error.get("error_type")
                or operation_error.get("message")
            )
        return str(
            parsed.get("error_type")
            or operation_error
            or event.get("error")
            or payload.get("error")
            or failed_outcome
        )

    explicit_error = event.get("error") or payload.get("error")
    if explicit_error:
        return str(explicit_error)

    if not parsed.get("is_envelope") and not operation:
        # 非 envelope（纯文本工具）→ 看结果文案
        return text_error(payload.get("result_summary") or parsed.get("error") or "")
    return ""

def call_ok(event: dict) -> bool:
    """一次工具调用是否成功（传输层 + 信封 + 文案三层都通过）。"""
    return not call_error(event)


def count_calls(events: list[dict]) -> tuple[int, int]:
    """(调用次数, 失败次数)；调用次数 = 真实工具调用数（每调用一行 trace）。"""
    calls = [e for e in events if e.get("event_type") == "tool_call"]
    return len(calls), sum(1 for e in calls if call_error(e))


def _percentile(sorted_ms: list[float], p: float) -> float:
    if not sorted_ms:
        return 0.0
    if len(sorted_ms) == 1:
        return sorted_ms[0]
    idx = min(len(sorted_ms) - 1, max(0, round(len(sorted_ms) * p / 100)))
    return sorted_ms[idx]


def aggregate_tools(events: list[dict], *, top_badcases: int = 20
                    ) -> dict[str, Any]:
    """从 trace 工具事件聚合 per-tool / per-mode 成功率、耗时、错误归因。

    events: TraceStore.get_thread() 返回的事件列表（含 event_type=="tool_call"）。
    返回结构：
      {per_tool: {name: {calls, succeeded, success_rate, p50_ms, p95_ms,
                          error_types: {type: count}, slow_calls}},
       per_mode: {...同上按 node 分组（react/plan/dispatcher）...},
       badcases: [{trace_id, seq, tool, args, error, duration_ms, ok}],
       overall: {...}}
    """
    calls = [e for e in events if e.get("event_type") == "tool_call"]
    # overall 形状固定（无工具调用时也带 success_rate）：report.assemble_report
    # 直接读 overall["success_rate"]，缺键会让整批评测在收尾阶段崩掉——
    # 「这一批没有任何工具调用」是合法结果，不是异常。
    result: dict[str, Any] = {
        "per_tool": {}, "per_mode": {}, "badcases": [],
        "overall": {"calls": len(calls), "succeeded": 0, "success_rate": 0.0,
                    "p50_ms": 0.0, "p95_ms": 0.0, "errors": 0},
    }
    if not calls:
        return result

    by_tool: dict[str, list[dict]] = defaultdict(list)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    failures: list[dict] = []

    for e in calls:
        tool = e.get("tool") or ""
        node = e.get("node") or "dispatcher"
        by_tool[tool].append(e)
        by_mode[node].append(e)
        reason = call_error(e)
        if reason:
            failures.append({
                "trace_id": e.get("trace_id", ""),
                "seq": e.get("seq"),
                "tool": tool,
                "args": (e.get("payload") or {}).get("args", ""),
                "error": reason,
                # 信封里没带 error_type（纯文本工具的降级文案）时回落到 reason，
                # 保证每条 badcase 都能显示失败类型。
                "error_type": (((e.get("payload") or {}).get("parsed") or {})
                               .get("error_type") or reason),
                "duration_ms": e.get("duration_ms"),
                "node": node,
            })

    for name, group in by_tool.items():
        result["per_tool"][name] = _stat_group(group)
    # per_mode 的总体（无工具名分割）
    for name, group in by_mode.items():
        result["per_mode"][name] = _stat_group(group)

    ok_calls = sum(1 for e in calls if call_ok(e))
    result["overall"].update({
        "succeeded": ok_calls,
        "success_rate": round(ok_calls / len(calls), 4),
        "p50_ms": round(_percentile(sorted(e.get("duration_ms") or 0 for e in calls), 50), 1),
        "p95_ms": round(_percentile(sorted(e.get("duration_ms") or 0 for e in calls), 95), 1),
        "errors": len(calls) - ok_calls,
    })
    failures.sort(key=lambda f: (f.get("duration_ms") or 0), reverse=True)
    result["badcases"] = failures[:top_badcases]
    return result


def _stat_group(group: list[dict]) -> dict[str, Any]:
    calls = list(group)
    if not calls:
        return {"calls": 0}
    ms = sorted(e.get("duration_ms") or 0 for e in calls)
    et: dict[str, int] = defaultdict(int)
    for e in calls:
        # 失败原因一律经 call_error 判定（信封/文案口径与 runner 一致），
        # 成功记 "succeeded" 桶，保持 error_types 形状稳定。
        et[call_error(e) or "succeeded"] += 1
    ok_calls = sum(1 for e in calls if call_ok(e))
    return {
        "calls": len(calls),
        "succeeded": ok_calls,
        "success_rate": round(ok_calls / len(calls), 4),
        "p50_ms": round(_percentile(ms, 50), 1),
        "p95_ms": round(_percentile(ms, 95), 1),
        "max_ms": round(max(ms), 1),
        "error_types": dict(et),
    }
