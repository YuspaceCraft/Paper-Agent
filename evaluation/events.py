"""
events.py — trace 事件类型与标准 emit 辅助。

事件命名与 observability.log_event 的 event 字段保持同构；agent 既有 log_event
经 sink 自动落 trace（node_start/node_end/node_error/tool_call/…），本模块提供
插桩新增的显式事件（intent / llm_call / retrieved_context / plan / final_answer
/ turn_start / turn_end 等）。

约定：事件 dict 顶层只放通用列字段（见 trace_store._COLUMNS），事件特有子字段
一律进 payload。emit_* 返回 None，内部捕获异常，插桩点绝不因 trace 失败而中断。
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from .trace_store import get_trace_store

# 显式新增的事件类型（既有 log_event 的事件名原样透传，不在此列）
EVENT_TYPES = frozenset({
    "turn_start", "intent", "llm_call", "retrieved_context",
    "plan", "plan_step", "plan_verify", "final_answer", "turn_end",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def record_event(event_type: str, node: str = "", **fields) -> None:
    """写入一个显式事件。顶层只含通用列；fields 中 `payload=` 会深合并进 payload。"""
    if "ok" in fields:
        raise ValueError("legacy 'ok' is not supported; use 'outcome'")
    ev: dict[str, Any] = {"ts": _now(), "event_type": event_type, "node": node}
    payload = dict(fields.pop("payload", {}) or {})
    for k, v in fields.items():
        if k in {
            "trace_id", "thread_id", "turn_seq", "seq", "ts", "event_type",
            "node", "duration_ms", "model", "intent", "error", "tool",
            "outcome", "source", "run_id", "parent_id", "payload",
        }:
            ev[k] = v
        else:
            payload[k] = v
    ev["payload"] = payload
    try:
        get_trace_store().record(ev)
    except Exception:  # noqa: BLE001 — 插桩绝不允许打断主链路
        pass


# ---- 各事件 emit 辅助（M2 插桩点调用）----


def emit_turn_start(*, thread_id: str, query: str = "", mode: str = "") -> None:
    record_event("turn_start", node="graph", thread_id=thread_id,
                 payload={"query": query[:2000], "mode": mode})


def emit_intent(*, intent: str = "", confidence: float | None = None,
                entities: list | None = None, focus_papers: list | None = None,
                needs_planning: bool | None = None,
                domain: str = "", node: str = "understand") -> None:
    record_event("intent", node=node, intent=intent,
                 payload={
                     "confidence": confidence, "entities": entities or [],
                     "focus_papers": focus_papers or [],
                     "needs_planning": needs_planning, "domain": domain,
                 })


def emit_llm_call(*, model: str = "", duration_ms: float | None = None,
                  mode: str = "invoke", node: str = "",
                  tokens: dict | None = None, error: str | None = None) -> None:
    record_event("llm_call", node=node, model=model,
                 duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
                 payload={"mode": mode, "tokens": tokens or {},
                          "error": error})


def emit_tool_call(*, tool: str = "", outcome: str = "succeeded",
                   duration_ms: float | None = None,
                   args: dict | str = "", error: str | None = None,
                   result_summary: str = "", node: str = "dispatcher",
                   parent_id: str | None = None,
                   parsed: dict | None = None,
                   operation: dict | None = None) -> None:
    """parsed: 工具结果信封解析（tool_contract.parse_tool_result 产出
    {is_envelope, outcome, error_type, error}——评测端"工具结果解析"透明化项）。"""
    record_event("tool_call", node=node, tool=tool,
                 outcome=outcome,
                 duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
                 parent_id=parent_id,
                 payload={"args": args if isinstance(args, str) else str(args),
                          "result_summary": result_summary[:800],
                          "error": error,
                          "parsed": parsed or {},
                          "operation": operation or {}})


def emit_retrieved_context(*, tool: str = "", query: str = "",
                           chunk_ids: list | None = None,
                           snapshot: str = "",
                           duration_ms: float | None = None,
                           outcome: str = "succeeded",
                           node: str = "dispatcher") -> None:
    record_event("retrieved_context", node=node, tool=tool, outcome=outcome,
                 duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
                 payload={"query": query[:2000], "chunk_ids": chunk_ids or [],
                          "snapshot": snapshot[:4000]})


def emit_plan(*, steps: list | None = None, mode: str = "", node: str = "plan") -> None:
    record_event("plan", node=node, payload={"steps": steps or [], "mode": mode})


def emit_plan_step(*, step_id: str | None = None, status: str = "",
                   detail: str = "", duration_ms: float | None = None,
                   node: str = "plan") -> None:
    record_event("plan_step", node=node,
                 duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
                 payload={"step_id": step_id, "status": status,
                          "detail": str(detail)[:2000]})


def emit_plan_verify(*, verification: dict | None = None, node: str = "verify") -> None:
    record_event("plan_verify", node=node, payload={"verification": verification or {}})


def emit_final_answer(*, answer: str = "", mode: str = "", intent: str = "",
                      node: str = "synthesize",
                      verification: dict | None = None,
                      plan_progress: int | None = None) -> None:
    record_event("final_answer", node=node, intent=intent,
                 payload={"answer": answer[:4000], "mode": mode,
                          "verification": verification or {},
                          "plan_progress": plan_progress})


def emit_turn_end(*, status: str = "ok", latency_ms: float | None = None,
                  error: str | None = None, node: str = "graph",
                  tokens_total: int | None = None,
                  query: str = "") -> None:
    record_event("turn_end", node=node, error=error,
                 duration_ms=round(latency_ms, 1) if latency_ms is not None else None,
                 payload={"status": status, "tokens_total": tokens_total,
                          "query": query[:2000]})
