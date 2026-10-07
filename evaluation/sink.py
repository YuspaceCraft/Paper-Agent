"""
sink.py — observability.log_event → TraceStore 的注册桥 + 评测上下文打标。

- attach()：把 agent/observability.log_event 的所有 payload 同步转发到
  TraceStore（幂等）。sink 异常由 observability 侧捕获，绝不反向打断日志。
- set_eval_ctx()：评测跑批对后续事件注入 {run_id, thread_id, turn_seq,
  source="eval"}，eval_ctx contextvar 定义在 trace_store.py，此处仅封装。
"""

from __future__ import annotations

from contextvars import ContextVar  # noqa: F401  (re-export 语义)

from .trace_store import eval_ctx as _ctx, get_trace_store

_attached = False


def set_eval_ctx(*, run_id: str | None = None, thread_id: str | None = None,
                 turn_seq: int = 1, qa_id: str | None = None,
                 source: str = "eval") -> None:
    """评测跑批打标：其后所有落 trace 的事件归属到该 run/线程。"""
    cur = dict(_ctx.get())
    if run_id:
        cur["run_id"] = run_id
    if thread_id:
        cur["thread_id"] = thread_id
    if qa_id:
        cur["qa_id"] = qa_id
    cur["turn_seq"] = turn_seq
    cur["source"] = source
    _ctx.set(cur)


def clear_eval_ctx() -> None:
    _ctx.set({})


def _sink(payload: dict) -> None:
    """observability.log_event → trace。额外为 eval 上下文补充 run/thread。"""
    if "event_type" not in payload and payload.get("event"):
        # log_event 使用 event 字段名，trace schema 用 event_type —— 归一化后透传
        payload = dict(payload)
        payload["event_type"] = payload.pop("event")
    get_trace_store().record(payload)


def attach() -> None:
    """幂等挂接：注册 sink，让一切 log_event 双写进 trace_store.db。"""
    global _attached
    if _attached:
        return
    from agent.observability import log_event  # noqa: F401

    from agent.observability import register_trace_sink
    register_trace_sink(_sink)
    _attached = True


def ensure_attached() -> bool:
    """attach() 的只读探测：已挂接返回 True，否则执行挂接。"""
    attach()
    return True