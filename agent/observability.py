"""
observability.py — 可观测性底座：trace_id + 结构化日志 + 计数器。

stdlib `logging` + `contextvars` trace_id + 内存计数器。无 OpenTelemetry/全家桶。
trace_id 在请求入口生成，经 contextvars 贯穿整轮，节点/工具/错误日志都带
trace_id，可凭它还原执行路径。

计数器：count/get_counters（累计值 + latency sum/max），不做 p50/p95 分位数
（分位数需样本序列，单用户本地场景 count+sum+max 足够）。
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps

# ---- trace_id ----

_trace_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-"
)


def set_trace_id(trace_id: str) -> None:
    _trace_var.set(trace_id)


def get_trace_id() -> str:
    return _trace_var.get()


# ---- counters (single-process; no lock — fine for a local agent) ----

_COUNTERS: dict[str, int] = {}
_LATENCY: dict[str, dict] = {}  # node -> {sum_ms, max_ms, n}


def count(metric: str, n: int = 1) -> None:
    _COUNTERS[metric] = _COUNTERS.get(metric, 0) + n


def get_counters() -> dict:
    return dict(_COUNTERS)


def _record_latency(node: str, ms: float) -> None:
    e = _LATENCY.setdefault(node, {"sum_ms": 0.0, "max_ms": 0.0, "n": 0})
    e["sum_ms"] += ms
    e["max_ms"] = max(e["max_ms"], ms)
    e["n"] += 1


def get_latency() -> dict:
    return {k: dict(v) for k, v in _LATENCY.items()}


# ---- structured logging ----

_LEVELS = {
    "debug": logging.DEBUG, "info": logging.INFO,
    "warning": logging.WARNING, "error": logging.ERROR,
}

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "agent") -> logging.Logger:
    lg = _loggers.get(name)
    if lg is None:
        lg = logging.getLogger(name)
        if not lg.handlers:
            h = logging.StreamHandler()
            # message is already a JSON string from log_event — emit verbatim
            h.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(h)
            lg.setLevel(logging.INFO)
            lg.propagate = False
        _loggers[name] = lg
    return lg


def log_event(event: str, node: str = "", level: str = "info", **fields) -> None:
    """Emit one structured JSON log line with trace_id/node/event/extra fields."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "level": level.upper(),
        "trace_id": get_trace_id(),
        "node": node,
        "event": event,
        **fields,
    }
    get_logger().log(
        _LEVELS.get(level, logging.INFO),
        json.dumps(payload, ensure_ascii=False, default=str),
    )


def timed(node: str):
    """Async decorator: log node start/end + latency; count errors on raise.

    functools.wraps preserves the wrapped node's __wrapped__/__name__, so
    LangGraph's signature introspection still sees the original (state, config).
    """
    def deco(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            log_event("node_start", node=node)
            t0 = time.time()
            try:
                result = await fn(*args, **kwargs)
                _record_latency(node, (time.time() - t0) * 1000)
                log_event(
                    "node_end", node=node,
                    duration_ms=round((time.time() - t0) * 1000, 1),
                )
                return result
            except Exception:
                count("errors")
                log_event("node_error", node=node, level="error")
                raise
        return wrapper
    return deco


def log_turn_summary(**fields) -> None:
    """Emit a turn-end summary with current counters merged in."""
    log_event("turn_end", node="graph", counters=get_counters(),
              latency=get_latency(), **fields)


# ---- self-check ----

if __name__ == "__main__":
    set_trace_id("check-abc123")
    assert get_trace_id() == "check-abc123"
    count("llm_calls")
    count("llm_calls")
    assert get_counters()["llm_calls"] == 2
    log_event("turn_end", node="graph", intent="test")  # must not raise
    print("Phase 0 observability self-check OK")
