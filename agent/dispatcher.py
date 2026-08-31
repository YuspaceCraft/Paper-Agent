"""
dispatcher.py — 统一工具调度器：超时 + 重试 + 审计。

在所有 Provider 之上加一层，不改 Provider 接口。每个工具调用统一经过：
- 超时（asyncio.wait_for，按工具配置兜底值，不覆盖工具内部 timeout）
- 重试（仅幂等工具 idempotentHint=True 对 transport 超时退避重试，上限 2）
- 审计（结构化日志，接入 observability）

保留 {"ok": false, "error_type": ...} 兼容格式——工具内部已按此返回，
dispatcher 不重写错误语义，只在超时/异常边界补充类型化信息。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from .observability import log_event, count
from .stream import emit, current_scope

# dispatcher 超时是「兜底」而非「主超时」：设得比工具内部 timeout 略大，
# 正常情况下工具自己先超时返回；dispatcher 只兜住真正挂死的调用（如 MCP）。
_DEFAULT_TIMEOUT = 130  # 覆盖 download_paper 的 120s httpx timeout
# v9: ingest_paper 由同步轮询（5min）改为异步启动（<1s 返回 task_id），
# 不再需要 310s 专线超时，落入默认兜底即可。
_TOOL_TIMEOUTS = {}


def _args_summary(args: dict) -> str:
    try:
        s = json.dumps(args, ensure_ascii=False)
    except Exception:
        s = str(args)
    return s[:200]


class ToolDispatcher:
    """统一工具调用入口。call_fn 签名：async (name, args) -> result。"""

    def __init__(self, call_fn, tooldefs: list):
        self._call_fn = call_fn
        self._defs = {td.name: td for td in tooldefs}

    def _timeout(self, name: str) -> int:
        return _TOOL_TIMEOUTS.get(name, _DEFAULT_TIMEOUT)

    async def call(self, name: str, args: dict) -> Any:
        td = self._defs.get(name)
        timeout = self._timeout(name)
        # 仅幂等工具对 transport 超时重试；非幂等工具失败即返回
        idempotent = bool(td and td.annotations.get("idempotentHint"))
        attempts = 3 if idempotent else 1

        # 层级可视化：仅在 subagent 内部（scope 非空）才向 SSE 推事件。
        # 父层（depth 0）工具已由 router/executor 各自 emit，这里不重复。
        # 子层（depth 1）工具是 subagent 内部调用，父层 astream 看不到，
        # 这是它们唯一的曝光通道。parent_id 指向本次 subagent 边界的 id。
        scope = current_scope()
        call_id = f"{name}-{uuid.uuid4().hex[:6]}" if scope else None
        start = time.time()
        if call_id:
            emit({"type": "tool_start", "id": call_id, "name": name,
                  "args": args, "parent_id": scope["id"]})

        for attempt in range(attempts):
            t0 = time.time()
            try:
                result = await asyncio.wait_for(
                    self._call_fn(name, args), timeout=timeout,
                )
                count("tools_called")
                log_event(
                    "tool_call", node="dispatcher", tool=name, ok=True,
                    duration_ms=round((time.time() - t0) * 1000, 1),
                    args=_args_summary(args),
                )
                if call_id:
                    emit({"type": "tool_end", "id": call_id, "name": name,
                          "status": "success", "result": str(result)[:4000],
                          "execution_time": round(time.time() - start, 2)})
                return result
            except asyncio.TimeoutError:
                log_event(
                    "tool_call", node="dispatcher", tool=name, ok=False,
                    duration_ms=round((time.time() - t0) * 1000, 1),
                    error="timeout", attempt=attempt + 1,
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # 1s, 2s 退避
                    continue
            except Exception as exc:
                log_event(
                    "tool_call", node="dispatcher", tool=name, ok=False,
                    duration_ms=round((time.time() - t0) * 1000, 1),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if call_id:
                    emit({"type": "tool_end", "id": call_id, "name": name,
                          "status": "error", "result": f"{type(exc).__name__}: {exc}",
                          "execution_time": round(time.time() - start, 2)})
                # 统一错误信封：工具抛异常（transport/MCP 崩溃等）时返回
                # 结构化错误而非向上抛，让 LLM 据 error_type/next 恢复。
                return json.dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "next": "Retry once or use a different tool.",
                    "error_type": "unknown",
                }, ensure_ascii=False)

        # 所有重试耗尽（仅 timeout 会走到这里）→ 返回兼容错误格式
        if call_id:
            emit({"type": "tool_end", "id": call_id, "name": name,
                  "status": "error", "result": f"timeout after {timeout}s",
                  "execution_time": round(time.time() - start, 2)})
        return json.dumps({
            "ok": False,
            "error": f"Tool '{name}' timed out after {timeout}s.",
            "next": "Retry once or use a different tool.",
            "error_type": "transient",
        }, ensure_ascii=False)
