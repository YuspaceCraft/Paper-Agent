"""
dispatcher.py — 统一工具调度器：超时 + 重试 + 权限 + 审计。

在所有 Provider 之上加一层，不改 Provider 接口。真正的治理逻辑在
agent/core/tool_gateway.py（ToolGateway.invoke：权限矩阵 / 审批判定 /
幂等键 / 超时 / 熔断 / 重试 / 脱敏审计），本模块只保留「面向 UI 与评测
的观察层」职责：
- SSE 事件（tool_start / tool_end，父子层级可视化）
- evaluation 事件：tool_call 由本模块的 log_event 单行落库（带结果摘要与信封
  解析），检索类工具额外写 retrieved_context（指标与链路回放）
- 与历史行为一致的兜底超时值（_DEFAULT_TIMEOUT / _TOOL_TIMEOUTS）

错误信封统一由 agent/tool_contract.py 定义（P6）——工具内部已按此返回，
gateway 只在超时/异常边界补充类型化信息，dispatcher 不重写错误语义。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from .observability import log_event, count
from .stream import emit, current_scope
from .core.execution_context import get_current_execution_context
from .core.tool_gateway import ToolGateway
from .core.tool_registry import ToolRegistry
from .core.result_utils import tool_ui_status

# dispatcher 超时是「兜底」而非「主超时」：设得比工具内部 timeout 略大，
# 正常情况下工具自己先超时返回；dispatcher 只兜住真正挂死的调用（如 MCP）。
_DEFAULT_TIMEOUT = 130  # 覆盖 download_paper 的 120s httpx timeout
# v9: ingest_paper 由同步轮询（5min）改为异步启动（<1s 返回 task_id），
# 不再需要 310s 专线超时，落入默认兜底即可。
_TOOL_TIMEOUTS = {}

# 检索类工具 → 额外写 retrieved_context 事件（评测端检索指标 hits / 上下文快照）
_RETRIEVAL_TOOLS = {"search_papers", "fetch_content"}
_SENSITIVE_ARG_PARTS = {
    "authorization", "credential", "key", "password", "secret", "token",
}


def _redact_arg_value(key: str, value: Any) -> Any:
    lowered = str(key).lower()
    if any(part in lowered for part in _SENSITIVE_ARG_PARTS):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_arg_value(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _redact_arg_value(key, child_value)
            for child_value in value
        ]
    return value


def _args_view(args: dict) -> dict:
    """Structured, redacted arguments for trace/evaluation payloads."""
    try:
        return _redact_arg_value("", args)
    except Exception:  # noqa: BLE001
        return {"_unserializable": True}


def parsed_result(result) -> dict:
    """结果信封解析视图（评测端「工具结果解析」透明化项）。

    非 envelope（纯文本工具）时 is_envelope=False；此时 ok 恒为 True，是否失败
    由 evaluation/metrics/tools.py 结合文本判定（唯一口径在那里）。
    """
    if result is None:
        return {}
    try:
        from .tool_contract import parse_tool_result
        pr = parse_tool_result(str(result))
        return {
            "is_envelope": pr.is_envelope,
            "error_type": pr.error_type or "", "error": (pr.error or "")[:300],
            "outcome": pr.outcome or "",
            "code": pr.code or "",
            "protocol_error": pr.protocol_error or "",
            "retryable": pr.retryable,
            "artifact_ids": [
                str(item.get("artifact_id") or "")
                for item in pr.artifacts
                if isinstance(item, dict) and item.get("artifact_id")
            ],
            "continues": bool(pr.continuation.get("next_offset")),
        }
    except Exception:  # noqa: BLE001
        return {}


def _trace_retrieval(tool: str, outcome: str, duration_ms: float, args: dict,
                     result=None) -> None:
    """检索类工具 → 额外写 retrieved_context 事件（评测端 hits / 上下文快照）。

    只在 evaluation 挂接后生效；任何异常静默吞掉，绝不影响工具主链路。
    """
    try:
        if outcome == "succeeded" and tool in _RETRIEVAL_TOOLS:
            from evaluation.events import emit_retrieved_context
            query = ""
            if isinstance(args, dict):
                query = str(args.get("query", ""))
            chunk_ids: list = []
            try:
                from .tool_contract import parse_tool_result
                pr = parse_tool_result(result)
                if (
                    pr.is_envelope
                    and pr.outcome == "succeeded"
                    and isinstance(pr.data, dict)
                ):
                    for h in (pr.data.get("results") or []):
                        if isinstance(h, dict) and h.get("chunk_id"):
                            chunk_ids.append(str(h["chunk_id"]))
            except Exception:  # noqa: BLE001
                pass
            emit_retrieved_context(
                tool=tool, query=query, chunk_ids=chunk_ids,
                snapshot=str(result), duration_ms=duration_ms,
                outcome=outcome,
            )
    except Exception:  # noqa: BLE001
        pass


def _artifact_meta(tool: str, result) -> dict:
    """Spill oversized evaluation results; non-eval runs return empty metadata."""
    try:
        from evaluation.artifacts import persist_if_large

        return persist_if_large(tool, str(result or ""))
    except Exception:  # noqa: BLE001
        return {}


class ToolDispatcher:
    """统一工具调用入口。call_fn 签名：async (name, args) -> result。"""

    def __init__(self, call_fn, tooldefs: list, registry=None, gateway=None,
                 idempotency_store=None):
        self._call_fn = call_fn
        self._defs = {td.name: td for td in tooldefs}
        # Registry 允许缺省（子图/测试直接构造）：缺省时从 tooldefs 现场派生，
        # 保证 gateway 的权限与版本判定始终有 spec 可依。
        self._registry = registry or ToolRegistry.from_tooldefs(tooldefs)
        self._gateway = gateway or ToolGateway(
            call_fn, self._registry, timeout_resolver=self._timeout,
            idempotency_store=idempotency_store,
        )

    def _timeout(self, name: str) -> int:
        return _TOOL_TIMEOUTS.get(name, _DEFAULT_TIMEOUT)

    @property
    def gateway(self) -> ToolGateway:
        """治理层入口（配置中心 / 测试 / 排障可读其决策记录）。"""
        return self._gateway

    async def call(self, name: str, args: dict) -> Any:
        spec = self._registry.get(name)

        # 层级可视化：仅在 subagent 内部（scope 非空）才向 SSE 推事件。
        # 父层（depth 0）工具已由 router/executor 各自 emit，这里不重复。
        # 子层（depth 1）工具是 subagent 内部调用，父层 astream 看不到，
        # 这是它们唯一的曝光通道。parent_id 指向本次 subagent 边界的 id。
        scope = current_scope()
        scope_id = scope["id"] if scope else None
        call_id = f"{name}-{uuid.uuid4().hex[:6]}" if scope else None
        start = time.time()
        if call_id:
            emit({"type": "tool_start", "id": call_id, "name": name,
                  "args": args, "parent_id": scope["id"]})

        ctx = get_current_execution_context()
        outcome = await self._gateway.invoke(name, args, ctx=ctx)
        operation = outcome.to_operation_result(spec=spec, ctx=ctx)
        wire_envelope = operation.to_envelope()
        if operation.outcome.value in {"succeeded", "partial"}:
            try:
                from .core.artifact_store import persist_text
                from .tool_contract import attach_artifact, paginate_tool_result

                artifact = persist_text(
                    wire_envelope,
                    tool_name=name,
                    execution_id=str(
                        getattr(ctx, "execution_id", "")
                        or getattr(ctx, "request_id", "")
                        or ""
                    ),
                    thread_id=str(getattr(ctx, "thread_id", "") or ""),
                )
                if artifact:
                    wire_envelope = attach_artifact(wire_envelope, artifact)
                wire_envelope = paginate_tool_result(
                    wire_envelope,
                    offset=int((args or {}).get("offset", 0) or 0),
                    max_chars=int(
                        (args or {}).get("max_chars", 6000) or 6000
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — projection must not fail calls
                log_event(
                    "tool_artifact_projection_failed",
                    node="dispatcher",
                    tool=name,
                    level="warning",
                    error=f"{type(exc).__name__}: {exc}",
                )
        duration_ms = outcome.duration_ms or (time.time() - start) * 1000
        parsed = parsed_result(wire_envelope)
        artifact = _artifact_meta(name, wire_envelope)
        # 唯一一条工具 trace 行：dispatcher 既有的 log_event("tool_call") 就是评估端
        # 的事实源（额外带上结果摘要与信封解析）。历史上这里再 emit 一次 tool_call，
        # 每次调用落两行 → 调用次数翻倍、成功率被摊平，已收敛为单行（2026-09-15）。
        log_event(
            "tool_call", node="dispatcher", tool=name,
            outcome=operation.outcome.value,
            duration_ms=round(duration_ms, 1), args=_args_view(args),
            tool_version=(spec.version if spec else "1"),
            side_effect=(spec.side_effect if spec else False),
            error=(
                ""
                if operation.outcome.value in {"succeeded", "partial", "skipped"}
                else (operation.error.code if operation.error else "tool_error")
            ),
            attempts=outcome.attempts,
            decision=outcome.decision.action,
            cache_hit=bool(outcome.cache_hit),
            parent_id=call_id,
            parsed=parsed,
            operation=operation.model_dump(mode="json"),
            **artifact,
        )
        if operation.outcome.value == "succeeded":
            count("tools_called")
            if outcome.cache_hit:
                count("tool_cache_hits")

        if call_id:
            emit({"type": "tool_end", "id": call_id, "name": name,
                  "status": tool_ui_status(operation.outcome.value),
                  "outcome": operation.outcome.value,
                  "code": operation.error.code if operation.error else "",
                  "retryable": (
                      operation.error.retryable if operation.error else False
                  ),
                  "result": wire_envelope[:4000],
                  "execution_time": round(time.time() - start, 2)})
        _trace_retrieval(
            name, outcome=operation.outcome.value,
            duration_ms=duration_ms, args=args, result=wire_envelope,
        )
        return wire_envelope
