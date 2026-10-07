"""
agent.py — Agent conversation API endpoints.

POST   /api/agent/chat         — send message, get full response
POST   /api/agent/chat/stream  — SSE streaming response
GET    /api/agent/health       — agent status + model info

ponytail: thin FastAPI wrapper around agent.graph. No business logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from ..schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentHealthResponse,
    AgentResumeRequest,
)
from agent.safety import sanitize_output
from agent.observability import set_trace_id, log_event, log_turn_summary
from agent.core.input_policy import guard_user_input
from agent.core.capability_policy import assess_capability
from agent.core.content_safety import classify_content_safety
from agent.core.completion_report import build_completion_report
from agent.core.output_policy import validate_final_output
from agent.core.result_utils import extract_final_answer, tool_ui_status

router = APIRouter(prefix="/api/agent", tags=["Agent"])

# ---- internal ----


def _strip_marker_segment(seg: str) -> str:
    """token 段级 marker 兜底过滤（不做 strip——模型 token 常以空格开头）。"""
    from agent.nodes import _FINAL_ANSWER_RE
    return _FINAL_ANSWER_RE.sub("", seg)


def _strip_marker(text: str) -> str:
    """Remove legacy [FINAL_ANSWER]/【FINAL_ANSWER】 protocol markers (P1).

    统一正则（大小写不敏感 + 全角括号 + final-answer 分隔符变体），
    非流式 /_chat 整条回答兜底过滤。
    """
    return _strip_marker_segment(text).strip()


def _pending_approval(result) -> dict | None:
    try:
        from agent.core.approval import pending_tool_approval
        return pending_tool_approval(result)
    except Exception:  # noqa: BLE001
        return None


def _final_answer_from_state(result) -> str:
    """Extract the last complete AI answer from a graph state/result object."""
    values = (
        result.get("values") if isinstance(result, dict)
        else getattr(result, "values", None)
    ) or result or {}
    if not isinstance(values, dict):
        return ""
    return _strip_marker(extract_final_answer(values.get("messages", [])))


def _needs_answer_replacement(streamed: str, final: str) -> bool:
    """True when partial streamed text must be replaced by the final answer."""
    streamed_text = str(streamed or "").strip()
    final_text = str(final or "").strip()
    return bool(streamed_text and final_text and streamed_text != final_text)


async def _get_agent():
    """Lazy-import agent — .env loaded at agent.graph import time."""
    from agent.graph import get_agent as _ga

    return await _ga()


def _ss_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _safe_error_message(code: str) -> str:
    if code == "turn_timeout":
        return "回答超时，请重试或简化问题。"
    return "Agent 执行失败，请稍后重试。"


def _completion(
    result: dict | None,
    *,
    query: str,
    answer: str,
    error: str | None = None,
) -> dict:
    """Build the shared completion/confidence report for one API turn."""
    validation = validate_final_output(query, answer)
    return build_completion_report(
        result or {},
        answer=answer,
        error=error,
        output_validation=validation.trace_view(),
    )


async def _cancel_tasks(*tasks) -> None:
    """Cancel and drain streaming helper tasks without leaking background work."""
    present = [task for task in tasks if task is not None]
    for task in present:
        if not task.done():
            task.cancel()
    if present:
        await asyncio.gather(*present, return_exceptions=True)


def _reg_thread(thread_id: str) -> None:
    """Register a trace mapping only for evaluation/single-run contexts."""
    try:
        from agent.observability import get_trace_id
        from evaluation.trace_store import eval_ctx, get_trace_store

        if not eval_ctx.get():
            return
        get_trace_store().set_thread_map(
            get_trace_id(), thread_id=thread_id, source="eval")
    except Exception:  # noqa: BLE001
        pass


def _emit_turn_trace(query: str, thread_id: str, t0: float,
                     status: str, answer: str = "", error: str | None = None) -> None:
    """线上路径 final_answer + turn_end 落 trace。"""
    try:
        from evaluation.events import emit_final_answer, emit_turn_end
        emit_final_answer(answer=answer[:4000], intent="")
        emit_turn_end(status=status, latency_ms=(time.monotonic() - t0) * 1000,
                      error=error, query=query)
    except Exception:  # noqa: BLE001
        pass


async def _iter_stream(stream, deadline: float):
    """Yield astream items until a total deadline.

    With subgraphs=True each item is (ns, (msg, metadata)). Per-item wait_for
    bounds any single stalled LLM/tool call; the deadline bounds the whole
    turn. Raises asyncio.TimeoutError on expiry.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        try:
            yield await asyncio.wait_for(stream.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return


# ---- endpoints ----


@router.post("/chat", response_model=AgentChatResponse)
async def chat(req: AgentChatRequest):
    """Send a message to the research literature agent (non-streaming)."""
    t0 = time.monotonic()
    input_decision = guard_user_input(req.query)
    if not input_decision.allowed:
        log_turn_summary(
            thread_id=req.thread_id, status="blocked",
            error=input_decision.code,
        )
        log_event(
            "input_blocked",
            node="router",
            thread_id=req.thread_id,
            code=input_decision.code,
            rules=list(input_decision.matched_rules),
        )
        _emit_turn_trace(
            req.query,
            req.thread_id,
            t0,
            "blocked",
            error=input_decision.code,
        )
        completion = _completion(
            None,
            query=req.query,
            answer=input_decision.message,
            error=input_decision.code,
        )
        return AgentChatResponse(
            answer=input_decision.message,
            thread_id=req.thread_id,
            error=input_decision.code,
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )
    query = input_decision.sanitized_text
    input_safety = await classify_content_safety(query)
    if input_safety.blocked:
        log_event(
            "content_safety_blocked",
            node="router",
            thread_id=req.thread_id,
            phase="input",
            **input_safety.trace_view(),
        )
        _emit_turn_trace(
            query, req.thread_id, t0, "blocked",
            answer=input_safety.safe_response, error="INPUT_BLOCKED",
        )
        completion = _completion(
            None,
            query=query,
            answer=input_safety.safe_response,
            error="INPUT_BLOCKED",
        )
        return AgentChatResponse(
            answer=input_safety.safe_response,
            thread_id=req.thread_id,
            error="INPUT_BLOCKED",
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )
    capability = assess_capability(query)
    if not capability.allowed:
        log_event(
            "capability_handoff",
            node="router",
            thread_id=req.thread_id,
            code=capability.code,
            rules=list(capability.matched_rules),
        )
        _emit_turn_trace(
            query,
            req.thread_id,
            t0,
            "handoff",
            error=capability.code,
        )
        completion = _completion(
            None,
            query=query,
            answer=capability.message,
            error=capability.code,
        )
        return AgentChatResponse(
            answer=capability.message,
            thread_id=req.thread_id,
            error=capability.code,
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )
    try:
        from agent.core.execution_context import (
            reset_current_execution_context,
            set_current_execution_context,
        )
        from agent.graph import prepare_turn

        agent = await _get_agent()
        config, ctx = prepare_turn(thread_id=req.thread_id, mode=req.mode)
        # 冻结的 turn 元数据：与 graph.run 同一条契约（root run_id == trace_id、
        # 身份/预算/prompt 绑定/工具版本写入 LangSmith metadata）。
        set_trace_id(ctx.trace_id)
        _reg_thread(req.thread_id)
        log_event("turn_start", node="router", thread_id=req.thread_id)
        ctx_token = set_current_execution_context(ctx)
        from langchain_core.messages import HumanMessage

        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "capability": capability.capability,
                        "requested_mode": req.mode,
                        **ctx.runtime_state(),
                    },
                    config=config,
                ),
                timeout=ctx.budget.turn_timeout_seconds,
            )
        finally:
            reset_current_execution_context(ctx_token)

        answer = _strip_marker(extract_final_answer(
            result.get("messages", [])
        ))

        approval = _pending_approval(result)
        if approval:
            completion = _completion(
                result,
                query=query,
                answer=answer,
                error="approval_required",
            )
            log_turn_summary(thread_id=req.thread_id, intent=result.get("intent", ""),
                             status="interrupted")
            _emit_turn_trace(query, req.thread_id, t0, "interrupted",
                             answer=answer, error="approval_required")
            return AgentChatResponse(
                answer="",
                intent=result.get("intent", ""),
                thread_id=req.thread_id,
                mode=result.get("mode", ""),
                requires_approval=True,
                approval=approval,
                task_status=completion,
                final_confidence=completion.get("confidence") or {},
            )

        log_turn_summary(thread_id=req.thread_id, intent=result.get("intent", ""),
                         status="ok")
        safety = await classify_content_safety(answer)
        if safety.blocked:
            completion = _completion(
                result,
                query=query,
                answer=safety.safe_response,
                error="OUTPUT_BLOCKED",
            )
            log_event(
                "output_blocked",
                node="router",
                thread_id=req.thread_id,
                **safety.trace_view(),
            )
            _emit_turn_trace(
                query, req.thread_id, t0, "blocked",
                answer=safety.safe_response, error="OUTPUT_BLOCKED",
            )
            return AgentChatResponse(
                answer=safety.safe_response,
                intent=result.get("intent", ""),
                thread_id=req.thread_id,
                mode=result.get("mode", ""),
                error="OUTPUT_BLOCKED",
                task_status=completion,
                final_confidence=completion.get("confidence") or {},
            )
        completion = _completion(result, query=query, answer=answer)
        _emit_turn_trace(query, req.thread_id, t0, "ok", answer=answer)
        return AgentChatResponse(
            answer=sanitize_output(answer),
            intent=result.get("intent", ""),
            thread_id=req.thread_id,
            mode=result.get("mode", ""),
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )

    except asyncio.TimeoutError:
        log_turn_summary(thread_id=req.thread_id, status="timeout")
        _emit_turn_trace(query, req.thread_id, t0, "timeout",
                         error="turn_timeout")
        answer = "（回答超时，请重试或简化问题。）"
        completion = _completion(
            None, query=query, answer=answer, error="turn_timeout",
        )
        return AgentChatResponse(
            answer=answer,
            thread_id=req.thread_id,
            error="turn_timeout",
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )
    except Exception as exc:
        log_turn_summary(thread_id=req.thread_id, status="error",
                         error=f"{type(exc).__name__}: {exc}")
        _emit_turn_trace(query, req.thread_id, t0, "error",
                         error="agent_runtime")
        completion = _completion(
            None, query=query, answer="", error="agent_runtime",
        )
        return AgentChatResponse(
            answer="",
            thread_id=req.thread_id,
            error="agent_runtime",
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )


@router.post("/resume", response_model=AgentChatResponse)
async def resume(req: AgentResumeRequest):
    """Resume a conversation paused at a tool-approval interrupt."""
    t0 = time.monotonic()
    try:
        from agent.graph import resume as resume_graph

        result = await resume_graph(
            thread_id=req.thread_id,
            approved=req.approved,
            mode=req.mode,
        )
        approval = _pending_approval(result)
        if approval:
            completion = _completion(
                result, query="", answer="", error="approval_required",
            )
            return AgentChatResponse(
                answer="",
                intent=result.get("intent", ""),
                thread_id=req.thread_id,
                mode=result.get("mode", ""),
                requires_approval=True,
                approval=approval,
                task_status=completion,
                final_confidence=completion.get("confidence") or {},
            )

        answer = _strip_marker(extract_final_answer(
            result.get("messages", [])
        ))
        error = result.get("error")
        if error:
            completion = _completion(
                result, query="", answer=answer, error=str(error),
            )
            log_turn_summary(thread_id=req.thread_id, status="error",
                             error=str(error))
            _emit_turn_trace("", req.thread_id, t0, "error",
                             answer=answer, error=str(error))
            return AgentChatResponse(
                answer=sanitize_output(answer),
                intent=result.get("intent", ""),
                thread_id=req.thread_id,
                mode=result.get("mode", ""),
                error=str(error),
                task_status=completion,
                final_confidence=completion.get("confidence") or {},
            )

        if not answer:
            completion = _completion(
                result, query="", answer="", error="empty_answer",
            )
            log_turn_summary(thread_id=req.thread_id, status="error",
                             error="empty_answer")
            _emit_turn_trace("", req.thread_id, t0, "error",
                             error="empty_answer")
            return AgentChatResponse(
                answer="",
                intent=result.get("intent", ""),
                thread_id=req.thread_id,
                mode=result.get("mode", ""),
                error="empty_answer",
                task_status=completion,
                final_confidence=completion.get("confidence") or {},
            )

        completion = _completion(result, query="", answer=answer)
        log_turn_summary(thread_id=req.thread_id,
                         intent=result.get("intent", ""), status="ok")
        _emit_turn_trace("", req.thread_id, t0, "ok", answer=answer)
        return AgentChatResponse(
            answer=sanitize_output(answer),
            intent=result.get("intent", ""),
            thread_id=req.thread_id,
            mode=result.get("mode", ""),
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )
    except Exception as exc:  # noqa: BLE001
        log_turn_summary(thread_id=req.thread_id, status="error",
                         error=f"{type(exc).__name__}: {exc}")
        _emit_turn_trace("", req.thread_id, t0, "error",
                         error="agent_runtime")
        completion = _completion(
            None, query="", answer="", error="agent_runtime",
        )
        return AgentChatResponse(
            answer="",
            thread_id=req.thread_id,
            error="agent_runtime",
            task_status=completion,
            final_confidence=completion.get("confidence") or {},
        )


@router.post("/chat/stream")
async def chat_stream(req: AgentChatRequest):
    """Send a message and stream the response via Server-Sent Events.

    v5: real token-level streaming via stream_mode="messages".
    LLM tokens stream as they're generated; tool calls emit structured
    start/end events so the client can render collapsible steps.

    SSE events:
      {"type":"token","content":"..."}              — LLM token chunk
      {"type":"tool_start","id":cid,"name":n,"args":{...}} — tool invoked
      {"type":"tool_end","id":cid,"name":n,"status":s,"result":"...","execution_time":t} — tool finished
      {"type":"mode","mode":"react|plan","source":"user|auto"} — 实际执行模式
      {"type":"plan","steps":[{id,description,target,depends_on,status}]} — 计划清单
      {"type":"plan_step","id":sid,"status":"running|done|failed|skipped"} — 单步 TODO 状态
      {"type":"plan_progress","done":n,"total":m}   — 计划完成度
      {"type":"plan_verify","status":s,"done":n,"total":m,"outstanding":[...]} — 计划验证报告
      {"type":"done"}                               — stream finished
      {"type":"error","message":"..."}              — error
    """
    agent = await _get_agent()
    from langchain_core.messages import HumanMessage, ToolMessage
    from agent.subagents import SUBAGENT_NAMES

    async def _stream():
        import uuid

        from agent.stream import set_event_queue, reset_event_queue

        last_answer = {"text": ""}
        emitted_token = {"value": False}
        streamed_answer = {"text": ""}
        t0 = time.monotonic()
        ev_token = None
        ctx_token = None
        msg_task = None
        ev_task = None
        stream = None
        try:
            input_decision = guard_user_input(req.query)
            if not input_decision.allowed:
                log_turn_summary(
                    thread_id=req.thread_id,
                    status="blocked",
                    error=input_decision.code,
                )
                log_event(
                    "input_blocked",
                    node="router",
                    thread_id=req.thread_id,
                    code=input_decision.code,
                    rules=list(input_decision.matched_rules),
                )
                _emit_turn_trace(
                    req.query,
                    req.thread_id,
                    t0,
                    "blocked",
                    error=input_decision.code,
                )
                completion = _completion(
                    None,
                    query=req.query,
                    answer=input_decision.message,
                    error=input_decision.code,
                )
                yield _ss_event({
                    "type": "task_status",
                    "task_status": completion,
                    "final_confidence": completion.get("confidence") or {},
                })
                yield _ss_event({
                    "type": "error",
                    "code": input_decision.code,
                    "message": input_decision.message,
                })
                yield _ss_event({"type": "done"})
                return
            query = input_decision.sanitized_text
            input_safety = await classify_content_safety(query)
            if input_safety.blocked:
                log_event(
                    "content_safety_blocked",
                    node="router",
                    thread_id=req.thread_id,
                    phase="input",
                    **input_safety.trace_view(),
                )
                _emit_turn_trace(
                    query, req.thread_id, t0, "blocked",
                    answer=input_safety.safe_response, error="INPUT_BLOCKED",
                )
                completion = _completion(
                    None,
                    query=query,
                    answer=input_safety.safe_response,
                    error="INPUT_BLOCKED",
                )
                yield _ss_event({
                    "type": "task_status",
                    "task_status": completion,
                    "final_confidence": completion.get("confidence") or {},
                })
                yield _ss_event({
                    "type": "error",
                    "code": "INPUT_BLOCKED",
                    "message": input_safety.safe_response,
                })
                yield _ss_event({"type": "done"})
                return
            capability = assess_capability(query)
            if not capability.allowed:
                log_event(
                    "capability_handoff",
                    node="router",
                    thread_id=req.thread_id,
                    code=capability.code,
                    rules=list(capability.matched_rules),
                )
                _emit_turn_trace(
                    query,
                    req.thread_id,
                    t0,
                    "handoff",
                    error=capability.code,
                )
                completion = _completion(
                    None,
                    query=query,
                    answer=capability.message,
                    error=capability.code,
                )
                yield _ss_event({
                    "type": "task_status",
                    "task_status": completion,
                    "final_confidence": completion.get("confidence") or {},
                })
                yield _ss_event({
                    "type": "error",
                    "code": capability.code,
                    "message": capability.message,
                })
                yield _ss_event({"type": "done"})
                return
            from agent.core.execution_context import (
                reset_current_execution_context,
                set_current_execution_context,
            )
            from agent.graph import prepare_turn

            config, ctx = prepare_turn(thread_id=req.thread_id, mode=req.mode)
            # 与 /chat 及 graph.run 同一契约：root run_id == 本地 trace_id，
            # 身份/预算/prompt 绑定/工具版本进 LangSmith metadata。
            set_trace_id(ctx.trace_id)
            _reg_thread(req.thread_id)
            t0 = time.monotonic()
            turn_timeout = ctx.budget.turn_timeout_seconds
            log_event("turn_start", node="router", thread_id=req.thread_id)
            ctx_token = set_current_execution_context(ctx)

            seen_tool_call_ids: set[str] = set()
            tool_start_times: dict[str, float] = {}

            # Two producers feed one output queue:
            #   _msg_pump — astream iteration → token/tool_start/tool_end (react mode)
            #   _ev_pump  — drains plan.py's event channel → plan/tool_start/tool_end
            # Both run as tasks; the generator yields from the shared queue in arrival
            # order. plan-mode steps (subagent tool calls in executor_node) bypass
            # stream_mode="messages", so they surface only via the event channel.
            events_q: asyncio.Queue = asyncio.Queue()
            out: asyncio.Queue = asyncio.Queue()
            ev_token = set_event_queue(events_q)
            _EV_STOP = object()
            _MSG_DONE = object()

            stream = agent.astream(
                {
                    "messages": [HumanMessage(content=query)],
                    "capability": capability.capability,
                    "requested_mode": req.mode,
                    **ctx.runtime_state(),
                },
                config=config,
                stream_mode="messages",
                subgraphs=True,  # tool loop lives in the "search" subgraph
            )

            async def _msg_pump():
                try:
                    async for ns, (msg, meta) in _iter_stream(
                        stream, time.monotonic() + turn_timeout,
                    ):
                        node = (meta or {}).get("langgraph_node", "")
                        # ── LLM token streaming ──
                        # (P5) 节点内手动 model.astream() 不产生 graph 级 messages
                        # 流的 AIMessageChunk——token 事件只有 _ev_pump 一条路
                        # （_stream_llm → event 队列）。此处不再有 chunk 分支；
                        # 若未来框架行为变化（chunk 进入本流），需在 emit token 时
                        # 与 event 队列去重，避免双份输出。这里只处理 graph 级
                        # 返回的完整 AIMessage（tool_calls 边界）与 ToolMessage。

                        # ── AI message（graph 节点最终返回的消息）──
                        if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "ai":
                            if not getattr(msg, "tool_calls", None) and str(msg.content).strip():
                                last_answer["text"] = str(msg.content).strip()
                            if msg.tool_calls:
                                # Only the "agent" node issues real tool calls; understand/
                                # plan use function-calling for structured output and would
                                # otherwise surface as bogus steps.
                                if node == "agent":
                                    for tc in msg.tool_calls:
                                        if tc.get("name", "") in SUBAGENT_NAMES:
                                            continue
                                        cid = tc.get("id", "")
                                        if cid not in seen_tool_call_ids:
                                            seen_tool_call_ids.add(cid)
                                            tool_start_times[cid] = time.monotonic()
                                            await out.put(_ss_event({
                                                "type": "tool_start",
                                                "id": cid,
                                                "name": tc.get("name", ""),
                                                "args": tc.get("args", {}),
                                                # 对话中心化：事件带 thread_id，前端据此归因对话绑定
                                                "thread_id": req.thread_id,
                                            }))

                        # ── Tool result ──
                        elif isinstance(msg, ToolMessage):
                            # 只处理父层 react 循环("tools" 节点)的 tool 完成。
                            # subagent 内部工具子图节点为 "subagent_tools"(见
                            # subagents._build_subgraph 注释)——其卡片由
                            # as_tool._call 边界 + ToolDispatcher 唯一发出,这里
                            # 跳过,否则会给每条叶子工具补一张孤儿 tool_end。
                            if node != "tools":
                                continue
                            # subagent 边界卡已由 as_tool._call 自己 emit。
                            if getattr(msg, "name", "") in SUBAGENT_NAMES:
                                continue
                            cid = getattr(msg, "tool_call_id", "") or ""
                            started = tool_start_times.get(cid)
                            from agent.tool_contract import parse_tool_result

                            parsed_tool = parse_tool_result(msg.content)
                            await out.put(_ss_event({
                                "type": "tool_end",
                                "id": cid,
                                "name": getattr(msg, "name", "") or "",
                                "status": (
                                    tool_ui_status(parsed_tool.outcome)
                                    if parsed_tool.is_envelope
                                    else getattr(msg, "status", "success")
                                ),
                                "outcome": (
                                    parsed_tool.outcome
                                    if parsed_tool.is_envelope else ""
                                ),
                                "result": str(msg.content)[:4000],
                                "execution_time": (
                                    round(time.monotonic() - started, 2) if started else None
                                ),
                                # 对话中心化：事件带 thread_id，前端据此归因对话绑定
                                "thread_id": req.thread_id,
                            }))
                except asyncio.TimeoutError:
                    await out.put(("err", "turn_timeout"))
                except Exception as exc:
                    log_event(
                        "stream_message_pump_failed",
                        node="api.agent",
                        level="warning",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    await out.put(("err", "agent_runtime"))
                finally:
                    await out.put(_MSG_DONE)

            async def _ev_pump():
                while True:
                    ev = await events_q.get()
                    if ev is _EV_STOP:
                        return
                    # Token events originate in nodes (via _stream_llm) — apply the
                    # same PII mask + legacy-marker 兜底过滤 (P1) the message pump
                    # would apply to LLM output.
                    if isinstance(ev, dict) and ev.get("type") == "token":
                        emitted_token["value"] = True
                        content = sanitize_output(
                            _strip_marker_segment(ev.get("content", "")))
                        streamed_answer["text"] += content
                        ev = {**ev, "content": content}
                    # 对话中心化：事件带 thread_id，前端据此归因对话绑定。
                    if isinstance(ev, dict):
                        ev = {**ev, "thread_id": req.thread_id}
                    await out.put(_ss_event(ev))

            msg_task = asyncio.create_task(_msg_pump())
            ev_task = asyncio.create_task(_ev_pump())

            error = None
            while True:
                item = await out.get()
                if item is _MSG_DONE:
                    break
                if isinstance(item, tuple) and item and item[0] == "err":
                    error = item[1]
                    continue
                yield item

            approval = None
            snapshot_answer = ""
            snapshot_values: dict = {}
            if not error:
                try:
                    snapshot = await agent.aget_state(config)
                    snapshot_values = (
                        snapshot.get("values") if isinstance(snapshot, dict)
                        else getattr(snapshot, "values", None)
                    ) or {}
                    approval = _pending_approval(snapshot)
                    snapshot_answer = _final_answer_from_state(snapshot)
                except Exception:  # noqa: BLE001 — state probe is observation-only
                    approval = None

            # Messages done — stop the event pump and flush its tail so the final
            # `done` always follows the last step event.
            await events_q.put(_EV_STOP)
            await ev_task
            while not out.empty():
                item = out.get_nowait()
                if item is _MSG_DONE:
                    continue
                if isinstance(item, tuple) and item and item[0] == "err":
                    if error is None:
                        error = item[1]
                    continue
                yield item

            if not approval and not error:
                final_text = sanitize_output(
                    snapshot_answer or last_answer["text"]
                )
                safety = await classify_content_safety(final_text)
                if safety.blocked:
                    log_event(
                        "output_blocked",
                        node="router",
                        thread_id=req.thread_id,
                        **safety.trace_view(),
                    )
                    final_text = safety.safe_response
                    last_answer["text"] = final_text
                    if emitted_token["value"]:
                        yield _ss_event({
                            "type": "replace_answer",
                            "content": final_text,
                        })
                    else:
                        yield _ss_event({
                            "type": "token",
                            "content": final_text,
                        })
                    completion = _completion(
                        snapshot_values,
                        query=query,
                        answer=final_text,
                        error="OUTPUT_BLOCKED",
                    )
                    yield _ss_event({
                        "type": "task_status",
                        "task_status": completion,
                        "final_confidence": completion.get("confidence") or {},
                    })
                    yield _ss_event({
                        "type": "error",
                        "code": "OUTPUT_BLOCKED",
                        "message": final_text,
                    })
                    _emit_turn_trace(
                        query, req.thread_id, t0, "blocked",
                        answer=final_text, error="OUTPUT_BLOCKED",
                    )
                    yield _ss_event({"type": "done"})
                    return
                if not emitted_token["value"] and final_text:
                    last_answer["text"] = final_text
                    yield _ss_event({
                        "type": "token",
                        "content": final_text,
                    })
                elif (
                    emitted_token["value"]
                    and final_text
                    and _needs_answer_replacement(
                        streamed_answer["text"], final_text,
                    )
                ):
                    last_answer["text"] = final_text
                    yield _ss_event({
                        "type": "replace_answer",
                        "content": final_text,
                    })

            if approval:
                completion = _completion(
                    snapshot_values,
                    query=query,
                    answer=last_answer["text"],
                    error="approval_required",
                )
                log_turn_summary(thread_id=req.thread_id, status="interrupted")
                _emit_turn_trace(query, req.thread_id, t0, "interrupted",
                                 answer=last_answer["text"],
                                 error="approval_required")
                yield _ss_event({
                    "type": "task_status",
                    "task_status": completion,
                    "final_confidence": completion.get("confidence") or {},
                })
                yield _ss_event({"type": "approval_required",
                                 "approval": approval,
                                 "thread_id": req.thread_id})
                yield _ss_event({"type": "done"})
            elif error:
                completion = _completion(
                    snapshot_values,
                    query=query,
                    answer=last_answer["text"],
                    error=error,
                )
                log_turn_summary(
                    thread_id=req.thread_id,
                    status=("timeout" if error == "turn_timeout" else "error"),
                    error=error,
                )
                _emit_turn_trace(
                    query, req.thread_id, t0,
                    "timeout" if error == "turn_timeout" else "error",
                    answer=last_answer["text"], error=error)
                yield _ss_event({
                    "type": "task_status",
                    "task_status": completion,
                    "final_confidence": completion.get("confidence") or {},
                })
                yield _ss_event({
                    "type": "error",
                    "code": error,
                    "message": _safe_error_message(error),
                })
            else:
                completion = _completion(
                    snapshot_values,
                    query=query,
                    answer=last_answer["text"],
                )
                log_turn_summary(thread_id=req.thread_id, status="ok")
                _emit_turn_trace(query, req.thread_id, t0, "ok",
                                 answer=last_answer["text"])
                yield _ss_event({
                    "type": "task_status",
                    "task_status": completion,
                    "final_confidence": completion.get("confidence") or {},
                })
                yield _ss_event({"type": "done"})

        except asyncio.TimeoutError:
            log_turn_summary(thread_id=req.thread_id, status="timeout")
            _emit_turn_trace(req.query, req.thread_id, t0, "timeout",
                             answer=last_answer["text"], error="turn_timeout")
            completion = _completion(
                None,
                query=req.query,
                answer=last_answer["text"],
                error="turn_timeout",
            )
            yield _ss_event({
                "type": "task_status",
                "task_status": completion,
                "final_confidence": completion.get("confidence") or {},
            })
            yield _ss_event({
                "type": "error",
                "code": "turn_timeout",
                "message": "回答超时，请重试或简化问题。",
            })
        except Exception as exc:
            log_turn_summary(thread_id=req.thread_id, status="error",
                             error=f"{type(exc).__name__}: {exc}")
            _emit_turn_trace(req.query, req.thread_id, t0, "error",
                             answer=last_answer["text"],
                             error="agent_runtime")
            completion = _completion(
                None,
                query=req.query,
                answer=last_answer["text"],
                error="agent_runtime",
            )
            yield _ss_event({
                "type": "task_status",
                "task_status": completion,
                "final_confidence": completion.get("confidence") or {},
            })
            yield _ss_event({
                "type": "error",
                "code": "agent_runtime",
                "message": "Agent 执行失败，请稍后重试。",
            })
        finally:
            await _cancel_tasks(msg_task, ev_task)
            if stream is not None:
                try:
                    await stream.aclose()
                except Exception:  # noqa: BLE001
                    pass
            if ev_token is not None:
                reset_event_queue(ev_token)
            if ctx_token is not None:
                reset_current_execution_context(ctx_token)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/health", response_model=AgentHealthResponse)
async def health(probe: bool = False):
    """Agent status + configuration info."""
    from agent.tools import ensure_tools
    from agent.nodes import _get_model
    from agent.core.model_gateway import model_candidate_metadata

    tools = await ensure_tools()
    model_name = os.getenv("LLM_MODEL", "qwen-plus")
    model = _get_model({"configurable": {}}, task="router")
    metadata = model_candidate_metadata(model)
    model_status = "configured"
    should_probe = probe or os.getenv(
        "AGENT_HEALTH_PROBE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if should_probe and hasattr(model, "probe"):
        probe_result = await model.probe()
        model_status = str(probe_result.get("status") or "degraded")
        metadata = {
            **metadata,
            "model_probe": probe_result,
        }
    return AgentHealthResponse(
        status="ready" if model_status == "ready" or model_status == "configured"
        else "degraded",
        model=model_name,
        tools=len(tools),
        model_status=model_status,
        model_candidates=list(metadata.get("model_candidates") or []),
    )
