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

from ..schemas import AgentChatRequest, AgentChatResponse, AgentHealthResponse
from agent.safety import sanitize_output
from agent.observability import set_trace_id, log_event, log_turn_summary

router = APIRouter(prefix="/api/agent", tags=["Agent"])

# ---- internal ----

_agent = None


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

# Whole-turn timeout — mirrors agent.graph.TURN_TIMEOUT. Bounds the entire
# request so a runaway loop can't hold the HTTP connection open forever.
# 默认 900s（与 graph.py 对齐）：复杂工具编排（max_steps 默认 30）是分钟级任务。
_TURN_TIMEOUT = float(os.getenv("AGENT_TURN_TIMEOUT", "900"))


async def _get_agent():
    """Lazy-import agent — .env loaded at agent.graph import time."""
    global _agent
    if _agent is None:
        from agent.graph import get_agent as _ga

        _agent = await _ga()
    return _agent


def _ss_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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
    try:
        import uuid
        set_trace_id(uuid.uuid4().hex[:12])
        log_event("turn_start", node="router", thread_id=req.thread_id)

        agent = await _get_agent()
        from langchain_core.messages import HumanMessage

        config = {"configurable": {"thread_id": req.thread_id}}
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=req.query)],
                 "requested_mode": req.mode},
                config=config,
            ),
            timeout=_TURN_TIMEOUT,
        )

        answer = ""
        for m in reversed(result.get("messages", [])):
            if hasattr(m, "type") and m.type == "ai" and hasattr(m, "content"):
                answer = _strip_marker(m.content)
                break

        log_turn_summary(thread_id=req.thread_id, intent=result.get("intent", ""),
                         status="ok")
        return AgentChatResponse(
            answer=sanitize_output(answer),
            intent=result.get("intent", ""),
            thread_id=req.thread_id,
            mode=result.get("mode", ""),
        )

    except asyncio.TimeoutError:
        log_turn_summary(thread_id=req.thread_id, status="timeout")
        return AgentChatResponse(
            answer="（回答超时，请重试或简化问题。）",
            thread_id=req.thread_id,
            error="turn_timeout",
        )
    except Exception as exc:
        log_turn_summary(thread_id=req.thread_id, status="error",
                         error=f"{type(exc).__name__}: {exc}")
        return AgentChatResponse(
            answer="",
            thread_id=req.thread_id,
            error=f"{type(exc).__name__}: {exc}",
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

        ev_token = None
        try:
            set_trace_id(uuid.uuid4().hex[:12])
            log_event("turn_start", node="router", thread_id=req.thread_id)

            config = {"configurable": {"thread_id": req.thread_id}}
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
                {"messages": [HumanMessage(content=req.query)],
                 "requested_mode": req.mode},
                config=config,
                stream_mode="messages",
                subgraphs=True,  # tool loop lives in the "search" subgraph
            )

            async def _msg_pump():
                try:
                    async for ns, (msg, meta) in _iter_stream(
                        stream, time.monotonic() + _TURN_TIMEOUT,
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
                            await out.put(_ss_event({
                                "type": "tool_end",
                                "id": cid,
                                "name": getattr(msg, "name", "") or "",
                                "status": getattr(msg, "status", "success"),
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
                    await out.put(("err", f"{type(exc).__name__}: {exc}"))
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
                        ev = {**ev, "content": sanitize_output(
                            _strip_marker_segment(ev.get("content", "")))}
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

            if error:
                log_turn_summary(
                    thread_id=req.thread_id,
                    status=("timeout" if error == "turn_timeout" else "error"),
                    error=error,
                )
                yield _ss_event({"type": "error", "message": error})
            else:
                log_turn_summary(thread_id=req.thread_id, status="ok")
                yield _ss_event({"type": "done"})

        except asyncio.TimeoutError:
            log_turn_summary(thread_id=req.thread_id, status="timeout")
            yield _ss_event({"type": "error", "message": "turn_timeout"})
        except Exception as exc:
            log_turn_summary(thread_id=req.thread_id, status="error",
                             error=f"{type(exc).__name__}: {exc}")
            yield _ss_event({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            if ev_token is not None:
                reset_event_queue(ev_token)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/health", response_model=AgentHealthResponse)
async def health():
    """Agent status + configuration info."""
    from agent.tools import ensure_tools

    tools = await ensure_tools()
    return AgentHealthResponse(
        status="ready",
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        tools=len(tools),
    )
