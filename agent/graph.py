"""
graph.py — LangGraph StateGraph build, compile, and convenience runner.

v5: AsyncSqliteSaver for cross-restart persistence + search subgraph
    encapsulating the agent ↔ tools ReAct loop.

Usage:
    from agent.graph import run

    result = await run("What is the loss function used in this paper?")
    answer = result["messages"][-1].content

    # Multi-turn: same thread_id preserves conversation via AsyncSqliteSaver
    result2 = await run("How does it compare to other methods?", thread_id="session_1")
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

# ponytail: load .env BEFORE any langchain imports — LangSmith reads
# LANGSMITH_TRACING_V2 at import time; loading after → trace hook not registered.
from dotenv import load_dotenv
_load_dotenv = load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .state import AgentState
from .nodes import (
    understand_node,
    memory_node,
    synthesize_node,
    chat_node,
    clarify_node,
    route_intent,
    domain_node,
)
from .resolution import resolve_node
from .search_loop import build_search_subgraph
from .plan import plan_node, executor_node, decide_mode
from .tools import ensure_tools

# ---- graph construction ----


def build_graph():
    """Build the agent StateGraph.

    v5: search subgraph encapsulates agent ↔ tools loop.
    Parent graph handles routing (understand → memory → route) and
    safety net (synthesize).

    Flow:
        START → understand → memory → route_intent
          ├─ literature_search → resolve → mode decision (decide_mode)
          │       ├─ react → search (subgraph) → synthesize → END
          │       └─ plan → plan_node → executor → synthesize → END
          ├─ general_chat → chat → END
          └─ needs_clarify / low confidence → clarify → END

    Search subgraph (agent/search_loop.py):
          agent → after_agent
            ├─ tool_calls + 已执行轮数 < max_steps → tools → agent (loop)
            ├─ 无 tool_calls 的文本 → exit（fast path，P1：不再有 marker 协议）
            └─ 已执行轮数 >= max_steps → exit (parent synthesizes)
    Step 上限由 state.max_steps 控制（默认 30），turn 级上限由
    state.max_turns / agent_node 守卫控制。
    """
    w = StateGraph(AgentState)

    w.add_node("understand", understand_node)
    w.add_node("memory", memory_node)
    w.add_node("resolve", resolve_node)
    w.add_node("domain", domain_node)
    w.add_node("search", build_search_subgraph())
    w.add_node("plan", plan_node)
    w.add_node("executor", executor_node)
    w.add_node("synthesize", synthesize_node)
    w.add_node("chat", chat_node)
    w.add_node("clarify", clarify_node)

    w.add_edge(START, "understand")
    w.add_edge("understand", "memory")

    w.add_conditional_edges(
        "memory",
        route_intent,
        {"resolve": "resolve", "chat": "chat", "clarify": "clarify"},
    )

    # 领域分流（v10）：决定 plan 通道的领域导向（paper/creation/coding），
    # react 路径零改动（domain 不影响 react 行为）。
    # react = existing single-ReAct path (zero regression);
    # plan = plan-and-execute for multi-paper / comparative / 写作/实验 queries.
    w.add_edge("resolve", "domain")
    w.add_conditional_edges(
        "domain",
        decide_mode,
        {"react": "search", "plan": "plan"},
    )
    w.add_edge("search", "synthesize")
    w.add_edge("plan", "executor")
    w.add_edge("executor", "synthesize")
    w.add_edge("synthesize", END)
    w.add_edge("chat", END)
    w.add_edge("clarify", END)

    return w


# ---- global instance ----

_checkpointer: AsyncSqliteSaver | None = None
_conn: object = None  # aiosqlite.Connection — kept alive for checkpointer lifetime


async def _get_checkpointer() -> AsyncSqliteSaver:
    """Lazy-init SQLite checkpointer for multi-turn conversation persistence.

    ponytail: AsyncSqliteSaver survives server restarts. Conversations
    persist to checkpoints.db in the project root.
    """
    global _checkpointer, _conn
    if _checkpointer is None:
        import aiosqlite
        _conn = await aiosqlite.connect("checkpoints.db")
        _checkpointer = AsyncSqliteSaver(_conn)
        await _checkpointer.setup()
    return _checkpointer


# ponytail: compile once with checkpointer for multi-turn persistence
_agent = None

# Whole-turn timeout — bounds the entire ainvoke, not just one LLM call.
# On timeout the runner returns a fallback answer instead of hanging.
# 默认 900s：复杂工具编排（max_steps 默认 30 步）叠加工具结果截断后仍可能
# 是分钟级任务；旧的 300s 在「逐篇验证 5 篇以上」场景成为实际截断点。
TURN_TIMEOUT = float(os.getenv("AGENT_TURN_TIMEOUT", "900"))


async def get_agent():
    """Get or create the compiled agent (with checkpointer + dynamic tools)."""
    global _agent
    if _agent is None:
        # ensure tools are assembled before graph build
        # (search subgraph reads tools at build time)
        await ensure_tools()
        graph = build_graph()
        _agent = graph.compile(checkpointer=await _get_checkpointer())
    return _agent


# ---- convenience runner ----


async def run(
    query: str,
    *,
    thread_id: str = "default",
    model: str | None = None,
) -> dict:
    """Run the agent with a single query.

    Args:
        query: User's question
        thread_id: Conversation session ID (same ID = same conversation)
        model: Override model name (default: LLM_MODEL env or qwen-plus)

    Returns:
        Final AgentState dict with messages, intent, etc.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    import uuid
    from .observability import set_trace_id, log_event, log_turn_summary

    set_trace_id(uuid.uuid4().hex[:12])
    log_event("turn_start", node="graph", thread_id=thread_id)

    agent = await get_agent()
    config: dict = {"configurable": {"thread_id": thread_id}}
    if model:
        config["configurable"]["model"] = model

    try:
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config,
            ),
            timeout=TURN_TIMEOUT,
        )
        log_turn_summary(
            thread_id=thread_id, intent=result.get("intent", ""), status="ok",
        )
        return result
    except asyncio.TimeoutError:
        log_turn_summary(thread_id=thread_id, status="timeout")
        return {
            "messages": [AIMessage(content="（回答超时，请重试或简化问题。）")],
            "intent": "general_chat",
            "error": "turn_timeout",
        }
