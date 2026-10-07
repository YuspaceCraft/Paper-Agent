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
import time
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
from .core.contracts import ExecutionContext
from .core.execution_context import (
    build_execution_context,
    get_current_execution_context,
    graph_version_from_nodes,
    reset_current_execution_context,
    set_current_execution_context,
    turn_metadata,
)
from .core.result_utils import current_turn_messages, extract_final_answer
from .nodes import (
    understand_node,
    memory_node,
    synthesize_node,
    chat_node,
    clarify_node,
    task_node,
    route_intent,
    domain_node,
)
from .context import context_node
from .resolution import resolve_node
from .search_loop import build_search_subgraph
from .plan import plan_node, executor_node, verify_node, decide_mode
from .tools import ensure_tools

# ---- graph construction ----


def build_graph():
    """Build the agent StateGraph.

    v5: search subgraph encapsulates agent ↔ tools loop.
    Parent graph handles routing (understand → memory → route) and
    safety net (synthesize).

    Flow:
        START → understand → memory → context → route_intent
          ├─ literature_search → resolve → mode decision (decide_mode,
          │       requested_mode 客户端覆盖优先)
          │       ├─ react → search (subgraph) → synthesize → END
          │       └─ plan → plan_node → executor → verify → synthesize → END
          ├─ general_chat → chat → END
          ├─ task_query → task (监督台：任务进度/详情，直达 task_registry) → END
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
    w.add_node("context", context_node)
    w.add_node("resolve", resolve_node)
    w.add_node("domain", domain_node)
    w.add_node("search", build_search_subgraph())
    w.add_node("plan", plan_node)
    w.add_node("executor", executor_node)
    w.add_node("verify", verify_node)
    w.add_node("synthesize", synthesize_node)
    w.add_node("chat", chat_node)
    w.add_node("clarify", clarify_node)
    w.add_node("task", task_node)

    w.add_edge(START, "understand")
    w.add_edge("understand", "memory")
    w.add_edge("memory", "context")

    w.add_conditional_edges(
        "context",
        route_intent,
        {"resolve": "resolve", "chat": "chat", "clarify": "clarify",
         "task": "task"},
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
    w.add_edge("executor", "verify")
    w.add_edge("verify", "synthesize")
    w.add_edge("synthesize", END)
    w.add_edge("chat", END)
    w.add_edge("clarify", END)
    w.add_edge("task", END)

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
_agent_lock = asyncio.Lock()

# Backward-compatible default for external scripts/tests. Runtime turns use
# ``ExecutionContext.budget.turn_timeout_seconds`` so the effective value is
# frozen at bootstrap instead of re-read mid-turn.
TURN_TIMEOUT = float(os.getenv("AGENT_TURN_TIMEOUT", "900"))


def invalidate_agent_graph() -> None:
    """Drop the compiled graph so the next turn uses the new tool snapshot."""
    global _agent
    _agent = None


async def get_agent():
    """Get or create the compiled agent (with checkpointer + dynamic tools)."""
    global _agent
    if _agent is None:
        async with _agent_lock:
            if _agent is None:
                # ensure tools are assembled before graph build
                # (search subgraph reads tools at build time)
                await ensure_tools()
                saver = await _get_checkpointer()
                # 领导-部门制：派发器挂共享 checkpointer → 子 agent 以 thread_id=task_id
                # 落同一 checkpoints.db（状态栈按 task 持久化，可 get_state/续跑）。
                from .supervisor import init_supervisor
                init_supervisor(saver)
                graph = build_graph()
                _agent = graph.compile(checkpointer=saver)
    return _agent


# ---- convenience runner ----


_graph_version_cache = ""


def get_graph_version() -> str:
    """Fingerprint of the graph layout, written into every trace."""
    global _graph_version_cache
    if not _graph_version_cache:
        try:
            names = [str(n) for n in build_graph().nodes]
        except Exception:  # noqa: BLE001
            names = []
        _graph_version_cache = graph_version_from_nodes(names)
    return _graph_version_cache


def prepare_turn(
    *,
    thread_id: str,
    model: str | None = None,
    mode: str | None = None,
    eval_dataset_id: str = "",
    execution_id: str | None = None,
) -> tuple[dict, ExecutionContext]:
    """Freeze one turn: LangSmith root run id == local trace_id, plus metadata.

    The returned config carries the full metadata contract (trace/thread/request
    ids, config revision, prompt and tool versions, graph version) so a run can
    be reproduced after the fact. ``ExecutionContext`` is the live companion:
    it holds the same config so child model/tool runs nest under this root.
    """
    import uuid

    run_id = uuid.uuid4()
    trace_id = run_id.hex
    ctx = build_execution_context(
        thread_id=thread_id,
        runnable_config=None,
        model=model,
        trace_id=trace_id,
        graph_version=get_graph_version(),
        eval_dataset_id=eval_dataset_id,
        execution_id=execution_id,
        require_initialized_tools=True,
    )
    config: dict = {
        "configurable": {"thread_id": thread_id},
        "run_id": run_id,
        "metadata": turn_metadata(ctx),
    }
    if model:
        config["configurable"]["model"] = model
    if mode:
        config["metadata"]["requested_mode"] = mode
    ctx.bind_runnable_config(config)
    return config, ctx


async def run(
    query: str,
    *,
    thread_id: str = "default",
    model: str | None = None,
    mode: str | None = None,
) -> dict:
    """Run the agent with a single query.

    Args:
        query: User's question
        thread_id: Conversation session ID (same ID = same conversation)
        model: Override model name (default: LLM_MODEL env or qwen-plus)
        mode: Override execution mode (auto/react/plan)——评测单样例可显式指定

    Returns:
        Final AgentState dict with messages, intent, etc.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    import time as _time
    from .observability import set_trace_id, get_trace_id, log_event, log_turn_summary
    from .core.input_policy import guard_user_input
    from .core.capability_policy import assess_capability
    from .core.content_safety import classify_content_safety

    input_decision = guard_user_input(query)
    if not input_decision.allowed:
        log_event(
            "input_blocked",
            node="graph",
            thread_id=thread_id,
            code=input_decision.code,
            rules=list(input_decision.matched_rules),
        )
        return _attach_completion_report({
            "messages": [AIMessage(content=input_decision.message)],
            "intent": "general_chat",
            "error": input_decision.code,
        }, query, error=input_decision.code)
    query = input_decision.sanitized_text
    input_safety = await classify_content_safety(query)
    if input_safety.blocked:
        log_event(
            "content_safety_blocked",
            node="graph",
            thread_id=thread_id,
            phase="input",
            **input_safety.trace_view(),
        )
        return _attach_completion_report({
            "messages": [AIMessage(content=input_safety.safe_response)],
            "intent": "needs_clarify",
            "error": "INPUT_BLOCKED",
        }, query, error="INPUT_BLOCKED")
    capability = assess_capability(query)
    if not capability.allowed:
        log_event(
            "capability_handoff",
            node="graph",
            thread_id=thread_id,
            code=capability.code,
            rules=list(capability.matched_rules),
        )
        return _attach_completion_report({
            "messages": [AIMessage(content=capability.message)],
            "intent": "needs_clarify",
            "capability": capability.capability,
            "error": capability.code,
        }, query, error=capability.code)

    # LangSmith 链路 join 键：顶层 ainvoke 用 run_id 强制根 run（langgraph 消费
    # RunnableConfig.run_id，节点/LLM/工具/子图全部嵌套其下），本地 trace_id =
    # 同一 run 的 hex → 本地事件与 LangSmith run 树双向可定位。
    # Tool assembly has to finish before freezing turn metadata; otherwise the
    # first trace records an empty registry and a false registry hash.
    agent = await get_agent()
    config, ctx = prepare_turn(thread_id=thread_id, model=model, mode=mode)
    trace_id = ctx.trace_id
    # 冻结的 turn 元数据（身份/预算/prompt 绑定/工具版本）整轮可见：节点、
    # 模型与 ToolGateway 都从这里读取，不再中途重读 env/YAML。
    ctx_token = set_current_execution_context(ctx)
    # 评测端采集：只有 eval_ctx 存在时才挂接本地 trace_store；生产链路
    # 以 LangSmith 为在线观测入口，不再双写本地结构化事件。
    try:
        from evaluation.trace_store import eval_ctx, get_trace_store

        if eval_ctx.get():
            from evaluation import sink as _eval_sink
            _eval_sink.attach()
            get_trace_store().set_thread_map(
                trace_id, thread_id=thread_id, source="eval",
                run_id=config["run_id"].hex,
            )
    except Exception as exc:  # noqa: BLE001
        log_event("trace_store_attach_failed", node="graph", level="warning",
                  error=f"{type(exc).__name__}: {exc}")
    set_trace_id(trace_id)
    log_event("turn_start", node="graph", thread_id=thread_id)
    try:
        from .core.configuration import build_configuration_snapshot

        snapshot = build_configuration_snapshot(model=model, unit_id=thread_id)
        log_event("configuration_snapshot", node="graph",
                  **snapshot.trace_metadata())
    except Exception as exc:  # noqa: BLE001 — snapshot is observational only
        log_event("configuration_snapshot_failed", node="graph", level="warning",
                  error=f"{type(exc).__name__}: {exc}")

    t0 = _time.monotonic()

    run_input: dict = {
        "messages": [HumanMessage(content=query)],
        "capability": capability.capability,
        **ctx.runtime_state(),
    }
    if mode:
        run_input["requested_mode"] = mode
    try:
        result = await asyncio.wait_for(
            agent.ainvoke(run_input, config=config),
            timeout=ctx.budget.turn_timeout_seconds,
        )
        from .core.approval import pending_tool_approval

        if pending_tool_approval(result):
            result = _attach_completion_report(
                result, query, error="approval_required",
            )
            log_turn_summary(thread_id=thread_id, intent=result.get("intent", ""),
                             status="interrupted")
            _trace_turn_end(query, result, thread_id, t0, status="interrupted",
                            error="approval_required")
            return result
        result = _attach_completion_report(result, query)
        log_turn_summary(
            thread_id=thread_id, intent=result.get("intent", ""), status="ok",
        )
        _trace_turn_end(query, result, thread_id, t0, status="ok")
        return result
    except asyncio.TimeoutError:
        log_turn_summary(thread_id=thread_id, status="timeout")
        _trace_turn_end(query, None, thread_id, t0, status="timeout")
        timeout_result = {
            "messages": [AIMessage(content="（回答超时，请重试或简化问题。）")],
            "intent": "general_chat",
            "error": "turn_timeout",
        }
        return _attach_completion_report(
            timeout_result, query, error="turn_timeout",
        )
    finally:
        reset_current_execution_context(ctx_token)


async def resume(
    *,
    thread_id: str,
    approved: bool,
    model: str | None = None,
    mode: str | None = None,
) -> dict:
    """Resume a graph paused at a tool-approval interrupt."""
    from langchain_core.messages import AIMessage
    from langgraph.types import Command

    from .observability import get_trace_id, log_event, log_turn_summary, set_trace_id

    agent = await get_agent()
    state_config = {"configurable": {"thread_id": thread_id}}
    previous = await agent.aget_state(state_config)
    previous_values = (
        previous.get("values") if isinstance(previous, dict)
        else getattr(previous, "values", {})
    ) or {}
    execution_id = str(previous_values.get("execution_id") or "")
    config, ctx = prepare_turn(
        thread_id=thread_id,
        model=model,
        mode=mode,
        execution_id=execution_id or None,
    )
    trace_id = ctx.trace_id
    ctx_token = set_current_execution_context(ctx)
    try:
        from evaluation.trace_store import eval_ctx, get_trace_store

        if eval_ctx.get():
            from evaluation import sink as _eval_sink
            _eval_sink.attach()
            get_trace_store().set_thread_map(
                trace_id, thread_id=thread_id, source="eval",
                run_id=config["run_id"].hex,
            )
    except Exception as exc:  # noqa: BLE001
        log_event("trace_store_attach_failed", node="graph", level="warning",
                  error=f"{type(exc).__name__}: {exc}")
    set_trace_id(trace_id)
    log_event("turn_resume", node="graph", thread_id=thread_id,
              approved=bool(approved))

    t0 = time.monotonic()
    try:
        from .core.approval import pending_tool_approval

        snapshot = await agent.aget_state(config)
        if not pending_tool_approval(snapshot):
            return {
                "messages": [AIMessage(content="当前会话没有待确认的工具操作。")],
                "error": "no_pending_approval",
            }
        result = await asyncio.wait_for(
            agent.ainvoke(
                Command(resume={"approved": bool(approved)}),
                config=config,
            ),
            timeout=ctx.budget.turn_timeout_seconds,
        )
        status = "interrupted" if pending_tool_approval(result) else "ok"
        result = _attach_completion_report(
            result,
            "",
            error="approval_required" if status == "interrupted" else None,
        )
        log_turn_summary(thread_id=thread_id, intent=result.get("intent", ""),
                         status=status)
        _trace_turn_end("", result, thread_id, t0,
                        status=status,
                        error="approval_required" if status == "interrupted" else None)
        return result
    except asyncio.TimeoutError:
        log_turn_summary(thread_id=thread_id, status="timeout")
        _trace_turn_end("", None, thread_id, t0, status="timeout")
        timeout_result = {
            "messages": [AIMessage(content="（审批续跑超时，请重试。）")],
            "error": "turn_timeout",
        }
        return _attach_completion_report(
            timeout_result, "", error="turn_timeout",
        )
    finally:
        reset_current_execution_context(ctx_token)


def _attach_completion_report(
    result: dict,
    query: str,
    *,
    error: str | None = None,
) -> dict:
    """Attach shared completion/confidence metadata without changing answers."""
    if not isinstance(result, dict):
        return result
    try:
        from .core.completion_report import build_completion_report
        from .core.output_policy import validate_final_output

        answer = extract_final_answer(result.get("messages", []))
        validation = validate_final_output(query, answer)
        report = build_completion_report(
            result,
            answer=answer,
            error=error or result.get("error"),
            output_validation=validation.trace_view(),
        )
        return {
            **result,
            "task_status": report,
            "final_confidence": report.get("confidence") or {},
        }
    except Exception:  # noqa: BLE001
        return result


def _trace_turn_end(query, result, thread_id, t0, *, status: str,
                    error: str | None = None) -> None:
    """写 final_answer + turn_end 两个显式 trace 事件（评测端链路闭合用）。

    从结果态反扫「无 tool_calls 的 AIMessage」作为最终回答（复用 subagents 的
    反扫约定）；提取 mode/intent/verification/plan_progress/token 估算。
    采集失败只 warn 不抛——绝不能影响回答本身。
    """
    try:
        from evaluation.events import emit_final_answer, emit_turn_end

        answer, mode, intent, verification, plan_progress = "", "", "", None, None
        tokens_total = None
        if isinstance(result, dict):
            mode = str(result.get("mode") or result.get("requested_mode") or "")
            intent = str(result.get("intent") or "")
            verification = result.get("verification")
            plan_progress = result.get("plan_progress")
            answer = extract_final_answer(result.get("messages", []))
            usage_message = next((
                message
                for message in reversed(current_turn_messages(
                    result.get("messages", [])
                ))
                if getattr(message, "type", "") == "ai"
                and not getattr(message, "tool_calls", None)
            ), None)
            usage = (
                getattr(usage_message, "response_metadata", {}).get(
                    "token_usage", {}
                )
                if usage_message else {}
            )
            if isinstance(usage, dict) and usage.get("total_tokens"):
                tokens_total = int(usage["total_tokens"])

        latency_ms = (time.monotonic() - t0) * 1000
        from evaluation.trace_store import eval_ctx

        if (
            not eval_ctx.get()
            and isinstance(result, dict)
            and str(result.get("mode") or "") == "plan"
        ):
            from .core.cost_history import record_plan_cost

            record_plan_cost(
                list(result.get("plan") or []),
                tokens_total=tokens_total,
                duration_ms=latency_ms,
            )
        emit_final_answer(answer=answer or "", mode=mode, intent=intent,
                          verification=verification, plan_progress=plan_progress)
        emit_turn_end(status=status, latency_ms=latency_ms,
                      tokens_total=tokens_total,
                      error=error or (None if status == "ok" else status),
                      query=query)
        if (
            not eval_ctx.get()
            and os.getenv("AGENT_POSTMORTEM", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        ):
            from .core.postmortem import record_turn_postmortem
            from .observability import get_trace_id

            postmortem = record_turn_postmortem(
                query=query,
                status=status,
                error=error,
                verification=(
                    verification if isinstance(verification, dict) else None
                ),
                trace_id=get_trace_id(),
            )
            if postmortem is not None:
                log_event(
                    "postmortem_candidate",
                    node="graph",
                    memory_id=postmortem.memory_id,
                    status=postmortem.status,
                )
    except Exception:  # noqa: BLE001
        pass
