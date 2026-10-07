"""Graph entrypoint contract checks that do not require an LLM or network."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

import agent.graph as graph
import agent.tools as tools
import evaluation.sink as sink
import evaluation.trace_store as trace_store
from agent.core.contracts import ToolSpec
from agent.core.tool_registry import ToolRegistry


def test_stream_final_answer_fallback_extracts_last_complete_ai_message():
    from web.api.routers.agent import _final_answer_from_state

    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "name": "fetch_content",
                "args": {"paper_name": "paper"},
                "id": "call-1",
            }]),
            AIMessage(content="complete final answer"),
        ],
    }
    assert _final_answer_from_state(state) == "complete final answer"


def test_stream_final_answer_fallback_does_not_reuse_previous_turn():
    from web.api.routers.agent import _final_answer_from_state

    state = {
        "messages": [
            AIMessage(content="old answer"),
            HumanMessage(content="new question"),
            AIMessage(content="", tool_calls=[{
                "name": "fetch_content",
                "args": {"paper_name": "paper"},
                "id": "call-2",
            }]),
        ],
    }
    assert _final_answer_from_state(state) == ""


def test_partial_stream_answer_detects_required_replacement():
    from web.api.routers.agent import _needs_answer_replacement

    assert _needs_answer_replacement("partial", "complete answer")
    assert not _needs_answer_replacement("complete", "complete")
    assert not _needs_answer_replacement("", "complete")


def test_run_registers_trace_thread_mapping(monkeypatch):
    calls: list[dict] = []

    class FakeAgent:
        async def ainvoke(self, run_input, config):
            return {
                "messages": [AIMessage(content="ok")],
                "intent": "general_chat",
                "mode": "react",
            }

    class FakeStore:
        def set_thread_map(self, trace_id, **meta):
            calls.append({"trace_id": trace_id, **meta})

    async def fake_get_agent():
        return FakeAgent()

    monkeypatch.setattr(graph, "get_agent", fake_get_agent)
    monkeypatch.setattr(trace_store, "get_trace_store", lambda: FakeStore())
    monkeypatch.setattr(sink, "attach", lambda: None)
    monkeypatch.setattr(
        tools, "_tool_registry", ToolRegistry([ToolSpec(name="probe_tool")])
    )
    monkeypatch.setattr(tools, "_built", True)

    token = trace_store.eval_ctx.set({"run_id": "eval-probe"})
    try:
        result = asyncio.run(graph.run("probe", thread_id="t-probe"))
    finally:
        trace_store.eval_ctx.reset(token)

    assert result["intent"] == "general_chat"
    assert len(calls) == 1
    assert calls[0]["thread_id"] == "t-probe"
    assert calls[0]["source"] == "eval"
    assert calls[0]["run_id"] == calls[0]["trace_id"]


def test_tool_reload_invalidates_compiled_graphs(monkeypatch):
    import agent.supervisor as supervisor
    from agent import tools

    async def fake_ensure_tools(**kwargs):
        return []

    old_agent = graph._agent
    old_cache = dict(supervisor._graph_cache)
    monkeypatch.setattr(tools, "ensure_tools", fake_ensure_tools)
    try:
        graph._agent = object()
        supervisor._graph_cache[("probe", False)] = object()
        asyncio.run(tools.reload_tools())
        assert graph._agent is None
        assert supervisor._graph_cache == {}
    finally:
        graph._agent = old_agent
        supervisor._graph_cache.clear()
        supervisor._graph_cache.update(old_cache)


def test_resume_reuses_the_original_execution_id(monkeypatch):
    captured: dict = {}

    class FakeAgent:
        async def aget_state(self, config):
            return {
                "values": {"execution_id": "turn-42"},
                "__interrupt__": [{"kind": "tool_approval"}],
            }

        async def ainvoke(self, command, config):
            captured.update(config)
            return {
                "messages": [AIMessage(content="resumed")],
                "intent": "general_chat",
                "mode": "react",
            }

    async def fake_get_agent():
        return FakeAgent()

    monkeypatch.setattr(graph, "get_agent", fake_get_agent)
    monkeypatch.setattr(
        tools, "_tool_registry", ToolRegistry([ToolSpec(name="probe_tool")])
    )
    monkeypatch.setattr(tools, "_built", True)

    result = asyncio.run(graph.resume(
        thread_id="resume-turn", approved=True,
    ))
    assert result["messages"][-1].content == "resumed"
    assert captured["metadata"]["execution_id"] == "turn-42"


def test_resume_endpoint_returns_resumed_answer(monkeypatch):
    from web.api.routers import agent as agent_router
    from web.api.schemas import AgentResumeRequest

    async def fake_resume(**_kwargs):
        return {
            "messages": [AIMessage(content="resumed through API")],
            "intent": "general_chat",
            "mode": "react",
        }

    monkeypatch.setattr(graph, "resume", fake_resume)
    result = asyncio.run(agent_router.resume(AgentResumeRequest(
        thread_id="resume-api", approved=True,
    )))
    assert result.answer == "resumed through API"
    assert result.error is None
