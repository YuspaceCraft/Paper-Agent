"""Phase 8 self-check — subagent runtime: as_tool summary extraction + subset.

Run: python agent/tests/test_subagents.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

import agent.subagents as sa
from agent.subagents import build_subagent, as_tool, SubagentArgs


def _fake_tool(name):
    class _A(BaseModel):
        q: str = ""

    async def _f(q: str = "") -> str:
        return f"{name}:{q}"

    return StructuredTool(name=name, description=name, args_schema=_A, coroutine=_f)


class _FakeSubgraph:
    """Minimal subgraph-like with ainvoke returning a fixed message list."""

    def __init__(self, content):
        self._content = content

    async def ainvoke(self, _input, config=None):
        return {"messages": [AIMessage(content=self._content)]}


def test_as_tool_extracts_summary():
    sg = _FakeSubgraph("the summary answer")
    t = as_tool("paper_reader", sg, "desc", SubagentArgs)
    out = asyncio.run(t.ainvoke({"task": "read RMNet loss"}))
    payload = json.loads(out)
    assert payload["outcome"] == "succeeded"
    assert payload["kind"] == "subagent"
    assert payload["data"]["answer"] == "the summary answer"


def test_as_tool_empty_fallback():
    sg = _FakeSubgraph("")
    t = as_tool("paper_reader", sg, "desc", SubagentArgs)
    out = asyncio.run(t.ainvoke({"task": "x"}))
    assert '"outcome": "failed"' in out, "empty subagent answer must degrade to an error"


def test_as_tool_marks_structured_error_answer_failed():
    sg = _FakeSubgraph(json.dumps({
        "error": "arxiv_api_unavailable",
        "detail": "arXiv API HTTP error: 406 Not Acceptable",
    }))
    t = as_tool("arxiv", sg, "desc", SubagentArgs)
    out = asyncio.run(t.ainvoke({"task": "find an agent paper"}))
    payload = json.loads(out)
    assert payload["outcome"] == "failed"
    assert payload["code"] == "arxiv_api_unavailable"
    assert "406 Not Acceptable" in payload["error"]


def test_as_tool_rejects_no_answer_placeholder():
    sg = _FakeSubgraph("No answer produced.")
    t = as_tool("arxiv", sg, "desc", SubagentArgs)
    out = asyncio.run(t.ainvoke({"task": "find an agent paper"}))
    payload = json.loads(out)
    assert payload["outcome"] == "failed"
    assert payload["code"] == "SUBAGENT_EMPTY_OUTPUT"


def test_subagent_terminal_error_skips_synthesis_llm(monkeypatch):
    from langchain_core.messages import ToolMessage
    import agent.nodes as nodes_mod

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("terminal subagent error must not trigger another LLM")

    monkeypatch.setattr(nodes_mod, "_get_model", fail_if_called)
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "name": "arxiv__search_papers",
                "args": {"query": "agent"},
                "id": "call-1",
            }]),
            ToolMessage(
                content=json.dumps({
                    "error": "arxiv_api_unavailable",
                    "detail": "arXiv API HTTP error: 406 Not Acceptable",
                }),
                tool_call_id="call-1",
                name="arxiv__search_papers",
            ),
        ],
    }
    result = asyncio.run(sa._subagent_synthesize(state, {}))
    payload = json.loads(result["messages"][0].content)
    assert payload["error"] == "arxiv_api_unavailable"


def test_subagent_no_recoverable_output_returns_error(monkeypatch):
    import agent.nodes as nodes_mod
    import evaluation.trace_wrap as trace_wrap

    async def fail_llm(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(nodes_mod, "_get_model", lambda *a, **k: object())
    monkeypatch.setattr(trace_wrap, "traced_ainvoke", fail_llm)
    result = asyncio.run(sa._subagent_synthesize({"messages": []}, {}))
    payload = json.loads(result["messages"][0].content)
    assert payload["error"] == "subagent_no_output"


def test_after_subagent_tools_short_circuits_terminal_error():
    from langchain_core.messages import ToolMessage

    state = {
        "messages": [
            ToolMessage(
                content=json.dumps({
                    "error": "arxiv_api_unavailable",
                    "detail": "406 Not Acceptable",
                }),
                tool_call_id="call-1",
                name="arxiv__search_papers",
            ),
        ],
    }
    assert sa._after_subagent_tools(state) == "synthesize"


def test_build_subagent_compiles():
    a = _fake_tool("fetch_content")
    sg, init = build_subagent("paper_reader", "sys", [a], max_steps=3)
    nodes = set(sg.get_graph().nodes)
    # v15: subagent 节点名带命名空间(与父层 react 循环的 "agent"/"tools" 区分),
    # 让 SSE 端 _msg_pump 能按 langgraph_node 排除 subagent 内部消息。
    assert {"subagent_agent", "subagent_tools"} <= nodes, \
        "subagent must have namespaced agent + tools nodes"
    assert init["subagent_system"] == "sys"
    assert init["bound_tools"] == ["fetch_content"]
    assert init["max_steps"] == 3


def test_build_subagents_names():
    # Claude Code 模式：库只读工具(search_papers/fetch_content)归父 agent，
    # subagent = arxiv(外网) / ingest(写) / creator(写作) / coder(实验编码, v10 Phase C)。
    fakes = [_fake_tool(n) for n in
             ("search_papers", "fetch_content", "download_paper", "ingest_paper",
              "arxiv__search_papers", "arxiv__get_paper_data",
              "arxiv__get_full_paper_text", "arxiv__list_categories",
              "arxiv__update_categories")]
    orig = sa.get_cached_tools
    sa.get_cached_tools = lambda: fakes
    try:
        tools = sa.build_subagents()
    finally:
        sa.get_cached_tools = orig
    assert {t.name for t in tools} == {"arxiv", "ingest", "creator", "coder"}


def test_build_subagents_skips_missing():
    # arxiv tools absent → arxiv subagent omitted (toolset empty)
    fakes = [_fake_tool(n) for n in ("download_paper", "ingest_paper")]
    orig = sa.get_cached_tools
    sa.get_cached_tools = lambda: fakes
    try:
        tools = sa.build_subagents()
    finally:
        sa.get_cached_tools = orig
    assert {t.name for t in tools} == {"ingest"}


def test_ingest_tools_destructive():
    from agent.providers.builtin_provider import BUILTIN_TOOLDEFS
    by = {t.name: t for t in BUILTIN_TOOLDEFS}
    assert by["ingest_paper"].annotations.get("readOnlyHint") is False
    assert by["download_paper"].annotations.get("readOnlyHint") is False


def test_as_tool_sets_scope():
    """as_tool must mark its subagent scope during the subgraph run and
    reset it afterwards, so leaf tool events can be tagged with a parent id."""
    from agent.stream import current_scope

    seen = {}

    class _ScopeSubgraph:
        async def ainvoke(self, _input, config=None):
            seen["scope"] = current_scope()
            return {"messages": [AIMessage(content="ok")]}

    t = as_tool("arxiv", _ScopeSubgraph(), "desc", SubagentArgs)
    asyncio.run(t.ainvoke({"task": "x"}))
    assert seen["scope"]["agent"] == "arxiv", "scope must be set during subgraph run"
    assert seen["scope"]["id"], "scope must carry a run_id for parent linkage"
    assert current_scope() is None, "scope must be reset after the subgraph run"


def test_as_tool_records_timing_trace():
    from agent import observability

    events = []
    observability.register_trace_sink(events.append)
    try:
        t = as_tool("arxiv", _FakeSubgraph("timed summary"), "desc", SubagentArgs)
        asyncio.run(t.ainvoke({"task": "x"}))
    finally:
        observability.unregister_trace_sink(events.append)

    timing = [event for event in events if event.get("event") == "tool_timing"]
    assert timing, events
    assert timing[-1]["tool"] == "arxiv"
    assert timing[-1]["duration_ms"] >= 0
    assert timing[-1]["outcome"] == "succeeded"


if __name__ == "__main__":
    test_as_tool_extracts_summary()
    test_as_tool_empty_fallback()
    test_as_tool_marks_structured_error_answer_failed()
    test_as_tool_rejects_no_answer_placeholder()
    test_subagent_terminal_error_skips_synthesis_llm()
    test_after_subagent_tools_short_circuits_terminal_error()
    test_build_subagent_compiles()
    test_build_subagents_names()
    test_build_subagents_skips_missing()
    test_ingest_tools_destructive()
    test_as_tool_sets_scope()
    test_as_tool_records_timing_trace()
    print("Phase 8 subagents self-check OK")
