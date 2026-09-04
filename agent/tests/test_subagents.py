"""Phase 8 self-check — subagent runtime: as_tool summary extraction + subset.

Run: python agent/tests/test_subagents.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import asyncio
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
    assert out == "the summary answer", "as_tool must return the subagent's answer"


def test_as_tool_empty_fallback():
    sg = _FakeSubgraph("")
    t = as_tool("paper_reader", sg, "desc", SubagentArgs)
    out = asyncio.run(t.ainvoke({"task": "x"}))
    assert '"ok": false' in out, "empty subagent answer must degrade to an error"


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


if __name__ == "__main__":
    test_as_tool_extracts_summary()
    test_as_tool_empty_fallback()
    test_build_subagent_compiles()
    test_build_subagents_names()
    test_build_subagents_skips_missing()
    test_ingest_tools_destructive()
    test_as_tool_sets_scope()
    print("Phase 8 subagents self-check OK")
