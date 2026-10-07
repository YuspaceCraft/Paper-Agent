"""Phase 1 self-check — loop governance guarantees termination.

Run: python agent/tests/test_loop.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.nodes import after_agent, build_tools_node
from agent.graph import TURN_TIMEOUT
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel


class _StuckMsg:
    """Fake AIMessage that still demands a tool call (the runaway case)."""
    type = "ai"
    content = ""
    tool_calls = [{"name": "search_papers", "args": {}, "id": "1"}]


class _DoneMsg:
    """Fake AIMessage with a final answer (no tool calls)."""
    type = "ai"
    content = "final answer"
    tool_calls = []


def test_max_steps_clamp():
    # max_steps 语义：撞上限那一步「已执行完成」后才停。已完成轮数 = iteration-1。
    # iteration=31 → 已完成 30 轮 >= 30 → 仍要工具 → synthesize（安全网终止）。
    stuck = {"messages": [_StuckMsg()], "iteration": 31, "max_steps": 30}
    assert after_agent(stuck) == "synthesize", "stuck loop must terminate"

    # iteration=30 → 已完成 29 轮 < 30 → 第 30 轮工具放行执行（不丢弃）。
    cap_bump = {"messages": [_StuckMsg()], "iteration": 30, "max_steps": 30}
    assert after_agent(cap_bump) == "tools", "cap-hitting call must still execute"

    under = {"messages": [_StuckMsg()], "iteration": 2, "max_steps": 30}
    assert after_agent(under) == "tools", "under budget should keep going"

    # 无工具调用 → 直接作答。
    done = {"messages": [_DoneMsg()], "iteration": 10, "max_steps": 30}
    assert after_agent(done) == "end", "final answer routes to end"


def test_empty_ai_response_routes_to_synthesis():
    state = {"messages": [AIMessage(content="", tool_calls=[])]}
    assert after_agent(state) == "synthesize"


def test_turn_timeout_configured():
    assert TURN_TIMEOUT > 0, "AGENT_TURN_TIMEOUT must resolve to a positive float"


def _fake_tool(name, out: str = "ok"):
    class _A(BaseModel):
        q: str = ""

    async def _f(q: str = "") -> str:
        return out

    return StructuredTool(name=name, description=name, args_schema=_A, coroutine=_f)


def test_tools_node_truncates_and_masks():
    # 大结果进 state 前截断（多轮循环输入不再爆炸）。
    node = build_tools_node([_fake_tool("big", out="x" * 20000)])
    state = {"messages": [AIMessage(
        content="", tool_calls=[{"name": "big", "args": {"q": "hi"}, "id": "t1"}])]}
    out = asyncio.run(node(state))
    tm = out["messages"][0]
    assert len(tm.content) <= 8100, "tool result must be truncated"
    assert "truncated" in tm.content, "truncated marker must be present"
    assert tm.tool_call_id == "t1"

    # 未知工具 → 错误信封而非抛穿 graph。
    state2 = {"messages": [AIMessage(
        content="", tool_calls=[{"name": "nope", "args": {}, "id": "t2"}])]}
    out2 = asyncio.run(node(state2))
    assert '"outcome": "failed"' in out2["messages"][0].content
    assert out2["messages"][0].status == "error"

    # 一条消息里多个独立 tool call → 并行执行，全部产出 ToolMessage。
    state3 = {"messages": [AIMessage(content="", tool_calls=[
        {"name": "a", "args": {}, "id": "c1"},
        {"name": "b", "args": {}, "id": "c2"},
    ])]}
    out3 = asyncio.run(build_tools_node([_fake_tool("a"), _fake_tool("b")])(state3))
    assert {m.tool_call_id for m in out3["messages"]} == {"c1", "c2"}


def test_agent_node_classifies_every_tool_in_a_batch(monkeypatch):
    import json
    import agent.nodes as nodes

    captured = {}

    async def fake_llm(model, messages, *, emit_tokens=True, config=None):
        captured["messages"] = messages
        return AIMessage(content="done")

    monkeypatch.setattr(nodes, "_get_bound_model", lambda *a, **k: object())
    monkeypatch.setattr(nodes, "_stream_llm", fake_llm)
    state = {
        "messages": [
            HumanMessage(content="read two things"),
            AIMessage(content="", tool_calls=[
                {"name": "a", "args": {}, "id": "a1"},
                {"name": "b", "args": {}, "id": "b1"},
            ]),
            ToolMessage(
                content=json.dumps({
                    "schema_version": "1.0",
                    "outcome": "failed",
                    "error": "boom",
                    "error_type": "transient",
                }),
                tool_call_id="a1",
                name="a",
            ),
            ToolMessage(
                content=json.dumps({
                    "schema_version": "1.0",
                    "outcome": "succeeded",
                    "data": {"ok": True},
                }),
                tool_call_id="b1",
                name="b",
            ),
        ],
        "iteration": 2,
    }

    asyncio.run(nodes.agent_node(state, {}))
    prompt_text = "\n".join(
        str(getattr(message, "content", ""))
        for message in captured["messages"]
    )
    assert "A tool call failed" in prompt_text


if __name__ == "__main__":
    test_max_steps_clamp()
    test_empty_ai_response_routes_to_synthesis()
    test_turn_timeout_configured()
    test_tools_node_truncates_and_masks()
    print("Phase 1 governance self-check OK")
