"""Hard context-window budgeting regressions."""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.core.contracts import Budget
from agent.core.token_budget import fit_for_context, fit_messages
from agent.nodes import _render_agent_context


def test_tool_call_and_result_fit_as_one_unit():
    messages = [
        SystemMessage(content="policy " * 80),
        HumanMessage(content="old question"),
        AIMessage(content="", tool_calls=[{
            "name": "search_papers",
            "args": {"query": "remote sensing"},
            "id": "call-1",
        }]),
        ToolMessage(
            content='{"outcome":"succeeded","data":{"text":"' + ("evidence " * 2000) + '"}}',
            tool_call_id="call-1",
            name="search_papers",
        ),
    ]

    fit = fit_messages(messages, max_tokens=900)
    types = [getattr(message, "type", "") for message in fit.messages]
    assert fit.input_tokens <= fit.budget_tokens
    assert "ai" in types and "tool" in types
    assert types.index("ai") < types.index("tool")


def test_newest_user_message_is_kept_and_truncated():
    fit = fit_messages([
        SystemMessage(content="policy " * 40),
        HumanMessage(content="latest " * 2000),
    ], max_tokens=180)

    assert fit.input_tokens <= fit.budget_tokens
    assert any(message.type == "human" for message in fit.messages)


def test_fit_for_context_uses_frozen_window_and_safety_ratio():
    class Ctx:
        budget = Budget(
            context_window_tokens=1000,
            max_output_tokens=100,
            token_safety_ratio=0.5,
        )

    fit = fit_for_context([
        SystemMessage(content="policy " * 100),
        HumanMessage(content="question " * 100),
    ], Ctx())
    assert fit.budget_tokens == 450
    assert fit.input_tokens <= fit.budget_tokens


def test_agent_context_does_not_reinject_retrieved_tool_messages():
    state = {
        "messages": [
            HumanMessage(content="question"),
            ToolMessage(
                content='{"outcome":"succeeded","data":{"text":"retrieved evidence"}}',
                tool_call_id="call-1",
                name="search_papers",
            ),
        ],
        "summary_cache": "compact history",
    }
    text, decision = _render_agent_context(state)
    assert "compact history" in text
    assert "retrieved evidence" not in text
    assert decision["rendered_zones"] == [
        "memory", "conversation_summary",
    ]
