from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent import observability
from agent.nodes import _trace_llm_inputs


class _FakeModel:
    model_name = "qwen-test"
    kwargs = {
        "tools": [
            {"function": {"name": "search_papers"}},
            {"function": {"name": "fetch_content"}},
        ],
    }


def test_llm_inputs_are_recorded_as_node_resources() -> None:
    events: list[dict] = []
    observability.register_trace_sink(events.append)
    try:
        _trace_llm_inputs(
            _FakeModel(),
            [
                SystemMessage(content="system prompt"),
                HumanMessage(content="user context"),
            ],
            {"metadata": {"langgraph_node": "agent"}},
        )
    finally:
        observability.unregister_trace_sink(events.append)

    assert len(events) == 1
    event = events[0]
    assert event["event"] == "llm_start"
    assert event["node"] == "agent"
    assert event["model"] == "qwen-test"
    assert event["prompt"] == "system prompt"
    assert event["tools"] == ["search_papers", "fetch_content"]
    assert "user context" in event["context_preview"]
