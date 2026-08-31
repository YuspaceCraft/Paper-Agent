"""
search_loop.py — search subgraph: agent ↔ tools ReAct loop.

Extracted from the parent graph so the agent/tool loop is encapsulated
and independently testable. The parent only sees "search subgraph → answer".

Usage (inside parent graph):
    from .search_loop import build_search_subgraph
    w.add_node("search", build_search_subgraph())
    w.add_edge("resolve", "search")
    w.add_edge("search", "synthesize")
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .nodes import agent_node, after_agent
from .tools import get_cached_tools


def build_search_subgraph():
    """Build the agent ↔ tools search loop as a compiled subgraph.

    Internal flow:
        agent → after_agent
          ├─ tool_calls + under max_iterations → tools → agent (loop)
          ├─ [FINAL_ANSWER] → END (fast path)
          └─ max_iterations exhausted → END (parent handles safety net)

    The subgraph is self-contained: it owns agent_node, ToolNode, and
    after_agent. The parent graph routes its output to synthesize or END.
    """
    tools = get_cached_tools()

    sg = StateGraph(AgentState)

    sg.add_node("agent", agent_node)
    sg.add_node("tools", ToolNode(tools))

    sg.set_entry_point("agent")

    sg.add_conditional_edges(
        "agent",
        after_agent,
        {"tools": "tools", "synthesize": END, "end": END},
    )
    sg.add_edge("tools", "agent")

    return sg.compile()
