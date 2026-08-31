"""Phase 1 self-check — loop governance guarantees termination.

Run: python agent/tests/test_loop.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.nodes import after_agent
from agent.graph import TURN_TIMEOUT


class _StuckMsg:
    """Fake AIMessage that still demands a tool call (the runaway case)."""
    type = "ai"
    content = ""
    tool_calls = [{"name": "search_papers", "args": {}, "id": "1"}]


def test_max_iterations_clamp():
    # At max_iterations, a message still requesting tools must route to
    # synthesize (safety net) rather than loop forever.
    stuck = {"messages": [_StuckMsg()], "iteration": 5, "max_iterations": 5}
    assert after_agent(stuck) == "synthesize", "stuck loop must terminate"

    under = {"messages": [_StuckMsg()], "iteration": 2, "max_iterations": 5}
    assert after_agent(under) == "tools", "under budget should keep going"


def test_turn_timeout_configured():
    assert TURN_TIMEOUT > 0, "AGENT_TURN_TIMEOUT must resolve to a positive float"


if __name__ == "__main__":
    test_max_iterations_clamp()
    test_turn_timeout_configured()
    print("Phase 1 governance self-check OK")
