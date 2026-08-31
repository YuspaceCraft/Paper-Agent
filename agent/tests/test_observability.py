"""Phase 0 self-check — trace_id + counters + graph builds with timed nodes.

Run: python agent/tests/test_observability.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent import observability as obs


def test_trace_id_and_counters():
    obs.set_trace_id("check-abc123")
    assert obs.get_trace_id() == "check-abc123"
    obs.count("llm_calls")
    obs.count("llm_calls")
    assert obs.get_counters()["llm_calls"] == 2
    obs.log_event("turn_end", node="graph", intent="test")  # must not raise


def test_graph_builds_with_timed_nodes():
    # @timed wraps nodes with functools.wraps; this proves the graph still
    # constructs (node registration doesn't break on the wrapped signature).
    from agent.graph import build_graph
    g = build_graph()
    assert g is not None


if __name__ == "__main__":
    test_trace_id_and_counters()
    test_graph_builds_with_timed_nodes()
    print("Phase 0 observability self-check OK")
