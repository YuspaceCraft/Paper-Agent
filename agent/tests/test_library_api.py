"""Self-check — library_api circuit breaker: dead local backend fails fast.

Run: python agent/tests/test_library_api.py
ponytail: assert-based, no framework, no live backend needed. The gate is
tested by pre-tripping the breaker; real connection behavior is covered by
the short connect timeout (api_timeout).
"""
import sys
import asyncio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import agent.library_api as la
import agent.providers.builtin_provider as bp


def test_breaker_state_transitions():
    la._DOWN_STATE.clear()
    assert la.api_is_down("http://x") is False
    la.api_mark_down("http://x")
    assert la.api_is_down("http://x") is True
    la.api_mark_up("http://x")
    assert la.api_is_down("http://x") is False


def test_breaked_backend_answers_fast():
    """熔断窗口期内的库工具调用必须速断，返回 backend_down 而非网络空等。"""
    la._DOWN_STATE.clear()
    la.api_mark_down(bp.API)
    try:
        for tool, args in [
            (bp.search_papers, {}),
            (bp.fetch_content, {"paper_name": "whatever"}),
            (bp.check_paper, {}),
        ]:
            out = asyncio.run(tool.ainvoke(args))
            data = json.loads(out)
            assert data.get("ok") is False, f"{tool.__name__} must not succeed while down"
            assert data.get("error_type") == "backend_down", tool.__name__
    finally:
        la._DOWN_STATE.clear()


def test_short_timeout_configured():
    t = la.api_timeout()
    assert t.connect <= 3.0, "local backend must not stall connect for a dead port"
    assert t.read >= t.connect


if __name__ == "__main__":
    test_breaker_state_transitions()
    test_breaked_backend_answers_fast()
    test_short_timeout_configured()
    print("library_api circuit breaker self-check OK")