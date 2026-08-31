"""Phase 3 self-check — ToolDispatcher timeout / retry / passthrough.

Run: python agent/tests/test_dispatcher.py
ponytail: assert-based, no framework, no backend calls.
"""
import asyncio
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.providers import ToolDef
import agent.dispatcher as dsp


def _td(name, idempotent):
    return ToolDef(name=name, description="", parameters={},
                   source="builtin",
                   annotations={"readOnlyHint": False,
                                "idempotentHint": idempotent})


def test_passthrough():
    async def ok(name, args):
        return "result-ok"
    d = dsp.ToolDispatcher(ok, [_td("ok", False)])
    assert asyncio.run(d.call("ok", {})) == "result-ok"


def test_timeout_returns_compat_json():
    async def slow(name, args):
        await asyncio.Event().wait()  # never completes
    old = dsp._DEFAULT_TIMEOUT
    dsp._DEFAULT_TIMEOUT = 0.05
    try:
        d = dsp.ToolDispatcher(slow, [_td("slow", False)])  # non-idempotent → 1 attempt
        r = asyncio.run(d.call("slow", {}))
    finally:
        dsp._DEFAULT_TIMEOUT = old
    assert '"error_type": "transient"' in r, r
    assert '"ok": false' in r, r


def test_idempotent_retries():
    calls = {"n": 0}

    async def slow(name, args):
        calls["n"] += 1
        await asyncio.Event().wait()  # never completes → always times out

    async def no_sleep(*a, **k):
        return None

    old = dsp._DEFAULT_TIMEOUT
    dsp._DEFAULT_TIMEOUT = 0.05
    d = dsp.ToolDispatcher(slow, [_td("slow", True)])  # idempotent → 2 retries
    try:
        with mock.patch("asyncio.sleep", new=no_sleep):  # skip backoff waits
            r = asyncio.run(d.call("slow", {}))
    finally:
        dsp._DEFAULT_TIMEOUT = old
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
    assert '"error_type": "transient"' in r, r


def test_emit_when_scoped():
    """Inside a subagent scope, dispatcher must emit tool_start/tool_end
    tagged with the subagent's parent_id; outside scope it must stay silent."""
    import agent.stream as st

    async def ok(name, args):
        return "r"

    d = dsp.ToolDispatcher(ok, [_td("ok", False)])

    async def run():
        q = asyncio.Queue()
        q_token = st.set_event_queue(q)
        s_token = st.set_scope("arxiv", "run-1")
        try:
            await d.call("ok", {})
        finally:
            st.reset_scope(s_token)
            st.reset_event_queue(q_token)
        return [q.get_nowait() for _ in range(q.qsize())]

    evs = asyncio.run(run())
    types = [e["type"] for e in evs]
    assert "tool_start" in types and "tool_end" in types, types
    starts = [e for e in evs if e["type"] == "tool_start"]
    assert starts and all(e.get("parent_id") == "run-1" for e in starts)

    # Outside scope (main level) → no emit, result still returned
    assert asyncio.run(d.call("ok", {})) == "r"


if __name__ == "__main__":
    test_passthrough()
    test_timeout_returns_compat_json()
    test_idempotent_retries()
    test_emit_when_scoped()
    print("Phase 3 dispatcher self-check OK")
