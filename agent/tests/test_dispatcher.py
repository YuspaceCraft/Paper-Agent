"""Phase 3 self-check — ToolDispatcher timeout / retry / passthrough.

Run: python agent/tests/test_dispatcher.py
ponytail: assert-based, no framework, no backend calls.
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.providers import ToolDef
import agent.dispatcher as dsp


def _td(name, idempotent, read_only=False):
    return ToolDef(name=name, description="", parameters={},
                   source="builtin",
                   annotations={"readOnlyHint": read_only,
                                "idempotentHint": idempotent})


def test_passthrough():
    async def ok(name, args):
        return "result-ok"
    d = dsp.ToolDispatcher(ok, [_td("ok", False)])
    payload = json.loads(asyncio.run(d.call("ok", {})))
    assert payload["outcome"] == "succeeded"
    assert payload["data"] == "result-ok"


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
    assert '"error_type": "tool_timeout"' in r, r
    assert '"code": "TOOL_TIMEOUT"' in r, r
    assert '"outcome": "timed_out"' in r, r


def test_idempotent_retries():
    calls = {"n": 0}

    async def slow(name, args):
        calls["n"] += 1
        await asyncio.Event().wait()  # never completes → always times out

    async def no_sleep(*a, **k):
        return None

    old = dsp._DEFAULT_TIMEOUT
    dsp._DEFAULT_TIMEOUT = 0.05
    d = dsp.ToolDispatcher(slow, [_td("slow", True, read_only=True)])
    try:
        with mock.patch("asyncio.sleep", new=no_sleep):  # skip backoff waits
            r = asyncio.run(d.call("slow", {}))
    finally:
        dsp._DEFAULT_TIMEOUT = old
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
    assert '"error_type": "tool_timeout"' in r, r


def test_timeout_retries_emit_one_logical_trace_event():
    """A retry is one user-visible tool attempt, not N evaluation failures.

    Trace 行只有一条：dispatcher 的 log_event("tool_call")（带信封解析），
    历史上这里还会再 emit 一条 → 调用次数翻倍、成功率被摊平。
    """
    from agent import observability

    calls = {"n": 0}
    traced = []

    async def slow(name, args):
        calls["n"] += 1
        await asyncio.Event().wait()

    async def no_sleep(*a, **k):
        return None

    old = dsp._DEFAULT_TIMEOUT
    dsp._DEFAULT_TIMEOUT = 0.02
    d = dsp.ToolDispatcher(slow, [_td("slow", True, read_only=True)])
    observability.register_trace_sink(traced.append)
    try:
        with mock.patch("asyncio.sleep", new=no_sleep):
            asyncio.run(d.call("slow", {}))
    finally:
        dsp._DEFAULT_TIMEOUT = old
        observability.unregister_trace_sink(traced.append)
    assert calls["n"] == 3
    tool_rows = [e for e in traced if e.get("event") == "tool_call"]
    assert len(tool_rows) == 1, tool_rows
    assert tool_rows[0]["outcome"] == "timed_out", tool_rows[0]
    assert "ok" not in tool_rows[0], tool_rows[0]
    assert tool_rows[0]["operation"]["outcome"] == "timed_out", tool_rows[0]
    # 信封解析随行落库（评测端失败归因的唯一口径）
    assert tool_rows[0]["parsed"]["error_type"] == "tool_timeout", tool_rows[0]


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
    payload = json.loads(asyncio.run(d.call("ok", {})))
    assert payload["data"] == "r"


def test_large_result_is_spilled_then_projected(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    body = "".join(f"{i:05d}\n" for i in range(2500))

    async def read_content(name, args):
        return body

    td = ToolDef(
        name="fetch_content",
        description="",
        parameters={
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "default": 0},
                "max_chars": {"type": "integer", "default": 6000},
            },
        },
        source="builtin",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    )
    dispatcher = dsp.ToolDispatcher(read_content, [td])
    payload = json.loads(asyncio.run(dispatcher.call(
        "fetch_content", {"offset": 0, "max_chars": 1000},
    )))

    assert payload["outcome"] == "succeeded"
    assert len(payload["data"]) == 1000
    assert payload["artifacts"][0]["artifact_id"]
    assert payload["continuation"]["next_offset"] == 1000
    assert (
        payload["continuation"]["artifact_id"]
        == payload["artifacts"][0]["artifact_id"]
    )
    from agent.core.artifact_store import read_page

    stored = read_page(
        payload["artifacts"][0]["artifact_id"], offset=0, max_chars=7000,
    )
    assert stored is not None
    assert "00000" in stored["text"]
    assert '"outcome": "succeeded"' in stored["text"]


if __name__ == "__main__":
    test_passthrough()
    test_timeout_returns_compat_json()
    test_idempotent_retries()
    test_timeout_retries_emit_one_logical_trace_event()
    test_emit_when_scoped()
    test_large_result_is_spilled_then_projected()
    print("Phase 3 dispatcher self-check OK")
