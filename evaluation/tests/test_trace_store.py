"""M1 self-check — trace_store 批量落库 + sink 转发的冒烟测试。

Run:  C:/Users/30811/miniconda3/envs/demo/python.exe evaluation/tests/test_trace_store.py
ponytail: assert-based, no framework, no LLM/backend calls（临时库文件）。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent import observability as obs
from evaluation import sink, trace_store


def _fresh_store(tmpdir: str) -> trace_store.TraceStore:
    st = trace_store.TraceStore(Path(tmpdir) / "trace_test.db")
    st._enabled = True
    return st


async def _exercise(store: trace_store.TraceStore) -> None:
    # record() 热路径：无 loop / 有 loop 两态都要不抛
    obs.set_trace_id("test-trace-0001")
    store.set_thread_map("test-trace-0001", thread_id="sess-1",
                         turn_seq=1, source="live")

    for i in range(200):
        store.record({"ts": "2026-01-01T00:00:00.000Z", "event_type": "node_end",
                      "node": "understand", "duration_ms": float(i), "n": i})

    await store.flush()
    evs = await store.get_trace("test-trace-0001")
    assert len(evs) == 200, len(evs)
    assert "ok" not in evs[0]
    seqs = [e["seq"] for e in evs]
    assert seqs == sorted(seqs), seqs[:5]

    # thread 归属 + get_thread
    thread = await store.get_thread("sess-1")
    assert len(thread) == 200
    assert thread[0]["thread_id"] == "sess-1"
    assert thread[0]["turn_seq"] == 1

    # query 过滤（event_type / node）
    hits = await store.query(event_type="node_end", limit=50)
    assert len(hits) == 50
    assert all(e["event_type"] == "node_end" for e in hits)

    # eval 上下文注入 run_id/index
    sink.set_eval_ctx(run_id="run-xyz", thread_id="evalqa:wi-1", source="eval")
    obs.set_trace_id("test-trace-0002")
    store.record({"ts": "2026-01-01T00:00:01.000Z", "event_type": "turn_start"})
    await store.flush()
    run = await store.query(run_id="run-xyz")
    assert len(run) == 1
    assert run[0]["thread_id"] == "evalqa:wi-1"
    assert run[0]["source"] == "eval"
    sink.clear_eval_ctx()

    # trace_id override：评测跑批确定性 trace
    obs.set_trace_id_override("qa-fixed-1")
    obs.set_trace_id("whatever")
    assert obs.get_trace_id() == "qa-fixed-1"
    obs.clear_trace_id_override()
    obs.set_trace_id("whatever")
    assert obs.get_trace_id() == "whatever"


async def _sink_forward(tmpdir: str) -> None:
    """attach 后 log_event 的（event→event_type）归一化 + 归属透传。"""
    store = trace_store.TraceStore(Path(tmpdir) / "trace_sink_test.db")
    store._enabled = True
    prev = trace_store._INSTANCES.pop("$default", None)
    trace_store._INSTANCES["$default"] = store
    try:
        sink.attach()
        obs.set_trace_id("test-sink-0001")
        store.set_thread_map("test-sink-0001", thread_id="sess-s", source="live")
        obs.log_event("understand_llm_failed", node="understand", error="boom")
        await store.flush()
        evs = await store.get_trace("test-sink-0001")
        assert len(evs) == 1
        assert evs[0]["event_type"] == "understand_llm_failed"
        assert evs[0]["thread_id"] == "sess-s"
        assert evs[0]["error"] == "boom"  # log_event 顶层字段 → trace 顶层列
    finally:
        await store.close()
        if prev is not None:
            trace_store._INSTANCES["$default"] = prev
        else:
            trace_store._INSTANCES.pop("$default", None)


async def _production_drop(tmpdir: str) -> None:
    """No eval ctx / thread mapping => production events stay out of local DB."""
    store = trace_store.TraceStore(Path(tmpdir) / "trace_prod_test.db")
    store._enabled = True
    store._store_live_events = False
    obs.set_trace_id("prod-trace-0001")
    store.record({"ts": "2026-01-01T00:00:00.000Z", "event_type": "final_answer"})
    await store.flush()
    assert await store.get_trace("prod-trace-0001") == []
    await store.close()


def run_all() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(tmp)
        try:
            asyncio.run(_exercise(store))
        finally:
            asyncio.run(store.close())
        print("trace_store 批量落库 self-check OK")

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_sink_forward(tmp))
        print("sink 转发 self-check OK")

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_production_drop(tmp))
        print("production trace drop self-check OK")


if __name__ == "__main__":
    os.environ.setdefault("AGENT_TRACE_ENABLED", "1")
    run_all()
