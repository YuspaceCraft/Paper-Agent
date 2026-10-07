"""M-live self-check — 评测实时进度（live 总线 + runner 实时指标 + SSE 数据源）。

Run:  C:/Users/30811/miniconda3/envs/demo/python.exe evaluation/tests/test_live.py
覆盖：
1. 总线：订阅回放、慢订阅者丢最旧、终态关闭、snapshot / is_running 语义；
2. stream_run_events：本进程实时路径 + 非本进程（DB 轮询）回退路径；
3. runner 集成：跑批过程中「进度行」可读且指标逐条更新，跑完被最终报告覆盖，
   实时聚合与最终报告同口径（recall@5 / cost / tokens）；
4. API 薄层：/api/eval 新路由已注册 + SSE 帧格式可被 EventSource 解析。

可选端到端：`EVAL_LIVE_HTTP=1` 再跑一条「真起 uvicorn + httpx 读 SSE 帧」的用例
（验证 Starlette 分帧/content-type/404，即前端 EventSource 实际看到的东西）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation import live, runner, datasets
from evaluation.config import EvalConfig
from evaluation.trace_store import TraceStore
from evaluation.sink import set_eval_ctx, clear_eval_ctx
from evaluation.events import record_event
from agent.observability import set_trace_id_override, clear_trace_id_override


async def _drain(q: asyncio.Queue, n: int = 1) -> list[dict]:
    """等 call_soon_threadsafe 投递后取 n 条（不阻塞等待新事件）。"""
    await asyncio.sleep(0)
    out = []
    for _ in range(n):
        out.append(q.get_nowait())
    return out


async def _close_store(store: TraceStore, prev) -> None:
    """测试收尾：关库 + 还原单例。必须在 finally 里调用——aiosqlite 的 worker 是
    非守护线程，漏关一次整个脚本就退不出去（表现为「测试挂死」）。"""
    import evaluation.trace_store as ts
    await asyncio.sleep(0.2)
    await store.close()
    await asyncio.sleep(0.2)              # Windows 文件锁：等 worker 退出
    ts._INSTANCES.pop("$default", None)
    if prev is not None:
        ts._INSTANCES["$default"] = prev


async def _test_bus() -> None:
    live.reset()
    feed, q1 = live.subscribe("run-a")
    live.publish("run-a", {"type": "run_started", "total": 2})
    live.publish("run-a", {"type": "aggregate", "done": 1, "total": 2,
                           "overall": {"recall@5": 0.5}})
    evs = await _drain(q1, 2)
    assert [e["type"] for e in evs] == ["run_started", "aggregate"], evs
    assert live.snapshot("run-a")["done"] == 1
    assert live.is_running("run-a")

    # 断线重连：新订阅者先拿到全部历史（回放）
    _feed2, q2 = live.subscribe("run-a")
    replay = await _drain(q2, 2)
    assert [e["type"] for e in replay] == ["run_started", "aggregate"], replay
    assert live.snapshot("run-a")["total"] == 2

    # 终态：feed 关闭，snapshot 带 status
    live.publish("run-a", {"type": "run_finished", "status": "done"})
    assert (await _drain(q1, 1))[0]["type"] == "run_finished"
    assert not live.is_running("run-a")
    assert live.snapshot("run-a")["status"] == "done"

    # 慢订阅者：队列满时丢最旧，绝不反压
    _feed3, q3 = live.subscribe("run-b")
    for i in range(live._QUEUE_MAX + 20):
        live.publish("run-b", {"type": "aggregate", "done": i})
    await asyncio.sleep(0)
    assert q3.qsize() == live._QUEUE_MAX
    assert q3.get_nowait()["done"] > 0     # 最旧的已被丢弃

    assert live.get_feed("run-unknown") is None
    assert live.snapshot("run-unknown") is None
    live.reset()


async def _test_stream_live() -> None:
    live.reset()

    async def _consume() -> list[dict]:
        return [ev async for ev in live.stream_run_events("run-s", keepalive_s=5)]

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)              # 先订阅（回放 + 实时都要覆盖）
    live.publish("run-s", {"type": "run_started", "total": 1})
    live.publish("run-s", {"type": "query_started", "index": 1, "total": 1})
    live.publish("run-s", {"type": "query_finished", "index": 1, "total": 1,
                           "category": "ok"})
    live.publish("run-s", {"type": "aggregate", "done": 1, "total": 1})
    live.publish("run-s", {"type": "run_finished", "status": "done"})
    evs = await asyncio.wait_for(task, timeout=5)
    assert [e["type"] for e in evs] == [
        "run_started", "query_started", "query_finished", "aggregate",
        "run_finished"], evs
    live.reset()


async def _test_stream_fallback() -> None:
    """非本进程的 run（CLI / 别的 worker）：按 eval_runs 行轮询，事件同构。"""
    live.reset()
    rows = [
        {"status": "running", "query_count": 0, "metadata": {"total": 2},
         "overall": {"progress": {"done": 0, "total": 2}}},
        {"status": "running", "query_count": 1, "metadata": {"total": 2},
         "overall": {"recall@5": 0.5}},
        {"status": "done", "query_count": 2, "metadata": {"total": 2},
         "overall": {"recall@5": 0.7}, "baseline_delta": {"gate": "pass"}},
    ]
    seen = {"n": 0}

    async def _fallback() -> dict:
        row = rows[min(seen["n"], len(rows) - 1)]
        seen["n"] += 1
        return row

    out = [ev async for ev in live.stream_run_events(
        "run-db", fallback=_fallback, poll_s=0.01)]
    types = [e["type"] for e in out]
    assert types[0] == "aggregate" and types[-1] == "run_finished", types
    assert types.count("aggregate") >= 2, types
    assert out[0]["done"] == 0 and out[0]["total"] == 2, out[0]
    assert out[-1]["gate"] == "pass" and out[-1]["overall"]["recall@5"] == 0.7

    async def _missing() -> None:
        return None

    lost = [ev async for ev in live.stream_run_events(
        "run-missing", fallback=_missing, poll_s=0.01)]
    assert lost[-1]["type"] == "run_failed", lost
    live.reset()


def _fake_agent_run(store, hits_for: dict):
    async def _run(query: str, thread_id: str, *, timeout: float = 900.0) -> dict:
        # 真实 agent 不碰评测上下文：run_id / trace_id 由 runner 在调用前设好，
        # 这里保持同样行为（否则「假 agent」会把事件写到别的 run 上）。
        record_event("turn_start", node="graph")
        record_event("intent", node="understand", intent="literature_search")
        record_event("retrieved_context", node="dispatcher", tool="search_papers",
                     query=query, chunk_ids=hits_for.get(query, []),
                     snapshot="ctx (fake)")
        record_event("tool_call", node="dispatcher", tool="search_papers",
                     outcome="succeeded", duration_ms=8.0, args={"query": query})
        record_event("llm_call", node="synthesize", model="qwen-plus",
                     duration_ms=200.0,
                     payload={"tokens": {"prompt_tokens": 300,
                                         "completion_tokens": 120,
                                         "total_tokens": 420}})
        record_event("final_answer", node="synthesize", answer="answer OK")
        record_event("turn_end", node="graph", status="ok")
        return {"messages": []}
    return _run


async def _test_runner_live(tmp: str) -> None:
    """跑批过程：事件序列 + 进度行指标随执行推进（前端两条路径同源）。"""
    live.reset()
    store = TraceStore(Path(tmp) / "live_test.db")
    store._enabled = True
    import evaluation.trace_store as ts
    prev = ts._INSTANCES.get("$default")
    ts._INSTANCES["$default"] = store

    manifest = Path(tmp) / "manifest.jsonl"
    manifest.write_text("\n".join([
        '{"id": "q1", "query": "dual stream feature", '
        '"ground_truth_ids": ["p1__chunk_0001", "p1__chunk_0002"], '
        '"difficulty_level": "medium", "generation_mode": "semantic"}',
        '{"id": "q2", "query": "loss function", '
        '"ground_truth_ids": ["p1__chunk_0003"], '
        '"difficulty_level": "easy", "generation_mode": "keyword"}',
    ]) + "\n", encoding="utf-8")
    qas = datasets.load_manifest(manifest)

    collected: list[dict] = []

    async def _collect() -> None:
        async for ev in live.stream_run_events("run-live", keepalive_s=5):
            collected.append(ev)

    collector = asyncio.create_task(_collect())
    await asyncio.sleep(0.05)

    mid_rows: list[tuple] = []

    async def _on_query_done(index, total, qm):
        # 模拟前端轮询：每条 QA 完成后立刻读 eval_runs 进度行
        conn = await store._ensure_conn()
        cur = await conn.execute(
            "SELECT status, query_count, overall, metadata FROM eval_runs"
            " WHERE run_id=?", ("run-live",))
        mid_rows.append(await cur.fetchone())

    orig = runner.agent_run
    runner.agent_run = _fake_agent_run(store, {
        "dual stream feature": ["p1__chunk_0001", "p1__chunk_0009"],
        "loss function": ["p1__chunk_0003"],
    })
    try:
        cfg = EvalConfig(max_queries=0, judge_sample_size=0,
                         run_dir=Path(tmp) / "runs")
        report = await runner.run_eval(cfg, qas, run_id="run-live",
                                       with_judge=False,
                                       on_query_done=_on_query_done)
    finally:
        runner.agent_run = orig

    try:
        await asyncio.wait_for(collector, timeout=5)
        types = [e["type"] for e in collected]
        assert types[0] == "run_started", types
        assert types.count("query_started") == 2, types
        assert types.count("query_finished") == 2, types
        assert types.count("aggregate") == 2, types
        assert types[-1] == "run_finished", types

        # 逐条明细：q1 命中 1/2（recall@5=0.5），q2 全中
        qf = [e for e in collected if e["type"] == "query_finished"]
        assert qf[0]["index"] == 1 and qf[0]["total"] == 2, qf[0]
        assert qf[0]["recall@5"] == 0.5, qf[0]
        assert qf[1]["recall@5"] == 1.0, qf[1]
        assert qf[0]["tokens"] == 420 and qf[1]["cost_usd"] >= 0, qf

        # 实时聚合与最终报告同口径（同一批纯函数）
        agg = [e for e in collected if e["type"] == "aggregate"][-1]
        assert agg["done"] == 2 and agg["total"] == 2, agg
        assert agg["overall"]["tokens_total"] == 840, agg["overall"]
        for key in ("recall@5", "mrr", "task_success_rate", "tool_success_rate",
                    "cost_usd", "tokens_total"):
            assert agg["overall"][key] == report["overall"][key], (
                key, agg, report["overall"])
        assert len(agg["recent"]) == 2 and agg["recent"][-1]["qid"] == "q2"
        assert agg["eta_s"] is None          # 跑完时无剩余

        # 跑批中：进度行 status=running 且 query_count/指标逐条推进
        assert len(mid_rows) == 2, mid_rows
        assert mid_rows[0][0] == "running" and mid_rows[0][1] == 1, mid_rows[0]
        first_overall = json.loads(mid_rows[0][2])
        assert first_overall["progress"] == {"done": 1, "total": 2}, first_overall
        assert first_overall["recall@5"] == 0.5, first_overall
        assert json.loads(mid_rows[0][3])["total"] == 2

        # 跑完：同一行被最终报告覆盖（不再有 progress 标记位）
        conn = await store._ensure_conn()
        cur = await conn.execute(
            "SELECT status, query_count, overall FROM eval_runs WHERE run_id=?",
            ("run-live",))
        final_row = await cur.fetchone()
        assert final_row[0] == "done" and final_row[1] == 2, final_row
        assert "progress" not in json.loads(final_row[2])
    finally:
        await _close_store(store, prev)
        live.reset()


def _test_api_surface() -> None:
    from web.api.routers import eval as eval_router

    routes = {(tuple(sorted(r.methods))[0], r.path)
              for r in eval_router.router.routes}
    for want in [("GET", "/api/eval/runs/{run_id}/stream"),
                 ("POST", "/api/eval/single/start"),
                 ("GET", "/api/eval/single/{thread_id}/stream"),
                 ("POST", "/api/eval/single")]:
        assert want in routes, want

    frame = eval_router._sse({"type": "aggregate", "done": 1, "中文": "✓"})
    assert frame.startswith("data: ") and frame.endswith("\n\n"), frame
    assert json.loads(frame[6:-2])["中文"] == "✓"


def _pin_eval_env(tmp: str):
    """把 EvalConfig.from_env 固定到临时 run_dir（避免测试写进仓库 eval_output）。

    返回原 classmethod 描述符本身，调用方 `EvalConfig.from_env = orig` 即可还原
    （多个用例依次调用时也幂等）。
    """
    from evaluation.config import EvalConfig
    original = EvalConfig.__dict__["from_env"]      # classmethod 描述符
    inner = original.__func__

    def _patched(**overrides):
        cfg = inner(EvalConfig, **overrides)
        cfg.run_dir = Path(tmp) / "runs"
        return cfg

    EvalConfig.from_env = _patched
    return original


async def _sse_frames(body_iterator, stop_types: set[str]) -> list[dict]:
    """消费 StreamingResponse 的 body_iterator，直到终态事件。"""
    frames: list[dict] = []
    async for chunk in body_iterator:
        ev = json.loads(str(chunk)[len("data: "):].strip())
        frames.append(ev)
        if ev.get("type") in stop_types:
            break
    return frames


async def _test_api_run_stream(tmp: str) -> None:
    """端到端：POST /api/eval/runs 起跑 → GET /runs/{id}/stream 收实时指标。"""
    from web.api.routers import eval as eval_router
    from evaluation.config import EvalConfig

    live.reset()
    store = TraceStore(Path(tmp) / "api_run.db")
    store._enabled = True
    import evaluation.trace_store as ts
    prev = ts._INSTANCES.get("$default")
    ts._INSTANCES["$default"] = store

    manifest = Path(tmp) / "api_manifest.jsonl"
    manifest.write_text(
        '{"id": "q1", "query": "loss function", "ground_truth_ids": ["p1__chunk_0003"]}\n'
        '{"id": "q2", "query": "dual stream feature", '
        '"ground_truth_ids": ["p1__chunk_0001"]}\n', encoding="utf-8")

    original_from_env = _pin_eval_env(tmp)
    orig_agent_run = runner.agent_run
    runner.agent_run = _fake_agent_run(store, {
        "loss function": ["p1__chunk_0003"],
        "dual stream feature": ["p1__chunk_0009"],
    })
    try:
        started = await eval_router.start_run(
            {"manifest": str(manifest), "limit": 0, "judge": False})
        run_id = started["run_id"]
        assert started["status"] == "running", started
        assert started["query_count"] == 2, started
        # 进度行必须在 POST 返回时就可读（前端订阅 SSE 不撞 404）
        row = await eval_router._get_report_row(run_id)
        assert row is not None and row[4] == "running", row

        stream = await eval_router.stream_run(run_id)
        frames = await _sse_frames(stream.body_iterator, {"run_finished", "run_failed"})
        types = [f["type"] for f in frames]
        assert types[0] == "run_started", types
        assert types.count("query_finished") == 2, types
        assert types[-1] == "run_finished", types
        agg = [f for f in frames if f["type"] == "aggregate"][-1]
        assert agg["done"] == 2 and agg["total"] == 2, agg
        assert len(agg["recent"]) == 2 and agg["recent"][-1]["qid"] == "q2", agg["recent"]
        # 跑完：报告行已是最终态（无 progress 标记）
        final = eval_router._report_dict(await eval_router._get_report_row(run_id))
        assert final["status"] == "done", final["status"]
        assert "progress" not in (final["overall"] or {}), final["overall"]
        assert final["overall"]["tokens_total"] == 840, final["overall"]
        # 不存在的 run → 404（EventSource 会看到连接失败，不会静默挂死）
        from fastapi import HTTPException
        try:
            await eval_router.stream_run("eval_nope")
            raise AssertionError("expected 404 for unknown run")
        except HTTPException as exc:
            assert exc.status_code == 404
    finally:
        runner.agent_run = orig_agent_run
        EvalConfig.from_env = original_from_env
        await _close_store(store, prev)
        live.reset()


async def _test_api_single_stream(tmp: str) -> None:
    """端到端：POST /api/eval/single/start → GET /single/{id}/stream 分阶段刷新。"""
    import agent.graph as graph_mod
    from web.api.routers import eval as eval_router
    from evaluation.config import EvalConfig
    from fastapi import HTTPException

    store = TraceStore(Path(tmp) / "api_single.db")
    store._enabled = True
    import evaluation.trace_store as ts
    prev = ts._INSTANCES.get("$default")
    ts._INSTANCES["$default"] = store

    calls = {"n": 0}

    async def _fake_graph_run(query: str, thread_id: str, mode=None, **kw) -> dict:
        calls["n"] += 1
        set_eval_ctx(run_id="", thread_id=thread_id, source="single")
        record_event("turn_start", node="graph", payload={"query": query})
        record_event("intent", node="understand", intent="literature_search")
        await asyncio.sleep(1.2)          # 慢链路：中途应能收到 running 快照
        record_event("tool_call", node="dispatcher", tool="search_papers",
                     outcome="succeeded", duration_ms=9.0, args={"query": query})
        record_event("final_answer", node="synthesize", answer="answer OK")
        record_event("turn_end", node="graph", status="ok")
        clear_eval_ctx()
        return {"context": {"active_project": "demo"}, "context_snapshot": "ctx"}

    original_from_env = _pin_eval_env(tmp)
    orig_run = graph_mod.run
    graph_mod.run = _fake_graph_run
    try:
        started = await eval_router.start_single({"query": "loss function"})
        thread_id = started["thread_id"]
        assert started["status"] == "running", started
        again = await eval_router.start_single(
            {"query": "loss function", "thread_id": thread_id})
        # 复用同一个 thread：不重开任务（后台 task 还没被调度，这里只保证不重跑）
        assert again.get("reused") is True and calls["n"] <= 1, (again, calls)

        stream = await eval_router.stream_single(thread_id)
        frames = await _sse_frames(stream.body_iterator, {"single_done", "single_error"})
        types = [f["type"] for f in frames]
        assert types[0] == "single_started", types
        assert types[-1] == "single_done", types
        snaps = [f for f in frames if f["type"] == "single_snapshot"]
        assert snaps[0]["status"] == "running", snaps[0]
        assert calls["n"] == 1, calls          # 复用线程不会重复跑同一条 query
        done = frames[-1]
        assert done["status"] == "done", done
        stages = [s["key"] for s in done["flow"]["stages"]]
        assert "answer" in stages and "tools" in stages, stages
        assert done["flow"]["metrics"]["cost_usd"] >= 0
        assert done["flow"]["readonly_trace_url"].endswith(f"single:{thread_id}")
        assert done["flow"]["context"]["active_project"] == "demo"

        trace = await eval_router.get_trace(f"single:{thread_id}")
        assert trace["event_count"] >= 5, trace["event_count"]

        try:
            await eval_router.stream_single("no-such-thread")
            raise AssertionError("expected 404 for unknown single run")
        except HTTPException as exc:
            assert exc.status_code == 404
    finally:
        graph_mod.run = orig_run
        EvalConfig.from_env = original_from_env
        await _close_store(store, prev)


async def _test_http_sse(tmp: str) -> None:
    """端到端 HTTP：真起 uvicorn，用 httpx 读 SSE 帧（`EVAL_LIVE_HTTP=1` 才跑）。

    前几个用例只验证 router 的 async generator；这一条验证 Starlette 确实把事件流
    按帧推给客户端（content-type / 分帧 / 404），也就是前端 EventSource 看到的东西。
    默认跳过：要占端口且依赖 httpx+uvicorn，不适合每次自检。
    """
    import socket
    import httpx
    import uvicorn
    from fastapi import FastAPI
    from web.api.routers import eval as eval_router

    live.reset()
    store = TraceStore(Path(tmp) / "http_sse.db")
    store._enabled = True
    import evaluation.trace_store as ts
    prev = ts._INSTANCES.get("$default")
    ts._INSTANCES["$default"] = store

    manifest = Path(tmp) / "http_manifest.jsonl"
    manifest.write_text(
        '{"id": "q1", "query": "loss function", "ground_truth_ids": ["p1__chunk_0003"]}\n'
        '{"id": "q2", "query": "dual stream feature", '
        '"ground_truth_ids": ["p1__chunk_0001"]}\n', encoding="utf-8")

    with socket.socket() as s:            # 让内核挑一个空闲端口
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    original_from_env = _pin_eval_env(tmp)
    orig_agent_run = runner.agent_run
    runner.agent_run = _fake_agent_run(store, {
        "loss function": ["p1__chunk_0003"],
        "dual stream feature": ["p1__chunk_0009"],
    })
    app = FastAPI()
    app.include_router(eval_router.router)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):              # 等 uvicorn 起来
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started, "uvicorn failed to start"

        base = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(base_url=base, timeout=60) as cli:
            started = (await cli.post("/api/eval/runs", json={
                "manifest": str(manifest), "limit": 0, "judge": False})).json()
            run_id = started["run_id"]
            assert started["status"] == "running", started

            frames: list[dict] = []
            async with cli.stream("GET", f"/api/eval/runs/{run_id}/stream") as resp:
                assert resp.status_code == 200, resp.status_code
                assert resp.headers["content-type"].startswith("text/event-stream"), resp.headers
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[len("data: "):])
                    frames.append(ev)
                    if ev["type"] in ("run_finished", "run_failed"):
                        break
            types = [f["type"] for f in frames]
            assert types[0] == "run_started" and types[-1] == "run_finished", types
            agg = [f for f in frames if f["type"] == "aggregate"][-1]
            assert agg["done"] == 2 and agg["total"] == 2, agg

            # 真实 HTTP 路径上的「实时指标 == 报告口径」
            report = (await cli.get(f"/api/eval/runs/{run_id}")).json()
            for key in ("recall@5", "mrr", "task_success_rate", "tool_success_rate",
                        "cost_usd", "tokens_total"):
                assert agg["overall"][key] == report["overall"][key], key

            unknown = await cli.get("/api/eval/runs/eval_nope/stream")
            assert unknown.status_code == 404, unknown.status_code
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=15)
        runner.agent_run = orig_agent_run
        EvalConfig.from_env = original_from_env
        await _close_store(store, prev)
        live.reset()


async def _go(tmp: str) -> None:
    await _test_bus()
    await _test_stream_live()
    await _test_stream_fallback()
    await _test_runner_live(tmp)
    _test_api_surface()
    await _test_api_run_stream(tmp)
    await _test_api_single_stream(tmp)
    if os.getenv("EVAL_LIVE_HTTP", "") in {"1", "true", "yes"}:
        await _test_http_sse(tmp)


def run_all() -> None:
    import shutil
    tmp = tempfile.mkdtemp(prefix="eval_live_test_")
    try:
        asyncio.run(_go(tmp))
    finally:
        for _ in range(5):
            try:
                shutil.rmtree(tmp)
                break
            except OSError:
                import time as _t
                _t.sleep(0.2)
    print("live 实时进度 self-check OK")


if __name__ == "__main__":
    import os
    os.environ.setdefault("AGENT_TRACE_ENABLED", "1")
    run_all()
