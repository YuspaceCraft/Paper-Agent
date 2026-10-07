"""跑批终态 / 完整用时 / 心跳收敛 self-check（零成本，mock agent_run）。

Run:  C:/Users/30811/miniconda3/envs/demo/python.exe evaluation/tests/test_run_state.py

覆盖三个历史缺陷：
1. 完整用时没有单独的启停计时（进度行里根本没有 duration 字段）；
2. 所有任务跑完后 eval_runs 行仍是 running，没有完成态；
3. 进程中断/取消后没人把行收敛成终态（运行中断 → 永远 running）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation import datasets, live as live_bus, runner
from evaluation.config import EvalConfig
from evaluation.events import record_event
from evaluation.live import row_event
from evaluation.sink import set_eval_ctx, clear_eval_ctx
from evaluation.trace_store import TraceStore, get_trace_store
from agent.observability import set_trace_id_override, clear_trace_id_override


def _manifest(tmp: str, n: int = 2) -> list[dict]:
    path = Path(tmp) / "state_manifest.jsonl"
    rows = [
        json.dumps({"id": f"q{i}", "query": f"query {i}",
                    "ground_truth_ids": [f"p1__chunk_000{i}"]})
        for i in range(1, n + 1)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return datasets.load_manifest(path)


def _fake_agent(store, *, fail: bool = False, sleep_s: float = 0.0):
    async def _run(query: str, thread_id: str, *, timeout: float = 900.0) -> dict:
        if sleep_s:
            await asyncio.sleep(sleep_s)
        if fail:
            raise RuntimeError("backend exploded")
        record_event("turn_start", node="graph")
        record_event("tool_call", node="dispatcher", tool="search_papers",
                     outcome="succeeded", duration_ms=5.0, args={"query": query},
                     payload={"parsed": {
                         "is_envelope": True, "outcome": "succeeded",
                     }})
        record_event("final_answer", node="synthesize", answer="answer OK")
        record_event("turn_end", node="graph", status="ok")
        return {"messages": []}
    return _run


async def _swap_store(tmp: str, name: str):
    import evaluation.trace_store as ts
    store = TraceStore(Path(tmp) / name)
    store._enabled = True
    prev = ts._INSTANCES.get("$default")
    ts._INSTANCES["$default"] = store
    return store, prev


async def _restore_store(store, prev) -> None:
    import evaluation.trace_store as ts
    await store.close()
    await asyncio.sleep(0.2)                      # Windows 文件锁
    if prev is not None:
        ts._INSTANCES["$default"] = prev
    else:
        ts._INSTANCES.pop("$default", None)


async def _row(store, run_id: str):
    conn = await store._ensure_conn()
    cur = await conn.execute("SELECT * FROM eval_runs WHERE run_id=?", (run_id,))
    row = await cur.fetchone()
    return dict(zip(runner._ROW_COLUMNS, row)) if row else None


async def _test_done_run_records_wall_clock(tmp: str) -> None:
    """正常跑完：行是 done + 完成态，完整用时是启停计时的墙钟差。"""
    live_bus.reset()
    store, prev = await _swap_store(tmp, "state_done.db")
    qas = _manifest(tmp)
    orig = runner.agent_run
    runner.agent_run = _fake_agent(store, sleep_s=0.05)
    try:
        cfg = EvalConfig(max_queries=0, judge_sample_size=0,
                         run_dir=Path(tmp) / "runs",
                         progress_heartbeat_s=0)
        report = await runner.run_eval(cfg, qas, run_id="run-done",
                                       with_judge=False)
    finally:
        runner.agent_run = orig

    assert report["status"] == "done", report["status"]
    assert report["started_at"] and report["finished_at"], report
    assert report["duration_s"] >= 0.05, report["duration_s"]
    assert report["overall"]["duration_s"] == report["duration_s"], report["overall"]
    # 逐条 QA 的墙钟样本 + 工具分子分母
    assert len(report["query_durations_s"]) == 2, report["query_durations_s"]
    assert report["overall"]["tool_calls"] == 2, report["overall"]
    assert report["overall"]["tool_failures"] == 0, report["overall"]
    assert report["overall"]["task_success"] == 2, report["overall"]
    assert report["task_metrics"]["failed"] == 0, report["task_metrics"]

    row = await _row(store, "run-done")
    assert row["status"] == "done", row["status"]
    assert row["finished_at"], row
    overall = json.loads(row["overall"])
    assert overall["duration_s"] == report["duration_s"], overall
    assert "progress" not in overall, overall
    await _restore_store(store, prev)


async def _test_query_exception_is_badcase(tmp: str) -> None:
    """单条 QA 抛异常：不炸整批，但该条必须算失败并进 badcase。"""
    live_bus.reset()
    store, prev = await _swap_store(tmp, "state_qerr.db")
    qas = _manifest(tmp)
    orig = runner.agent_run
    runner.agent_run = _fake_agent(store, fail=True)
    try:
        cfg = EvalConfig(max_queries=0, judge_sample_size=0,
                         run_dir=Path(tmp) / "runs", progress_heartbeat_s=0)
        report = await runner.run_eval(cfg, qas, run_id="run-qerr",
                                       with_judge=False)
    finally:
        runner.agent_run = orig

    assert report["status"] == "done", report["status"]
    assert report["task_metrics"]["failed"] == 2, report["task_metrics"]
    assert report["overall"]["task_success_rate"] == 0.0, report["overall"]
    cats = [b["category"] for b in report["badcases"]]
    assert cats == ["run_error", "run_error"], report["badcases"]
    assert report["badcases"][0]["run_error"], report["badcases"][0]
    assert report["badcases"][0]["success"] is False, report["badcases"][0]
    await _restore_store(store, prev)


async def _test_crash_in_report_phase_leaves_failed_row(tmp: str) -> None:
    """收尾阶段崩掉（judge / 报告组装）：行必须收敛到 failed，而不是停在 running。

    这正是线上观察到的现象：5 条任务全跑完，行却永远 running。
    """
    live_bus.reset()
    store, prev = await _swap_store(tmp, "state_failed.db")
    qas = _manifest(tmp)
    orig_agent, orig_report = runner.agent_run, runner.assemble_report
    runner.agent_run = _fake_agent(store)

    def _boom(**_kw):
        raise RuntimeError("report boom")

    runner.assemble_report = _boom
    try:
        cfg = EvalConfig(max_queries=0, judge_sample_size=0,
                         run_dir=Path(tmp) / "runs", progress_heartbeat_s=0)
        try:
            await runner.run_eval(cfg, qas, run_id="run-failed", with_judge=False)
            raise AssertionError("expected the runner to re-raise")
        except RuntimeError as exc:
            assert "report boom" in str(exc), exc
    finally:
        runner.agent_run, runner.assemble_report = orig_agent, orig_report

    row = await _row(store, "run-failed")
    assert row["status"] == "failed", row
    assert row["finished_at"], row
    notes = json.loads(row["notes"] or "[]")
    assert any("report boom" in n for n in notes), notes
    # 已跑完的部分指标不丢：前端报告页仍能看到 done=2
    overall = json.loads(row["overall"])
    assert overall["progress"] == {"done": 2, "total": 2}, overall
    assert row["query_count"] == 2, row
    # 状态事件：终态 + failed
    state = live_bus.snapshot("run-failed") or {}
    assert state.get("type") == "run_failed" and state.get("status") == "failed", state
    await _restore_store(store, prev)


async def _test_cancelled_run_marks_interrupted(tmp: str) -> None:
    """被取消（关掉 App / 重启 worker）：行收敛到 interrupted。"""
    live_bus.reset()
    store, prev = await _swap_store(tmp, "state_cancel.db")
    qas = _manifest(tmp)
    orig = runner.agent_run
    runner.agent_run = _fake_agent(store, sleep_s=30.0)
    try:
        cfg = EvalConfig(max_queries=0, judge_sample_size=0,
                         run_dir=Path(tmp) / "runs", progress_heartbeat_s=0)
        task = asyncio.create_task(runner.run_eval(
            cfg, qas, run_id="run-cancel", with_judge=False))
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        runner.agent_run = orig

    row = await _row(store, "run-cancel")
    assert row["status"] == "interrupted", row
    assert row["finished_at"], row
    await _restore_store(store, prev)


async def _test_reconcile_stale_runs(tmp: str) -> None:
    """进程直接消失（无心跳）→ 读路径把 running 行收敛成 interrupted。"""
    import time
    live_bus.reset()
    store, prev = await _swap_store(tmp, "state_stale.db")
    await runner.upsert_run_row(
        store, run_id="run-stale", dataset_id="ds", status="running",
        query_count=2, started_at=time.time() - 1000,
        overall={"recall@5": 0.5, "progress": {"done": 2, "total": 5}},
        metadata={"total": 5, "heartbeat": time.time() - 1000})
    await runner.upsert_run_row(
        store, run_id="run-fresh", dataset_id="ds", status="running",
        query_count=1, started_at=time.time() - 5,
        overall={"progress": {"done": 1, "total": 5}},
        metadata={"total": 5, "heartbeat": time.time()})

    done = await runner.reconcile_stale_runs(store, stale_after_s=60)
    assert done == ["run-stale"], done

    stale = await _row(store, "run-stale")
    assert stale["status"] == "interrupted", stale["status"]
    assert stale["finished_at"], stale
    # 部分指标保留：前端仍能看到跑到哪一步 + 已经算出来的指标
    assert json.loads(stale["overall"])["recall@5"] == 0.5, stale["overall"]
    assert json.loads(stale["metadata"]).get("total") == 5, stale["metadata"]
    notes = json.loads(stale["notes"] or "[]")
    assert any("heartbeat" in n for n in notes), notes

    fresh = await _row(store, "run-fresh")
    assert fresh["status"] == "running", fresh
    await _restore_store(store, prev)


def test_row_event_duration_comes_from_timestamps() -> None:
    """DB 轮询路径（CLI / 别的 worker 的 run）也要有完整用时。"""
    row = {"status": "done", "query_count": 2, "metadata": {"total": 2},
           "started_at": "2026-09-15T10:00:00+00:00",
           "finished_at": "2026-09-15T10:00:30+00:00",
           "overall": {"recall@5": 0.5}, "baseline_delta": {"gate": "pass"},
           "notes": [], "badcases": [{"query_id": "q1"}]}
    ev = row_event("run-x", row)
    assert ev["type"] == "run_finished", ev
    assert ev["duration_s"] == 30.0, ev
    assert ev["badcase_count"] == 1, ev

    running = row_event("run-x", {**row, "status": "running",
                                  "finished_at": None})
    assert running["type"] == "aggregate", running
    assert running["elapsed_s"] > 0, running


async def _go(tmp: str) -> None:
    await _test_done_run_records_wall_clock(tmp)
    await _test_query_exception_is_badcase(tmp)
    await _test_crash_in_report_phase_leaves_failed_row(tmp)
    await _test_cancelled_run_marks_interrupted(tmp)
    await _test_reconcile_stale_runs(tmp)
    test_row_event_duration_comes_from_timestamps()


def run_all() -> None:
    import shutil
    tmp = tempfile.mkdtemp(prefix="eval_state_test_")
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
    print("跑批终态/计时 self-check OK")


if __name__ == "__main__":
    import os
    os.environ.setdefault("AGENT_TRACE_ENABLED", "1")
    run_all()
