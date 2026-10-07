"""
eval.py — Agent 评测 API（薄封装 evaluation 包）。

GET    /api/eval/traces/{trace_id}                 — 完整链路事件链
GET    /api/eval/thread/{thread_id}                — 会话各 turn（trace_id 列表）
GET    /api/eval/runs                              — 最近评测 run 列表
GET    /api/eval/runs/{run_id}                     — run 报告全文
GET    /api/eval/runs/{run_id}/badcases            — 逐条 badcase 表
POST   /api/eval/runs                              — 起后台评测跑批（202）
POST   /api/eval/prune                             — 清理 trace
POST   /api/eval/manifest/from-trace               — 线上 badcase 回流评测集
GET    /api/eval/runs/{run_id}/stream              — SSE：跑批过程实时进度 + 实时指标
POST   /api/eval/single                            — 单样例透明评测（阻塞，返回完整 flow）
POST   /api/eval/single/start                      — 单样例评测（非阻塞，返回 thread_id）
GET    /api/eval/single/{thread_id}/stream         — SSE：单样例分阶段实时视图

挂载于 web/api/main.py include_router（prefix=/api/eval）。

实时进度（2026-09-15）：跑批不再是「启动后等最后一份报告」——runner 每完成一条
QA 就发 query_finished + aggregate（累计指标/token/成本/ETA），本模块把它们转成
SSE；同一份聚合同时写进 eval_runs 进度行（status=running），所以 SSE 断线或
跨进程（CLI 跑批）时，列表 / 报告页轮询也能看到随执行更新的指标。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from evaluation.config import EvalConfig
from evaluation.trace_store import get_trace_store
from evaluation import datasets as _datasets

router = APIRouter(prefix="/api/eval", tags=["eval"])

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                "Connection": "keep-alive"}
_SINGLE_POLL_S = 0.8        # 单样例：轮询 trace 事件的间隔（阶段实时出现）
_SINGLE_STREAM_MAX_S = 1800  # 单样例流上限（与 TURN_TIMEOUT 同量级）


def _sse(event: dict) -> str:
    """一条 SSE 数据帧（EventSource 直接 JSON.parse）。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ---- 查询（只读） ----


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    events = await get_trace_store().get_trace(trace_id)
    if not events:
        raise HTTPException(404, f"trace not found: {trace_id}")
    return {"trace_id": trace_id, "event_count": len(events), "events": events}


@router.get("/thread/{thread_id}")
async def get_thread(thread_id: str):
    store = get_trace_store()
    turns = await store.list_turns(thread_id)
    events = await store.get_thread(thread_id)
    return {"thread_id": thread_id, "turns": turns, "events": events}


@router.get("/runs")
async def list_runs(limit: int = 20):
    store = get_trace_store()
    await _reconcile_stale(store)
    conn = await store._ensure_conn()
    cur = await conn.execute(
        "SELECT run_id, dataset_id, status, query_count, started_at, overall,"
        " finished_at, badcases"
        " FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,))
    rows = await cur.fetchall()
    out = []
    for r in rows:
        ov = {}
        try:
            ov = json.loads(r[5] or "{}")
        except ValueError:
            pass
        badcases = []
        try:
            badcases = json.loads(r[7] or "[]")
        except ValueError:
            pass
        out.append({
            "run_id": r[0], "dataset_id": r[1], "status": r[2],
            "query_count": r[3], "started_at": r[4], "overall": ov,
            "finished_at": r[6],
            # 完整用时（running 时是「已用」）：起始/结束都由 runner 打点
            "duration_s": _duration_s(r[4], r[6]),
            "badcase_count": len(badcases),
        })
    return {"runs": out, "count": len(out)}


def _duration_s(started_at, finished_at) -> Optional[float]:
    """起止 ISO 串 → 秒（running 中按「到现在」算已用时长）。"""
    started = _epoch(started_at)
    if started is None:
        return None
    finished = _epoch(finished_at) or time.time()
    return round(finished - started, 1)


def _epoch(ts) -> Optional[float]:
    from datetime import datetime
    if ts in (None, ""):
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


async def _reconcile_stale(store) -> None:
    """读路径收敛：心跳过期的 running 行 → interrupted（失败只降级）。"""
    try:
        from evaluation.runner import reconcile_stale_runs
        await reconcile_stale_runs(
            store, stale_after_s=EvalConfig.from_env().stale_run_after_s)
    except Exception:  # noqa: BLE001 — 收敛是观测通道的兜底
        pass


async def _get_report_row(run_id: str):
    store = get_trace_store()
    conn = await store._ensure_conn()
    cur = await conn.execute(
        "SELECT * FROM eval_runs WHERE run_id=?", (run_id,))
    return await cur.fetchone()


_REPORT_COLUMNS = ["run_id", "dataset_id", "started_at", "finished_at",
                   "status", "query_count", "metadata", "overall", "dimension",
                   "badcases", "judge", "tool_metrics", "task_metrics",
                   "baseline_delta", "cost_estimate", "notes"]

_REPORT_JSON = {"metadata", "overall", "dimension", "badcases", "judge",
                "tool_metrics", "task_metrics", "baseline_delta",
                "cost_estimate", "notes"}


def _report_dict(row) -> dict:
    d = dict(zip(_REPORT_COLUMNS, row))
    for k in _REPORT_JSON:
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (ValueError, TypeError):
                pass
    # 报告页也要能显示完整用时（与列表同一个口径）
    if d.get("duration_s") is None:
        d["duration_s"] = _duration_s(d.get("started_at"), d.get("finished_at"))
    return d


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    await _reconcile_stale(get_trace_store())
    row = await _get_report_row(run_id)
    if not row:
        raise HTTPException(404, f"run not found: {run_id}")
    return _report_dict(row)


@router.get("/runs/{run_id}/badcases")
async def get_run_badcases(run_id: str):
    row = await _get_report_row(run_id)
    if not row:
        raise HTTPException(404, f"run not found: {run_id}")
    d = _report_dict(row)
    bad = d.get("badcases") or []
    return {"run_id": run_id, "badcases": bad, "count": len(bad)}


# ---- 单样例透明评测（第一层：完整流程可视化） ----


# 单样例跑批登记表：thread_id → {status, query, mode, result, error, started, finished}
# 进程内、短生命周期（一条 turn）；重启/跨进程订阅时 stream 端点仍能从 trace 事件
# 重建分阶段视图（只少 graph 返回的 context 快照）。
_SINGLE_RUNS: dict[str, dict] = {}
_SINGLE_KEEP_S = 1800


def _prune_single_runs() -> None:
    """清理已结束且超过保留期的登记项（内存上限；进行中的不动）。"""
    now = time.time()
    for tid, rec in list(_SINGLE_RUNS.items()):
        if rec.get("status") == "running":
            continue
        age = now - float(rec.get("finished") or rec.get("started") or now)
        if age > _SINGLE_KEEP_S:
            _SINGLE_RUNS.pop(tid, None)


def _single_args(body: dict) -> tuple[str, str, Optional[str], dict[str, str]]:
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query required")
    thread_id = body.get("thread_id") or f"eval-single-{uuid.uuid4().hex[:6]}"
    raw_overrides = body.get("prompt_overrides") or {}
    overrides = {
        str(key): str(value)
        for key, value in raw_overrides.items()
        if str(key).strip() and str(value).strip()
    } if isinstance(raw_overrides, dict) else {}
    return query, thread_id, body.get("mode"), overrides


async def _single_turn(thread_id: str, query: str, mode: Optional[str],
                       prompt_overrides: dict[str, str] | None = None) -> None:
    """跑一条完整链路（生产路径），结果/错误登记进 _SINGLE_RUNS。"""
    from evaluation.sink import set_eval_ctx, clear_eval_ctx
    from evaluation.sink import attach as attach_trace_sink
    from agent.observability import (set_trace_id_override,
                                     clear_trace_id_override)
    from evaluation.config import EvalConfig
    from agent.prompt_store import clear_prompt_overrides, set_prompt_overrides

    rec = _SINGLE_RUNS.setdefault(thread_id, {})
    cfg = EvalConfig.from_env()
    attach_trace_sink()
    set_trace_id_override(f"single:{thread_id}")
    set_eval_ctx(run_id="", thread_id=thread_id, qa_id=f"single:{thread_id}",
                 source="single")
    set_prompt_overrides(prompt_overrides)
    try:
        from agent.graph import run as _graph_run, TURN_TIMEOUT
        timeout = min(cfg.turn_timeout, TURN_TIMEOUT) if TURN_TIMEOUT else cfg.turn_timeout
        rec["result"] = await asyncio.wait_for(
            _graph_run(query, thread_id=thread_id, mode=mode), timeout=timeout)
        rec["status"] = "done"
    except asyncio.TimeoutError:
        rec["result"] = {"intent": "general_chat", "error": "turn_timeout",
                         "context_snapshot": ""}
        rec["status"] = "failed"
        rec["error"] = "turn_timeout"
    except Exception as exc:  # noqa: BLE001
        rec["result"] = {"intent": "", "error": f"{type(exc).__name__}: {exc}",
                         "context_snapshot": ""}
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        clear_prompt_overrides()
        clear_trace_id_override()
        clear_eval_ctx()
        rec["finished"] = time.time()


async def _single_flow(thread_id: str, rec: dict, query: str) -> dict:
    """当前 trace 事件 → 分阶段 flow（未跑完时 result=None，只出已完成阶段）。

    完成判定双保险：登记表状态（本进程）或 trace 里出现 turn_end（重启/跨进程）。
    """
    from evaluation.flow import build_flow
    from evaluation.report import estimate_cost
    from evaluation.config import EvalConfig

    store = get_trace_store()
    await store.flush()
    events = await store.get_thread(thread_id)
    trace_id = f"single:{thread_id}"
    finished = (rec.get("status") in ("done", "failed")
                or any(e.get("event_type") == "turn_end" for e in events))
    # 完整用时 = 单独计时：起跑登记 started，跑完登记 finished（不累加链路各步耗时）
    started = rec.get("started")
    ended = rec.get("finished")
    wall_ms: Optional[float] = None
    if started:
        wall_ms = ((ended if (finished and ended) else time.time()) - started) * 1000
    flow = build_flow(events, rec.get("result") if finished else None,
                      query=query, wall_ms=wall_ms)
    cost = estimate_cost([e for e in events if e.get("event_type") == "llm_call"],
                         EvalConfig.from_env().prices)
    flow["metrics"]["cost_usd"] = cost["cost_usd"]
    flow["metrics"]["elapsed_s"] = round(wall_ms / 1000.0, 1) if wall_ms else None
    flow["metrics"]["trace_id"] = trace_id
    flow["metrics"]["event_count"] = len(events)
    flow["thread_id"] = thread_id
    flow["status"] = rec.get("status") or ("done" if finished else "running")
    if rec.get("error"):
        flow["error"] = rec["error"]
    if finished:
        flow["readonly_trace_url"] = f"/api/eval/traces/{trace_id}"
    return flow


@router.post("/single")
async def run_single(body: dict):
    """跑一个 query 的完整链路并返回分阶段透明视图（阻塞到跑完）。

    body: {query, thread_id?, mode?(auto/react/plan)}
    返回 build_flow 结构：stages（意图/上下文/计划/工具/回答，各含耗时与 token）+
    metrics（总时长/总token/估算占比/成本/工具成败/任务成败）+ context。
    长时间（工具编排分钟级）由调用方等待；与 /api/agent/chat 同样的 TURN_TIMEOUT。
    需要「边跑边看」用 POST /single/start + GET /single/{thread_id}/stream。
    """
    query, thread_id, mode, prompt_overrides = _single_args(body)
    _prune_single_runs()
    _SINGLE_RUNS[thread_id] = {"status": "running", "query": query, "mode": mode,
                               "prompt_overrides": prompt_overrides,
                               "started": time.time()}
    await _single_turn(thread_id, query, mode, prompt_overrides)
    return await _single_flow(thread_id, _SINGLE_RUNS[thread_id], query)


@router.post("/single/start")
async def start_single(body: dict):
    """非阻塞启动单样例链路评测 → {thread_id, trace_id}（202 语义）。

    进度/指标用 GET /api/eval/single/{thread_id}/stream 订阅：EventSource 断线
    重连只重读 trace 事件，不会重跑这条 query。
    """
    query, thread_id, mode, prompt_overrides = _single_args(body)
    rec = _SINGLE_RUNS.get(thread_id)
    if rec and rec.get("status") == "running":
        return {"outcome": "succeeded", "thread_id": thread_id, "status": "running",
                "reused": True}
    _prune_single_runs()
    _SINGLE_RUNS[thread_id] = {"status": "running", "query": query, "mode": mode,
                               "prompt_overrides": prompt_overrides,
                               "started": time.time()}
    asyncio.create_task(_single_turn(thread_id, query, mode, prompt_overrides))
    return {"outcome": "succeeded", "thread_id": thread_id, "trace_id": f"single:{thread_id}",
            "status": "running"}


@router.get("/single/{thread_id}/stream")
async def stream_single(thread_id: str):
    """SSE：单样例链路的分阶段实时视图（阶段/耗时/token/工具逐步出现）。

    事件：single_started → single_snapshot（每 ~0.8s，含 flow 快照）
    → single_done | single_error。
    """
    rec = _SINGLE_RUNS.get(thread_id)
    store = get_trace_store()
    await store.flush()
    events = await store.get_thread(thread_id)
    if rec is None and not events:
        raise HTTPException(404, f"single run not found: {thread_id}")
    query = (rec or {}).get("query") or ""

    async def _gen():
        started = time.time()
        yield _sse({"type": "single_started", "thread_id": thread_id,
                    "trace_id": f"single:{thread_id}"})
        while True:
            flow = await _single_flow(thread_id, rec or {}, query)
            status = str(flow.get("status") or "running")
            frame = {"thread_id": thread_id, "status": status,
                     "elapsed_s": round(time.time() - started, 1), "flow": flow}
            yield _sse({"type": "single_snapshot", **frame})
            if status in ("done", "failed"):
                yield _sse({"type": "single_done", **frame})
                return
            if time.time() - started > _SINGLE_STREAM_MAX_S:
                yield _sse({"type": "single_error", "thread_id": thread_id,
                            "error": "stream_timeout"})
                return
            await asyncio.sleep(_SINGLE_POLL_S)

    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


# ---- 跑批（后台任务） ----


@router.post("/runs")
async def start_run(body: dict):
    manifest = body.get("manifest") or ""
    if not manifest:
        raise HTTPException(400, "manifest path required")
    limit = int(body.get("limit") or 0)
    with_judge = bool(body.get("judge", True)) and not body.get("no_judge", False)
    run_id = f"eval_{uuid.uuid4().hex[:8]}"

    try:
        qas = _datasets.deduplicate(_datasets.load_manifest(manifest))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"manifest load failed: {type(exc).__name__}: {exc}")
    if limit:
        qas = qas[:limit]
    dataset_id = _datasets.dataset_id_for(qas)
    dataset_validation = _datasets.validate_ground_truth_coverage(qas)

    # 先把「进行中」行落库：前端拿到 run_id 后立刻能订阅 SSE（避免 404 竞态），
    # 也能在不支持 SSE 的场景直接用列表/报告页轮询看进度。
    await _upsert_running_row(run_id, dataset_id, total=len(qas))

    async def _background() -> None:
        from evaluation.runner import run_eval
        try:
            cfg = EvalConfig.from_env(max_queries=limit)
            await run_eval(cfg, qas, run_id=run_id, dataset_id=dataset_id,
                           with_judge=with_judge)
        except Exception as exc:  # noqa: BLE001
            from evaluation.live import publish as _live_publish
            _live_publish(run_id, {"type": "run_failed", "run_id": run_id,
                                   "error": f"{type(exc).__name__}: {exc}"})
            await _mark_run_failed(run_id, dataset_id, exc)

    asyncio.create_task(_background())
    return {"outcome": "succeeded", "run_id": run_id, "dataset_id": dataset_id,
            "query_count": len(qas), "status": "running",
            "dataset_validation": dataset_validation}


async def _upsert_running_row(run_id: str, dataset_id: str, total: int) -> None:
    """启动即写「进行中」行（进度行失败不影响跑批）。"""
    from evaluation.runner import ensure_running_row
    try:
        await ensure_running_row(get_trace_store(), run_id, dataset_id, total)
    except Exception:  # noqa: BLE001
        pass


async def _mark_run_failed(run_id: str, dataset_id: str, exc: BaseException) -> None:
    """跑批在进入 runner 前就失败时补一条 failed 行（进度行/报告页可见）。"""
    from evaluation.runner import upsert_run_row
    try:
        await upsert_run_row(
            get_trace_store(), run_id=run_id, dataset_id=dataset_id,
            status="failed", query_count=0, started_at=time.time(),
            notes=[f"{type(exc).__name__}: {exc}"])
    except Exception:  # noqa: BLE001
        pass


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE：跑批过程实时进度 + 实时指标（本进程内存总线优先）。

    事件：run_started → query_started / query_finished → aggregate（累计指标 /
    token / 成本 / ETA）→ judge_started|judge_finished → run_finished | run_failed。
    非本进程的 run（CLI / 别的 worker）退回每 2s 轮询 eval_runs 行，事件同构，
    指标粒度到「已完成条数」；EventSource 断线重连先收历史回放。
    """
    from evaluation.live import stream_run_events

    row = await _get_report_row(run_id)
    if not row:
        raise HTTPException(404, f"run not found: {run_id}")

    async def _fallback() -> Optional[dict]:
        r = await _get_report_row(run_id)
        return _report_dict(r) if r else None

    async def _gen():
        async for ev in stream_run_events(run_id, fallback=_fallback):
            yield _sse(ev)

    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@router.post("/prune")
async def prune(body: dict):
    older = int(body.get("older_than_days") or 30)
    run_id = body.get("run_id")
    deleted = await get_trace_store().prune(older_than_days=older, run_id=run_id)
    return {"outcome": "succeeded", "deleted": deleted}


@router.post("/feedback")
async def create_feedback(body: dict):
    """Write one LangSmith feedback entry for a completed evaluation trace."""
    from evaluation.feedback import submit_feedback

    trace_id = str(body.get("trace_id") or "").strip()
    key = str(body.get("key") or "").strip()
    if not trace_id or not key:
        raise HTTPException(400, "trace_id and key are required")
    score = body.get("score")
    try:
        if score is not None:
            score = float(score)
    except (TypeError, ValueError):
        raise HTTPException(400, "score must be numeric")
    try:
        return await submit_feedback(
            trace_id=trace_id,
            key=key,
            score=score,
            comment=str(body.get("comment") or ""),
            run_id=str(body.get("run_id") or ""),
            project_name=str(body.get("project") or ""),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"LangSmith feedback failed: {exc}")


# ---- 线上 badcase 回流评测集 ----


@router.post("/manifest/from-trace")
async def manifest_from_trace(body: dict):
    trace_id = body.get("trace_id") or ""
    if not trace_id:
        raise HTTPException(400, "trace_id required")
    events = await get_trace_store().get_trace(trace_id)
    if not events:
        raise HTTPException(404, f"trace not found: {trace_id}")

    query = ""
    answer = ""
    chunk_ids: list[str] = []
    for e in events:
        payload = e.get("payload") or {}
        if e.get("event_type") == "turn_end" and payload.get("query"):
            query = payload["query"]
        if e.get("event_type") == "final_answer" and payload.get("answer"):
            answer = payload["answer"]
        if e.get("event_type") == "retrieved_context":
            chunk_ids.extend(payload.get("chunk_ids") or [])
    if not query or not answer:
        raise HTTPException(422, "trace lacks query/answer (not a complete turn)")

    out = Path(__file__).resolve().parent.parent.parent.parent / "eval_output" / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "live_candidates.jsonl"
    qid = f"live-{trace_id}"
    row = {
        "id": qid, "query": query, "ground_truth_ids": list(dict.fromkeys(chunk_ids)),
        "metadata_filters": {}, "difficulty_level": "unknown",
        "generation_mode": "live_trace",
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"outcome": "succeeded", "written": str(target), "qa_id": qid,
            "chunk_ids": len(row["ground_truth_ids"])}
