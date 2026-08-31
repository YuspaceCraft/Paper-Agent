"""
background.py — Agent-driven background task endpoints.

POST   /api/agent/ingest          — start 入库 (ingest): returns task_id immediately
GET    /api/agent/tasks           — list top-level background tasks (the "任务栈")
GET    /api/agent/tasks/{task_id} — one task's status
POST   /api/agent/notify/stream   — SSE: stream a task-completion notice from the notifier LLM

入库 is presented as ONE atomic background task: `_run_ingest` drives the parse
pipeline (_run_pipeline) then the vectorization (_run_indexing) in-place on the
same task_id, with a continuous `stage` (parse → index) and live progress. There
are no internal sub-tasks — each "ingest" shows up as exactly one row.

API layer stays thin: validation → schedule background thread → return JSON.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from ..schemas import (
    TaskStatus, AgentIngestRequest, AgentIngestResult, AgentNotifyRequest,
)
from . import (
    _task_create, _task_update, _task_get, _task_list, _task_public,
    _subscribe_task_stream, _unsubscribe_task_stream, set_task_loop,
)
from .pdf import _locate_and_dedup, _run_pipeline
from .index import _run_indexing
from agent.safety import sanitize_output
from agent.stream import set_event_queue, reset_event_queue

router = APIRouter(prefix="/api/agent", tags=["Agent background tasks"])


# ---- composite task serialization ----
# 对外序列化统一走 routers/__init__._task_public（notify→bool、隐藏 parent、
# result 解 JSON），供 /tasks 与 /tasks/stream（SSE 推送）共用。


# ---- unified ingest runner ----

def _run_ingest(task_id: str, paper_name: str, pdf_path: str):
    """后台执行「入库」：解析 PDF → 写入向量库，同一个任务的完整过程。

    vs 旧实现：不再由「解析任务 + 索引任务」两个内部子任务拼接，而是直接在
    同一 task_id 上依次执行 _run_pipeline（解析）与 _run_indexing（向量化），
    stage/进度连续推进 —— 对 agent 和用户而言，入库是不可分割的一个操作。
    """
    try:
        _task_update(task_id, status="running", stage="parse", progress="开始解析…")

        pdf, paper_name, _, existing = _locate_and_dedup(paper_name, pdf_path)
        # 去重快判只在「确实已索引」时成立。registered-but-not-indexed 的论文
        # （旧 /api/pdf/process 仅解析流注册过：catalog indexed=false 但
        # dedup:hash 键已存在）如果直接短路，任务秒报「已在库中」而向量库毫无
        # 数据——正是「入库说完成、检查说未入库」这种矛盾状态的源头。
        if existing and bool(existing.get("indexed")):  # 与 reader.py bool() 同源，展示/判定不脱节
            _task_update(task_id, status="done", stage="", progress="Complete",
                         result=json.dumps({
                             "status": "indexed",
                             "paper_name": existing.get("title", paper_name) or paper_name,
                             "message": "该论文已在库中（重复，无需重新处理）",
                         }, ensure_ascii=False))
            return

        # Stage 1 — 解析。output dir 已有 rag_chunks.json 则跳过 docling 重解析，
        #   直接复用（meta 沿用 catalog 现有元数据，避免 register_indexed 空覆盖）；
        #   否则走 _run_pipeline（register=False 不写目录，收尾由 register_indexed()
        #   一次置真，避免中间态/失败残留误导性「已解析未入库」终态，方案 B）。
        rag_path = f"pdf_pipeline/output/{paper_name}/rag_chunks.json"
        parse_meta = None
        if Path(rag_path).exists():
            if existing:
                parse_meta = {
                    "metadata": dict(existing),
                    "page_count": existing.get("page_count", 0),
                    "chunk_count": existing.get("chunk_count", 0),
                }
        else:
            reg_data = _run_pipeline(task_id, pdf, paper_name, register=False)
            if (_task_get(task_id) or {}).get("status") == "failed":
                return  # _run_pipeline 已写失败状态
            parse_meta = reg_data or None

        # Stage 2 — 向量化入库（收尾注册+置 indexed 一次完成）
        _task_update(task_id, stage="index", progress="解析完成，正在向量化入库…")
        _run_indexing(task_id, rag_path, "", meta=parse_meta)
        t = _task_get(task_id) or {}
        if t.get("status") == "failed":
            return

        # 统一对外结果（覆盖 _run_indexing 的 stats —— 入库的单一结果形态）
        result = (t.get("result") or {}) if isinstance(t.get("result"), dict) else {}
        chunk_count = result.get("total_chunks") or result.get("total") or 0
        _task_update(task_id, status="done", stage="", progress="Complete",
                     result=json.dumps({
                         "status": "indexed", "paper_name": paper_name,
                         "chunk_count": chunk_count,
                         "message": "已完成入库，可在知识库中检索。",
                         "stages": {
                             "parse": {"status": "done"},
                             "index": {"status": "done", "chunk_count": chunk_count},
                         },
                     }, ensure_ascii=False))
    except HTTPException as exc:
        _task_update(task_id, status="failed", stage="",
                     error=f"{exc.status_code}: {exc.detail}")
    except Exception as exc:
        _task_update(task_id, status="failed", stage="",
                     error=f"{type(exc).__name__}: {exc}")


# ---- endpoints ----

@router.post("/ingest", response_model=AgentIngestResult)
async def agent_ingest(req: AgentIngestRequest):
    """Start 入库 (parse PDF → vector index) as ONE background task.

    Returns the composite task_id immediately; the whole job runs in a daemon
    thread (stage: parse → index). PDF/duplicate checks happen inside
    _locate_and_dedup; a failing task reports its error on the task's error field.
    """
    paper_name = req.paper_name.strip()
    task_id = uuid.uuid4().hex[:12]
    _task_create(task_id, kind="ingest", paper_name=paper_name,
                 notify="1" if req.notify else "")
    thread = threading.Thread(
        target=_run_ingest,
        args=(task_id, paper_name, req.pdf_path),
        daemon=True,
    )
    thread.start()
    return AgentIngestResult(task_id=task_id, paper_name=paper_name, status="running")


@router.get("/tasks", response_model=list[TaskStatus])
async def list_tasks(limit: int = 30):
    """List top-level background tasks, newest first (the task stack)."""
    return [TaskStatus(**_task_public(t)) for t in _task_list(limit)]


@router.get("/tasks/stream")
async def task_stream():
    """SSE: live background-task feed.

    Sends the current top-level task snapshot first, then a `task_update` event
    each time a task is created or its state crosses a point (parse → index →
    done/failed). EventSource auto-reconnects on drop and re-receives the
    snapshot. Keepalive comment frames stop proxies from closing the idle
    connection. Event shape: {"type": "task_snapshot"|"task_update", "task": {...}}
    """
    loop = asyncio.get_running_loop()
    set_task_loop(loop)
    q: asyncio.Queue = asyncio.Queue()
    _subscribe_task_stream(q)

    async def _events():
        try:
            for t in _task_list(30):
                yield f"data: {json.dumps({'type': 'task_snapshot', 'task': _task_public(t)}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                else:
                    yield f"data: {ev}\n\n"
        finally:
            _unsubscribe_task_stream(q)

    return StreamingResponse(_events(), media_type="text/event-stream")


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def task_status(task_id: str):
    """Get one background task's status."""
    task = _task_get(task_id)
    if task is None:
        raise HTTPException(404, f"Task {task_id} not found")
    return TaskStatus(**_task_public(task))


@router.post("/notify/stream")
async def notify_stream(req: AgentNotifyRequest):
    """SSE: stream a short task-completion notice — "agent" informing the user.

    The task facts are injected as structured context; the notifier LLM turns
    them into a 1-2 sentence message. Events: token…, done | error.
    """
    from agent.notifier import stream_task_notify

    async def _gen():
        ev_token = None
        try:
            events_q: asyncio.Queue = asyncio.Queue()
            out: asyncio.Queue = asyncio.Queue()
            ev_token = set_event_queue(events_q)
            _EV_STOP = object()
            _NOTIFY_DONE = object()

            async def _ev_pump():
                while True:
                    ev = await events_q.get()
                    if ev is _EV_STOP:
                        return
                    if isinstance(ev, dict) and ev.get("type") == "token":
                        ev = {**ev, "content": sanitize_output(ev.get("content", ""))}
                    await out.put(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n")

            async def _notify():
                try:
                    await stream_task_notify(req.task)
                finally:
                    await out.put(_NOTIFY_DONE)

            ev_task = asyncio.create_task(_ev_pump())
            n_task = asyncio.create_task(_notify())

            while True:
                item = await out.get()
                if item is _NOTIFY_DONE:
                    break
                yield item

            await events_q.put(_EV_STOP)
            await ev_task
            await n_task
            yield "data: {\"type\": \"done\"}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False)}\n\n"
        finally:
            if ev_token is not None:
                reset_event_queue(ev_token)

    return StreamingResponse(_gen(), media_type="text/event-stream")