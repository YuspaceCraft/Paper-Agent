"""
routers — shared infrastructure for API routers.

Redis connection + task state management used by pdf.py and index.py.
Redis down → task state falls back to in-memory dict (survives worker restart
but not server restart). In-memory only, full persistence requires Redis.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from urllib.parse import urlparse as _urlparse, unquote as _unquote

import redis as _redis_lib

_REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
_TASK_TTL = 3600  # auto-expire completed tasks after 1 hour

# ---- Redis connection (lazy, shared) ----

_redis: "_redis_lib.Redis | None" = None
_redis_ok: bool | None = None  # tri-state: None=untested, True=alive, False=dead


def _get_redis() -> "_redis_lib.Redis | None":
    """Lazy Redis connection, RESP2 for old Redis compatibility."""
    global _redis, _redis_ok
    if _redis_ok is False:
        return None
    if _redis is not None:
        try:
            _redis.ping()
            _redis_ok = True
            return _redis
        except Exception:
            _redis_ok = False
            _redis = None
            return None

    try:
        parsed = _urlparse(_REDIS_URL)
        _redis = _redis_lib.Redis(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 6379,
            protocol=2,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        if parsed.password:
            _redis.execute_command("AUTH", _unquote(parsed.password))
        _redis.ping()
        _redis_ok = True
        return _redis
    except Exception:
        _redis_ok = False
        _redis = None
        return None


# ---- Task state (Redis primary, in-memory fallback) ----

# ponytail: fallback dict for when Redis is down — survives within one worker
_FALLBACK: dict[str, dict] = {}


def _task_create(task_id: str, **fields: str | None) -> None:
    """Create a new task entry."""
    now = str(time.time())
    data = {"task_id": task_id, "created_at": now, "updated_at": now,
            "status": "pending", "progress": "", "error": "", "result": ""}
    data.update({k: v or "" for k, v in fields.items()})

    r = _get_redis()
    if r:
        key = f"task:{task_id}"
        r.hset(key, mapping=data)
        r.expire(key, _TASK_TTL)
        r.zadd("task:list", {task_id: time.time()})
    else:
        _FALLBACK[task_id] = data

    publish_task_update(task_id)


def _task_update(task_id: str, **fields: str | None) -> None:
    """Partial update task fields."""
    now = str(time.time())
    r = _get_redis()
    if r:
        key = f"task:{task_id}"
        mapping = {k: (v or "") for k, v in fields.items()}
        mapping["updated_at"] = now
        r.hset(key, mapping=mapping)
    elif task_id in _FALLBACK:
        _FALLBACK[task_id].update({k: v or "" for k, v in fields.items()})
        _FALLBACK[task_id]["updated_at"] = now

    publish_task_update(task_id)


def _task_get(task_id: str) -> dict | None:
    """Get full task state."""
    r = _get_redis()
    if r:
        data = r.hgetall(f"task:{task_id}")
        if data:
            # parse result JSON if stored as string; empty string = None
            raw = data.get("result", "")
            if raw and isinstance(raw, str):
                try:
                    data["result"] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    data["result"] = None
            elif not raw:
                data["result"] = None
            return data
        return None
    data = _FALLBACK.get(task_id)
    if data and not data.get("result"):
        data["result"] = None
    return data


# ponytail: zset members never expire when the hash does, so _task_list prunes
# stale ids as it scans (bounded by limit — a long-dormant queue is cleaned on
# demand rather than on write). Multi-worker: zset is shared, pruning is a
# just-in-time sweep and safe to run from any worker.
def _task_list(limit: int = 30) -> list[dict]:
    """List recent top-level tasks, newest first.

    Internal tasks marked with a `parent` field (created by the composite
    ingest orchestrator, see routers/background.py) are hidden — the frontend's
    task center shows one entry per top-level task (e.g. one "ingest" task per
    paper) instead of its parse/index sub-tasks.
    """
    out: list[dict] = []
    r = _get_redis()
    if r:
        ids = r.zrevrange("task:list", 0, -1)
        for tid in ids:
            if len(out) >= limit:
                break
            task = _task_get(tid)
            if task is None:
                r.zrem("task:list", tid)  # expired hash → prune stale member
                continue
            if task.get("parent"):
                continue
            out.append(task)
        return out
    # In-memory fallback: newest first (single worker)
    items = sorted(
        (d for d in _FALLBACK.values() if not d.get("parent")),
        key=lambda d: float(d.get("created_at", 0) or 0),
        reverse=True,
    )
    return items[:limit]


# ---- live task event bus (SSE push) ----
# _task_create/_task_update 之后把该任务的快照广播给所有连到
# /api/agent/tasks/stream 的 SSE 客户端。publish 会被后台线程（_run_pipeline /
# _run_indexing / _run_ingest）和事件循环协程两处调用：跨线程投递统一经由
# loop.call_soon_threadsafe 落到事件循环线程，避免裸操作 asyncio.Queue 的线程
# 安全问题。单 worker 假设（uvicorn 默认；Electron 后端单进程）。

_TASK_LISTENERS: set["asyncio.Queue"] = set()
_TASK_LISTEN_LOCK = threading.Lock()
_TASK_LOOP: "asyncio.AbstractEventLoop | None" = None
_TASK_EVENT_LOCK = threading.Lock()  # serialize snapshot reads from bg threads


def _task_public(t: dict) -> dict:
    """对外序列化：notify str→bool，隐藏内部 parent，result 解 JSON。"""
    out = dict(t)
    out["notify"] = out.get("notify") in ("1", "true", "True")
    out.setdefault("stage", "")
    out.pop("parent", None)
    result = out.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            result = None
    out["result"] = result if result else None
    return out


def _subscribe_task_stream(q) -> None:
    with _TASK_LISTEN_LOCK:
        _TASK_LISTENERS.add(q)


def _unsubscribe_task_stream(q) -> None:
    with _TASK_LISTEN_LOCK:
        _TASK_LISTENERS.discard(q)


def set_task_loop(loop) -> None:
    """记录主事件循环（首个 SSE 连接建立时）——后台线程经它投递事件。"""
    global _TASK_LOOP
    _TASK_LOOP = loop


def publish_task_update(task_id: str) -> None:
    """广播单个任务当前状态到所有活跃 SSE 客户端。

    无活跃客户端时快速 no-op。后台线程调用时 _TASK_LOOP 必须在场（首个 SSE
    连接已注册）；线程间以 call_soon_threadsafe 执行 queue.put_nowait。
    """
    if not _TASK_LISTENERS:
        return
    loop = _TASK_LOOP
    if loop is None or loop.is_closed():
        return
    task = _task_get(task_id)
    if task is None:
        return
    event = json.dumps(
        {"type": "task_update", "task": _task_public(task)}, ensure_ascii=False
    )
    for q in list(_TASK_LISTENERS):
        try:
            loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception:
            continue
