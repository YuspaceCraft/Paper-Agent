"""
live.py — 评测过程的实时进度总线（SSE 数据源，评估端）。

一条评测跑批是「逐条 QA、逐条出指标」的长任务（几十条 × 分钟级）；只看跑完的
run_summary.json，意味着第 3 条就坏掉的检索要等到最后才知道。本模块把跑批过程
变成可订阅的事件流：

    runner.run_eval() --publish--> RunFeed --subscribe--> /api/eval/runs/{id}/stream

事件类型（event["type"]）：
    run_started    {run_id, dataset_id, total, started_at}
    query_started  {index, total, qid, query}
    query_finished {index, total, qid, category, success, mrr, recall@5, ndcg@10,
                    tool_errors, duration_s, tokens, cost_usd, trace_id, run_error}
    aggregate      {done, total, elapsed_s, eta_s, overall{...}, recent[...]}
    judge_started / judge_finished {sample_size, context_hit, ...}
    run_finished   {run_id, status, duration_s, query_count, overall, gate}
    run_failed     {run_id, error}
    heartbeat      {run_id, ts}                    ← 无进展时的保活帧

约定（与平台设计 §7「采集绝不上热路径阻塞」一致）：
- publish 永不向调用方抛异常；缓冲与订阅者都在内存（单 worker 假设，uvicorn 默认）。
- 跨线程安全：publish 可能来自后台线程 → 已登记事件循环时经 call_soon_threadsafe 投递。
- 慢订阅者丢最旧事件（队列上限），绝不反压跑批。
- 内存总线只覆盖本进程发起的 run；CLI / 别的 worker 的 run 由 stream_run_events
  的 fallback（读 eval_runs 行）轮询补齐，两条路径产出同构事件。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, AsyncIterator, Awaitable, Callable

TERMINAL_TYPES = {"run_finished", "run_failed"}

_MAX_BUFFER = 400    # 每个 run 保留的事件上限（断线重连回放用）
_QUEUE_MAX = 300     # 单订阅者队列上限（慢消费者丢最旧）
_MAX_FEEDS = 64      # 进程内保留的 run 数上限（先淘汰已结束的最旧 feed）


class RunFeed:
    """单次跑批的进度缓冲 + 订阅者集合。"""

    __slots__ = ("run_id", "events", "subscribers", "loop", "state", "done",
                 "publishing", "updated_at")

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.state: dict[str, Any] = {"run_id": run_id, "status": "running"}
        self.done = False
        self.publishing = False   # 本进程是否有该 run 的发布者
        self.updated_at = time.time()


_FEEDS: dict[str, RunFeed] = {}
_LOCK = threading.Lock()


# ---- feed 管理 ----

def _feed(run_id: str) -> RunFeed:
    with _LOCK:
        feed = _FEEDS.get(run_id)
        if feed is None:
            if len(_FEEDS) >= _MAX_FEEDS:
                _evict_locked()
            feed = RunFeed(run_id)
            _FEEDS[run_id] = feed
        return feed


def _evict_locked() -> None:
    """淘汰已结束且最久未更新的 feed；进行中的 run 永不淘汰。"""
    finished = [f for f in _FEEDS.values() if f.done]
    if not finished:
        return
    for f in sorted(finished, key=lambda x: x.updated_at)[: max(1, len(finished) // 2)]:
        _FEEDS.pop(f.run_id, None)


def get_feed(run_id: str) -> RunFeed | None:
    """读 feed（不创建）。"""
    return _FEEDS.get(run_id)


def reset() -> None:
    """清空总线（测试用）。"""
    with _LOCK:
        _FEEDS.clear()


# ---- 发布 ----

def publish(run_id: str, event: dict) -> None:
    """广播一条进度事件；任何异常都吞掉——进度绝不能影响跑批。"""
    try:
        if not run_id:
            return
        feed = _feed(run_id)
        feed.publishing = True
        feed.updated_at = time.time()
        feed.events.append(event)
        if len(feed.events) > _MAX_BUFFER:
            del feed.events[: len(feed.events) - _MAX_BUFFER]
        if event.get("type") == "aggregate":
            feed.state = {**feed.state, **event}
        elif event.get("type") in TERMINAL_TYPES:
            feed.done = True
            feed.state = {**feed.state, **event,
                          "status": event.get("status") or event["type"]}
        _dispatch(feed, event)
    except Exception:  # noqa: BLE001 — 进度通道不得影响评测主链路
        pass


def _put_drop_oldest(q: asyncio.Queue, event: dict) -> None:
    while True:
        try:
            q.put_nowait(event)
            return
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                return


def _dispatch(feed: RunFeed, event: dict) -> None:
    with _LOCK:
        queues = list(feed.subscribers)
    loop = feed.loop
    for q in queues:
        try:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(_put_drop_oldest, q, event)
            else:
                _put_drop_oldest(q, event)
        except RuntimeError:
            continue


# ---- 订阅 ----

def subscribe(run_id: str) -> tuple[RunFeed, asyncio.Queue]:
    """订阅一个 run：先回放缓冲（断线重连补齐历史），再收实时事件。"""
    feed = _feed(run_id)
    try:
        feed.loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
    for ev in list(feed.events):
        _put_drop_oldest(q, ev)
    with _LOCK:
        feed.subscribers.add(q)
    return feed, q


def unsubscribe(run_id: str, q: asyncio.Queue) -> None:
    with _LOCK:
        feed = _FEEDS.get(run_id)
        if feed is not None:
            feed.subscribers.discard(q)


def snapshot(run_id: str) -> dict | None:
    """最近一次状态快照（聚合指标）；无记录返回 None。"""
    feed = _FEEDS.get(run_id)
    return dict(feed.state) if feed is not None else None


def is_running(run_id: str) -> bool:
    """本进程是否仍在跑该 run（无 feed = False，交由调用方查库）。"""
    feed = _FEEDS.get(run_id)
    return bool(feed and not feed.done and feed.publishing)


# ---- 事件流（SSE 数据源）----

def row_event(run_id: str, row: dict) -> dict:
    """eval_runs 行 → 与总线同构的事件（非本进程 run 的轮询路径）。

    duration_s 由行内 started_at / finished_at 现算（跑批自己启停计时），
    不再依赖行里有没有这个字段——这正是「完整用时」在轮询路径上的唯一来源。
    """
    status = str(row.get("status") or "")
    metadata = row.get("metadata") or {}
    total = int((metadata or {}).get("total") or row.get("query_count") or 0)
    overall = row.get("overall") or {}
    duration_s = row.get("duration_s")
    if duration_s is None:
        started = _epoch(row.get("started_at"))
        finished = _epoch(row.get("finished_at"))
        if started is not None:
            duration_s = round((finished or time.time()) - started, 1)
    if status in ("running", "pending"):
        return {
            "type": "aggregate", "source": "db", "run_id": run_id,
            "status": status, "done": int(row.get("query_count") or 0),
            "total": total, "overall": overall, "recent": [],
            "started_at": row.get("started_at") or "",
            "elapsed_s": duration_s,
        }
    failed = status != "done"
    return {
        "type": "run_failed" if failed else "run_finished",
        "source": "db", "run_id": run_id, "status": status,
        "query_count": int(row.get("query_count") or 0),
        "overall": overall, "gate": (row.get("baseline_delta") or {}).get("gate"),
        "duration_s": duration_s,
        "finished_at": row.get("finished_at") or "",
        "badcase_count": len(row.get("badcases") or []),
        "error": "; ".join(row.get("notes") or []) if failed else "",
    }


def _epoch(ts) -> float | None:
    """ISO 串 / epoch 秒 → epoch 秒（解析不了返回 None）。"""
    if ts is None or ts == "":
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


async def stream_run_events(
    run_id: str,
    *,
    fallback: Callable[[], Awaitable[dict | None]] | None = None,
    poll_s: float = 2.0,
    keepalive_s: float = 15.0,
) -> AsyncIterator[dict]:
    """把一次跑批的进度变成事件流（到 run_finished / run_failed 结束）。

    - 本进程发起的 run：直接吃内存总线（实时、含逐条明细）。
    - 非本进程（CLI / 别的 worker）：按 poll_s 轮询 fallback 给的 eval_runs 行，
      产出同构的 aggregate / run_finished 事件（指标粒度到「已完成的条数」）。
    - fallback=None：只等实时事件（纯内存消费者，如测试 / 进程内 CLI 打印）。
    """
    feed, q = subscribe(run_id)
    try:
        while True:
            if not feed.publishing and fallback is not None:
                row = await fallback()
                if row is None:
                    yield {"type": "run_failed", "run_id": run_id,
                           "error": "run not found"}
                    return
                event = row_event(run_id, row)
                yield event
                if event["type"] in TERMINAL_TYPES:
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=poll_s)
                except asyncio.TimeoutError:
                    continue
            else:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=keepalive_s)
                except asyncio.TimeoutError:
                    yield {"type": "heartbeat", "run_id": run_id, "ts": time.time()}
                    continue
            yield ev
            if ev.get("type") in TERMINAL_TYPES:
                return
    finally:
        unsubscribe(run_id, q)
