"""
trace_store.py — 全链路 trace 结构化存储（评估端唯一事实源）。

设计要点：
- 独立 trace_store.db（aiosqlite，仿 agent/supervisor.py 惰性单例模式），
  不与 checkpoints.db / task_store.db 撞事务；
- 事件先入内存缓冲，后台批量 flush（每 50 条或 1s），record() 同步返回、
  绝不在 agent 热路径 await 写库；flush 失败静默降级（缓冲回灌、上限 5000）；
- 归属（thread_id/turn_seq/source/run_id）由两条来源合并：
    1) eval 模式：evaluation.sink.set_eval_ctx 显式注入（优先）
    2) 线上模式：_thread_map{trace_id→归属}（graph.run 在 set_trace_id 处登记）
- 所有 log_event 经 observability sink 转发落到这里，等价于既有 JSON 行日志的
  结构化双写；评估 runner / API 通过 get_trace/get_thread/query 重建链路。

开关：AGENT_TRACE_ENABLED != "0"（env 默认开）。
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("evaluation.trace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DB = Path(os.getenv("AGENT_TRACE_DB", str(PROJECT_ROOT / "trace_store.db")))

# 缓冲上限：超出丢最旧并告警（采集失败绝不能影响主链路）
_BUFFER_MAX = 5000
_FLUSH_BATCH = 50
_FLUSH_INTERVAL_S = 1.0
_PAYLOAD_MAX_CHARS = 100_000  # 单事件 payload 上限（防大 result 撑爆 row）

# ---- eval 上下文（评测跑批打标）----
eval_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "eval_ctx", default={}
)

_COLUMNS = [
    "trace_id", "thread_id", "turn_seq", "seq", "ts", "event_type", "node",
    "duration_ms", "model", "intent", "error", "tool", "outcome",
    "source", "run_id", "parent_id", "payload",
]
_INSERT_SQL = (
    "INSERT INTO trace_events ("
    + ", ".join(_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" * len(_COLUMNS))
    + ")"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clip(s: Any, limit: int) -> str:
    t = json.dumps(s, ensure_ascii=False, default=str) if not isinstance(s, str) else s
    return t if len(t) <= limit else t[:limit]


class TraceStore:
    """单例事件存储。record() 同步；flush/get_* 均为协程（评测/API 在循环内用）。"""

    def __init__(self, db_path: Path | str = TRACE_DB) -> None:
        self._db_path = str(db_path)
        self._conn: Any = None            # aiosqlite.Connection（惰性建，绑定创建它的 loop）
        self._lock = asyncio.Lock()       # 连接初始化互斥
        self._buffer_lock = threading.Lock()
        self._buffer: list[tuple] = []    # 已组装好排序列的 row（含附加字段进 payload）
        self._seq_map: dict[str, int] = {}     # trace_id -> 已用 seq
        self._thread_map: dict[str, dict] = {}  # trace_id -> {thread_id, turn_seq, source, run_id}
        self._flush_task: asyncio.Task | None = None
        self._background: set[asyncio.Task] = set()
        self._closed = False
        self._enabled = os.getenv("AGENT_TRACE_ENABLED", "1") != "0"
        # Production is LangSmith-first.  The local structured store is kept for
        # evaluation runs and explicit live-debug sessions only.
        self._store_live_events = (
            os.getenv("AGENT_TRACE_STORE_LIVE", "0") not in {"", "0", "false"}
        )

    # ---- 生命周期 ----

    async def _ensure_conn(self) -> Any:
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is None:
                import aiosqlite
                self._conn = await aiosqlite.connect(self._db_path)
                await self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS trace_events ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  trace_id TEXT NOT NULL,"
                    "  thread_id TEXT NOT NULL DEFAULT '',"
                    "  turn_seq INTEGER NOT NULL DEFAULT 1,"
                    "  seq INTEGER NOT NULL,"
                    "  ts TEXT NOT NULL,"
                    "  event_type TEXT NOT NULL,"
                    "  node TEXT NOT NULL DEFAULT '',"
                    "  duration_ms REAL,"
                    "  model TEXT, intent TEXT, error TEXT, tool TEXT,"
                    "  outcome TEXT,"
                    "  source TEXT NOT NULL DEFAULT 'live',"
                    "  run_id TEXT NOT NULL DEFAULT '',"
                    "  parent_id TEXT,"
                    "  payload TEXT NOT NULL DEFAULT '{}',"
                    "  UNIQUE(trace_id, seq)"
                    ")"
                )
                await self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ev_trace ON trace_events(trace_id, seq)"
                )
                await self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ev_thread ON trace_events(thread_id, ts)"
                )
                await self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ev_type_ts ON trace_events(event_type, ts)"
                )
                await self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ev_run ON trace_events(run_id, seq)"
                )
                cur = await self._conn.execute("PRAGMA table_info(trace_events)")
                columns = {str(row[1]) for row in await cur.fetchall()}
                if "outcome" not in columns:
                    await self._conn.execute(
                        "ALTER TABLE trace_events ADD COLUMN outcome TEXT"
                    )
                if "ok" in columns:
                    await self._conn.execute(
                        "UPDATE trace_events SET outcome = CASE"
                        " WHEN ok = 1 THEN 'succeeded'"
                        " WHEN ok = 0 THEN 'failed'"
                        " ELSE outcome END"
                        " WHERE outcome IS NULL"
                    )
                await self._conn.commit()
                await self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_runs ("
                    "  run_id TEXT PRIMARY KEY,"
                    "  dataset_id TEXT NOT NULL,"
                    "  started_at TEXT NOT NULL, finished_at TEXT,"
                    "  status TEXT NOT NULL, query_count INTEGER,"
                    "  metadata TEXT NOT NULL DEFAULT '{}',"
                    "  overall TEXT, dimension TEXT, badcases TEXT,"
                    "  judge TEXT, tool_metrics TEXT, task_metrics TEXT,"
                    "  baseline_delta TEXT, cost_estimate TEXT, notes TEXT"
                    ")"
                )
                await self._conn.commit()
        return self._conn

    def set_thread_map(self, trace_id: str, **meta) -> None:
        """登记 trace_id → 归属。graph.run 在 set_trace_id 处调用（线上路径）。"""
        self._thread_map[trace_id] = {
            "thread_id": meta.get("thread_id", ""),
            "turn_seq": meta.get("turn_seq", 1),
            "source": meta.get("source", "live"),
            "run_id": meta.get("run_id", ""),
        }

    def clear_thread_map(self, trace_id: str) -> None:
        self._thread_map.pop(trace_id, None)

    # ---- 采集（同步，热路径安全）----

    def record(self, ev: dict, ts: str | None = None) -> None:
        """接收一个已带 event_type/ts 的事件 dict，补全归属与 seq 后入缓冲。

        绝不 await；返回前不触碰数据库。event_type 缺失的行直接丢弃。
        """
        if not self._enabled or self._closed:
            return
        event_type = ev.get("event_type") or ""
        if not event_type:
            return

        trace_id = ev.get("trace_id") or ""
        if not trace_id:
            from agent.observability import get_trace_id
            trace_id = get_trace_id()
        if not trace_id or trace_id == "-":
            return

        ctx = eval_ctx.get()
        meta = self._thread_map.get(trace_id) or {}
        if not ctx and not meta and not self._store_live_events:
            return
        thread_id = ctx.get("thread_id") or meta.get("thread_id") or ""
        turn_seq = int(ctx.get("turn_seq") or meta.get("turn_seq") or 1)

        payload: dict = ev.get("payload") or {}
        # 附加字段（事件特有子字段）收进 payload JSON
        extra_keys = set(ev) - {
            "trace_id", "ts", "event_type", "node", "duration_ms", "model",
            "intent", "error", "tool", "outcome", "source", "run_id",
            "parent_id", "payload",
        }
        for k in extra_keys:
            payload.setdefault(k, ev[k])

        seq = self._seq_map.get(trace_id, 0) + 1
        self._seq_map[trace_id] = seq
        outcome = _event_outcome(ev, payload)
        row = (
            trace_id,
            thread_id,
            turn_seq,
            seq,
            ev.get("ts") or ts or _now_iso(),
            event_type,
            ev.get("node") or "",
            ev.get("duration_ms"),
            ev.get("model"),
            ev.get("intent"),
            ev.get("error"),
            ev.get("tool"),
            outcome,
            ev.get("source") or ctx.get("source") or meta.get("source") or "live",
            ev.get("run_id") or ctx.get("run_id") or meta.get("run_id") or "",
            ev.get("parent_id"),
            _clip(payload, _PAYLOAD_MAX_CHARS),
        )
        with self._buffer_lock:
            dropped = 0
            if len(self._buffer) >= _BUFFER_MAX:
                del self._buffer[: len(self._buffer) - _BUFFER_MAX]
                dropped = 1
            self._buffer.append(row)
        if dropped:
            logger.warning("trace buffer overflow: oldest events dropped")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # 无事件循环（同步 CLI）→ 等下次 flush；上限内不丢
        if self._flush_task is None:
            self._flush_task = loop.create_task(self._flush_loop())
            self._background.add(self._flush_task)
            self._flush_task.add_done_callback(self._background.discard)

    # ---- 落盘 ----

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_FLUSH_INTERVAL_S)
                await self._flush()
        except asyncio.CancelledError:
            await self._flush()
            raise

    async def _flush(self, force: bool = False) -> None:
        if self._closed or self._buffer is None:
            return
        with self._buffer_lock:
            if not self._buffer and not force:
                return
            rows, self._buffer = self._buffer, []
        if not rows:
            return
        try:
            conn = await self._ensure_conn()
            await conn.executemany(_INSERT_SQL, rows)
            await conn.commit()
        except Exception as exc:  # noqa: BLE001 — flush 失败必须静默降级
            logger.warning("trace flush failed: %s (buffering %d rows)", exc, len(rows))
            with self._buffer_lock:
                self._buffer = rows + self._buffer
                if len(self._buffer) > _BUFFER_MAX:
                    del self._buffer[: len(self._buffer) - _BUFFER_MAX]

    async def flush(self) -> None:
        """评测 runner 落报告前强制刷盘（不等待后台无限循环任务）。"""
        await self._flush(force=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self._flush(force=True)
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    # ---- 查询（评测端）----

    async def get_trace(self, trace_id: str) -> list[dict]:
        """按 seq 顺序返回整个 trace 事件链。"""
        conn = await self._ensure_conn()
        cur = await conn.execute(
            "SELECT trace_id, thread_id, turn_seq, seq, ts, event_type, node,"
            " duration_ms, model, intent, error, tool, outcome, source, run_id,"
            " parent_id, payload FROM trace_events"
            " WHERE trace_id=? ORDER BY seq",
            (trace_id,),
        )
        return [_row_to_dict(r) for r in await cur.fetchall()]

    async def get_thread(self, thread_id: str, turn_seq: int | None = None) -> list[dict]:
        """某会话（thread_id）一个 turn（缺省=最新）的完整事件链。"""
        conn = await self._ensure_conn()
        if turn_seq is None:
            cur = await conn.execute(
                "SELECT COALESCE(MAX(turn_seq), 1) FROM trace_events WHERE thread_id=?",
                (thread_id,),
            )
            row = await cur.fetchone()
            turn_seq = int(row[0]) if row and row[0] else 1
        cur = await conn.execute(
            "SELECT trace_id, thread_id, turn_seq, seq, ts, event_type, node,"
            " duration_ms, model, intent, error, tool, outcome, source, run_id,"
            " parent_id, payload FROM trace_events"
            " WHERE thread_id=? AND turn_seq=? ORDER BY seq",
            (thread_id, turn_seq),
        )
        return [_row_to_dict(r) for r in await cur.fetchall()]

    async def list_turns(self, thread_id: str, limit: int = 20) -> list[dict]:
        """会话所有 turn 摘要（trace_id/turn_seq/首末事件时间/节点数）。"""
        conn = await self._ensure_conn()
        cur = await conn.execute(
            "SELECT trace_id, turn_seq, MIN(ts), COUNT(*) FROM trace_events"
            " WHERE thread_id=? GROUP BY trace_id, turn_seq ORDER BY turn_seq DESC LIMIT ?",
            (thread_id, limit),
        )
        return [
            {"trace_id": r[0], "turn_seq": r[1], "started_at": r[2], "event_count": r[3]}
            for r in await cur.fetchall()
        ]

    async def query(
        self,
        *,
        thread_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
        tool: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """组合过滤查询（全为可选）。"""
        where, params = [], []
        if thread_id:
            where.append("thread_id = ?")
            params.append(thread_id)
        if trace_id:
            where.append("trace_id = ?")
            params.append(trace_id)
        if run_id:
            where.append("run_id = ?")
            params.append(run_id)
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        if source:
            where.append("source = ?")
            params.append(source)
        if tool:
            where.append("tool = ?")
            params.append(tool)
        sql = (
            "SELECT trace_id, thread_id, turn_seq, seq, ts, event_type, node,"
            " duration_ms, model, intent, error, tool, outcome, source, run_id,"
            " parent_id, payload FROM trace_events"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        conn = await self._ensure_conn()
        cur = await conn.execute(sql, params)
        return [_row_to_dict(r) for r in await cur.fetchall()]

    async def count(self) -> int:
        conn = await self._ensure_conn()
        cur = await conn.execute("SELECT COUNT(*) FROM trace_events")
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def prune(self, *, older_than_days: int = 30, run_id: str | None = None) -> int:
        """清理逻辑：older_than_days 前的 live 事件删除；eval 事件由 run_id 显式删。"""
        conn = await self._ensure_conn()
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        cur = await conn.execute(
            "DELETE FROM trace_events WHERE source='live' AND ts < ?",
            (cutoff_iso,),
        )
        deleted = cur.rowcount
        if run_id:
            cur = await conn.execute(
                "DELETE FROM trace_events WHERE run_id = ?", (run_id,)
            )
            deleted += cur.rowcount
        await conn.commit()
        return deleted


# ---- 单例 ----

_INSTANCES: dict[str, TraceStore] = {}
_atexit_registered = False


def _close_store_best_effort(st: "TraceStore") -> None:
    """atexit：进程退出时关闭连接（否则 aiosqlite 的 worker 线程非 daemon，
    一次性 CLI（list/prune/qrels 等）会阻塞进程退出）。"""
    try:
        if st._conn is not None and not st._closed:
            try:
                asyncio.get_running_loop()
                return  # 事件循环仍活跃（服务端）——交给服务生命周期关闭
            except RuntimeError:
                pass
            asyncio.run(st.close())
    except Exception:  # noqa: BLE001
        pass


def get_trace_store(db_path: Path | str | None = None) -> TraceStore:
    global _atexit_registered
    key = str(db_path) if db_path else "$default"
    st = _INSTANCES.get(key)
    if st is None:
        st = TraceStore(db_path) if db_path else TraceStore()
        _INSTANCES[key] = st
    if not _atexit_registered:
        import atexit
        _atexit_registered = True
        atexit.register(_close_store_best_effort, st)
    return st


def _row_to_dict(row) -> dict:
    """将查询行映射为事件 dict；payload 尽量反序列化为 dict。"""
    keys = [
        "trace_id", "thread_id", "turn_seq", "seq", "ts", "event_type", "node",
        "duration_ms", "model", "intent", "error", "tool", "outcome", "source",
        "run_id", "parent_id", "payload",
    ]
    d = dict(zip(keys, row))
    try:
        d["payload"] = json.loads(d["payload"] or "{}")
    except (ValueError, TypeError):
        pass
    return d


def _event_outcome(ev: dict, payload: dict) -> str | None:
    """Resolve the public outcome from the event or structured operation."""
    if ev.get("outcome"):
        return str(ev["outcome"])
    operation = payload.get("operation")
    if isinstance(operation, dict) and operation.get("outcome"):
        return str(operation["outcome"])
    parsed = payload.get("parsed")
    if isinstance(parsed, dict) and parsed.get("outcome"):
        return str(parsed["outcome"])
    return None


def new_run_id() -> str:
    return f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
