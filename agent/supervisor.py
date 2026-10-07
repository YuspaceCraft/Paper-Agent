"""
supervisor.py — 领导-部门制监督派发器（LangGraph 原生组件实现）。

主 agent（领导）把长期任务以 (task_id, task_content) 派发给隔离子 agent（部门）。
部门 = 现有 subagents.build_subagent 子图 + 独立 LangGraph thread（thread_id=task_id）。

LangGraph 原语 vs 需求映射（避免自研）：

| 需求                       | 组件                                                                 |
|---------------------------|----------------------------------------------------------------------|
| 子 agent 隔离（自有工具/自存消息） | 现有 build_subagent 子图 + 独立 thread                            |
| 状态栈：执行哪个任务、进行到哪步  | AsyncSqliteSaver：worker 以 thread_id=task_id 运行；aget_state 读     |
|                           | values(messages/iteration) + next(待跑节点) + tasks(interrupt 块)     |
| 任务注册/监督元数据           | AsyncSqliteStore namespace ("tasks", task_id) 跨线程共享              |
| 领导干预                    | interrupt() 门禁 + Command(resume=reply, thread_id=task_id) 续跑      |
| 后台并发 / 跨轮次存活         | asyncio.create_task + 每舱独立 checkpoint thread                     |

用法（一般经工具调用，见 providers/task_provider.py）：
    await init_supervisor(checkpointer)          # graph.get_agent 启动时挂共享 checkpointer
    task_id = await dispatch("creator", "写第3章", "…", parent_thread="sess_1")
    card    = await progress(task_id)            # 读状态栈（next / iteration / interrupted）
    out     = await collect(task_id)             # 产出（无 tool_calls 的末条 AIMessage）
    await resume(task_id, "同意，引用 RMNet 的表3")  # 处理 interrupt 暂停
    await cancel(task_id)
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage  # noqa: F401
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from .core.contracts import (
    AgentError,
    ErrorType,
    OperationKind,
    OperationOutcome,
    OperationResult,
    ResultMeta,
)
from .core.result_utils import extract_final_answer, structured_error_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_STORE_DB = PROJECT_ROOT / "task_store.db"

# ---- 模块级单例 ----

_checkpointer: BaseCheckpointSaver | None = None
_store = None  # AsyncSqliteStore——惰性建，事件循环内使用
_store_conn = None  # aiosqlite.Connection（store 生命周期）
_running: dict[str, asyncio.Task] = {}   # task_id -> 在跑的后台任务（供 cancel/孤儿判定）
_graphs: dict[str, object] = {}          # task_id -> CompiledStateGraph（供 resume/get_state）
_graph_cache: dict[tuple[str, bool], object] = {}  # (role, leader_gate) -> CompiledStateGraph
_bg: set[asyncio.Task] = set()
_store_lock: asyncio.Lock | None = None
_recovery_lock: asyncio.Lock | None = None
_transition_lock = asyncio.Lock()
_last_recovery_scan = 0.0

TASK_LEASE_SECONDS = float(os.getenv("AGENT_TASK_LEASE_SECONDS", "60"))
TASK_HEARTBEAT_SECONDS = float(
    os.getenv("AGENT_TASK_HEARTBEAT_SECONDS", "15")
)
RECOVERY_SCAN_SECONDS = float(
    os.getenv("AGENT_TASK_RECOVERY_SCAN_SECONDS", "5")
)
_WORKER_OWNER = (
    f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _task_result(
    *,
    task_id: str,
    role: str,
    outcome: OperationOutcome,
    duration_ms: float,
    data: dict | None = None,
    error: AgentError | None = None,
    effect_applied: str | None = None,
) -> dict:
    result = OperationResult(
        kind=OperationKind.TASK,
        operation_id=task_id,
        outcome=outcome,
        data=data,
        error=error,
        meta=ResultMeta(
            tool_name=role,
            duration_ms=duration_ms,
            attempt=1,
            max_attempts=1,
            effect_applied=effect_applied,
        ),
    )
    return result.model_dump(mode="json")


def _lease_deadline() -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=TASK_LEASE_SECONDS)
    ).isoformat(timespec="seconds")


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _lease_expired(meta: dict) -> bool:
    deadline = _parse_time(meta.get("lease_expires_at", ""))
    if deadline is None:
        return True
    return deadline <= datetime.now(timezone.utc)


def init_supervisor(checkpointer: BaseCheckpointSaver | None) -> None:
    """启动时挂共享 checkpointer（与对话同一 AsyncSqliteSaver，worker 线程落同一
    checkpoints.db）。None 时仅 Store/文件系统能力可用（/api/tasks 列表场景）。"""
    global _checkpointer
    _checkpointer = checkpointer


def reset_worker_graph_cache() -> None:
    """Invalidate idle worker templates after tools/config are reloaded.

    Running task objects stay in ``_graphs`` until they finish; only the
    role-level cache is replaced for future dispatches.
    """
    _graph_cache.clear()


async def shutdown_supervisor() -> None:
    """Cancel background workers and close the supervisor store cleanly."""
    global _store, _store_conn, _store_lock, _checkpointer

    tasks = {t for t in [*_running.values(), *_bg] if not t.done()}
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    _running.clear()
    _graphs.clear()
    _graph_cache.clear()
    _bg.clear()

    conn = _store_conn
    _store = None
    _store_conn = None
    _store_lock = None
    _checkpointer = None
    if conn is not None:
        try:
            await conn.close()
        except Exception:
            pass


async def _get_store():
    """惰性建 AsyncSqliteStore（持 aiosqlite 连接；必须 in-loop）。"""
    global _store, _store_lock, _store_conn
    if _store is None:
        if _store_lock is None:
            _store_lock = asyncio.Lock()
        async with _store_lock:
            if _store is None:
                import aiosqlite
                from langgraph.store.sqlite import AsyncSqliteStore
                # isolation_level=None（autocommit）：AsyncSqliteStore 自身管理
                # BEGIN/COMMIT，默认模式会撞「cannot start a transaction within a
                # transaction」（TROUBLESHOOTING「supervisor / SqliteStore 事务」）。
                _store_conn = await aiosqlite.connect(
                    str(TASK_STORE_DB), isolation_level=None)
                _store = AsyncSqliteStore(_store_conn)
                await _store.setup()
    return _store


# ---- worker 图构建（按 role 缓存） ----

def _worker_graph(role: str, leader_gate: bool):
    """按 (role, leader_gate) 构建/复用已编译子图。子图与 as_tool 同步子 agent 同
    源同构（subagents.SUBAGENTS 表），仅多挂 checkpointer 与可选 gate。"""
    key = (role, leader_gate)
    g = _graph_cache.get(key)
    if g is not None:
        return g
    from .subagents import SUBAGENTS, build_subagent
    from .config import get_limits

    spec = next((s for s in SUBAGENTS if s.name == role), None)
    if spec is None:
        raise ValueError(f"unknown subagent role: {role}")
    limits = get_limits()
    cfg = limits.subagents.get(role) if limits else None
    max_steps = cfg.max_steps if cfg else spec.max_steps

    tools = _tools_for(spec.tools)
    system = spec.system_prompt
    if leader_gate:
        tools = tools + [_request_review_tool()]
        system += (
            "\n\nYou may call request_review(question) to PAUSE and consult the "
            "leader (main agent) when you need approval, missing information, or "
            "a decision the task cannot make alone. After the leader replies, "
            "request_review returns their answer — continue from there. Never use "
            "it for routine sub-steps."
        )
    graph, init_state = build_subagent(
        spec.name, system, tools, max_steps=max_steps,
        checkpointer=_checkpointer, leader_gate=leader_gate,
    )
    _graph_cache[key] = (graph, init_state)
    return _graph_cache[key]


def _tools_for(names: list[str]):
    from .tools import get_base_tools
    base = {t.name: t for t in get_base_tools()}
    return [base[n] for n in names if n in base]


def _request_review_tool():
    from langchain_core.tools import StructuredTool
    return StructuredTool.from_function(request_review)


# request_review 工具本体（也可在 subagents.py 引入，此处内联保持自包含）
async def request_review(question: str) -> str:
    """Pause this task and ask the leader (main agent) for input or a decision."""
    return "PENDING_LEADER_REVIEW"


# ---- store 元数据读写 ----

_META_KEYS = (
    "task_id", "role", "title", "input_preview", "parent_thread",
    "status", "created_at", "updated_at", "error", "output_preview",
    "leader_gate", "model", "attempt", "worker_id", "lease_owner",
    "heartbeat_at", "lease_expires_at", "recoverable",
    "result",
)


async def _meta_put(task_id: str, **fields) -> None:
    """写/合并任务元数据（output 全文保留，其余键截断）。None 值显式清空/忽略。"""
    store = await _get_store()
    meta = await _meta_get(task_id) or {}
    if "task_id" not in fields and "task_id" not in meta:
        fields["task_id"] = task_id
    for k, v in fields.items():
        if v is None:
            meta.pop(k, None)
        else:
            meta[k] = v
    meta["updated_at"] = _now()
    await store.aput(("tasks", task_id), "meta", _pick(meta))


def _pick(m: dict) -> dict:
    """store 落盘视图：基础键截断 + 完整 output（collect 用）。"""
    keep = {k: m.get(k) for k in _META_KEYS}
    keep["task_id"] = str(keep.get("task_id") or "")
    keep["role"] = str(keep.get("role") or "")
    keep["title"] = str(keep.get("title") or "")
    keep["input_preview"] = str(keep.get("input_preview") or "")[:500]
    keep["parent_thread"] = str(keep.get("parent_thread") or "")
    keep["status"] = str(keep.get("status") or "unknown")
    keep["created_at"] = str(keep.get("created_at") or "")
    keep["updated_at"] = str(keep.get("updated_at") or "")
    keep["error"] = str(keep.get("error") or "")[:500]
    keep["output_preview"] = str(keep.get("output_preview") or "")[:800]
    keep["leader_gate"] = bool(keep.get("leader_gate"))
    keep["model"] = str(keep.get("model") or "")
    try:
        keep["attempt"] = int(keep.get("attempt") or 0)
    except (TypeError, ValueError):
        keep["attempt"] = 0
    for key in (
        "worker_id", "lease_owner", "heartbeat_at", "lease_expires_at",
    ):
        keep[key] = str(keep.get(key) or "")
    keep["recoverable"] = bool(keep.get("recoverable"))
    keep["result"] = (
        keep.get("result") if isinstance(keep.get("result"), dict) else None
    )
    out = m.get("output")
    keep["output"] = (out or "")[:200_000]
    return keep


async def _meta_get(task_id: str) -> dict | None:
    store = await _get_store()
    try:
        item = await store.aget(("tasks", task_id), "meta")
    except Exception:
        return None
    if item is None:
        return None
    return dict(item.value) if isinstance(item.value, dict) else {"value": item.value}


async def _meta_scan() -> list[dict]:
    """namespace tasks 下全部条目，最新优先。"""
    store = await _get_store()
    try:
        items = await store.asearch(("tasks",), limit=200)
    except Exception:
        return []
    out = []
    for it in items:
        v = it.value
        if isinstance(v, dict):
            out.append(dict(v))
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


async def recover_expired_leases(*, force: bool = False) -> list[str]:
    """Mark dead-worker tasks orphaned once per process.

    A live local task or an unexpired lease is left untouched. Interrupted
    tasks never enter this path because they are intentionally not running.
    """
    global _recovery_lock, _last_recovery_scan
    now = time.monotonic()
    if (
        not force
        and _last_recovery_scan
        and now - _last_recovery_scan < max(0.0, RECOVERY_SCAN_SECONDS)
    ):
        return []
    if _recovery_lock is None:
        _recovery_lock = asyncio.Lock()
    recovered: list[str] = []
    async with _recovery_lock:
        for meta in await _meta_scan():
            task_id = str(meta.get("task_id") or "")
            stored = str(meta.get("status") or "")
            if not task_id or stored not in ("pending", "running"):
                continue
            local = _running.get(task_id)
            if local is not None and not local.done():
                continue
            if not _lease_expired(meta):
                continue
            await _meta_put(
                task_id,
                status="orphaned",
                recoverable=True,
                worker_id=None,
                lease_owner=None,
                heartbeat_at=None,
                lease_expires_at=None,
                error="worker lease expired; task can be recovered",
            )
            recovered.append(task_id)
        _last_recovery_scan = time.monotonic()
    return recovered


# ---- 后台 runner ----

async def _heartbeat_loop(task_id: str, owner: str) -> None:
    """Renew the persisted lease while this worker remains the owner."""
    while True:
        await asyncio.sleep(max(1.0, TASK_HEARTBEAT_SECONDS))
        meta = await _meta_get(task_id)
        if (
            not meta
            or meta.get("status") != "running"
            or meta.get("lease_owner") != owner
        ):
            return
        await _meta_put(
            task_id,
            heartbeat_at=_now(),
            lease_expires_at=_lease_deadline(),
        )


async def _run_worker(
    task_id: str, run_input, config: dict, graph,
    lease_owner: str | None = None,
) -> None:
    """后台执行 worker 图。隔离事件流 + 状态推进 + 产出落 store。

    run_input: init_state dict（首次）或 Command(resume=reply)（interrupt 续跑）。
    结束后若 checkpoint 处于 interrupted（有 interrupt 块）→ 状态 interrupted，
    保留 graph 供 resume；否则 done/failed，写产出，清 graph 缓存。
    """
    from .stream import set_event_queue, reset_event_queue, set_scope, reset_scope
    from .observability import log_event

    q_tok = set_event_queue(None)   # 后台子图事件不泄漏进任何 SSE 回合
    s_tok = set_scope("compartment", task_id)
    ctx_tok = None
    started = time.time()
    owner = lease_owner or _WORKER_OWNER
    heartbeat: asyncio.Task | None = None
    try:
        from .core.execution_context import (
            build_execution_context,
            reset_current_execution_context,
            set_current_execution_context,
        )

        worker_ctx = build_execution_context(
            thread_id=task_id,
            request_id=task_id,
            execution_id=f"task:{task_id}",
            roles={"user"},
        )
        worker_ctx.bind_runnable_config(config)
        ctx_tok = set_current_execution_context(worker_ctx)
        meta = await _meta_get(task_id) or {}
        role = str(meta.get("role") or "")
        attempt = int(meta.get("attempt") or 0) + 1
        await _meta_put(
            task_id,
            status="running",
            attempt=attempt,
            worker_id=owner,
            lease_owner=owner,
            heartbeat_at=_now(),
            lease_expires_at=_lease_deadline(),
            recoverable=False,
            error=None,
        )
        heartbeat = asyncio.create_task(_heartbeat_loop(task_id, owner))
        _bg.add(heartbeat)
        heartbeat.add_done_callback(_bg.discard)
        log_event("subagent_start", node="supervisor", task_id=task_id)
        try:
            # 子 agent 舱也是独立 LangSmith 根 run：每次 ainvoke 新 run_id
            #（不污染共享 config，resume 复用同 config 时 lineage 不串）。
            result = await graph.ainvoke(
                run_input, {**config, "run_id": uuid.uuid4()})
        except asyncio.CancelledError:
            error = AgentError(
                error_type=ErrorType.TASK,
                code="TASK_CANCELLED",
                message="Task execution was cancelled.",
                user_message="任务已取消。",
                retryable=False,
                recovery_action="Resume or dispatch a new task if needed.",
                tool_name=role,
                effect_applied="unknown",
            )
            result_payload = _task_result(
                task_id=task_id,
                role=role,
                outcome=OperationOutcome.CANCELLED,
                duration_ms=round((time.time() - started) * 1000, 1),
                error=error,
                effect_applied="unknown",
            )
            await _meta_put(
                task_id, status="cancelled", worker_id=None,
                lease_owner=None, heartbeat_at=None, lease_expires_at=None,
                error=error.user_message, result=result_payload,
            )
            log_event("subagent_result", node="supervisor", task_id=task_id,
                      status="cancelled",
                      duration_ms=round((time.time() - started) * 1000, 1))
            raise
        except Exception as exc:
            error = AgentError(
                error_type=ErrorType.TASK,
                code="TASK_EXECUTION_FAILED",
                message=f"{type(exc).__name__} while running task.",
                user_message="子任务执行失败。",
                retryable=False,
                recovery_action="Inspect the task trace and retry explicitly.",
                tool_name=role,
                effect_applied="unknown",
            )
            result_payload = _task_result(
                task_id=task_id,
                role=role,
                outcome=OperationOutcome.FAILED,
                duration_ms=round((time.time() - started) * 1000, 1),
                error=error,
                effect_applied="unknown",
            )
            await _meta_put(
                task_id, status="failed",
                error=error.user_message, result=result_payload,
                output_preview="", worker_id=None, lease_owner=None,
                heartbeat_at=None, lease_expires_at=None,
            )
            log_event("subagent_result", node="supervisor", task_id=task_id,
                      status="failed", error=error.code,
                      duration_ms=round((time.time() - started) * 1000, 1))
            _finish(task_id)
            return

        # 正常返回：可能是 done，也可能是 interrupt 暂停（Leader Gate）
        # aget_state 防失败：无 checkpointer 的图（测试/极端）仅回退 ainvoke 结果。
        snap = None
        interrupted = False
        try:
            snap = await graph.aget_state(config)
            interrupted = bool(getattr(snap, "tasks", None))
        except Exception:
            pass
        if interrupted:
            result_payload = _task_result(
                task_id=task_id,
                role=role,
                outcome=OperationOutcome.INTERRUPTED,
                duration_ms=round((time.time() - started) * 1000, 1),
                data={"question": _pending_question(snap)},
            )
            await _meta_put(
                task_id, status="interrupted", recoverable=True,
                worker_id=None, lease_owner=None, heartbeat_at=None,
                lease_expires_at=None, result=result_payload,
            )
            log_event("subagent_result", node="supervisor", task_id=task_id,
                      status="interrupted",
                      duration_ms=round((time.time() - started) * 1000, 1))
            # 保留 graph 缓存，供 resume(); _running 移除（不再自动跑）
            _running.pop(task_id, None)
            _graphs[task_id] = graph
            return

        if isinstance(result, dict):
            output = _extract_output(result)
        else:
            output = _extract_output(getattr(snap, "values", {}) if snap else {})
        answer_error = structured_error_payload(output) if output else None
        if output and answer_error is None:
            result_payload = _task_result(
                task_id=task_id,
                role=role,
                outcome=OperationOutcome.SUCCEEDED,
                duration_ms=round((time.time() - started) * 1000, 1),
                data={"output": output},
                effect_applied="yes",
            )
            status = "done"
            error_text = None
        else:
            code = (
                str(answer_error.get("error") or "SUBAGENT_EMPTY_OUTPUT")
                if answer_error else "SUBAGENT_EMPTY_OUTPUT"
            )
            detail = (
                str(answer_error.get("detail") or answer_error.get("error") or "")
                if answer_error else ""
            )
            error = AgentError(
                error_type=ErrorType.SUBAGENT,
                code=code,
                message=detail or "Subagent produced no final output.",
                user_message=(
                    f"子任务执行失败：{detail}"
                    if detail else "子任务没有生成最终结果。"
                ),
                retryable=False,
                recovery_action="Retry with a more explicit task.",
                tool_name=role,
            )
            result_payload = _task_result(
                task_id=task_id,
                role=role,
                outcome=OperationOutcome.FAILED,
                duration_ms=round((time.time() - started) * 1000, 1),
                error=error,
            )
            status = "failed"
            error_text = error.user_message
        await _meta_put(
            task_id, status=status, output=output, error=error_text,
            result=result_payload,
            worker_id=None, lease_owner=None, heartbeat_at=None,
            lease_expires_at=None, recoverable=False,
        )
        log_event("subagent_result", node="supervisor", task_id=task_id,
                  status=status, output_preview=output[:800],
                  duration_ms=round((time.time() - started) * 1000, 1))
        _graphs.pop(task_id, None)
        _finish(task_id)
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        if ctx_tok is not None:
            reset_current_execution_context(ctx_tok)
        reset_event_queue(q_tok)
        reset_scope(s_tok)


def _extract_output(state: dict) -> str:
    return extract_final_answer(state.get("messages") or [])


def _pending_question(snapshot) -> Any:
    tasks = getattr(snapshot, "tasks", None) or ()
    if not tasks:
        return None
    interrupts = getattr(tasks[0], "interrupts", None) or ()
    return getattr(interrupts[0], "value", None) if interrupts else None


def _finish(task_id: str) -> None:
    _running.pop(task_id, None)
    _graphs.pop(task_id, None)


# ---- 监督 API（供工具层与 task_node 调用） ----

async def dispatch(role: str, title: str, task: str,
                   parent_thread: str = "", model: str | None = None,
                   leader_gate: bool = False) -> str:
    """派发一个隔离子 agent 任务，立即返回 task_id（后台运行）。"""
    graph, init_state = _worker_graph(role, leader_gate)
    task_id = uuid.uuid4().hex[:10]
    config = {"configurable": {"thread_id": task_id}}
    if model:
        config["configurable"]["model"] = model
    run_input = dict(init_state)
    run_input["messages"] = [HumanMessage(content=task or title or "")]
    await _meta_put(
        task_id, role=role, title=title or role,
        input_preview=task or "", parent_thread=parent_thread,
        status="pending", error=None, output_preview="",
        created_at=_now(), model=model or "", leader_gate=leader_gate,
        attempt=0, worker_id=_WORKER_OWNER, lease_owner=_WORKER_OWNER,
        heartbeat_at=_now(), lease_expires_at=_lease_deadline(),
        recoverable=True,
    )
    from .observability import log_event
    from .subagents import SUBAGENTS

    spec = next((item for item in SUBAGENTS if item.name == role), None)

    log_event("subagent_dispatch", node="supervisor", task_id=task_id,
              role=role, title=title or role, parent_thread=parent_thread,
              task_preview=(task or "")[:800], model=model or "",
              tools=list(spec.tools) if spec else [],
              prompt_preview=(spec.system_prompt[:800] if spec else ""))

    _graphs[task_id] = graph  # 运行期即可读状态栈（_finish/interrupted 才释放）
    bg = asyncio.create_task(_run_worker(
        task_id, run_input, config, graph, lease_owner=_WORKER_OWNER,
    ))
    _running[task_id] = bg
    _bg.add(bg)
    bg.add_done_callback(_bg.discard)
    return task_id


async def progress(task_id: str) -> dict:
    """读单个任务的状态栈：store 状态 + checkpoint（next/iteration/messages/
    interrupted 问题）。任务不存在 → 抛 ValueError。"""
    await recover_expired_leases()
    meta = await _meta_get(task_id)
    if not meta:
        raise ValueError(f"task '{task_id}' not found")
    card = dict(meta)
    card["status"] = _effective_status(task_id, meta.get("status", "unknown"), meta)
    # 栈: running 期间每 super-step 落一个 checkpoint → next 节点即「进行到哪」
    graph = _graphs.get(task_id)
    if graph is None:
        graph = await _graph_for_meta(meta)
    if graph is not None and card["status"] in (
        "running", "interrupted", "done", "orphaned",
    ):
        try:
            snap = await graph.aget_state(
                {"configurable": {"thread_id": task_id}})
            card["next"] = list(getattr(snap, "next", [])) or []
            values = getattr(snap, "values", {}) or {}
            card["message_count"] = len(values.get("messages") or [])
            card["iteration"] = values.get("iteration", 0)
            tasks = getattr(snap, "tasks", []) or []
            if tasks:
                intr = getattr(tasks[0], "interrupts", None) or ()
                if intr:
                    card["interrupted"] = True
                    card["question"] = getattr(intr[0], "value", None)
        except Exception:
            pass
    return card


async def _graph_for_meta(meta: dict) -> object | None:
    """从 store 元数据重建 worker 图引用（进程重启后 resume 用）——重编译一次。
    性能可接受（单任务低频）；避免长期持有图导致跨重启存根残留。"""
    role = meta.get("role", "")
    if not role:
        return None
    try:
        graph, _ = _worker_graph(role, bool(meta.get("leader_gate")))
    except Exception:
        return None
    _graphs[meta["task_id"]] = graph
    return graph


def _effective_status(task_id: str, stored: str, meta: dict) -> str:
    """Resolve local execution state against the persisted worker lease."""
    if stored in ("pending", "running"):
        t = _running.get(task_id)
        if t is not None and not t.done():
            return stored
        if not _lease_expired(meta):
            return stored
        return "orphaned"
    return stored


async def collect(task_id: str) -> str:
    """取任务产出（done 状态，完整 output）。未完成/失败给出明确错误。"""
    card = await progress(task_id)
    status = card.get("status", "")
    if status == "done":
        meta = await _meta_get(task_id)
        return str((meta or {}).get("output") or "")
    if status == "interrupted":
        raise ValueError(
            f"task '{task_id}' is interrupted and waiting for leader input "
            f"(question: {card.get('question')}) — use task_resume to continue")
    if status == "failed":
        raise ValueError(f"task '{task_id}' failed: {card.get('error')}")
    raise ValueError(f"task '{task_id}' has no output yet (status={status})")


async def cancel(task_id: str) -> None:
    """干预：取消在跑后台任务，标记 cancelled。"""
    meta = await _meta_get(task_id) or {}
    error = AgentError(
        error_type=ErrorType.TASK,
        code="TASK_CANCELLED",
        message="Task was cancelled by the leader.",
        user_message="任务已取消。",
        retryable=False,
        recovery_action="Resume or dispatch a new task if needed.",
        tool_name=str(meta.get("role") or ""),
        effect_applied="unknown",
    )
    result_payload = _task_result(
        task_id=task_id,
        role=str(meta.get("role") or ""),
        outcome=OperationOutcome.CANCELLED,
        duration_ms=0.0,
        error=error,
        effect_applied="unknown",
    )
    await _meta_put(
        task_id, status="cancelled", worker_id=None, lease_owner=None,
        heartbeat_at=None, lease_expires_at=None, recoverable=False,
        error=error.user_message, result=result_payload,
    )
    t = _running.get(task_id)
    if t and not t.done():
        t.cancel()
    _finish(task_id)


async def resume(task_id: str, reply: str) -> bool:
    """领导干预：向 interrupt 暂停的任务回复并续跑（Command(resume=reply)）。"""
    async with _transition_lock:
        return await _resume_locked(task_id, reply)


async def _resume_locked(task_id: str, reply: str) -> bool:
    await recover_expired_leases()
    existing = _running.get(task_id)
    if existing is not None and not existing.done():
        return False
    meta = await _meta_get(task_id)
    if not meta:
        raise ValueError(f"task '{task_id}' not found")
    if meta.get("status") != "interrupted":
        raise ValueError(f"task '{task_id}' is not interrupted")
    graph = _graphs.get(task_id)
    if graph is None:
        graph = await _graph_for_meta(meta)
    if graph is None:
        raise ValueError(f"task '{task_id}' has no resumable interrupt")
    config = {"configurable": {"thread_id": task_id}}
    if meta.get("model"):
        config["configurable"]["model"] = str(meta["model"])
    bg = asyncio.create_task(_run_worker(
        task_id, Command(resume=reply), config, graph,
        lease_owner=_WORKER_OWNER,
    ))
    _running[task_id] = bg
    _bg.add(bg)
    bg.add_done_callback(_bg.discard)
    return True


async def recover(task_id: str) -> bool:
    """Reattach an orphaned task to its checkpoint and continue execution."""
    async with _transition_lock:
        return await _recover_locked(task_id)


async def _recover_locked(task_id: str) -> bool:
    await recover_expired_leases(force=True)
    existing = _running.get(task_id)
    if existing is not None and not existing.done():
        return False
    meta = await _meta_get(task_id)
    if not meta:
        raise ValueError(f"task '{task_id}' not found")
    if meta.get("status") != "orphaned":
        raise ValueError(f"task '{task_id}' is not orphaned")
    graph = await _graph_for_meta(meta)
    if graph is None:
        raise ValueError(f"task '{task_id}' cannot rebuild its worker graph")
    config = {"configurable": {"thread_id": task_id}}
    if meta.get("model"):
        config["configurable"]["model"] = str(meta["model"])
    bg = asyncio.create_task(_run_worker(
        task_id, None, config, graph, lease_owner=_WORKER_OWNER,
    ))
    _running[task_id] = bg
    _bg.add(bg)
    bg.add_done_callback(_bg.discard)
    return True


async def list_tasks(kind: str = "") -> list[dict]:
    """全部派发任务（store 元数据快照，latest first；列表不携带完整 output）。"""
    await recover_expired_leases()
    metas = await _meta_scan()
    if kind:
        metas = [m for m in metas
                 if m.get("role") == kind or m.get("kind") == kind]
    out = []
    for m in metas:
        card = dict(m)
        card.pop("output", None)  # 列表视图不背完整产出（collect 单独取）
        result = card.pop("result", None)
        if isinstance(result, dict):
            card["outcome"] = str(result.get("outcome") or "")
            error = result.get("error")
            if isinstance(error, dict):
                card["error_code"] = str(error.get("code") or "")
                card["retryable"] = bool(error.get("retryable"))
        card["status"] = _effective_status(card.get("task_id", ""),
                                           card.get("status", "unknown"), m)
        out.append(card)
    return out
