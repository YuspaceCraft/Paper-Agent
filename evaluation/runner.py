"""
runner.py — 评测跑批：manifest → agent 全链路 → 收集 trace → 分阶段指标 → 报告。

流程（每 QA）：
  set_eval_ctx(run_id, thread_id=evalqa:{run_id}:{qid}, source=eval)
  set_trace_id_override(f"{run_id}:{qid}")      → 确定性 trace_id（整轮可检索）
  agent.graph.run(query, thread_id=evalqa:{run_id}:{qid}) → 生产路径
  get_trace({run_id}:{qid})                       → 只读本次 run 的链路事件
  compute_per_query()                             → 检索/工具/任务/badcase 归因

聚合后：retrieval.aggregate + tools.aggregate + task 汇总 + 可选 judge 采样
→ assemble_report → 落 eval_output/runs/<run_id>/run_summary.json + eval_runs 表。

默认小样本（用户决策）；逐条超时护栏；--resume 断点续跑；judge 走 executor
线程不阻塞事件循环。

实时进度（2026-09-15）：每完成一条 QA 就向 evaluation/live.py 的进程内总线发布
query_finished + aggregate（累计指标 / token / 成本 / ETA），并把同一份聚合写进
eval_runs 进度行（status=running，overall.progress 给出完成度）——前端 SSE 与
轮询两条路径拿到的都是「随执行更新的指标」，且与最终报告同口径。

计时与终态（2026-09-15 修订）：
- 完整用时 = 跑批开始 → 跑批结束的墙钟差（runner 自己启停计时，写进报告与
  eval_runs 行）；执行链路里每一步的 duration_ms 只用于阶段归因，不再相加当总时长。
- 跑批一定会留下终态行：正常结束 done；异常 failed；被取消/中断 interrupted。
  长任务期间周期性刷新 metadata.heartbeat，读路径据此把「进程已消失仍挂在
  running」的行收敛成 interrupted（reconcile_stale_runs）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from datetime import datetime as _dt
from datetime import timezone as _tz
from pathlib import Path
from typing import Any

from .config import EvalConfig
from .metrics import judge as judge_mod
from .metrics.tools import aggregate_tools
from .metrics.retrieval import (aggregate as retr_aggregate,
                                classify_badcase,
                                per_query_metrics)
from .metrics.contracts import evaluate_task_contract
from .metrics.task import aggregate_task_metrics, task_success
from . import datasets
from .report import assemble_report, estimate_cost, run_metadata
from . import live as live_bus

_RUN_COLUMNS = ("overall", "dimension", "badcases", "judge", "tool_metrics",
                "task_metrics", "baseline_delta", "cost")

# eval_runs 表列序（SELECT * 用；与 _COLUMNS 的 SQL 顺序一致）
_ROW_COLUMNS = ("run_id", "dataset_id", "started_at", "finished_at", "status",
                "query_count", "metadata", *_RUN_COLUMNS, "notes")


def _now_iso() -> str:
    return _dt.now(_tz.utc).isoformat(timespec="seconds")


def _iso(ts: float | str | None) -> str | None:
    """epoch 秒 / ISO 串 → ISO 串；None 原样返回。"""
    if isinstance(ts, str):
        return ts or None
    if ts is None:
        return None
    return _dt.fromtimestamp(float(ts), _tz.utc).isoformat(timespec="seconds")


def _epoch(ts: float | str | None) -> float | None:
    """ISO 串 / epoch 秒 → epoch 秒；解析不出来返回 None。"""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return _dt.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


async def agent_run(query: str, thread_id: str, *, timeout: float = 900.0) -> dict:
    """完整走生产路径（checkpointer / 超时 / tools 装配全真实）。"""
    from agent.graph import run as _graph_run, TURN_TIMEOUT
    effective = min(timeout, TURN_TIMEOUT) if TURN_TIMEOUT else timeout
    return await asyncio.wait_for(
        _graph_run(query, thread_id=thread_id), timeout=effective)


def _hits_from(events: list[dict]) -> list[str]:
    """检索命中 = retrieved_context 按 seq 收集唯一 chunk_id（首次出现次序）。"""
    hits: list[str] = []
    seen: set[str] = set()
    for e in sorted(events, key=lambda x: x.get("seq", 0)):
        if e.get("event_type") != "retrieved_context":
            continue
        for chunk_id in (e.get("payload") or {}).get("chunk_ids") or []:
            cid = str(chunk_id)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            hits.append(cid)
    return hits


def _find_last_answer(events: list[dict]) -> str:
    fa = next(iter([e for e in reversed(events)
                    if e.get("event_type") == "final_answer"]), None)
    return (fa.get("payload") or {}).get("answer", "") if fa else ""


def _find_intent(events: list[dict]) -> str:
    ev = next(iter([e for e in reversed(events)
                    if e.get("event_type") == "intent"]), None)
    if not ev:
        return ""
    return str(ev.get("intent") or (ev.get("payload") or {}).get("intent") or "")


def _context_snapshot(events: list[dict], limit: int = 4000) -> str:
    parts: list[str] = []
    used = 0
    for e in sorted(events, key=lambda x: x.get("seq", 0)):
        if e.get("event_type") != "retrieved_context":
            continue
        snap = (e.get("payload") or {}).get("snapshot") or ""
        if snap:
            parts.append(snap)
            used += len(snap)
        if used >= limit:
            break
    return "\n---\n".join(parts)[:limit]


# ---- 实时进度（live 总线 + eval_runs 进度行）----

def _publish(run_id: str, event: dict) -> None:
    """广播一条进度事件；任何异常都不影响跑批。"""
    try:
        live_bus.publish(run_id, event)
    except Exception:  # noqa: BLE001
        pass


def _query_row(index: int, total: int, qid: str, query: str, qm: dict, *,
               duration_s: float, tokens: int, cost_usd: float) -> dict:
    """逐条 QA 的实时行（query_finished / aggregate.recent 共用）。"""
    return {
        "index": index, "total": total, "qid": qid, "query": query,
        "task_type": qm.get("task_type") or "",
        "difficulty": qm.get("difficulty_level") or "",
        "task_length": qm.get("task_length") or "",
        "category": qm.get("category") or "", "success": bool(qm.get("success")),
        "mrr": qm.get("mrr"), "recall@5": qm.get("recall@5"),
        "ndcg@10": qm.get("ndcg@10"), "tool_errors": qm.get("tool_errors") or 0,
        "duration_s": round(duration_s, 1), "tokens": tokens,
        "cost_usd": round(cost_usd, 4), "trace_id": qm.get("trace_id") or "",
        "step_count": qm.get("step_count") or 0,
        "run_error": qm.get("run_error") or "",
        "contract_success": qm.get("contract_success"),
        "contract_failures": [
            item.get("name") for item in qm.get("contract_failures") or []
        ],
    }


async def _upsert_progress_row(store, *, run_id: str, dataset_id: str, total: int,
                               started: float, done: int = 0,
                               overall: dict | None = None,
                               task_metrics: dict | None = None,
                               tool_metrics: dict | None = None,
                               cost: dict | None = None) -> None:
    """把「进行中」的进度写进 eval_runs 行。

    这样即便前端不走 SSE（不同进程的 CLI 跑批 / 断线），列表与报告页轮询也能
    看到实时指标；行内 metadata.total 给出总条数，overall.progress 给出完成度。
    """
    try:
        await upsert_run_row(
            store, run_id=run_id, dataset_id=dataset_id, status="running",
            query_count=done, started_at=started,
            overall=overall or {"progress": {"done": done, "total": total}},
            task_metrics=task_metrics, tool_metrics=tool_metrics, cost=cost,
            metadata={**run_metadata(), "dataset_id": dataset_id,
                      "total": total, "live": True,
                      "heartbeat": round(time.time(), 3)},
        )
    except Exception:  # noqa: BLE001 — 进度是观测通道，失败只降级
        pass


# ---- 心跳 / 终态收敛 ----

_HEARTBEATS: dict[str, asyncio.Task] = {}


def start_heartbeat(store, *, run_id: str, interval: float) -> None:
    """起一个心跳任务：周期性刷新进度行 metadata.heartbeat（幂等）。"""
    if interval <= 0 or not run_id or run_id in _HEARTBEATS:
        return
    _HEARTBEATS[run_id] = asyncio.create_task(
        _heartbeat_loop(store, run_id=run_id, interval=interval))


def stop_heartbeat(run_id: str) -> None:
    task = _HEARTBEATS.pop(run_id, None)
    if task is not None and not task.done():
        task.cancel()


async def _heartbeat_loop(store, *, run_id: str, interval: float) -> None:
    """长 query（分钟级）期间也保持心跳新鲜，避免被读路径误判成中断。"""
    try:
        while True:
            await asyncio.sleep(interval)
            await _touch_heartbeat(store, run_id=run_id)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — 心跳失败不影响跑批
        pass


async def _touch_heartbeat(store, *, run_id: str) -> None:
    conn = await store._ensure_conn()
    cur = await conn.execute(
        "SELECT metadata FROM eval_runs WHERE run_id=? AND status='running'",
        (run_id,))
    row = await cur.fetchone()
    if not row:
        return
    meta: dict = {}
    try:
        meta = json.loads(row[0] or "{}")
    except ValueError:
        meta = {}
    meta["heartbeat"] = round(time.time(), 3)
    await conn.execute("UPDATE eval_runs SET metadata=? WHERE run_id=?",
                       (json.dumps(meta, ensure_ascii=False), run_id))
    await conn.commit()


async def finalize_run_row(store, *, run_id: str, dataset_id: str = "",
                           status: str, note: str = "",
                           started=None) -> None:
    """把非终态行收敛成终态（保留已算出的部分指标 + 失败原因）。

    幂等：已是 done / failed / interrupted 的行不会被覆盖。跑批异常、被取消、
    进程重启都会走到这里——「所有任务结束后仍只有 running 态」就是缺了这一步。
    """
    try:
        conn = await store._ensure_conn()
        cur = await conn.execute("SELECT * FROM eval_runs WHERE run_id=?",
                                 (run_id,))
        row = await cur.fetchone()
        keep: dict[str, Any] = {}
        if row:
            d = dict(zip(_ROW_COLUMNS, row))
            if str(d.get("status") or "") in ("done", "failed", "interrupted"):
                return
            for key in _RUN_COLUMNS:
                raw = d.get(key)
                try:
                    keep[key] = json.loads(raw) if raw else None
                except ValueError:
                    keep[key] = None
            for key, default in (("metadata", {}), ("notes", [])):
                try:
                    keep[key] = json.loads(d.get(key) or "null")
                except ValueError:
                    keep[key] = None
                keep[key] = keep[key] if keep[key] is not None else default
            keep["dataset_id"] = d.get("dataset_id") or dataset_id
            keep["query_count"] = int(d.get("query_count") or 0)
            started = d.get("started_at") or started
        keep.setdefault("dataset_id", dataset_id)
        keep.setdefault("query_count", 0)
        keep.setdefault("metadata", {})
        keep.setdefault("notes", [])
        finished = time.time()
        overall = dict(keep.get("overall") or {})
        overall["status"] = status
        overall["progress"] = {
            "done": keep["query_count"],
            "total": int((keep.get("metadata") or {}).get("total") or 0)}
        await upsert_run_row(
            store, run_id=run_id, dataset_id=keep["dataset_id"], status=status,
            query_count=keep["query_count"], started_at=started,
            finished_at=finished, overall=overall,
            dimension=keep.get("dimension"), badcases=keep.get("badcases"),
            judge=keep.get("judge"), tool_metrics=keep.get("tool_metrics"),
            task_metrics=keep.get("task_metrics"),
            baseline_delta=keep.get("baseline_delta"), cost=keep.get("cost"),
            metadata=keep.get("metadata"),
            notes=list(keep.get("notes") or []) + ([note] if note else []))
    except Exception:  # noqa: BLE001 — 收敛失败也不能掩盖原始异常
        pass


async def reconcile_stale_runs(store, *, stale_after_s: float = 180.0,
                               ) -> list[str]:
    """把「进程已消失却仍挂在 running」的行收敛成 interrupted（读路径调用）。

    判据：status=running ∧ 本进程没有该 run 的发布者 ∧ metadata.heartbeat
    （缺省回退 started_at）早于 stale_after_s 之前。前端列表 / 报告页因此不会
    无限显示 running——坏掉的 run 会以终态 + 部分指标出现在报告里。
    """
    out: list[str] = []
    if stale_after_s <= 0:
        return out
    try:
        conn = await store._ensure_conn()
        cur = await conn.execute(
            "SELECT run_id, started_at, metadata FROM eval_runs"
            " WHERE status IN ('running','pending')")
        rows = await cur.fetchall()
    except Exception:  # noqa: BLE001 — 读路径的尽力而为
        return out
    now = time.time()
    for run_id, started_at, metadata in rows:
        try:
            if live_bus.is_running(run_id):      # 本进程还在跑，别动
                continue
            meta: dict = {}
            try:
                meta = json.loads(metadata or "{}")
            except ValueError:
                meta = {}
            hb = _epoch(meta.get("heartbeat")) or _epoch(started_at) or now
            if now - float(hb) <= stale_after_s:
                continue
            await finalize_run_row(
                store, run_id=run_id, status="interrupted",
                note=(f"interrupted: no heartbeat for {int(now - hb)}s"
                      " (run process gone)"), started=started_at)
            _publish(run_id, {"type": "run_failed", "run_id": run_id,
                              "status": "interrupted",
                              "error": "run interrupted (no heartbeat)"})
            out.append(run_id)
        except Exception:  # noqa: BLE001 — 单行失败不影响其它行
            continue
    return out


async def ensure_running_row(store, run_id: str, dataset_id: str,
                             total: int) -> None:
    """启动时先落「进行中」行（供 API / CLI 在 run_eval 之前调用）。

    作用是消掉竞态：前端拿到 run_id 就订阅 SSE，此时行已存在（否则 404）；
    同时让不支持 SSE 的客户端从第一秒起就能轮询到进度。
    """
    await _upsert_progress_row(store, run_id=run_id, dataset_id=dataset_id,
                               total=total, started=time.time())


async def _publish_progress(store, *, run_id: str, dataset_id: str,
                            query_row: dict, per_query: list[dict],
                            tool_events: list[dict], cost_total: float,
                            tokens_total: int, series: list[dict],
                            started: float) -> None:
    """逐条 QA 完成后：发明细 + 累计聚合，并把进度写进 eval_runs 行。

    实时数字与最终报告（assemble_report）共用同一批纯函数与同名口径，
    避免「界面上的数」与「报告里的数」两套算法。
    """
    try:
        retr_agg = retr_aggregate(per_query)
        tool_metrics = aggregate_tools(tool_events)
        ok_tasks = [q for q in per_query if q.get("success")]
        t_overall = tool_metrics.get("overall") or {}
        contract_rows = [
            row for row in per_query if row.get("contract_enabled", False)
        ]
        contract_success = sum(
            1 for row in contract_rows if row.get("contract_success"))
        done, total = len(per_query), int(query_row.get("total") or 0)
        elapsed = time.time() - started
        overall = {
            **(retr_agg.get("overall") or {}),
            # 工具/任务成功率都是「成功数 / 总数」算出来的比值，同时透出分子分母
            "tool_success_rate": t_overall.get("success_rate", 0.0),
            "tool_calls": t_overall.get("calls", 0),
            "tool_failures": t_overall.get("errors", 0),
            "task_success_rate": round(len(ok_tasks) / max(done, 1), 4),
            "task_success": len(ok_tasks),
            "contract_success_rate": (
                round(contract_success / len(contract_rows), 4)
                if contract_rows else None
            ),
            "contract_success": contract_success,
            "contract_checks": len(contract_rows),
            "cost_usd": cost_total,
            "tokens_total": tokens_total,
            "progress": {"done": done, "total": total},
        }
        _publish(run_id, {"type": "query_finished", "run_id": run_id, **query_row})
        _publish(run_id, {
            "type": "aggregate", "run_id": run_id, "dataset_id": dataset_id,
            "status": "running", "done": done, "total": total,
            "elapsed_s": round(elapsed, 1),
            "eta_s": (round(elapsed / done * (total - done), 1)
                      if done and total > done else None),
            "overall": overall, "last": query_row, "recent": series[-10:],
        })
        await _upsert_progress_row(
            store, run_id=run_id, dataset_id=dataset_id, total=total,
            started=started, done=done, overall=overall,
            task_metrics={"task_success_rate": overall["task_success_rate"],
                          "query_count": done},
            tool_metrics=tool_metrics,
            cost={"cost_usd": cost_total, "tokens_total": tokens_total})
    except Exception:  # noqa: BLE001 — 进度是观测通道，失败只降级
        pass


async def run_eval(cfg: EvalConfig, qas: list[dict], *,
                   run_id: str | None = None, dataset_id: str | None = None,
                   with_judge: bool = True, on_progress=None,
                   on_query_done=None) -> dict:
    """跑批入口。qas 去重后执行；返回 run 报告并落盘 eval_runs。

    全过程向 live 总线发布进度（run_started / query_started / query_finished /
    aggregate / judge_* / run_finished），失败发 run_failed 后原样抛出。

    无论成功、异常还是被取消，eval_runs 行都会收敛到终态（done / failed /
    interrupted）——不会留下永远 running 的进度行。
    """
    from .trace_store import new_run_id, get_trace_store
    rid = run_id or new_run_id()
    started = time.time()
    try:
        return await _run_impl(cfg, qas, run_id=rid, dataset_id=dataset_id,
                               with_judge=with_judge, on_progress=on_progress,
                               on_query_done=on_query_done)
    except BaseException as exc:  # noqa: BLE001 — 含 CancelledError（取消/重启）
        status = ("interrupted" if isinstance(exc, asyncio.CancelledError)
                  else "failed")
        note = f"{status}: {type(exc).__name__}: {exc}"[:300]
        await finalize_run_row(get_trace_store(), run_id=rid,
                               dataset_id=dataset_id or "",
                               status=status, note=note, started=started)
        _publish(rid, {"type": "run_failed", "run_id": rid, "status": status,
                       "error": note})
        raise
    finally:
        stop_heartbeat(rid)


async def _run_impl(cfg: EvalConfig, qas: list[dict], *, run_id: str,
                    dataset_id: str | None = None, with_judge: bool = True,
                    on_progress=None, on_query_done=None) -> dict:
    """run_eval 的实现体（run_id 已解析，进度事件由调用方兜底）。"""
    from .trace_store import get_trace_store
    from .sink import attach as attach_trace_sink, set_eval_ctx, clear_eval_ctx
    from agent.observability import (set_trace_id_override,
                                     clear_trace_id_override)

    attach_trace_sink()
    store = get_trace_store()
    qas = datasets.deduplicate(qas)
    if cfg.max_queries > 0:
        qas = qas[: cfg.max_queries]
    dataset_id = dataset_id or datasets.dataset_id_for(qas)

    run_dir = cfg.run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    per_query: list[dict] = []
    series: list[dict] = []          # 逐条实时行（SSE aggregate.recent）
    llm_events: list[dict] = []      # 实时累计：token / 成本
    tool_events: list[dict] = []     # 实时累计：工具成功率
    query_durations: list[float] = []  # 逐条 QA 的墙钟耗时（秒）
    status = "done"
    started = time.time()            # 完整用时起点（跑批开始）
    total = len(qas)

    _publish(run_id, {"type": "run_started", "run_id": run_id,
                      "dataset_id": dataset_id, "total": total,
                      "started_at": _now_iso(), "status": "running"})
    await _upsert_progress_row(store, run_id=run_id, dataset_id=dataset_id,
                               total=total, started=started)
    start_heartbeat(store, run_id=run_id,
                    interval=cfg.progress_heartbeat_s)

    for i, qa in enumerate(qas, start=1):
        qid = qa.get("id", f"qa_{i}")
        query = qa.get("query", "")
        # A run must not inherit checkpoints or traces from a previous run with
        # the same QA id.  A re-run of the same run_id deliberately resumes it.
        thread_id = f"evalqa:{run_id}:{qid}"
        if on_progress:
            on_progress(i, total, qid)
        _publish(run_id, {"type": "query_started", "index": i, "total": total,
                          "qid": qid, "query": query})
        turn_started = time.time()

        set_eval_ctx(run_id=run_id, thread_id=thread_id, qa_id=qid, source="eval")
        trace_id = f"{run_id}:{qid}"
        set_trace_id_override(trace_id)
        error: str | None = None
        try:
            await agent_run(query, thread_id, timeout=cfg.turn_timeout)
        except asyncio.TimeoutError:
            error = "turn_timeout"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            clear_trace_id_override()
            clear_eval_ctx()

        await store.flush()
        events = await store.get_trace(trace_id)

        # ── 单条任务指标 ──
        hits = _hits_from(events)
        qa_filters = qa.get("metadata_filters") or {}
        meta = {
            "query_id": qid,
            "query": qa.get("query", ""),
            "task_type": qa.get("task_type") or "legacy_retrieval",
            "task_length": qa.get("task_length") or "unspecified",
            "difficulty_level": (
                qa.get("difficulty_level") or "unknown"),
            "content_type": (
                qa_filters.get("content_type") or "unknown"),
            "generation_mode": (
                qa.get("generation_mode") or "unknown"),
        }
        retrieval_enabled = datasets.retrieval_enabled(qa)
        if retrieval_enabled:
            retr = per_query_metrics(
                hits, qa.get("ground_truth_ids", []), meta=meta)
        else:
            retr = {**meta, "retrieval_enabled": False}
        # 任务成功率口径：run_error（超时/异常）与工具失败都算未达成
        task = task_success(events, run_error=error)
        tools = [e for e in events if e.get("event_type") == "tool_call"]
        t_fail = task.get("tool_failures", 0)
        duration_s = time.time() - turn_started
        query_durations.append(duration_s)
        q_llm = [e for e in events if e.get("event_type") == "llm_call"]
        q_cost = estimate_cost(q_llm, cfg.prices)

        qm: dict[str, Any] = {**retr, **task}
        qm["trace_id"] = trace_id
        qm["tool_errors"] = t_fail
        qm["step_count"] = len(tools) + len(q_llm)
        qm["duration_s"] = round(duration_s, 1)
        qm["answer"] = _find_last_answer(events)
        qm["intent"] = _find_intent(events)
        qm["prompt_tokens"] = int(q_cost.get("prompt_tokens") or 0)
        qm["completion_tokens"] = int(q_cost.get("completion_tokens") or 0)
        qm["tokens_total"] = int(q_cost.get("tokens_total") or 0)
        qm["cost_usd"] = float(q_cost.get("cost_usd") or 0.0)
        if error:
            qm["run_error"] = error

        contract = evaluate_task_contract(
            qa, events, observed=qm, query_started_at=turn_started)
        qm.update(contract)
        if not contract["contract_success"] and qm.get("success"):
            failed_checks = [
                item.get("name") for item in contract.get("contract_failures") or []
            ]
            qm["success"] = False
            qm["status"] = "degraded" if qm.get("has_answer") else "failed"
            qm["task_error"] = (
                "contract failed: " + ", ".join(failed_checks[:8])
            )
        qm["category"] = classify_badcase(qm, mrr_threshold=0.0)
        per_query.append(qm)

        with (run_dir / "per_query.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(qm, ensure_ascii=False) + "\n")

        # ── 逐条实时进度：明细行 + 累计聚合（SSE 与 eval_runs 进度行同源）──
        llm_events.extend(q_llm)
        tool_events.extend(tools)
        row = _query_row(i, total, qid, query, qm,
                         duration_s=duration_s,
                         tokens=int(q_cost.get("tokens_total") or 0),
                         cost_usd=float(q_cost.get("cost_usd") or 0.0))
        series.append(row)
        run_cost = estimate_cost(llm_events, cfg.prices)
        await _publish_progress(
            store, run_id=run_id, dataset_id=dataset_id, query_row=row,
            per_query=per_query, tool_events=tool_events,
            cost_total=float(run_cost.get("cost_usd") or 0.0),
            tokens_total=int(run_cost.get("tokens_total") or 0),
            series=series, started=started)
        if on_query_done:
            try:
                hook = on_query_done(i, total, qm)
                if inspect.isawaitable(hook):
                    await hook
            except Exception:  # noqa: BLE001 — 进度回调失败不影响跑批
                pass

    # ── 聚合 ──
    all_events = await store.query(run_id=run_id, limit=100000)
    if not all_events and (llm_events or tool_events):
        # 报告口径与实时口径同源：正常按 run_id 从库里取全量事件；取不到（sink 未
        # 挂 / run_id 不匹配）时退回逐条累计的事件，避免报告里的成本与工具成功率
        # 突然掉成 0，与刚才实时进度里看到的数对不上。
        all_events = list(llm_events) + list(tool_events)
    retr_agg = retr_aggregate(per_query)
    tool_metrics = aggregate_tools(all_events)
    task_metrics = aggregate_task_metrics(per_query)
    task_metrics["query_count"] = len(per_query)

    judge_result: dict = {}
    if with_judge and per_query:
        _publish(run_id, {"type": "judge_started", "run_id": run_id,
                          "sample_size": min(cfg.judge_sample_size, len(per_query))})
        judge_result = await _run_judge(cfg, store, per_query)
        _publish(run_id, {"type": "judge_finished", "run_id": run_id,
                          **judge_result})

    # Self-contained offline archive: reports remain inspectable after the local
    # trace DB or LangSmith retention window is cleared.
    trace_archive = run_dir / "trace_events.jsonl"
    with trace_archive.open("w", encoding="utf-8") as handle:
        for event in sorted(all_events, key=lambda item: (
                item.get("trace_id", ""), item.get("seq", 0))):
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    finished = time.time()          # 完整用时终点（跑批结束）
    report = assemble_report(
        run_id=run_id, dataset_id=dataset_id, status=status,
        query_count=len(per_query), metadata={**run_metadata(),
                                              "dataset_id": dataset_id,
                                              "ordering": "tool_call_sequence",
                                              "trace_archive": str(trace_archive)},
        events=all_events, per_query=per_query,
        retrieval_agg=retr_agg, tool_metrics=tool_metrics,
        task_metrics=task_metrics, judge_result=judge_result, cfg=cfg,
        started_at=started, finished_at=finished,
        query_durations=query_durations,
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    await insert_eval_run(store, report, started_at=started)
    await store.flush()
    _publish(run_id, {
        "type": "run_finished", "run_id": run_id, "status": status,
        "query_count": len(per_query), "duration_s": report["duration_s"],
        "finished_at": report["finished_at"],
        "overall": report["overall"],
        "gate": (report.get("baseline_delta") or {}).get("gate"),
        "badcase_count": len(report.get("badcases") or []),
        "cost_usd": report["overall"].get("cost_usd"),
    })

    # LangSmith run 树导出（平台设计 §8.1）：把每次评测的 run 树落到
    # eval_output/runs/<run_id>/langsmith_runs.jsonl，报告在 trace 过保留期后
    # 仍可复现。默认关闭——导出要访问网络且 SDK 索引有延迟，正式跑批/发布时
    # 用 EVAL_EXPORT_LANGSMITH=1 打开；失败只告警，绝不影响报告。
    if os.getenv("EVAL_EXPORT_LANGSMITH", "") not in {"", "0", "false"}:
        from agent.core.trace_export import export_best_effort
        for qm in per_query:
            trace_id = str(qm.get("trace_id") or "")
            if trace_id:
                await asyncio.to_thread(export_best_effort, trace_id)

    return report


async def upsert_run_row(store, *, run_id: str, dataset_id: str, status: str,
                         query_count: int = 0, started_at=None,
                         finished_at=None, overall=None, dimension=None,
                         badcases=None, judge=None, tool_metrics=None,
                         task_metrics=None, baseline_delta=None, cost=None,
                         metadata=None, notes=None) -> None:
    """写/更新 eval_runs 行（同 run_id 覆盖，进度行与报告行同一路径）。

    running 阶段只填已算出的字段（其余留 NULL）；跑完由 insert_eval_run 覆盖为
    完整报告，因此「进度行」与「最终行」同构，前端不需要两套解析。
    """
    import json as _j
    payload = {"overall": overall, "dimension": dimension, "badcases": badcases,
               "judge": judge, "tool_metrics": tool_metrics,
               "task_metrics": task_metrics, "baseline_delta": baseline_delta,
               "cost": cost}
    values = tuple(
        _j.dumps(payload[k], ensure_ascii=False)
        if payload.get(k) is not None else None
        for k in _RUN_COLUMNS)
    conn = await store._ensure_conn()
    await conn.execute(
        "INSERT OR REPLACE INTO eval_runs (run_id, dataset_id, started_at,"
        " finished_at, status, query_count, metadata, overall, dimension,"
        " badcases, judge, tool_metrics, task_metrics, baseline_delta,"
        " cost_estimate, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, dataset_id, _iso(started_at) or _now_iso(), _iso(finished_at),
         status, query_count, _j.dumps(metadata or {}, ensure_ascii=False),
         *values, _j.dumps(notes or [], ensure_ascii=False)))
    await conn.commit()


async def insert_eval_run(store, report: dict, *, started_at=None) -> None:
    """写最终报告行（幂等，同 run_id 重跑覆盖）。

    finished_at 用报告里 runner 打点的收尾时间（与 duration_s 同源，不再各取各
    的 time.time()）；overall 里带 duration_s，前端列表/报告页无需再算一遍。
    """
    await upsert_run_row(
        store, run_id=report["run_id"], dataset_id=report["dataset_id"],
        status=report.get("status", "done"),
        query_count=report.get("query_count", 0),
        started_at=started_at or report.get("started_at"),
        finished_at=report.get("finished_at") or time.time(),
        overall=report.get("overall"),
        dimension=report.get("dimension"), badcases=report.get("badcases"),
        judge=report.get("judge"), tool_metrics=report.get("tool_metrics"),
        task_metrics=report.get("task_metrics"),
        baseline_delta=report.get("baseline_delta"), cost=report.get("cost"),
        metadata=report.get("metadata"), notes=report.get("notes"))


async def _run_judge(cfg: EvalConfig, store, per_query: list[dict]) -> dict:
    """采样 judge（badcase 优先 + 随机补齐），LLM 调用放 executor 不阻塞循环。

    样本数据（query/answer/context/intent）在 async 内预取，仅纯函数入线程。
    """
    if cfg.judge_budget_usd <= 0:
        return {"skipped": True, "reason": "judge budget disabled"}
    picked = (sorted(per_query, key=lambda q: q.get("category") == "ok")
              )[: cfg.judge_sample_size]

    samples: list[dict] = []
    for q in picked:
        trace_id = str(q.get("trace_id") or "")
        events = await store.get_trace(trace_id) if trace_id else []
        samples.append({
            "query": q.get("query", ""),
            "answer": q.get("answer", ""),
            "context": _context_snapshot(events),
            "intent": q.get("intent", q.get("verification_status", "")),
            "prefer": q.get("category") != "ok",
        })

    def _judge_blocking() -> dict:
        return judge_mod.judge_answer_vs_context(
            samples, model=cfg.judge_model,
            base_url="", api_key_env="DASHSCOPE_API_KEY",
            max_samples=cfg.judge_sample_size)

    try:
        return await asyncio.to_thread(_judge_blocking)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---- recovery 测试跑批 ----

async def run_recovery_test(cfg: EvalConfig, qas: list[dict]) -> dict:
    """中断恢复批量测试（复用测评集前几条的 query 走故障注入）。"""
    from .metrics.task import recovery_test, aggregate_recovery
    results = []
    for qa in (qas or [])[:3]:
        thread_id = f"evalqa:{qa.get('id', 'x')}:recovery"
        try:
            res = await recovery_test(
                qa.get("query", ""), thread_id=thread_id,
                run_fn=lambda q, t: agent_run(q, t, timeout=cfg.turn_timeout))
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            results.append({"error": f"{type(exc).__name__}: {exc}",
                            "recovered": False})
    return aggregate_recovery(results)
