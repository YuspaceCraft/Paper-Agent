"""
report.py — run 报告组装、成本估算、回归基线对比。

- estimate_cost(llm_events, prices)：由真实 token usage × 价表估算 USD。
- baseline：eval_output/runs/.baseline.json 按 dataset_id 存最近 k 次 overall，
  新 run 对比最优基线：降幅 >5% → warn；Recall@5 < 0.6 → block。
- build_report()：组装 run 报告 schema（见计划文件）。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import EvalConfig, PROJECT_ROOT

BASELINE_PATH = PROJECT_ROOT / "eval_output" / "runs" / ".baseline.json"
_BASELINE_KEEP = 6
_REGRESSION_PCT = 5.0
_BASELINE_METRIC_VERSION = 2  # unique-hit retrieval + run-isolated task metrics


def estimate_cost(llm_events: list[dict], prices: dict) -> dict:
    """累计 llm_call 事件 tokens × 价表 → USD 估算；estimated 标注降权提示。"""
    total = {"prompt": 0, "completion": 0}
    per_model: dict[str, dict] = {}
    per_node: dict[str, dict] = {}
    model_calls: dict[str, int] = {}
    estimated_calls = 0
    for e in llm_events:
        tok = (e.get("payload") or {}).get("tokens") or {}
        p = int(tok.get("prompt_tokens") or 0)
        c = int(tok.get("completion_tokens") or 0)
        if tok.get("estimated"):
            estimated_calls += 1
        if not (p or c):
            continue
        model = e.get("model") or "default"
        node = e.get("node") or "unknown"
        rate_i, rate_o = prices.get(model) or prices.get("default", (0.001, 0.002))
        call_cost = (p * rate_i + c * rate_o) / 1000.0
        total["prompt"] += p
        total["completion"] += c
        model_calls[model] = model_calls.get(model, 0) + 1
        m = per_model.setdefault(model, {"prompt": 0, "completion": 0, "cost_usd": 0.0})
        m["prompt"] += p
        m["completion"] += c
        m["cost_usd"] += call_cost
        n = per_node.setdefault(
            node, {"prompt": 0, "completion": 0, "cost_usd": 0.0, "calls": 0},
        )
        n["prompt"] += p
        n["completion"] += c
        n["cost_usd"] += call_cost
        n["calls"] += 1
    cost = (total["prompt"] * 0.001 + total["completion"] * 0.002) / 1000.0
    if per_model:
        cost = sum(m["cost_usd"] for m in per_model.values())
    return {
        "prompt_tokens": total["prompt"],
        "completion_tokens": total["completion"],
        "tokens_total": total["prompt"] + total["completion"],
        "cost_usd": round(cost, 4),
        "estimated_calls": estimated_calls,
        "model_calls": model_calls,
        "per_model": per_model,
        "per_node": per_node,
    }


def cache_metrics(events: list[dict]) -> dict:
    """Tool read-cache utilization from trace rows."""
    calls = [e for e in events if e.get("event_type") == "tool_call"]
    hits = 0
    for event in calls:
        payload = event.get("payload") or {}
        if event.get("cache_hit") or payload.get("cache_hit"):
            hits += 1
    total = len(calls)
    return {
        "calls": total,
        "hits": hits,
        "hit_rate": round(hits / total, 4) if total else 0.0,
    }


def run_metadata() -> dict:
    """agent 代码 / 模型 / 配置指纹（评测可复现性）。"""
    commit = ""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()[:12]
    except Exception:  # noqa: BLE001
        pass
    cfg_hash = ""
    try:
        from agent import config_store
        raw = {ns: config_store.get_ns(ns) for ns in ("experiment", "tools", "skills")}
        cfg_hash = __import__("hashlib").md5(
            json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        pass
    tools_hash = ""
    try:
        from agent.tools import get_cached_tools
        names = sorted(t.name for t in get_cached_tools())
        tools_hash = __import__("hashlib").md5(",".join(names).encode()).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        pass
    model_routes = {}
    try:
        import agent.core.model_router as model_router

        model_routes = model_router.resolve_model_routes(
            os.getenv("LLM_MODEL", "qwen-plus"),
            small_model=os.getenv("AGENT_MODEL_SMALL", ""),
        )
    except Exception:  # noqa: BLE001
        pass
    return {
        "commit": commit,
        "model": os.getenv("LLM_MODEL", "qwen-plus"),
        "model_routes": model_routes,
        "config_fingerprint": cfg_hash,
        "tools_fingerprint": tools_hash,
    }


# ---- 回归基线 ----

def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {}


def _save_baseline(data: dict) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def _update_baseline(dataset_id: str, overall: dict, run_id: str) -> None:
    bl = _load_baseline()
    runs = [
        row for row in (bl.get(dataset_id) or [])
        if row.get("metric_version") == _BASELINE_METRIC_VERSION
    ]
    runs.append({
        "run_id": run_id,
        "metric_version": _BASELINE_METRIC_VERSION,
        "overall": overall,
    })
    runs.sort(key=lambda r: r.get("overall", {}).get("mrr", 0), reverse=True)
    bl[dataset_id] = runs[:_BASELINE_KEEP]
    _save_baseline(bl)


def _best_baseline(dataset_id: str) -> dict | None:
    runs = [
        row for row in (_load_baseline().get(dataset_id) or [])
        if row.get("metric_version") == _BASELINE_METRIC_VERSION
    ]
    return runs[0]["overall"] if runs else None


def compare_baseline(dataset_id: str, overall: dict, run_id: str) -> dict:
    """与最优基线对比 → {metric: {prev, now, delta, flag}} + 门禁结论。"""
    prev = _best_baseline(dataset_id)
    out: dict[str, dict] = {}
    flags = []
    recall5 = overall.get("recall@5")
    recall5_block = (
        isinstance(recall5, (int, float))
        and not isinstance(recall5, bool)
        and recall5 < 0.6
    )
    if prev:
        for key, now in overall.items():
            if (
                not isinstance(now, (int, float))
                or isinstance(now, bool)
                or key not in prev
            ):
                continue
            p = prev[key]
            if p == 0:
                delta = None
            else:
                delta = round((now - p) / p * 100, 2)
            flag = "pass"
            if delta is not None and delta < -_REGRESSION_PCT:
                flag = "warn"
            if key == "recall@5" and recall5_block:
                flag = "block"
            out[key] = {"prev": p, "now": now,
                        **({"delta_pct": delta} if delta is not None else {}),
                        "flag": flag}
    if recall5_block:
        flags.append("recall@5<0.6")
    # Persist only after comparing, otherwise a new best run compares to itself
    # and always reports a zero delta.
    _update_baseline(dataset_id, overall, run_id)
    gate = "pass"
    if flags:
        gate = "block"
    elif any(d["flag"] == "warn" for d in out.values()):
        gate = "warn"
    return {"because_best": prev,
            "deltas": out,
            "gate": gate,
            "warnings": flags}


def assemble_report(*, run_id, dataset_id, status, query_count, metadata,
                    events, per_query, retrieval_agg, tool_metrics, task_metrics,
                    judge_result, cfg: EvalConfig, started_at=None,
                    finished_at=None, query_durations=None) -> dict:
    """把分阶段结果组装成 run 报告（含计时、成本与基线对比）。

    started_at / finished_at：跑批起止的墙钟时间（runner 在启动与收尾处打点）。
    query_durations：逐条 QA 的墙钟耗时（秒），用于 p50/p95 —— 不用「把链路里
    每一步的 duration_ms 相加」这种累加口径当耗时。
    """
    badcases = [
        {**q, "trace_id": q.get("trace_id", "")}
        for q in per_query if q.get("category") not in (None, "ok")
    ][:50]
    cost = estimate_cost(events, cfg.prices)
    cache = cache_metrics(events)
    successful_tasks = int(task_metrics.get("success") or 0)
    cost_per_task = (
        round(cost["cost_usd"] / successful_tasks, 4)
        if successful_tasks else None
    )
    tokens_per_task = (
        round(cost["tokens_total"] / successful_tasks, 1)
        if successful_tasks else None
    )
    durations = [float(d) for d in (query_durations or []) if d is not None]
    timing = timing_stats(durations)
    if started_at is not None and finished_at is not None:
        duration_s = round(float(finished_at) - float(started_at), 1)
    else:
        duration_s = timing["total_s"]
    t_overall = tool_metrics["overall"]
    delta = compare_baseline(dataset_id,
                             {**retrieval_agg["overall"],
                              "tool_success_rate": t_overall["success_rate"],
                              "task_success_rate": task_metrics.get("task_success_rate", 0.0)},
                             run_id)
    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "status": status,
        "query_count": query_count,
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "duration_s": duration_s,
        "metadata": metadata,
        "overall": {
            **retrieval_agg["overall"],
            # 工具/任务成功率都是「成功数 / 调用（条目）数」算出来的比值；
            # 分子分母一并透出，便于核对「失败 N 次成功率是不是真的 1」。
            "tool_success_rate": t_overall["success_rate"],
            "tool_calls": t_overall.get("calls", 0),
            "tool_failures": t_overall.get("errors", 0),
            "task_success_rate": task_metrics.get("task_success_rate", 0.0),
            "task_success": task_metrics.get("success"),
            "task_degraded": task_metrics.get("degraded"),
            "task_failed": task_metrics.get("failed"),
            "contract_success_rate": task_metrics.get("contract_success_rate"),
            "contract_success": task_metrics.get("contract_success"),
            "contract_checks": task_metrics.get("contract_checks"),
            "recovery_rate": task_metrics.get("recovery_rate"),
            # 检索聚合已经计算了 bootstrap CI；保留在 overall 内供对比页直接使用。
            "confidence_intervals": retrieval_agg.get("confidence_intervals", {}),
            "duration_s": duration_s,
            "answer_latency_p50_s": timing["p50_s"],
            "answer_latency_p95_s": timing["p95_s"],
            "answer_latency_max_s": timing["max_s"],
            "cost_usd": cost["cost_usd"],
            "tokens_total": cost["tokens_total"],
            "cost_per_successful_task_usd": cost_per_task,
            "tokens_per_successful_task": tokens_per_task,
            "cache_hit_rate": cache["hit_rate"],
            "cache_hits": cache["hits"],
            "cache_calls": cache["calls"],
        },
        "dimension": retrieval_agg["dimension_breakdown"],
        "task_dimension": task_metrics.get("dimension", []),
        "judge": judge_result or {},
        "badcases": badcases,
        "tool_metrics": tool_metrics,
        "task_metrics": task_metrics,
        "baseline_delta": delta,
        "cost": {**cost, "cache": cache},
        "query_durations_s": [round(d, 1) for d in durations],
        "notes": ["ordering=agent tool-call sequence"],
    }


def _iso(ts) -> str | None:
    """epoch 秒 → UTC ISO 串（已是字符串则原样返回）。"""
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(
        timespec="seconds")


def timing_stats(durations: list[float]) -> dict:
    """逐条 QA 墙钟耗时 → {p50_s, p95_s, max_s, total_s}（无样本时全 None）。"""
    if not durations:
        return {"p50_s": None, "p95_s": None, "max_s": None, "total_s": None}
    ds = sorted(durations)
    idx95 = min(len(ds) - 1, max(0, int(len(ds) * 0.95 + 0.999) - 1))
    return {
        "p50_s": round(ds[len(ds) // 2], 2),
        "p95_s": round(ds[idx95], 2),
        "max_s": round(ds[-1], 2),
        "total_s": round(sum(ds), 1),
    }
