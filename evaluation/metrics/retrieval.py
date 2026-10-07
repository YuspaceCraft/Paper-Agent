"""
metrics/retrieval.py — 检索质量指标（QA 评测集 → Recall/Precision@K、MRR、NDCG@K）。

复用 retrieval_orchestrator.evaluator 的纯函数（_recall/_precision/_ndcg/_mrr/
_hit_rate/_bootstrap_ci/_dimension_breakdown），不复制逻辑。

语义（trace 驱动的 agent 检索评测）：
- hits = trace 中 retrieved_context 事件按 seq 序的 chunk_id 列表 —— 顺序即
  工具调用序（与 retrieval_engine 静态评测不同：不是一次检索的 top-k，而是
  agent 跨多次工具调用实际看到的上下文顺序）。
- ground_truth_ids 来自评测 manifest（复用 eval_manifest_merged.jsonl 格式）。
- 逐条 badcase 导出：补齐现有 evaluator 只输出 top-5 failure_analysis 的缺口。
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from retrieval_orchestrator.evaluator import (  # noqa: F401（复用既有实现）
    _bootstrap_ci,
    _dimension_breakdown,
    _hit_rate,
    _mrr,
    _ndcg,
    _precision,
    _recall,
)

DEFAULT_K_VALUES = [5, 10, 20]


def per_query_metrics(hits: list[str], ground_truth: list[str],
                      k_values: list[int] | None = None,
                      meta: dict | None = None) -> dict:
    """单条 QA 的检索指标（hits 顺序 = trace 工具调用序）。"""
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    qm: dict[str, Any] = dict(meta or {})
    qm["retrieval_enabled"] = True
    qm["ground_truth_ids"] = ground_truth
    for k in k_values:
        qm[f"recall@{k}"] = round(_recall(hits, ground_truth, k), 4)
        qm[f"precision@{k}"] = round(_precision(hits, ground_truth, k), 4)
        qm[f"ndcg@{k}"] = round(_ndcg(hits, ground_truth, k), 4)
        qm[f"hit@{k}"] = round(_hit_rate(hits, ground_truth, k), 4)
    qm["mrr"] = round(_mrr(hits, ground_truth), 4)
    return qm


def aggregate(query_metrics: list[dict], k_values: list[int] | None = None) -> dict:
    """总体指标 + bootstrap CI + 维度分解（复用 evaluator 纯函数）。"""
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    eligible = [
        q for q in query_metrics
        if q.get("retrieval_enabled", True)
    ]
    skipped_count = len(query_metrics) - len(eligible)
    if not eligible:
        return {
            "overall": {
                **{
                    f"{metric}@{k}": 0.0
                    for k in k_values
                    for metric in ("recall", "precision", "ndcg", "hit")
                },
                "mrr": 0.0,
            },
            "query_count": 0,
            "skipped_query_count": skipped_count,
            "confidence_intervals": {"note": "no queries"},
            "dimension_breakdown": [],
        }
    n = len(eligible)
    overall: dict[str, float] = {}
    for k in k_values:
        for m in ("recall", "precision", "ndcg", "hit"):
            overall[f"{m}@{k}"] = round(
                mean(q[f"{m}@{k}"] for q in eligible), 4)
    overall["mrr"] = round(mean(q["mrr"] for q in eligible), 4)
    return {
        "overall": overall,
        "query_count": n,
        "skipped_query_count": skipped_count,
        "confidence_intervals": _bootstrap_ci(eligible, k_values, n),
        "dimension_breakdown": _dimension_breakdown(eligible, k_values),
    }


def classify_badcase(qm: dict, *, mrr_threshold: float = 0.0,
                     top_k: int = 5) -> str:
    """把单条 QA 归入 badcase 类别（评测报告分类用）。

    类别顺序（先因后果，任务级优先于检索级）：
      run_error      执行中断（超时/异常），指标不可信
      no_answer      没有产出回答
      tool_fail      有回答但工具调用失败（降级回答）
      task_fail      任务未达成（plan 验证不过等）
      retrieval_fail 检索完全没命中
      low_mrr        命中但排序太差

    历史缺陷：只按检索指标分类 → 工具全挂、任务没达成的条目会被归成 ok，
    失败任务不进 badcase（2026-09-15 修订）。
    """
    if qm.get("run_error"):
        return "run_error"
    if not str(qm.get("answer", "") or "").strip():
        return "no_answer"
    if qm.get("tool_errors"):
        return "tool_fail"
    if qm.get("success") is False:
        return "task_fail"
    if not qm.get("retrieval_enabled", True):
        return "ok"
    if qm.get(f"recall@{top_k}", 1.0) == 0.0:
        return "retrieval_fail"
    if qm.get("mrr", 1.0) <= mrr_threshold:
        return "low_mrr"
    return "ok"


def export_badcases(query_metrics: list[dict], *, top_n: int = 50,
                    mrr_threshold: float = 0.0, top_k: int = 5) -> list[dict]:
    """逐条 badcase 导出（按 mrr 升序 + 类别）。补齐 evaluator 只出 top-5 缺口。"""
    scored = []
    for qm in query_metrics:
        cat = classify_badcase(qm, mrr_threshold=mrr_threshold, top_k=top_k)
        if cat == "ok":
            continue
        scored.append({**qm, "category": cat})
    scored.sort(key=lambda q: (q.get("mrr", 0.0), q.get("query_id", "")))
    return scored[:top_n]


def export_qrels(manifest_path: Path | str, out_path: Path | str,
                 max_rel: int = 1) -> None:
    """导出 TREC qrels：`query_id Q0 chunk_id rel`。"""
    lines: list[str] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qa = json.loads(line)
            qid = qa.get("id", "")
            for cid in qa.get("ground_truth_ids", []):
                lines.append(f"{qid} Q0 {cid} {max_rel}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
