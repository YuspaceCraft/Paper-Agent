"""
evaluator.py — 自动化评估计算器
================================

Computes:
  - Recall@K, Precision@K, MRR, NDCG@K
  - Per-dimension breakdown (difficulty_level, content_type, chunk length)
  - Optional LLM-as-Judge for Context Precision / Faithfulness
  - Low-score failure case analysis

Outputs evaluation_report.json.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import random
from pathlib import Path
from typing import Any

RANDOM_SEED = 42


# ================================================================
# Core Metrics
# ================================================================

def _dcg(relevance: list[int], k: int | None = None) -> float:
    """Discounted Cumulative Gain."""
    if k is not None:
        relevance = relevance[:k]
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevance)
    )


def _ndcg(hits: list[str], ground_truth: list[str], k: int) -> float:
    """NDCG@K: normalized by ideal DCG."""
    gt_set = set(ground_truth)
    rel = [1 if hit in gt_set else 0 for hit in hits[:k]]
    ideal_rel = sorted([1] * min(len(ground_truth), k) + [0] * max(0, k - len(ground_truth)), reverse=True)
    dcg_val = _dcg(rel)
    idcg_val = _dcg(ideal_rel)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def _recall(hits: list[str], ground_truth: list[str], k: int) -> float:
    """Recall@K: fraction of ground_truth found in top-K hits."""
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0
    found = sum(1 for h in hits[:k] if h in gt_set)
    return found / len(gt_set)


def _precision(hits: list[str], ground_truth: list[str], k: int) -> float:
    """Precision@K: fraction of top-K hits that are relevant."""
    gt_set = set(ground_truth)
    if k == 0:
        return 0.0
    found = sum(1 for h in hits[:k] if h in gt_set)
    return found / k


def _mrr(hits: list[str], ground_truth: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant hit."""
    gt_set = set(ground_truth)
    for i, h in enumerate(hits):
        if h in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def _hit_rate(hits: list[str], ground_truth: list[str], k: int) -> float:
    """Hit@K: 1 if any ground_truth in top-K, 0 otherwise."""
    gt_set = set(ground_truth)
    return 1.0 if any(h in gt_set for h in hits[:k]) else 0.0


# ================================================================
# Evaluation Entry Point
# ================================================================

def evaluate(
    manifest_path: str | Path,
    results_path: str | Path,
    k_values: list[int] | None = None,
    llm_judge: bool = False,
    llm_model: str = "",
    llm_base_url: str = "",
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    chunks_path: str | Path | None = None,
) -> dict:
    """Run evaluation on a retrieval results file.

    Returns evaluation report dict (same as written to evaluation_report.json).
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # Load data
    manifest = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa = json.loads(line)
                manifest[qa["id"]] = qa

    results = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    # Load chunks for failure analysis (optional)
    chunks_map = {}
    if chunks_path:
        data = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        chunks_map = {c["chunk_id"]: c for c in data.get("chunks", [])}

    # Compute per-query metrics
    query_metrics = []
    recall_sum: dict[int, float] = {k: 0.0 for k in k_values}
    precision_sum: dict[int, float] = {k: 0.0 for k in k_values}
    ndcg_sum: dict[int, float] = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    hit_sum: dict[int, float] = {k: 0.0 for k in k_values}

    for r in results:
        qid = r["query_id"]
        qa = manifest.get(qid, {})
        gt = qa.get("ground_truth_ids", r.get("ground_truth_ids", []))
        hits = [h.get("chunk_id", "") for h in r.get("hits", [])]

        qm = {
            "query_id": qid,
            "query": r.get("query", qa.get("query", "")),
            "ground_truth_ids": gt,
            "difficulty_level": qa.get("difficulty_level", "unknown"),
            "content_type": qa.get("metadata_filters", {}).get("content_type", ""),
            "generation_mode": qa.get("generation_mode", ""),
        }
        for k in k_values:
            rec = _recall(hits, gt, k)
            prec = _precision(hits, gt, k)
            nd = _ndcg(hits, gt, k)
            hit = _hit_rate(hits, gt, k)
            qm[f"recall@{k}"] = round(rec, 4)
            qm[f"precision@{k}"] = round(prec, 4)
            qm[f"ndcg@{k}"] = round(nd, 4)
            qm[f"hit@{k}"] = round(hit, 4)
            recall_sum[k] += rec
            precision_sum[k] += prec
            ndcg_sum[k] += nd
            hit_sum[k] += hit

        mrr = _mrr(hits, gt)
        qm["mrr"] = round(mrr, 4)
        mrr_sum += mrr

        query_metrics.append(qm)

    n = max(len(results), 1)

    # Aggregate metrics
    overall = {}
    for k in k_values:
        overall[f"recall@{k}"] = round(recall_sum[k] / n, 4)
        overall[f"precision@{k}"] = round(precision_sum[k] / n, 4)
        overall[f"ndcg@{k}"] = round(ndcg_sum[k] / n, 4)
        overall[f"hit@{k}"] = round(hit_sum[k] / n, 4)
    overall["mrr"] = round(mrr_sum / n, 4)

    # Confidence intervals (95%, bootstrap)
    cis = _bootstrap_ci(query_metrics, k_values, n)

    # Per-dimension breakdown
    breakdown = _dimension_breakdown(query_metrics, k_values)

    # Failure case analysis (bottom 5 by MRR)
    sorted_by_mrr = sorted(query_metrics, key=lambda q: q["mrr"])
    failures = []
    for qm in sorted_by_mrr[:5]:
        qid = qm["query_id"]
        gt_ids = qm["ground_truth_ids"]
        hit_ids = [
            h.get("chunk_id", "") for h in
            next((r.get("hits", []) for r in results if r["query_id"] == qid), [])
        ][:10]

        failure = {
            "query_id": qid,
            "query": qm["query"],
            "mrr": qm["mrr"],
            "expected_chunks": gt_ids,
            "actual_top5": hit_ids[:5],
            "analysis": _diagnose_failure(gt_ids, hit_ids, chunks_map),
        }
        failures.append(failure)

    # Optional LLM-as-Judge
    llm_scores = {}
    if llm_judge and llm_model:
        llm_scores = _llm_judge_eval(
            results, manifest, chunks_map, llm_model, llm_base_url, llm_api_key_env,
        )

    report = {
        "overall_metrics": overall,
        "confidence_intervals": cis,
        "dimension_breakdown": breakdown,
        "failure_analysis": failures,
        "query_count": n,
        "k_values": k_values,
    }
    if llm_scores:
        report["llm_judge"] = llm_scores

    return report


def _bootstrap_ci(query_metrics: list[dict], k_values: list[int], n: int) -> dict:
    """Bootstrap 95% confidence intervals for key metrics."""
    random.seed(RANDOM_SEED)
    if n < 10:
        return {"note": "too few queries for bootstrap (< 10)"}

    ci: dict[str, dict] = {}
    for k in k_values:
        for metric in ["recall", "precision", "ndcg"]:
            key = f"{metric}@{k}"
            values = [q[key] for q in query_metrics]
            means = []
            for _ in range(1000):
                sample = random.choices(values, k=n)
                means.append(statistics.mean(sample))
            means.sort()
            ci[key] = {
                "mean": round(statistics.mean(values), 4),
                "ci95_low": round(means[25], 4),
                "ci95_high": round(means[974], 4),
            }

    mrr_values = [q["mrr"] for q in query_metrics]
    mrr_means = []
    for _ in range(1000):
        sample = random.choices(mrr_values, k=n)
        mrr_means.append(statistics.mean(sample))
    mrr_means.sort()
    ci["mrr"] = {
        "mean": round(statistics.mean(mrr_values), 4),
        "ci95_low": round(mrr_means[25], 4),
        "ci95_high": round(mrr_means[974], 4),
    }
    return ci


def _dimension_breakdown(query_metrics: list[dict], k_values: list[int]) -> list[dict]:
    """Per-dimension metric breakdowns."""
    dims = ["difficulty_level", "content_type", "generation_mode"]
    breakdowns = []

    for dim in dims:
        groups: dict[str, list[dict]] = {}
        for q in query_metrics:
            val = q.get(dim, "unknown") or "unknown"
            groups.setdefault(val, []).append(q)

        for val, qs in sorted(groups.items()):
            entry = {"dimension": dim, "value": val, "count": len(qs)}
            for k in k_values:
                for metric in ["recall", "precision", "ndcg", "mrr"]:
                    key = f"{metric}@{k}" if metric != "mrr" else "mrr"
                    if key in qs[0]:
                        entry[f"{metric}@{k}" if metric != "mrr" else "mrr"] = round(
                            statistics.mean(q[key] for q in qs), 4,
                        )
            breakdowns.append(entry)

    return breakdowns


def _diagnose_failure(
    gt_ids: list[str],
    hit_ids: list[str],
    chunks_map: dict,
) -> str:
    """Diagnose why retrieval failed for this query."""
    reasons = []
    for gid in gt_ids:
        if gid not in hit_ids:
            # Check if chunk exists
            gc = chunks_map.get(gid)
            if gc:
                ct = gc.get("content_type", "body")
                tc = gc.get("token_count", 0)
                reasons.append(
                    f"Chunk {gid} (type={ct}, tokens={tc}) not in top results — "
                    f"possible embedding mismatch or content too specific"
                )
            else:
                reasons.append(f"Chunk {gid} not found in corpus")
    if not reasons:
        reasons.append("All ground-truth chunks retrieved but MRR low — ranking issue")
    return "; ".join(reasons)


def _extract_score(text: str) -> int | None:
    """Extract a 1-5 integer score from LLM output, robust to formatting variations."""
    stripped = text.strip()
    # Try bare integer first
    if stripped.isdigit() and 1 <= int(stripped) <= 5:
        return int(stripped)
    # Try regex: find the first 1-5 digit that looks like a score
    m = re.search(r'\b([1-5])\b', stripped)
    if m:
        return int(m.group(1))
    return None


def _llm_judge_eval(
    results: list[dict],
    manifest: dict,
    chunks_map: dict,
    llm_model: str,
    llm_base_url: str,
    llm_api_key_env: str,
    sample_size: int = 10,
) -> dict:
    """LLM-as-Judge for top-1 result: context precision and faithfulness.

    ponytail: sample up to `sample_size` queries to avoid excessive API cost.
    """
    import os
    from openai import OpenAI

    api_key = os.getenv(llm_api_key_env, "")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv(llm_api_key_env, "")
        except ImportError:
            pass

    client = OpenAI(api_key=api_key, base_url=llm_base_url or None)

    # Sample queries
    sample = results[:]
    random.seed(RANDOM_SEED)
    if len(sample) > sample_size:
        sample = random.sample(sample, sample_size)

    precision_scores = []
    faithfulness_scores = []
    _dropped_precision = 0
    _dropped_faithfulness = 0

    for r in sample:
        qa = manifest.get(r["query_id"], {})
        query = r.get("query", qa.get("query", ""))
        top1 = r["hits"][0] if r.get("hits") else None
        if not top1:
            continue

        cid = top1.get("chunk_id", "")
        chunk = chunks_map.get(cid, {})
        content = chunk.get("content", "")[:2000]

        # Context precision: does the retrieved chunk help answer the query?
        cp_prompt = (
            "Rate whether the following text chunk contains information relevant "
            "to answering the query. Give a score from 1 (irrelevant) to 5 (perfectly relevant). "
            "Output ONLY the integer score.\n\n"
            f"Query: {query}\n\nChunk: {content}\n\nScore (1-5):"
        )
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": cp_prompt}],
                temperature=0,
                max_tokens=10,
            )
            score = _extract_score(resp.choices[0].message.content)
            if score is not None:
                precision_scores.append(score / 5.0)
            else:
                _dropped_precision += 1
        except Exception:
            _dropped_precision += 1

        # Faithfulness: is the chunk content factually consistent?
        faith_prompt = (
            "Rate whether the following text chunk from an academic paper is internally "
            "coherent and factually consistent (not gibberish or garbled text). "
            "Give a score from 1 (incoherent/garbled) to 5 (perfectly coherent). "
            "Output ONLY the integer score.\n\n"
            f"Chunk: {content}\n\nScore (1-5):"
        )
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": faith_prompt}],
                temperature=0,
                max_tokens=10,
            )
            score = _extract_score(resp.choices[0].message.content)
            if score is not None:
                faithfulness_scores.append(score / 5.0)
            else:
                _dropped_faithfulness += 1
        except Exception:
            _dropped_faithfulness += 1

    # Warn if drop rate exceeds 10%
    total_tried = len(precision_scores) + _dropped_precision
    if total_tried > 0 and _dropped_precision / total_tried > 0.1:
        print(f"  [LLM-JUDGE] WARNING: {_dropped_precision}/{total_tried} context precision scores unparseable")
    total_tried = len(faithfulness_scores) + _dropped_faithfulness
    if total_tried > 0 and _dropped_faithfulness / total_tried > 0.1:
        print(f"  [LLM-JUDGE] WARNING: {_dropped_faithfulness}/{total_tried} faithfulness scores unparseable")

    return {
        "context_precision": round(statistics.mean(precision_scores), 4) if precision_scores else None,
        "faithfulness": round(statistics.mean(faithfulness_scores), 4) if faithfulness_scores else None,
        "sample_size": len(precision_scores),
        "dropped_scores": {
            "precision": _dropped_precision,
            "faithfulness": _dropped_faithfulness,
        },
    }
