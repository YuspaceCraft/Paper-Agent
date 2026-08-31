"""
fusion.py — 混合检索融合策略
==============================

RRF (Reciprocal Rank Fusion): 对排名而非分数进行融合，免归一化。
Weighted: 归一化分数后加权求和。
"""
from __future__ import annotations


def rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Combine dense and sparse results via Reciprocal Rank Fusion.

    Robust to different score distributions — uses rank position, not raw scores.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        docs[cid] = r

    for rank, r in enumerate(sparse_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in docs:
            docs[cid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**docs.get(cid, {}), "chunk_id": cid, "score": s}
        for cid, s in ranked
    ]


def weighted_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = 0.7,
    top_k: int = 10,
) -> list[dict]:
    """Combine via weighted sum of normalized scores.

    ponytail: min-max normalization per result list, then weighted sum.
    """
    def _normalize(results: list[dict]) -> dict[str, float]:
        if not results:
            return {}
        scores_list = [r["score"] for r in results]
        smin, smax = min(scores_list), max(scores_list)
        if smax == smin:
            return {r["chunk_id"]: 0.5 for r in results}
        return {r["chunk_id"]: (r["score"] - smin) / (smax - smin) for r in results}

    dn = _normalize(dense_results)
    sn = _normalize(sparse_results)

    all_ids = set(dn) | set(sn)
    docs: dict[str, dict] = {}
    for r in dense_results:
        docs[r["chunk_id"]] = r
    for r in sparse_results:
        if r["chunk_id"] not in docs:
            docs[r["chunk_id"]] = r

    combined = {
        cid: dense_weight * dn.get(cid, 0) + (1 - dense_weight) * sn.get(cid, 0)
        for cid in all_ids
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"chunk_id": cid, "score": s, **docs.get(cid, {})} for cid, s in ranked]
