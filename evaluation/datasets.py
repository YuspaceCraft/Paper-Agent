"""
datasets.py — 评测集加载 / 版本化 / 去重 / 注册表。

评测集格式（与 retrieval_orchestrator.eval_manifest 兼容）：
  JSONL，每条 {id, query, ground_truth_ids[], metadata_filters{}, difficulty_level,
               generation_mode}
-dataset 版本：manifest 缺 dataset_id 时按 sha1(query 序列) 隐式算；注册表
  eval_output/datasets/registry.json 记录显式版本号与来源/难度/review 状态。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

_REGISTRY_PATH = PROJECT_ROOT / "eval_output" / "datasets" / "registry.json"


def _norm_query(q: str) -> str:
    return re.sub(r"[\W_]+", "", q.casefold())


def retrieval_enabled(qa: dict) -> bool:
    """Whether retrieval metrics apply to this task.

    Explicit ``evaluation.retrieval`` wins. Legacy manifests fall back to the
    presence of ground-truth IDs, preserving their previous behavior.
    """
    evaluation = qa.get("evaluation")
    if isinstance(evaluation, dict) and "retrieval" in evaluation:
        return bool(evaluation.get("retrieval"))
    return bool(qa.get("ground_truth_ids"))


def load_manifest(path: str | Path) -> list[dict]:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    qas: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qas.append(json.loads(line))
    return qas


def validate_ground_truth_coverage(
    qas: list[dict],
    rag_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check that manifest ground truths exist in the current RAG corpus.

    A stale manifest otherwise produces a deterministic all-zero retrieval
    score regardless of model or query-rewrite quality.
    """
    path = Path(rag_path) if rag_path else (
        PROJECT_ROOT / "eval_output" / "all_rag_chunks.json"
    )
    if not path.exists():
        return {
            "available": False,
            "rag_path": str(path),
            "error": "rag_chunks_not_found",
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "available": False,
            "rag_path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }

    chunk_ids = {
        str(chunk.get("chunk_id", ""))
        for chunk in payload.get("chunks", [])
        if chunk.get("chunk_id")
    }
    retrieval_qas = [qa for qa in qas if retrieval_enabled(qa)]
    ground_truth = {
        str(cid)
        for qa in retrieval_qas
        for cid in qa.get("ground_truth_ids", [])
        if cid
    }
    matched = ground_truth & chunk_ids
    missing = ground_truth - chunk_ids
    stale_query_ids = [
        str(qa.get("id") or f"qa_{index}")
        for index, qa in enumerate(qas, start=1)
        if retrieval_enabled(qa)
        and qa.get("ground_truth_ids")
        and not (set(map(str, qa["ground_truth_ids"])) & chunk_ids)
    ]

    return {
        "available": True,
        "rag_path": str(path),
        "query_count": len(qas),
        "retrieval_query_count": len(retrieval_qas),
        "non_retrieval_query_count": len(qas) - len(retrieval_qas),
        "chunk_count": len(chunk_ids),
        "ground_truth_count": len(ground_truth),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "coverage": round(len(matched) / max(len(ground_truth), 1), 4),
        "stale_query_count": len(stale_query_ids),
        "stale_query_ids": stale_query_ids[:20],
    }


def deduplicate(qas: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for qa in qas:
        key = _norm_query(qa.get("query", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(qa)
    return out


def dataset_id_for(qas: list[dict], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    h = hashlib.sha1(
        "\x1f".join(qa.get("query", "") for qa in qas).encode("utf-8")).hexdigest()[:12]
    return f"sha1-{h}"


def compute_paper_set(qas: list[dict]) -> str:
    """paper_set 指纹：从 GT chunk_id 前缀（paper__chunk_n）推导涉及论文。"""
    papers = set()
    for qa in qas:
        for cid in qa.get("ground_truth_ids", []):
            p = str(cid).rsplit("__chunk_", 1)[0] if "__chunk_" in str(cid) else str(cid)
            if p:
                papers.add(p)
    h = hashlib.md5("\x1f".join(sorted(papers)).encode("utf-8")).hexdigest()[:8]
    return h


def registry_update(dataset_id: str, **meta) -> dict:
    reg = {}
    if _REGISTRY_PATH.exists():
        try:
            reg = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            reg = {}
    reg[dataset_id] = {
        "created_at": meta.pop("created_at", None) or __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        **meta,
    }
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    return reg[dataset_id]


def summarize(qas: list[dict]) -> dict[str, Any]:
    """评测集概览（难度 mix / 模式 / 覆盖论文数）。"""
    task_types = sorted({
        str(qa.get("task_type") or "legacy_retrieval") for qa in qas})
    task_lengths = sorted({
        str(qa.get("task_length") or "unspecified") for qa in qas})
    return {
        "count": len(qas),
        "difficulty_mix": {
            v: sum(1 for q in qas if q.get("difficulty_level") == v)
            for v in ("easy", "medium", "hard")
        },
        "task_type_mix": {
            task_type: sum(
                1 for qa in qas
                if str(qa.get("task_type") or "legacy_retrieval") == task_type
            )
            for task_type in task_types
        },
        "task_length_mix": {
            task_length: sum(
                1 for qa in qas
                if str(qa.get("task_length") or "unspecified") == task_length
            )
            for task_length in task_lengths
        },
        "retrieval_query_count": sum(1 for qa in qas if retrieval_enabled(qa)),
        "non_retrieval_query_count": sum(
            1 for qa in qas if not retrieval_enabled(qa)),
        "generation_modes": sorted({
            q.get("generation_mode", "unknown") for q in qas}),
        "paper_set": compute_paper_set(qas),
        "queried_papers": sorted({
            str(cid).rsplit("__chunk_", 1)[0] if "__chunk_" in str(cid) else str(cid)
            for qa in qas for cid in qa.get("ground_truth_ids", [])
            if cid}),
    }
