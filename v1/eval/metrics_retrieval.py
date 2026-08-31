"""
metrics_retrieval.py — 检索质量评测指标
==================
纯算法指标，不依赖 LLM。

指标列表：
  Hit Rate@K   — top-K 中是否有相关文档（覆盖率）
  MRR@K        — 第一个相关文档的倒数排名（平均倒数排名）
  Precision@K  — top-K 中相关文档的占比
  Recall@K     — top-K 中相关文档数 / 总相关文档数
  NDCG@K       — 归一化折损累计增益（考虑排名位置的权重）

相关性判定：
  检索到的文档的 metadata["filename"] 与 ground-truth 文件名做规范化子串匹配。
"""

import math
import unicodedata

from langchain_core.documents import Document


def _normalize_filename(name: str) -> str:
    """Unicode NFC 规范化 + 小写 + 去空白。"""
    return unicodedata.normalize("NFC", name).strip().lower()


def is_relevant(doc: Document, gt_sources: list[str]) -> bool:
    """单个文档是否与 ground-truth 来源匹配。

    匹配规则：文档的 metadata["filename"] 包含 ground-truth 来源文件名
    （子串匹配）。规范化后比较，容忍文件名编码差异。

    参数:
        doc:        检索到的文档
        gt_sources: 已规范化的 ground-truth 文件名列表

    返回:
        bool: True 表示文档与至少一个 ground-truth 来源相关
    """
    doc_filename = _normalize_filename(doc.metadata.get("filename", ""))
    if not doc_filename:
        return False

    for gt_name in gt_sources:
        if gt_name in doc_filename:
            return True
    return False


# ---------------------------------------------------------------------------
#  单查询指标
# ---------------------------------------------------------------------------

def _relevance_labels(
    retrieved: list[Document],
    gt_sources: list[str],
) -> list[int]:
    """将 top-K 文档列表转为 0/1 相关性标签列表。"""
    return [1 if is_relevant(doc, gt_sources) else 0 for doc in retrieved]


def _hit_at_k(retrieved: list[Document], gt_sources: list[str], k: int) -> int:
    """top-K 中是否有相关文档（0 或 1）。"""
    labels = _relevance_labels(retrieved[:k], gt_sources)
    return 1 if sum(labels) > 0 else 0


def _mrr_at_k(retrieved: list[Document], gt_sources: list[str], k: int) -> float:
    """第一个相关文档的倒数排名。"""
    labels = _relevance_labels(retrieved[:k], gt_sources)
    for rank, label in enumerate(labels, 1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def _precision_at_k(retrieved: list[Document], gt_sources: list[str], k: int) -> float:
    """top-K 中相关文档占比。"""
    if k == 0:
        return 0.0
    labels = _relevance_labels(retrieved[:k], gt_sources)
    return sum(labels) / k


def _recall_at_k(
    retrieved: list[Document],
    gt_sources: list[str],
    total_relevant: int,
    k: int,
) -> float:
    """top-K 中相关文档数 / 该查询的总相关文档数。"""
    if total_relevant == 0:
        return 0.0
    labels = _relevance_labels(retrieved[:k], gt_sources)
    return min(1.0, sum(labels) / total_relevant)


def _ndcg_at_k(retrieved: list[Document], gt_sources: list[str], k: int) -> float:
    """归一化折损累计增益（使用二值相关性 0/1）。"""
    labels = _relevance_labels(retrieved[:k], gt_sources)

    # DCG
    dcg = 0.0
    for i, rel in enumerate(labels):
        if rel > 0:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 log₂(rank+1)，rank 从 1 开始

    # IDCG（理想排序：所有相关 doc 都排在最前面）
    num_rel = min(sum(labels), k)
    idcg = 0.0
    for i in range(num_rel):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
#  批量指标聚合
# ---------------------------------------------------------------------------

def hit_rate_at_k(
    retrieved_all: list[list[Document]],
    gt_all: list[list[str]],
    k: int = 5,
) -> float:
    """Hit Rate@K — 至少命中一次的比例。"""
    if not retrieved_all:
        return 0.0
    hits = sum(_hit_at_k(r, g, k) for r, g in zip(retrieved_all, gt_all))
    return hits / len(retrieved_all)


def mrr(
    retrieved_all: list[list[Document]],
    gt_all: list[list[str]],
    k: int = 5,
) -> float:
    """MRR@K — 平均倒数排名。"""
    if not retrieved_all:
        return 0.0
    total = sum(_mrr_at_k(r, g, k) for r, g in zip(retrieved_all, gt_all))
    return total / len(retrieved_all)


def precision_at_k(
    retrieved_all: list[list[Document]],
    gt_all: list[list[str]],
    k: int = 5,
) -> float:
    """Precision@K — 平均精确率。"""
    if not retrieved_all:
        return 0.0
    total = sum(_precision_at_k(r, g, k) for r, g in zip(retrieved_all, gt_all))
    return total / len(retrieved_all)


def recall_at_k(
    retrieved_all: list[list[Document]],
    gt_all: list[list[str]],
    total_relevant_per_query: list[int],
    k: int = 5,
) -> float:
    """Recall@K — 平均召回率。"""
    if not retrieved_all:
        return 0.0
    total = 0.0
    for r, g, tr in zip(retrieved_all, gt_all, total_relevant_per_query):
        total += _recall_at_k(r, g, tr, k)
    return total / len(retrieved_all)


def ndcg_at_k(
    retrieved_all: list[list[Document]],
    gt_all: list[list[str]],
    k: int = 5,
) -> float:
    """NDCG@K — 平均归一化折损累计增益。"""
    if not retrieved_all:
        return 0.0
    total = sum(_ndcg_at_k(r, g, k) for r, g in zip(retrieved_all, gt_all))
    return total / len(retrieved_all)


# ---------------------------------------------------------------------------
#  便捷聚合函数
# ---------------------------------------------------------------------------

def compute_all_retrieval_metrics(
    retrieved_all: list[list[Document]],
    gt_all: list[list[str]],
    k_values: list[int] | None = None,
) -> dict:
    """计算所有检索指标。

    参数:
        retrieved_all: 每个查询检索到的文档列表
        gt_all:        每个查询的 ground-truth 来源文件名列表
        k_values:      要计算的 K 值列表（默认 [1, 3, 5, 10]）

    返回:
        dict: 如 {"hit_rate@3": 0.9, "mrr@5": 0.83, "precision@5": 0.68, ...}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # 计算每个查询的总相关文档数（用于 Recall）
    total_relevant_per_query = [len(gt) for gt in gt_all]

    metrics: dict = {}
    for k in k_values:
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_all, gt_all, k)
        metrics[f"mrr@{k}"] = mrr(retrieved_all, gt_all, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_all, gt_all, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_all, gt_all, total_relevant_per_query, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_all, gt_all, k)

    return metrics


def compute_per_query_retrieval(
    retrieved: list[Document],
    gt_sources: list[str],
    k: int = 5,
) -> dict:
    """单查询的检索指标（用于报告中的 per_query 详情）。

    返回:
        dict: {"hit": bool, "rank": int or None, "precision": float, "recall": float, "ndcg": float}
    """
    labels = _relevance_labels(retrieved[:k], gt_sources)
    total_relevant = len(gt_sources)

    # 找到第一个相关文档的排名
    first_rank = None
    for rank, label in enumerate(labels, 1):
        if label == 1:
            first_rank = rank
            break

    return {
        "hit": sum(labels) > 0,
        "first_rank": first_rank,
        "precision": sum(labels) / k if k > 0 else 0.0,
        "recall": sum(labels) / total_relevant if total_relevant > 0 else 0.0,
        "ndcg": _ndcg_at_k(retrieved, gt_sources, k),
    }
