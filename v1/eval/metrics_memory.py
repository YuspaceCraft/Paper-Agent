"""
metrics_memory.py — 记忆质量评测指标
=================
覆盖长期记忆(LTM)的四大评测维度：

  1. 提取质量 (Extraction Quality)  — LLM-judge：提取的事实是否准确、非冗余
  2. 检索质量 (Retrieval Quality)   — 算法指标：Precision@K / Recall@K / MRR@K
  3. 记忆影响 (Memory Impact)       — LLM-judge：LTM 是否改善了答案质量
  4. 一致性   (Consistency)          — 算法指标：记忆是否存在语义重复或矛盾

依赖：
  - eval.judge.LLMJudge — 提取质量和记忆影响的 LLM 裁判
  - src.embedder — 一致性检测的嵌入模型（可选）
"""

import math
import numpy as np

from eval.judge import LLMJudge


# ================================================================
#  1. 记忆提取质量 (Extraction Quality) — LLM-judge
# ================================================================

def compute_memory_extraction_quality(
    judge: LLMJudge,
    conversations: list[str],
    extracted_facts_all: list[list[str]],
) -> dict:
    """评估从对话中提取的 LTM 事实的准确性。

    参数:
        judge:               LLMJudge 实例
        conversations:       每轮对话的文本记录
        extracted_facts_all: 每轮对话提取的 LTM 事实列表

    返回:
        dict: {
            "extraction_accuracy_avg": 0.85,
            "per_conversation": [...],
            "missed_important_facts_all": [...],
        }
    """
    results = []
    accuracies = []
    all_missed = []

    total = len(conversations)
    for i, (conv, facts) in enumerate(zip(conversations, extracted_facts_all)):
        print(f"  [MEM-EVAL] 提取质量评测 ({i + 1}/{total})...")
        result = judge.evaluate_memory_extraction(conv, facts)
        results.append(result)
        accuracies.append(result.get("overall_accuracy", 0.0))
        all_missed.extend(result.get("missed_important_facts", []))

    avg_accuracy = sum(accuracies) / max(len(accuracies), 1)

    print(f"[MEM-EVAL] [OK] 提取质量评测完成: "
          f"平均准确性={avg_accuracy:.3f}")

    return {
        "extraction_accuracy_avg": avg_accuracy,
        "per_conversation": results,
        "missed_important_facts_all": all_missed,
    }


# ================================================================
#  2. 记忆检索质量 (Retrieval Quality) — 算法指标
# ================================================================

def _is_ltm_relevant(retrieved_fact: str, gt_facts: list[str]) -> bool:
    """判断检索到的 LTM 事实是否与 ground-truth 匹配（子串模糊匹配）。"""
    r = retrieved_fact.strip().lower()
    for gt in gt_facts:
        g = gt.strip().lower()
        if g in r or r in g:
            return True
    return False


def _ltm_precision_at_k(
    retrieved: list[str], gt: list[str], k: int,
) -> float:
    if k == 0:
        return 0.0
    labels = [1 if _is_ltm_relevant(r, gt) else 0 for r in retrieved[:k]]
    return sum(labels) / k


def _ltm_recall_at_k(
    retrieved: list[str], gt: list[str], k: int,
) -> float:
    if not gt:
        return 0.0
    labels = [1 if _is_ltm_relevant(r, gt) else 0 for r in retrieved[:k]]
    return min(1.0, sum(labels) / len(gt))


def _ltm_mrr(retrieved: list[str], gt: list[str], k: int) -> float:
    for rank, r in enumerate(retrieved[:k], 1):
        if _is_ltm_relevant(r, gt):
            return 1.0 / rank
    return 0.0


def compute_memory_retrieval_quality(
    all_retrieved_ltm: list[list[str]],
    all_gt_ltm: list[list[str]],
    k_values: list[int] | None = None,
) -> dict:
    """计算 LTM 检索的 Precision@K / Recall@K / MRR@K。

    参数:
        all_retrieved_ltm: 每个查询检索到的 LTM 事实内容列表
        all_gt_ltm:        每个查询的 ground-truth 相关 LTM 事实列表
        k_values:          要计算的 K 值列表

    返回:
        dict: {"precision@3": 0.80, "recall@3": 0.75, "mrr@3": 0.90, ...}
    """
    if k_values is None:
        k_values = [1, 3, 5]

    if not all_retrieved_ltm:
        return {}

    n = len(all_retrieved_ltm)
    metrics = {}

    for k in k_values:
        precisions = [_ltm_precision_at_k(r, g, k) for r, g in zip(all_retrieved_ltm, all_gt_ltm)]
        recalls = [_ltm_recall_at_k(r, g, k) for r, g in zip(all_retrieved_ltm, all_gt_ltm)]
        mrrs = [_ltm_mrr(r, g, k) for r, g in zip(all_retrieved_ltm, all_gt_ltm)]

        metrics[f"ltm_precision@{k}"] = sum(precisions) / n
        metrics[f"ltm_recall@{k}"] = sum(recalls) / n
        metrics[f"ltm_mrr@{k}"] = sum(mrrs) / n

    print(f"[MEM-EVAL] [OK] 记忆检索评测完成: "
          f"P@3={metrics.get('ltm_precision@3', 0):.3f}, "
          f"R@3={metrics.get('ltm_recall@3', 0):.3f}")

    return metrics


# ================================================================
#  3. 记忆影响 (Memory Impact) — LLM-judge
# ================================================================

def compute_memory_impact(
    judge: LLMJudge,
    queries: list[str],
    answers_with_ltm: list[str],
    answers_without_ltm: list[str],
    ltm_used_all: list[str],
) -> dict:
    """评估 LTM 对答案质量的改善程度。

    参数:
        judge:              LLMJudge 实例
        queries:            用户查询列表
        answers_with_ltm:   使用 LTM 的答案列表
        answers_without_ltm: 未使用 LTM 的答案列表
        ltm_used_all:       每个查询使用的 LTM 描述

    返回:
        dict: {"impact_avg": 0.67, "impact_normalized_avg": 0.56, "per_query": [...]}
    """
    results = []
    scores = []
    normalized = []

    total = len(queries)
    for i, (q, a_with, a_without, ltm) in enumerate(
        zip(queries, answers_with_ltm, answers_without_ltm, ltm_used_all)
    ):
        print(f"  [MEM-EVAL] 记忆影响评测 ({i + 1}/{total})...")
        result = judge.evaluate_memory_impact(q, a_with, a_without, ltm)
        results.append(result)
        scores.append(result.get("impact_score", 0))
        normalized.append(result.get("impact_normalized", 0.33))

    impact_avg = sum(scores) / max(len(scores), 1)
    norm_avg = sum(normalized) / max(len(normalized), 1)

    # 统计分布
    distribution = {"worse": 0, "same": 0, "better": 0, "much_better": 0}
    for r in results:
        label = r.get("impact_label", "same")
        if label in distribution:
            distribution[label] += 1

    print(f"[MEM-EVAL] [OK] 记忆影响评测完成: "
          f"avg={impact_avg:.1f} (norm={norm_avg:.3f}), "
          f"分布={distribution}")

    return {
        "impact_avg": impact_avg,
        "impact_normalized_avg": norm_avg,
        "impact_distribution": distribution,
        "per_query": results,
    }


# ================================================================
#  4. 记忆一致性 (Consistency) — 算法指标
# ================================================================

def compute_memory_consistency(
    all_ltm_facts: list[str],
    similarity_threshold: float = 0.85,
) -> dict:
    """检测 LTM 事实之间的语义重复（不一致/冗余）。

    使用嵌入向量的余弦相似度检测近似重复的事实。
    相似度超过阈值的记为"潜在重复"。

    参数:
        all_ltm_facts:        所有 LTM 事实的内容列表
        similarity_threshold:  判定重复的余弦相似度阈值

    返回:
        dict: {
            "total_facts": int,
            "duplicate_pairs": int,
            "duplicate_details": [...],
            "consistency_score": 0.95,  # 1 - duplicate_ratio
        }
    """
    n = len(all_ltm_facts)
    if n <= 1:
        return {
            "total_facts": n,
            "duplicate_pairs": 0,
            "duplicate_details": [],
            "consistency_score": 1.0,
        }

    # 尝试加载嵌入模型
    embeddings = None
    try:
        from agent.embedder import get_embeddings
        embeddings = get_embeddings()
    except Exception:
        pass

    if embeddings is None:
        # 无嵌入模型，使用简单的文本重叠检测
        return _consistency_by_text_overlap(all_ltm_facts, threshold=0.7)

    # 嵌入所有 facts
    try:
        vectors = [embeddings.embed_query(f) for f in all_ltm_facts]
    except Exception:
        return _consistency_by_text_overlap(all_ltm_facts, threshold=0.7)

    # 计算 pairwise cosine similarity
    duplicate_pairs = []
    vec_array = np.array(vectors)
    norms = np.linalg.norm(vec_array, axis=1)

    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] == 0 or norms[j] == 0:
                continue
            cosine = float(np.dot(vec_array[i], vec_array[j]) / (norms[i] * norms[j]))
            if cosine >= similarity_threshold:
                duplicate_pairs.append({
                    "fact_a": all_ltm_facts[i][:100],
                    "fact_b": all_ltm_facts[j][:100],
                    "similarity": round(cosine, 4),
                })

    dup_count = len(duplicate_pairs)
    max_pairs = n * (n - 1) / 2
    consistency = 1.0 - (dup_count / max_pairs) if max_pairs > 0 else 1.0

    print(f"[MEM-EVAL] [OK] 一致性检测: {n} 条事实, "
          f"{dup_count} 对潜在重复, 一致性={consistency:.3f}")

    return {
        "total_facts": n,
        "duplicate_pairs": dup_count,
        "duplicate_details": duplicate_pairs,
        "consistency_score": consistency,
    }


def _consistency_by_text_overlap(
    facts: list[str], threshold: float = 0.7,
) -> dict:
    """基于文本字符重叠的简单重复检测（无需嵌入模型）。"""
    n = len(facts)
    duplicate_pairs = []
    for i in range(n):
        fi = facts[i].strip().lower()
        if not fi:
            continue
        for j in range(i + 1, n):
            fj = facts[j].strip().lower()
            if not fj:
                continue
            # Jaccard 字符级相似度
            si, sj = set(fi), set(fj)
            if not si or not sj:
                continue
            jaccard = len(si & sj) / len(si | sj)
            if jaccard >= threshold:
                duplicate_pairs.append({
                    "fact_a": facts[i][:100],
                    "fact_b": facts[j][:100],
                    "jaccard_similarity": round(jaccard, 4),
                })

    dup_count = len(duplicate_pairs)
    max_pairs = n * (n - 1) / 2
    consistency = 1.0 - (dup_count / max_pairs) if max_pairs > 0 else 1.0

    print(f"[MEM-EVAL] [OK] 一致性检测(text): {n} 条事实, "
          f"{dup_count} 对潜在重复, 一致性={consistency:.3f}")

    return {
        "total_facts": n,
        "duplicate_pairs": dup_count,
        "duplicate_details": duplicate_pairs,
        "consistency_score": consistency,
    }


# ================================================================
#  聚合函数
# ================================================================

def compute_all_memory_metrics(
    judge: LLMJudge,
    conversations: list[str],
    extracted_facts_all: list[list[str]],
    all_retrieved_ltm: list[list[str]],
    all_gt_ltm: list[list[str]],
    queries: list[str],
    answers_with_ltm: list[str],
    answers_without_ltm: list[str],
    ltm_used_all: list[str],
    all_ltm_facts: list[str],
) -> dict:
    """计算所有记忆质量指标（一站式聚合）。

    返回:
        dict: {
            "extraction": {...},
            "retrieval": {...},
            "impact": {...},
            "consistency": {...},
        }
    """
    print(f"\n[MEM-EVAL] [START] 开始记忆质量评测")

    # 提取质量
    extraction = compute_memory_extraction_quality(
        judge, conversations, extracted_facts_all,
    )

    # 检索质量
    retrieval = compute_memory_retrieval_quality(
        all_retrieved_ltm, all_gt_ltm,
    )

    # 记忆影响
    impact = compute_memory_impact(
        judge, queries, answers_with_ltm, answers_without_ltm, ltm_used_all,
    )

    # 一致性
    consistency = compute_memory_consistency(all_ltm_facts)

    print(f"[MEM-EVAL] [OK] 记忆质量评测全部完成")

    return {
        "extraction": extraction,
        "retrieval": retrieval,
        "impact": impact,
        "consistency": consistency,
    }
