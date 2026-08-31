"""
metrics_generation.py — 生成质量评测指标
====================
基于 LLM-as-Judge 的生成质量评测。

三大维度：
  1. 忠实度 (Faithfulness)  — 答案是否基于上下文（检测幻觉），分数 [0, 1]
  2. 答案相关性 (Relevance) — 答案是否切题，分数 [0, 1]（已归一化）
  3. 正确性 (Correctness)   — 答案与参考答案的语义一致性，分数 [0, 1]（已归一化）

依赖：
  - eval.judge.LLMJudge — LLM 裁判
"""

from eval.judge import LLMJudge


def compute_faithfulness(
    judge: LLMJudge,
    queries: list[str],
    answers: list[str],
    contexts: list[str],
) -> list[dict]:
    """评估每条答案的忠实度。

    返回:
        list[dict]: 每条查询的忠实度结果，包含 claims 详情和 score
    """
    results = []
    for i, (q, a, ctx) in enumerate(zip(queries, answers, contexts)):
        print(f"  [GEN-EVAL] 忠实度评测 ({i + 1}/{len(queries)})...")
        result = judge.evaluate_faithfulness(q, a, ctx)
        results.append(result)
    return results


def compute_answer_relevance(
    judge: LLMJudge,
    queries: list[str],
    answers: list[str],
) -> list[dict]:
    """评估每条答案的相关性（1-5 分，自动归一化到 [0, 1]）。

    返回:
        list[dict]: 每条查询的相关性结果
    """
    results = []
    for i, (q, a) in enumerate(zip(queries, answers)):
        print(f"  [GEN-EVAL] 相关性评测 ({i + 1}/{len(queries)})...")
        result = judge.evaluate_answer_relevance(q, a)
        results.append(result)
    return results


def compute_correctness(
    judge: LLMJudge,
    queries: list[str],
    answers: list[str],
    references: list[str],
) -> list[dict]:
    """评估每条答案的正确性（1-5 分，自动归一化到 [0, 1]）。

    返回:
        list[dict]: 每条查询的正确性结果
    """
    results = []
    for i, (q, a, ref) in enumerate(zip(queries, answers, references)):
        print(f"  [GEN-EVAL] 正确性评测 ({i + 1}/{len(queries)})...")
        result = judge.evaluate_correctness(q, a, ref)
        results.append(result)
    return results


def compute_all_generation_metrics(
    judge: LLMJudge,
    queries: list[str],
    answers: list[str],
    contexts: list[str],
    references: list[str],
) -> dict:
    """计算所有生成质量指标。

    参数:
        judge:      LLMJudge 实例
        queries:    用户问题列表
        answers:    LLM 生成的答案列表
        contexts:   检索到的上下文列表（拼接后的文档内容）
        references: 参考答案列表

    返回:
        dict: {
            "faithfulness_avg": 0.85,
            "faithfulness_per_query": [...],
            "answer_relevance_avg": 0.80,
            "answer_relevance_per_query": [...],
            "correctness_avg": 0.75,
            "correctness_per_query": [...],
        }
    """
    print(f"[GEN-EVAL] [START] 开始生成质量评测 ({len(queries)} 条查询)")

    # 忠实度
    faithfulness_results = compute_faithfulness(judge, queries, answers, contexts)
    faithfulness_avg = sum(r.get("score", 0.0) for r in faithfulness_results) / max(len(faithfulness_results), 1)

    # 答案相关性
    relevance_results = compute_answer_relevance(judge, queries, answers)
    relevance_avg = sum(r.get("score_normalized", 0.0) for r in relevance_results) / max(len(relevance_results), 1)
    relevance_raw_avg = sum(r.get("score", 1) for r in relevance_results) / max(len(relevance_results), 1)

    # 正确性（只对有 reference_answer 的查询评测）
    correctness_results = compute_correctness(judge, queries, answers, references)
    correctness_avg = sum(r.get("score_normalized", 0.0) for r in correctness_results) / max(len(correctness_results), 1)
    correctness_raw_avg = sum(r.get("score", 1) for r in correctness_results) / max(len(correctness_results), 1)

    print(f"[GEN-EVAL] [OK] 生成质量评测完成: "
          f"忠实度={faithfulness_avg:.3f}, "
          f"相关性={relevance_raw_avg:.1f}/5 ({relevance_avg:.3f}), "
          f"正确性={correctness_raw_avg:.1f}/5 ({correctness_avg:.3f})")

    return {
        "faithfulness_avg": faithfulness_avg,
        "faithfulness_per_query": faithfulness_results,
        "answer_relevance_avg": relevance_avg,
        "answer_relevance_raw_avg": relevance_raw_avg,
        "answer_relevance_per_query": relevance_results,
        "correctness_avg": correctness_avg,
        "correctness_raw_avg": correctness_raw_avg,
        "correctness_per_query": correctness_results,
    }
