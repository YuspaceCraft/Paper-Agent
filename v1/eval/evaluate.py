"""
evaluate.py — 评测编排器
==========
将各个评测模块串联成完整的评测工作流。

这是 eval/ 的"指挥中心"——它不实现具体指标，
而是按照正确的顺序调用各个模块。

三条评测管道：
  run_retrieval_evaluation()  — 检索质量评测
  run_generation_evaluation() — 生成质量评测（含检索）
  run_e2e_evaluation()        — 端到端评测（检索 + 生成 + 延迟）

重用现有 src.* 模块（不修改任何现有文件）：
  - src.embedder.get_embeddings()
  - src.store.load_vector_store() / store_exists()
  - src.retriever.create_retriever_with_rerank() / Reranker / format_retrieved_docs()
  - src.generator.create_llm() / create_rag_chain()
  - src.config.*
"""

import time
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from config import (
    TOP_K,
    RERANK_CANDIDATE_K,
    RERANK_FINAL_K,
    LLM_MODEL,
)
from agent.embedder import get_embeddings
from agent.store import load_vector_store, store_exists
from agent.retriever import (
    create_retriever,
    create_retriever_with_rerank,
    format_retrieved_docs,
)
from agent.generator import create_llm, create_rag_chain

from eval.dataset import load_dataset, EvalSample
from eval.metrics_retrieval import compute_all_retrieval_metrics, compute_per_query_retrieval
from eval.judge import LLMJudge
from eval.metrics_generation import compute_all_generation_metrics
from eval.metrics_memory import compute_all_memory_metrics
from eval.report import print_report


# ================================================================
#  管道初始化（共享组件）
# ================================================================

def _setup_components(use_rerank: bool = True):
    """初始化共享管道组件。

    返回:
        dict: {"embeddings", "vector_store", "retriever", "reranker"}
    """
    # 检查向量数据库
    if not store_exists():
        raise FileNotFoundError(
            "[ERR] 未找到向量数据库！\n"
            "请先运行文档导入: python -m src.cli ingest"
        )

    print("[EVAL] [SETUP] 正在初始化管道组件...")

    # 嵌入模型
    embeddings = get_embeddings()

    # 向量数据库
    vector_store = load_vector_store(embeddings)

    # 检索器（可选 Reranker）
    retriever = None
    reranker = None
    if use_rerank:
        retriever = create_retriever_with_rerank(
            vector_store,
            reranker=None,  # 自动创建独立的 Reranker，不与 embedding 共享
        )
    else:
        retriever = create_retriever(vector_store, k=TOP_K)

    print("[EVAL] [OK] 管道组件就绪")
    return {
        "embeddings": embeddings,
        "vector_store": vector_store,
        "retriever": retriever,
        "reranker": reranker,
    }


# ================================================================
#  1. 检索质量评测
# ================================================================

def run_retrieval_evaluation(
    dataset_path: str | Path,
    top_k: int = 5,
    use_rerank: bool = True,
    candidate_k: int | None = None,
    final_k: int | None = None,
) -> dict:
    """检索质量评测。

    流程:
      1. 加载数据集
      2. 初始化检索管道
      3. 对每条查询执行检索，收集结果
      4. 计算 Hit Rate / MRR / NDCG / Precision@K / Recall@K

    参数:
        dataset_path: JSONL 数据集路径
        top_k:        最终返回的文档数
        use_rerank:   是否使用二阶段检索（粗筛 + 精排）
        candidate_k:  粗筛候选数（默认使用 config 中的值）
        final_k:      精排后数量（默认使用 config 中的值）

    返回:
        dict: 完整评测结果，可用于 report.py 输出
    """
    dataset_path = Path(dataset_path)
    print("\n" + "=" * 60)
    print(f"  [EVAL] 检索质量评测")
    print(f"  数据集: {dataset_path.name}")
    print("=" * 60)

    # 加载数据集
    samples = load_dataset(dataset_path)
    if not samples:
        print("[EVAL] [ERR] 数据集中无有效样本")
        return {}

    # 过滤：只保留有 relevant_sources 的样本
    retrieval_samples = [s for s in samples if s.has_retrieval_gt()]
    if not retrieval_samples:
        print("[EVAL] [ERR] 数据集中无带 relevant_sources 的样本")
        return {}

    # 初始化管道
    try:
        components = _setup_components(use_rerank=use_rerank)
    except FileNotFoundError as e:
        print(f"[EVAL] [ERR] {e}")
        return {}

    retriever = components["retriever"]

    # 如果用户指定了 candidate_k / final_k，覆盖默认值
    effective_k = top_k
    c_k = candidate_k or RERANK_CANDIDATE_K
    f_k = final_k or RERANK_FINAL_K

    # 每条查询执行检索
    queries = []
    retrieved_all = []
    gt_all = []
    per_query = []
    failed = []

    total = len(retrieval_samples)
    for i, sample in enumerate(retrieval_samples):
        print(f"\n[EVAL] 检索评测 ({i + 1}/{total}): [{sample.id}] {sample.question[:60]}...")
        queries.append(sample.question)

        try:
            # 检索（二阶段或单阶段）
            docs = retriever.invoke(sample.question)[:effective_k]
            retrieved_all.append(docs)

            gt_names = sample.relevant_sources
            gt_all.append(gt_names)

            # 单查询指标
            pq = compute_per_query_retrieval(docs, gt_names, k=effective_k)
            per_query.append({
                "id": sample.id,
                "question": sample.question,
                "retrieved_sources": [
                    d.metadata.get("filename", "?") for d in docs
                ],
                "retrieval_metrics": pq,
            })
            print(f"    [OK] 检索到 {len(docs)} 个文档, "
                  f"命中: {pq['hit']}, 精确率: {pq['precision']:.3f}")

        except Exception as e:
            print(f"    [ERR] 检索失败: {e}")
            retrieved_all.append([])
            gt_all.append(sample.relevant_sources)
            failed.append({"id": sample.id, "error": str(e)})
            per_query.append({
                "id": sample.id,
                "question": sample.question,
                "retrieved_sources": [],
                "retrieval_metrics": {"hit": False, "first_rank": None, "precision": 0.0, "recall": 0.0, "ndcg": 0.0},
            })

    # 计算聚合指标
    k_values = [k for k in [1, 3, 5, 10] if k <= effective_k]
    if effective_k not in k_values:
        k_values.append(effective_k)
    retrieval_metrics = compute_all_retrieval_metrics(retrieved_all, gt_all, k_values)

    print("\n" + "-" * 40)
    print("[EVAL] [OK] 检索质量评测完成")
    for k, v in retrieval_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 按类别分层
    by_category = _compute_category_breakdown(
        retrieval_samples, retrieved_all, gt_all, effective_k
    )

    return {
        "meta": {
            "dataset": str(dataset_path),
            "num_queries": total,
            "num_success": total - len(failed),
            "num_failed": len(failed),
            "timestamp": datetime.now().isoformat(),
            "llm_model": LLM_MODEL,
            "use_rerank": use_rerank,
            "top_k": effective_k,
            "eval_type": "retrieval",
        },
        "retrieval": retrieval_metrics,
        "failed_queries": failed,
        "per_query": per_query,
        "by_category": by_category,
    }


# ================================================================
#  2. 生成质量评测
# ================================================================

def run_generation_evaluation(
    dataset_path: str | Path,
    use_rerank: bool = True,
) -> dict:
    """生成质量评测。

    流程:
      1. 加载数据集（必须有 reference_answer）
      2. 初始化检索 + 生成管道
      3. 对每条查询：检索 → 生成答案 → LLM-judge 评分

    参数:
        dataset_path: JSONL 数据集路径
        use_rerank:   是否使用二阶段检索

    返回:
        dict: 完整评测结果
    """
    dataset_path = Path(dataset_path)
    print("\n" + "=" * 60)
    print(f"  [EVAL] 生成质量评测")
    print(f"  数据集: {dataset_path.name}")
    print("=" * 60)

    # 加载数据集
    samples = load_dataset(dataset_path)
    if not samples:
        print("[EVAL] [ERR] 数据集中无有效样本")
        return {}

    # 过滤：只保留有 reference_answer 的样本
    gen_samples = [s for s in samples if s.has_generation_gt()]
    if not gen_samples:
        print("[EVAL] [ERR] 数据集中无带 reference_answer 的样本")
        return {}

    # 初始化管道
    try:
        components = _setup_components(use_rerank=use_rerank)
    except FileNotFoundError as e:
        print(f"[EVAL] [ERR] {e}")
        return {}

    retriever = components["retriever"]
    llm = create_llm()
    chain = create_rag_chain(retriever, llm)
    judge = LLMJudge(llm=llm)

    queries = []
    all_answers = []
    all_contexts = []
    all_references = []
    per_query = []
    failed = []

    total = len(gen_samples)
    for i, sample in enumerate(gen_samples):
        print(f"\n[EVAL] 生成评测 ({i + 1}/{total}): [{sample.id}] {sample.question[:60]}...")
        queries.append(sample.question)

        try:
            # 检索
            docs = retriever.invoke(sample.question)
            context = format_retrieved_docs(docs)
            all_contexts.append(context)

            # 生成
            print(f"  [GENERATE] 正在生成答案...")
            answer = chain.invoke(sample.question)
            all_answers.append(answer)
            all_references.append(sample.reference_answer)

            # 简略记录（judge 评测在批量阶段进行）
            per_query.append({
                "id": sample.id,
                "question": sample.question,
                "generated_answer": answer,
                "reference_answer": sample.reference_answer,
                "retrieved_sources": [
                    d.metadata.get("filename", "?") for d in docs
                ],
            })
            print(f"  [OK] 答案长度: {len(answer)} 字符")

        except Exception as e:
            print(f"  [ERR] 生成失败: {e}")
            all_answers.append("")
            all_contexts.append("")
            all_references.append(sample.reference_answer)
            failed.append({"id": sample.id, "error": str(e)})
            per_query.append({
                "id": sample.id,
                "question": sample.question,
                "generated_answer": "",
                "reference_answer": sample.reference_answer,
                "retrieved_sources": [],
            })

    # 批量 LLM-judge 评测
    print(f"\n{'─' * 40}")
    print(f"[EVAL] 开始 LLM-Judge 评测 ({total - len(failed)} 条有效答案)...")
    gen_metrics = compute_all_generation_metrics(
        judge,
        queries,
        all_answers,
        all_contexts,
        all_references,
    )

    # 将 per-query judge 结果合并到 per_query
    for i, pq in enumerate(per_query):
        if i < len(gen_metrics["faithfulness_per_query"]):
            pq["faithfulness"] = gen_metrics["faithfulness_per_query"][i].get("score", 0.0)
            pq["faithfulness_detail"] = gen_metrics["faithfulness_per_query"][i]
        if i < len(gen_metrics["answer_relevance_per_query"]):
            pq["relevance"] = gen_metrics["answer_relevance_per_query"][i].get("score_normalized", 0.0)
            pq["relevance_raw"] = gen_metrics["answer_relevance_per_query"][i].get("score", 1)
        if i < len(gen_metrics["correctness_per_query"]):
            pq["correctness"] = gen_metrics["correctness_per_query"][i].get("score_normalized", 0.0)
            pq["correctness_raw"] = gen_metrics["correctness_per_query"][i].get("score", 1)

    return {
        "meta": {
            "dataset": str(dataset_path),
            "num_queries": total,
            "num_success": total - len(failed),
            "num_failed": len(failed),
            "timestamp": datetime.now().isoformat(),
            "llm_model": LLM_MODEL,
            "use_rerank": use_rerank,
            "eval_type": "generation",
        },
        "generation": {
            "faithfulness_avg": gen_metrics["faithfulness_avg"],
            "answer_relevance_avg": gen_metrics["answer_relevance_avg"],
            "answer_relevance_raw_avg": gen_metrics["answer_relevance_raw_avg"],
            "correctness_avg": gen_metrics["correctness_avg"],
            "correctness_raw_avg": gen_metrics["correctness_raw_avg"],
        },
        "failed_queries": failed,
        "per_query": per_query,
    }


# ================================================================
#  3. 端到端评测
# ================================================================

def run_e2e_evaluation(
    dataset_path: str | Path,
    use_rerank: bool = True,
    top_k: int = 5,
) -> dict:
    """端到端评测（检索 + 生成 + 延迟）。

    流程:
      1. 加载完整数据集
      2. 初始化管道
      3. 对每条查询：
         a. 计时检索
         b. 计时生成
         c. 计算检索指标
         d. LLM-judge 评测生成质量
      4. 聚合所有指标

    参数:
        dataset_path: JSONL 数据集路径
        use_rerank:   是否使用二阶段检索
        top_k:        最终返回的文档数

    返回:
        dict: 完整评测结果（检索 + 生成 + 延迟）
    """
    dataset_path = Path(dataset_path)
    print("\n" + "=" * 60)
    print(f"  [EVAL] 端到端评测")
    print(f"  数据集: {dataset_path.name}")
    print("=" * 60)

    # 加载数据集
    samples = load_dataset(dataset_path)
    if not samples:
        print("[EVAL] [ERR] 数据集中无有效样本")
        return {}

    # 初始化管道
    try:
        components = _setup_components(use_rerank=use_rerank)
    except FileNotFoundError as e:
        print(f"[EVAL] [ERR] {e}")
        return {}

    retriever = components["retriever"]
    llm = create_llm()
    chain = create_rag_chain(retriever, llm)
    judge = LLMJudge(llm=llm)

    # 收集各阶段数据
    queries = []
    retrieved_all = []
    gt_all = []
    all_answers = []
    all_contexts = []
    all_references = []
    retrieval_times = []
    generation_times = []
    per_query = []
    failed = []

    total = len(samples)
    for i, sample in enumerate(samples):
        print(f"\n[EVAL] E2E ({i + 1}/{total}): [{sample.id}] {sample.question[:60]}...")
        queries.append(sample.question)

        try:
            # --- Step 1: 检索 ---
            t0 = time.perf_counter()
            docs = retriever.invoke(sample.question)[:top_k]
            t1 = time.perf_counter()
            retrieval_s = t1 - t0
            retrieval_times.append(retrieval_s)

            retrieved_all.append(docs)
            context = format_retrieved_docs(docs)
            all_contexts.append(context)

            gt_names = sample.relevant_sources if sample.has_retrieval_gt() else []
            gt_all.append(gt_names)

            # --- Step 2: 生成 ---
            t2 = time.perf_counter()
            answer = chain.invoke(sample.question)
            t3 = time.perf_counter()
            generation_s = t3 - t2
            generation_times.append(generation_s)

            all_answers.append(answer)
            all_references.append(sample.reference_answer if sample.has_generation_gt() else "")

            # --- Step 3: 单查询指标 ---
            retrieval_pq = compute_per_query_retrieval(docs, gt_names, k=top_k) if gt_names else {}

            per_query.append({
                "id": sample.id,
                "question": sample.question,
                "retrieved_sources": [
                    d.metadata.get("filename", "?") for d in docs
                ],
                "generated_answer": answer,
                "reference_answer": sample.reference_answer,
                "retrieval_metrics": retrieval_pq,
                "latency_s": {
                    "retrieval": round(retrieval_s, 3),
                    "generation": round(generation_s, 3),
                    "total": round(retrieval_s + generation_s, 3),
                },
            })

            print(f"    检索: {retrieval_s:.2f}s | 生成: {generation_s:.2f}s | "
                  f"总计: {retrieval_s + generation_s:.2f}s")

        except Exception as e:
            print(f"  [ERR] E2E 失败: {e}")
            retrieved_all.append([])
            gt_all.append(sample.relevant_sources if sample.has_retrieval_gt() else [])
            all_answers.append("")
            all_contexts.append("")
            all_references.append(sample.reference_answer if sample.has_generation_gt() else "")
            failed.append({"id": sample.id, "error": str(e)})
            per_query.append({
                "id": sample.id,
                "question": sample.question,
                "retrieved_sources": [],
                "generated_answer": "",
                "reference_answer": sample.reference_answer,
                "retrieval_metrics": {},
                "latency_s": {"retrieval": 0, "generation": 0, "total": 0},
            })

    # --- 聚合检索指标 ---
    retrieval_metrics = {}
    if any(gt_all):
        k_values = [k for k in [1, 3, 5, 10] if k <= top_k]
        if top_k not in k_values:
            k_values.append(top_k)
        retrieval_metrics = compute_all_retrieval_metrics(retrieved_all, gt_all, k_values)

    # --- LLM-judge 评测生成质量 ---
    gen_metrics = {}
    has_generation = any(all_answers) and any(all_references)
    if has_generation:
        print(f"\n{'─' * 40}")
        print(f"[EVAL] 开始 LLM-Judge 评测...")
        gen_metrics = compute_all_generation_metrics(
            judge,
            queries,
            all_answers,
            all_contexts,
            all_references,
        )

        # 合并 judge 结果
        for i, pq in enumerate(per_query):
            if i < len(gen_metrics.get("faithfulness_per_query", [])):
                pq["faithfulness"] = gen_metrics["faithfulness_per_query"][i].get("score", 0.0)
                pq["faithfulness_detail"] = gen_metrics["faithfulness_per_query"][i]
            if i < len(gen_metrics.get("answer_relevance_per_query", [])):
                pq["relevance"] = gen_metrics["answer_relevance_per_query"][i].get("score_normalized", 0.0)
                pq["relevance_raw"] = gen_metrics["answer_relevance_per_query"][i].get("score", 1)
            if i < len(gen_metrics.get("correctness_per_query", [])):
                pq["correctness"] = gen_metrics["correctness_per_query"][i].get("score_normalized", 0.0)
                pq["correctness_raw"] = gen_metrics["correctness_per_query"][i].get("score", 1)

    # --- 延迟统计 ---
    latency_stats = {}
    if retrieval_times:
        latency_stats = {
            "avg_retrieval_s": sum(retrieval_times) / len(retrieval_times),
            "avg_generation_s": sum(generation_times) / max(len(generation_times), 1),
            "avg_total_s": (sum(retrieval_times) + sum(generation_times)) / max(len(retrieval_times), 1),
            "total_s": sum(retrieval_times) + sum(generation_times),
        }

    # --- 按类别分层 ---
    by_category = {}
    if any(gt_all):
        by_category = _compute_category_breakdown(
            samples, retrieved_all, gt_all, top_k
        )

    print("\n" + "=" * 60)
    print(f"[EVAL] [OK] 端到端评测完成")

    return {
        "meta": {
            "dataset": str(dataset_path),
            "num_queries": total,
            "num_success": total - len(failed),
            "num_failed": len(failed),
            "timestamp": datetime.now().isoformat(),
            "llm_model": LLM_MODEL,
            "use_rerank": use_rerank,
            "top_k": top_k,
            "eval_type": "e2e",
        },
        "retrieval": retrieval_metrics,
        "generation": {
            "faithfulness_avg": gen_metrics.get("faithfulness_avg"),
            "answer_relevance_avg": gen_metrics.get("answer_relevance_avg"),
            "answer_relevance_raw_avg": gen_metrics.get("answer_relevance_raw_avg"),
            "correctness_avg": gen_metrics.get("correctness_avg"),
            "correctness_raw_avg": gen_metrics.get("correctness_raw_avg"),
        },
        "latency": latency_stats,
        "failed_queries": failed,
        "per_query": per_query,
        "by_category": by_category,
    }


# ================================================================
#  辅助函数
# ================================================================

def _compute_category_breakdown(
    samples: list[EvalSample],
    retrieved_all: list[list[Document]],
    gt_all: list[list[str]],
    top_k: int,
) -> dict:
    """按 metadata.category 分层统计检索指标。"""
    from collections import defaultdict

    categories: dict[str, dict] = defaultdict(lambda: {
        "retrieved": [],
        "gt": [],
        "count": 0,
    })

    for sample, docs, gt in zip(samples, retrieved_all, gt_all):
        cat = sample.metadata.get("category", "other") if sample.metadata else "other"
        categories[cat]["retrieved"].append(docs)
        categories[cat]["gt"].append(gt)
        categories[cat]["count"] += 1

    result = {}
    for cat, data in categories.items():
        if data["count"] > 0:
            k_values = [k for k in [1, 3, 5] if k <= top_k]
            result[cat] = {
                "count": data["count"],
                "metrics": compute_all_retrieval_metrics(
                    data["retrieved"], data["gt"], k_values
                ),
            }

    return result


# ================================================================
#  4. 记忆质量评测
# ================================================================

def run_memory_evaluation(
    dataset_path: str | Path,
    use_rerank: bool = True,
) -> dict:
    """记忆质量评测。

    流程:
      1. 加载数据集（含 conversation / expected_memories / query）
      2. 初始化管道
      3. 对每条样本：
         a. 模拟多轮对话 → 提取 LTM 事实
         b. 对 follow-up query：检索 LTM + 生成答案
         c. 生成无 LTM 的对照答案
      4. 计算四大维度的记忆质量指标

    参数:
        dataset_path: JSONL 数据集路径
        use_rerank:   是否使用二阶段检索

    返回:
        dict: 完整评测结果
    """
    dataset_path = Path(dataset_path)
    print("\n" + "=" * 60)
    print(f"  [EVAL] 记忆质量评测")
    print(f"  数据集: {dataset_path.name}")
    print("=" * 60)

    # 加载数据集
    samples = load_dataset(dataset_path)
    if not samples:
        print("[EVAL] [ERR] 数据集中无有效样本")
        return {}

    # 初始化管道
    try:
        components = _setup_components(use_rerank=use_rerank)
    except FileNotFoundError as e:
        print(f"[EVAL] [ERR] {e}")
        return {}

    retriever = components["retriever"]
    llm = create_llm()
    judge = LLMJudge(llm=llm)

    # 收集各阶段数据
    conversations = []
    extracted_facts_all = []
    all_retrieved_ltm = []
    all_gt_ltm = []
    queries = []
    answers_with_ltm = []
    answers_without_ltm = []
    ltm_used_all = []
    all_ltm_facts_flat = []
    per_query = []
    failed = []

    # 没有 LTM 的简单 RAG 链（用于对照）
    chain_without_ltm = create_rag_chain(retriever, llm)

    total = len(samples)
    for i, sample in enumerate(samples):
        print(f"\n[EVAL] 记忆评测 ({i + 1}/{total}): [{sample.id}]")

        try:
            # --- Step 1: 从 conversation 提取 LTM ---
            conv_text = _format_conversation(sample)
            conversations.append(conv_text)

            # 模拟逐轮提取 LTM
            turns = _parse_conversation_turns(sample)
            extracted = []
            from agent.memory import HybridMemory
            from agent.generator import create_hybrid_rag_chain

            hm = HybridMemory(stm_window_size=5)
            for turn_num, (user_msg, ai_msg) in enumerate(turns):
                facts = hm.long_term.extract_from_exchange(
                    user_msg, ai_msg, llm, turn_number=turn_num,
                )
                extracted.extend(facts)
            extracted_facts_all.append(extracted)
            all_ltm_facts_flat.extend(extracted)

            # --- Step 2: 执行 follow-up query ---
            query = sample.question
            queries.append(query)
            gt_ltm = [s for s in sample.relevant_sources if s]  # 复用 relevant_sources 存 GT LTM
            all_gt_ltm.append(gt_ltm)

            # 检索 LTM 相关记忆
            relevant_ltm = hm.long_term.retrieve(query)
            ltm_content = [s.content for s in relevant_ltm]
            all_retrieved_ltm.append(ltm_content)

            ltm_text = "\n".join(f"- {c}" for c in ltm_content) if ltm_content else "（无相关长期记忆）"
            ltm_used_all.append(ltm_text)

            # 有 LTM 的答案
            docs = retriever.invoke(query)
            from agent.retriever import format_retrieved_docs
            from agent.generator import create_hybrid_rag_chain

            context_text = format_retrieved_docs(docs)
            hybrid_chain = create_hybrid_rag_chain(retriever, llm)
            ctx = hm.get_context_for_query(query)
            answer_with = hybrid_chain.invoke({
                "question": query,
                "history": ctx["history"],
                "summary": ctx["summary"],
                "long_term_memory": ctx["long_term_memory"],
                "state": ctx["state"],
            })
            answers_with_ltm.append(answer_with)

            # 无 LTM 的对照答案
            answer_without = chain_without_ltm.invoke(query)
            answers_without_ltm.append(answer_without)

            per_query.append({
                "id": sample.id,
                "question": query,
                "extracted_facts": extracted,
                "retrieved_ltm": ltm_content,
                "gt_ltm": gt_ltm,
                "answer_with_ltm": answer_with,
                "answer_without_ltm": answer_without,
                "ltm_used": ltm_text,
            })

            print(f"  提取 {len(extracted)} 条 LTM, "
                  f"检索 {len(ltm_content)} 条相关记忆")

        except Exception as e:
            print(f"  [ERR] 记忆评测失败: {e}")
            failed.append({"id": sample.id, "error": str(e)})
            conversations.append("")
            extracted_facts_all.append([])
            all_retrieved_ltm.append([])
            all_gt_ltm.append([])
            queries.append(sample.question)
            answers_with_ltm.append("")
            answers_without_ltm.append("")
            ltm_used_all.append("")
            per_query.append({"id": sample.id, "error": str(e)})

    # --- 计算所有记忆指标 ---
    mem_metrics = compute_all_memory_metrics(
        judge=judge,
        conversations=conversations,
        extracted_facts_all=extracted_facts_all,
        all_retrieved_ltm=all_retrieved_ltm,
        all_gt_ltm=all_gt_ltm,
        queries=queries,
        answers_with_ltm=answers_with_ltm,
        answers_without_ltm=answers_without_ltm,
        ltm_used_all=ltm_used_all,
        all_ltm_facts=all_ltm_facts_flat,
    )

    print("\n" + "=" * 60)
    print(f"[EVAL] [OK] 记忆质量评测完成")

    return {
        "meta": {
            "dataset": str(dataset_path),
            "num_queries": total,
            "num_success": total - len(failed),
            "num_failed": len(failed),
            "timestamp": datetime.now().isoformat(),
            "llm_model": LLM_MODEL,
            "use_rerank": use_rerank,
            "eval_type": "memory",
        },
        "memory": mem_metrics,
        "failed_queries": failed,
        "per_query": per_query,
    }


def _format_conversation(sample: EvalSample) -> str:
    """从 EvalSample 的 metadata 中提取对话记录并格式化为文本。"""
    conv = sample.metadata.get("conversation", [])
    if not conv:
        return ""
    lines = []
    for msg in conv:
        role = "用户" if msg.get("role") == "human" else "助手"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _parse_conversation_turns(sample: EvalSample) -> list[tuple[str, str]]:
    """从 EvalSample 的 metadata 中解析对话轮次 (user_msg, ai_msg) 对。"""
    conv = sample.metadata.get("conversation", [])
    turns = []
    human_msg = None
    for msg in conv:
        if msg.get("role") == "human":
            human_msg = msg.get("content", "")
        elif msg.get("role") == "ai" and human_msg is not None:
            turns.append((human_msg, msg.get("content", "")))
            human_msg = None
    return turns
