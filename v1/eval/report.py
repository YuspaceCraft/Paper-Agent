"""
report.py — 评测报告生成器
========
将评测结果格式化为人类可读的控制台输出和机器可读的 JSON。

输出格式：
  1. 控制台报告 — 模仿 src/cli.py 中 cmd_status() 的风格
  2. JSON 报告   — 包含完整的 per-query 细节，适合程序化分析
"""

import json
from datetime import datetime
from pathlib import Path


# ================================================================
#  控制台报告
# ================================================================

def _section_header(title: str, width: int = 60) -> str:
    """生成段落标题行。"""
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def _format_float(value: float, decimals: int = 4) -> str:
    """格式化浮点数，处理 None。"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def format_console_report(results: dict) -> str:
    """生成控制台格式的评测报告。

    参数:
        results: evaluate.py 返回的完整结果字典

    返回:
        str: 格式化后的控制台报告
    """
    lines = []
    sep = "=" * 60

    # 标题
    lines.append(sep)
    lines.append("  RAG 评测报告")
    lines.append(sep)

    # 元信息
    meta = results.get("meta", {})
    lines.append(f"  数据集:       {meta.get('dataset', 'N/A')}")
    lines.append(f"  评测查询数:   {meta.get('num_queries', 0)}")
    lines.append(f"  成功:         {meta.get('num_success', 0)}")
    lines.append(f"  失败:         {meta.get('num_failed', 0)}")
    lines.append(f"  评测时间:     {meta.get('timestamp', datetime.now().isoformat())}")
    lines.append(f"  配置:         LLM={meta.get('llm_model', 'N/A')}, "
                       f"Rerank={meta.get('use_rerank', True)}, "
                       f"Top-K={meta.get('top_k', 'N/A')}")

    # 检索质量
    retrieval = results.get("retrieval", {})
    if retrieval:
        lines.append(_section_header("检索质量 (Retrieval Quality)"))
        metrics_order = [
            ("hit_rate", "Hit Rate"),
            ("mrr", "MRR (平均倒数排名)"),
            ("precision", "Precision (精确率)"),
            ("recall", "Recall (召回率)"),
            ("ndcg", "NDCG (归一化折损累计增益)"),
        ]
        for key_prefix, label in metrics_order:
            k_values = sorted(set(
                int(k.split("@")[1])
                for k in retrieval.keys()
                if k.startswith(f"{key_prefix}@")
            ))
            for k_val in k_values:
                metric_key = f"{key_prefix}@{k_val}"
                value = retrieval.get(metric_key)
                if value is not None:
                    lines.append(f"  {label}@{k_val:>2}:  {_format_float(value)}")

    # 生成质量
    generation = results.get("generation", {})
    if generation:
        lines.append(_section_header("生成质量 (Generation Quality)"))
        lines.append(f"  忠实度 (Faithfulness):      {_format_float(generation.get('faithfulness_avg'))}"
                     f"  ({_format_float(generation.get('faithfulness_avg', 0) * 100, 1)}%)")
        lines.append(f"  答案相关性 (Relevance):      {_format_float(generation.get('answer_relevance_raw_avg'), 1)} / 5"
                     f"  ({_format_float(generation.get('answer_relevance_avg', 0) * 100, 1)}%)")
        lines.append(f"  正确性 (Correctness):        {_format_float(generation.get('correctness_raw_avg'), 1)} / 5"
                     f"  ({_format_float(generation.get('correctness_avg', 0) * 100, 1)}%)")

    # 记忆质量
    memory = results.get("memory", {})
    if memory:
        lines.append(_section_header("记忆质量 (Memory Quality)"))

        # 提取质量
        extraction = memory.get("extraction", {})
        if extraction:
            acc = extraction.get("extraction_accuracy_avg")
            lines.append(f"  提取准确率 (Extraction Accuracy): {_format_float(acc)}"
                         f"  ({_format_float((acc or 0) * 100, 1)}%)")

        # 检索质量
        retrieval_mem = memory.get("retrieval", {})
        if retrieval_mem:
            for k in [1, 3, 5]:
                p = retrieval_mem.get(f"ltm_precision@{k}")
                r = retrieval_mem.get(f"ltm_recall@{k}")
                m = retrieval_mem.get(f"ltm_mrr@{k}")
                if p is not None:
                    lines.append(f"  记忆精确率@{k} (Precision):     {_format_float(p)}")
                if r is not None:
                    lines.append(f"  记忆召回率@{k} (Recall):        {_format_float(r)}")
                if m is not None and k == 3:
                    lines.append(f"  记忆MRR@{k}:                    {_format_float(m)}")

        # 记忆影响
        impact = memory.get("impact", {})
        if impact:
            imp_avg = impact.get("impact_avg")
            imp_norm = impact.get("impact_normalized_avg")
            dist = impact.get("impact_distribution", {})
            label = ""
            if imp_avg is not None:
                if imp_avg > 0.5:
                    label = " (显著改善)"
                elif imp_avg > 0:
                    label = " (轻微改善)"
                elif imp_avg == 0:
                    label = " (无影响)"
                else:
                    label = " (负面影响)"
            lines.append(f"  记忆影响分数 (Impact):          {_format_float(imp_avg)}{label}")
            lines.append(f"  影响分布:                       worse={dist.get('worse',0)}, "
                         f"same={dist.get('same',0)}, better={dist.get('better',0)}, "
                         f"much_better={dist.get('much_better',0)}")

        # 一致性
        consistency = memory.get("consistency", {})
        if consistency:
            cs = consistency.get("consistency_score")
            dup = consistency.get("duplicate_pairs", 0)
            total_f = consistency.get("total_facts", 0)
            lines.append(f"  一致性分数 (Consistency):       {_format_float(cs)}"
                         f"  ({dup} 对重复 / {total_f} 条事实)")

    # 延迟
    latency = results.get("latency", {})
    if latency:
        lines.append(_section_header("延迟 (Latency)"))
        lines.append(f"  平均检索耗时:     {_format_float(latency.get('avg_retrieval_s'), 2)}s")
        lines.append(f"  平均生成耗时:     {_format_float(latency.get('avg_generation_s'), 2)}s")
        lines.append(f"  平均总耗时:       {_format_float(latency.get('avg_total_s'), 2)}s")
        if latency.get("total_s"):
            lines.append(f"  总耗时:           {_format_float(latency.get('total_s'), 2)}s")

    # 失败查询
    failed = results.get("failed_queries", [])
    if failed:
        lines.append(_section_header("失败查询 (Failed Queries)"))
        for fq in failed:
            lines.append(f"  [{fq.get('id', '?')}] {fq.get('error', 'unknown')}")
    else:
        lines.append(_section_header("失败查询: (无)"))

    # 按类别分层（如果有）
    by_category = results.get("by_category")
    if by_category:
        lines.append(_section_header("按类别分层 (By Category)"))
        for cat, cat_results in by_category.items():
            lines.append(f"\n  [{cat}] ({cat_results.get('count', 0)} 条查询)")
            for k, v in cat_results.get("metrics", {}).items():
                lines.append(f"    {k}: {_format_float(v)}")

    lines.append(f"\n{sep}")

    return "\n".join(lines)


# ================================================================
#  JSON 报告
# ================================================================

def _make_json_safe(obj):
    """递归将对象转为 JSON 可序列化格式。"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    return str(obj)


def write_json_report(results: dict, output_path: str | Path) -> None:
    """将评测结果写入 JSON 文件。

    参数:
        results:     evaluate.py 返回的完整结果字典
        output_path: JSON 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_safe = _make_json_safe(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, ensure_ascii=False, indent=2)

    print(f"[REPORT] [OK] JSON 报告已写入: {output_path}")


def print_report(
    results: dict,
    output_path: str | Path | None = None,
) -> None:
    """打印控制台报告 + 可选写入 JSON 文件。

    参数:
        results:     evaluate.py 返回的完整结果字典
        output_path: 可选，JSON 输出路径
    """
    # 控制台输出
    console = format_console_report(results)
    print(console)

    # JSON 输出
    if output_path:
        write_json_report(results, output_path)
