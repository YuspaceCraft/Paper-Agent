"""
cli.py — 评测命令行接口
======
评测模块的入口。用 python -m eval.cli <命令> 启动。

可用命令：
  evaluate-retrieval   — 检索质量评测
  evaluate-generation  — 生成质量评测
  evaluate-e2e         — 端到端评测
  list-datasets        — 列出可用数据集
"""

import argparse
import sys
from pathlib import Path


def cmd_evaluate_retrieval(args):
    """处理 'evaluate-retrieval' 子命令"""
    from eval.evaluate import run_retrieval_evaluation
    from eval.report import print_report

    results = run_retrieval_evaluation(
        dataset_path=args.dataset,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
        candidate_k=args.candidate_k or None,
        final_k=args.final_k or None,
    )

    if results:
        print_report(results, output_path=args.output)


def cmd_evaluate_generation(args):
    """处理 'evaluate-generation' 子命令"""
    from eval.evaluate import run_generation_evaluation
    from eval.report import print_report

    results = run_generation_evaluation(
        dataset_path=args.dataset,
        use_rerank=not args.no_rerank,
    )

    if results:
        print_report(results, output_path=args.output)


def cmd_evaluate_e2e(args):
    """处理 'evaluate-e2e' 子命令"""
    from eval.evaluate import run_e2e_evaluation
    from eval.report import print_report

    results = run_e2e_evaluation(
        dataset_path=args.dataset,
        use_rerank=not args.no_rerank,
        top_k=args.top_k,
    )

    if results:
        print_report(results, output_path=args.output)


def cmd_evaluate_memory(args):
    """处理 'evaluate-memory' 子命令"""
    from eval.evaluate import run_memory_evaluation
    from eval.report import print_report

    results = run_memory_evaluation(
        dataset_path=args.dataset,
        use_rerank=not args.no_rerank,
    )

    if results:
        print_report(results, output_path=args.output)


def cmd_list_datasets(args):
    """处理 'list-datasets' 子命令：列出可用评测数据集"""
    # 搜索项目根目录下的 evalsets/
    project_root = Path(__file__).resolve().parent.parent
    evalsets_dir = project_root / "evalsets"

    print("\n" + "=" * 60)
    print("  可用评测数据集")
    print("=" * 60)

    if not evalsets_dir.exists():
        print(f"  目录不存在: {evalsets_dir}")
        print(f"  [TIP] 创建 evalsets/ 目录并放入 .jsonl 文件")
        print("=" * 60)
        return

    jsonl_files = sorted(evalsets_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  目录为空: {evalsets_dir}")
        print(f"  [TIP] 放入 .jsonl 格式的评测数据集")
        print("=" * 60)
        return

    for f in jsonl_files:
        # 统计行数
        try:
            with open(f, "r", encoding="utf-8") as fp:
                lines = [l for l in fp if l.strip() and not l.strip().startswith("#")]
            count = len(lines)
        except Exception:
            count = "?"

        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}")
        print(f"    路径:   {f}")
        print(f"    大小:   {size_kb:.1f} KB")
        print(f"    行数:   {count}")

        # 推断评测类型
        has_retrieval = False
        has_generation = False
        try:
            import json
            for line in lines[:3]:
                data = json.loads(line.strip())
                if "relevant_sources" in data:
                    has_retrieval = True
                if "reference_answer" in data:
                    has_generation = True
            types = []
            if has_retrieval:
                types.append("检索")
            if has_generation:
                types.append("生成")
            print(f"    适合:   {', '.join(types) if types else '未知'}")
        except Exception:
            pass
        print()

    print("=" * 60)


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        prog="python -m eval.cli",
        description="RAG Agent 评测系统 — 检索质量、生成质量、端到端评测",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ----- evaluate-retrieval -----
    p_retrieval = subparsers.add_parser(
        "evaluate-retrieval",
        help="检索质量评测（Hit Rate / MRR / NDCG / Precision / Recall）",
    )
    p_retrieval.add_argument(
        "--dataset", "-d",
        required=True,
        help="评测数据集路径（JSONL 格式）",
    )
    p_retrieval.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="最终返回的文档数（默认: 5）",
    )
    p_retrieval.add_argument(
        "--no-rerank",
        action="store_true",
        help="禁用二阶段检索（仅使用向量粗筛）",
    )
    p_retrieval.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="粗筛候选数（覆盖 config 中的 RERANK_CANDIDATE_K）",
    )
    p_retrieval.add_argument(
        "--final-k",
        type=int,
        default=None,
        help="精排后数量（覆盖 config 中的 RERANK_FINAL_K）",
    )
    p_retrieval.add_argument(
        "--output", "-o",
        default=None,
        help="JSON 报告输出路径（可选）",
    )

    # ----- evaluate-generation -----
    p_generation = subparsers.add_parser(
        "evaluate-generation",
        help="生成质量评测（忠实度 / 相关性 / 正确性）",
    )
    p_generation.add_argument(
        "--dataset", "-d",
        required=True,
        help="评测数据集路径（JSONL 格式，需含 reference_answer）",
    )
    p_generation.add_argument(
        "--no-rerank",
        action="store_true",
        help="禁用二阶段检索（仅使用向量粗筛）",
    )
    p_generation.add_argument(
        "--output", "-o",
        default=None,
        help="JSON 报告输出路径（可选）",
    )

    # ----- evaluate-e2e -----
    p_e2e = subparsers.add_parser(
        "evaluate-e2e",
        help="端到端评测（检索 + 生成 + 延迟追踪）",
    )
    p_e2e.add_argument(
        "--dataset", "-d",
        required=True,
        help="评测数据集路径（JSONL 格式）",
    )
    p_e2e.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="最终返回的文档数（默认: 5）",
    )
    p_e2e.add_argument(
        "--no-rerank",
        action="store_true",
        help="禁用二阶段检索（仅使用向量粗筛）",
    )
    p_e2e.add_argument(
        "--output", "-o",
        default=None,
        help="JSON 报告输出路径（可选）",
    )

    # ----- evaluate-memory -----
    p_memory = subparsers.add_parser(
        "evaluate-memory",
        help="记忆质量评测（提取准确性 / 检索精确率 / 记忆影响 / 一致性）",
    )
    p_memory.add_argument(
        "--dataset", "-d",
        required=True,
        help="评测数据集路径（JSONL 格式，需含 conversation + expected_memories）",
    )
    p_memory.add_argument(
        "--no-rerank",
        action="store_true",
        help="禁用二阶段检索（仅使用向量粗筛）",
    )
    p_memory.add_argument(
        "--output", "-o",
        default=None,
        help="JSON 报告输出路径（可选）",
    )

    # ----- list-datasets -----
    subparsers.add_parser(
        "list-datasets",
        help="列出 evalsets/ 目录下可用的评测数据集",
    )

    # 解析参数
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n[TIP] 快速开始:")
        print("  1. python -m eval.cli list-datasets")
        print("  2. python -m eval.cli evaluate-retrieval -d evalsets/sample_retrieval.jsonl")
        print("  3. python -m eval.cli evaluate-generation -d evalsets/sample_generation.jsonl")
        print("  4. python -m eval.cli evaluate-e2e -d evalsets/sample_e2e.jsonl -o report.json")
        print("  5. python -m eval.cli evaluate-memory -d evalsets/sample_memory.jsonl")
        sys.exit(0)

    # 分发到对应的处理函数
    handlers = {
        "evaluate-retrieval": cmd_evaluate_retrieval,
        "evaluate-generation": cmd_evaluate_generation,
        "evaluate-e2e": cmd_evaluate_e2e,
        "evaluate-memory": cmd_evaluate_memory,
        "list-datasets": cmd_list_datasets,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
