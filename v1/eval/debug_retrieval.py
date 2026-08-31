"""
debug_retrieval.py — 检索失败诊断工具
==================
用于定位 Stage 1 检索质量瓶颈：哪些 query 没命中、为什么。
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import _setup_components
from eval.dataset import load_dataset
from agent.retriever import create_retriever
from config import TOP_K


def debug_hit_failures(dataset_path: str, top_k: int = 5):
    """诊断 Hit@1 失败的 query。"""
    samples = load_dataset(dataset_path)
    retrieval_samples = [s for s in samples if s.has_retrieval_gt()]

    print(f"\n{'='*60}")
    print(f"  检索失败诊断 ({len(retrieval_samples)} 条查询)")
    print(f"{'='*60}")

    # 仅用 Stage 1（无 reranker）
    components = _setup_components(use_rerank=False)
    retriever = components["retriever"]

    failures = []
    for i, sample in enumerate(retrieval_samples):
        docs = retriever.invoke(sample.question)[:top_k]

        # 判断 Hit@1
        doc_filename = docs[0].metadata.get("filename", "") if docs else ""
        gt_filenames = [s.lower() for s in sample.relevant_sources]
        hit1 = any(g in doc_filename.lower() for g in gt_filenames)

        if not hit1:
            failures.append({
                "id": sample.id,
                "question": sample.question,
                "gt_sources": sample.relevant_sources,
                "top1_file": doc_filename,
                "top1_preview": docs[0].page_content[:200] if docs else "(空)",
                "top3_files": [d.metadata.get("filename", "?") for d in docs[:3]],
            })

    print(f"\n[DIAG] Hit@1 失败: {len(failures)}/{len(retrieval_samples)} "
          f"({100*len(failures)/len(retrieval_samples):.1f}%)\n")

    for f in failures:
        print(f"{'─'*60}")
        print(f"[{f['id']}] {f['question'][:100]}")
        print(f"  GT 文件:    {f['gt_sources']}")
        print(f"  Top-1 文件: {f['top1_file']}")
        print(f"  Top-3 文件: {f['top3_files']}")
        print(f"  Top-1 内容预览: {f['top1_preview'][:150]}...")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="诊断检索失败")
    parser.add_argument("dataset", help="JSONL 数据集路径")
    args = parser.parse_args()
    debug_hit_failures(args.dataset)
