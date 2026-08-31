"""
eval/ — RAG Agent 评测模块
==========================

评测三大维度：
  1. 检索质量评测 — Hit Rate, MRR, NDCG, Precision@K, Recall@K
  2. 生成质量评测 — 忠实度、相关性、正确性（LLM-as-Judge）
  3. 端到端评测   — 完整管道 + 延迟追踪

用法：
  python -m eval.cli evaluate-retrieval --dataset evalsets/sample_retrieval.jsonl
  python -m eval.cli evaluate-generation --dataset evalsets/sample_generation.jsonl
  python -m eval.cli evaluate-e2e --dataset evalsets/sample_e2e.jsonl
"""

__version__ = "0.1.0"
