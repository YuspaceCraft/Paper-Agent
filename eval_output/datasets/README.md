# Evaluation Datasets

## RAG 与论文操作任务集 v3 (current)
- Date: 2026-09-19
- File: `agent_rag_paper_ops_v3.jsonl`
- Generator: `python -m evaluation.task_dataset`
- Shape: 4 task families x 3 difficulty levels x 3 task lengths = 36 tasks
- Families: `rag_retrieval`, `paper_download`, `paper_read`, `paper_ingest`
- Difficulty: easy=12, medium=12, hard=12
- Length: short=12, medium=12, long=12
- Retrieval GT: 18 tasks; non-retrieval side-effect tasks: 18 tasks
- Includes: tool/argument contracts, answer evidence checks, artifacts,
  time/step/token budgets, and ingestion postconditions
- Stateful tasks require an isolated workspace/backend; do not run
  `paper_ingest` against the production index.

Design notes and metric definitions:
`docs/agent任务评测数据集v3.md`.

## manifest_v1_115qa.jsonl (stale)
- Date: 2026-07-20
- Papers: 5 (RMNet, MV-CC, BLIP-CC, DEM, Pix4Cap)
- Chunks: 123
- Modes: keyword=50, semantic=50, cross_chunk=15
- LLM: kimi-k2.6 @ DashScope
- Total: 115 QA pairs
- Status: historical dataset. Its chunk IDs are not present in the current
  `eval_output/all_rag_chunks.json`; using it now produces Recall=0 by construction.

## manifest_v2_current_kb_50qa.jsonl
- Generated from the current 241-chunk local corpus
- Modes: keyword=50
- Ground-truth coverage against the current corpus: 50/50

## Usage
  python -m retrieval_orchestrator evaluate --config retrieval_orchestrator/evaluation.yaml
  python -m evaluation run --manifest eval_output/datasets/manifest_v2_current_kb_50qa.jsonl
  python -m evaluation run --manifest eval_output/datasets/agent_rag_paper_ops_v3.jsonl --dataset agent-rag-paperops-v3
