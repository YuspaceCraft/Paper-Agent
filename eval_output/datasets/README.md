# Evaluation Datasets

## manifest_v1_115qa.jsonl
- Date: 2026-07-20
- Papers: 5 (RMNet, MV-CC, BLIP-CC, DEM, Pix4Cap)
- Chunks: 123
- Modes: keyword=50, semantic=50, cross_chunk=15
- LLM: kimi-k2.6 @ DashScope
- Total: 115 QA pairs

## Usage
  python -m retrieval_orchestrator evaluate --config retrieval_orchestrator/evaluation.yaml --manifest eval_output/datasets/manifest_v1_115qa.jsonl
