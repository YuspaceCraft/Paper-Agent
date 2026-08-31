# Retrieval Evaluation Guide

## 快速开始

```bash
# 1. 生成评估数据集 (keyword 模式, 无需 LLM)
python -m retrieval_orchestrator generate \
  --rag-path pdf_pipeline/output/RMNet/rag_chunks.json \
  --output eval_output/eval_manifest.jsonl \
  --modes keyword --samples 50

# 2. 人工审核 QA 对 (可选)
python -m retrieval_orchestrator review \
  --manifest eval_output/eval_manifest.jsonl

# 3. 完整评估流程
python -m retrieval_orchestrator evaluate --config retrieval_orchestrator/evaluation.yaml

# 4. 快速测试单条检索
python -m retrieval_orchestrator search \
  --query "dual stream CNN VMamba feature extraction" \
  --top-k 10
```

## 评估报告解读

`evaluation_report.json` 包含以下部分：

| 字段 | 说明 |
|------|------|
| `overall_metrics` | 全局 Recall@K / Precision@K / NDCG@K / MRR |
| `confidence_intervals` | Bootstrap 95% 置信区间 |
| `dimension_breakdown` | 按 difficulty_level / content_type / generation_mode 分组指标 |
| `failure_analysis` | 5 个最低分 query 的诊断分析 |
| `llm_judge` | (可选) LLM 对 Top-1 结果的 Context Precision / Faithfulness 评分 |

### 关键指标含义

- **Recall@K**: ground_truth 中有多少比例在 Top-K 结果中。数值越高 = 覆盖越全。
- **MRR**: 第一个相关结果的倒数排名。越高 = 排序越好。
- **NDCG@K**: 考虑排序位置的折损累积增益。越高 = 排序越优。
- **Hit@K**: Top-K 中至少命中一个 ground_truth 的 query 比例。

### 诊断低分案例

失败分析中常见问题及对应上游修复方向：

| 失败原因 | 根因 | 修复方向 |
|---------|------|---------|
| Chunk 不在向量库中 | 该 chunk 未被索引 | 检查 indexer pipeline 是否覆盖所有 chunk |
| Chunk 类型特殊 (formula/table) | 短文本嵌入质量差 | 增加 textualization 上下文，提高 formula_chunk token 预算 |
| 关键词查询匹配不到 | 关键词不在 embedding 语义范围内 | 增加 HyDE 改写或 query 扩展 |
| 跨块推理失败 | 单块不包含完整信息 | 调整 neighbor_window 参数，增加上下文拼接 |
| 高 token_count 的 chunk 召回低 | 长文本嵌入稀释 | 降低 chunk_tokens 或增加重叠 |

## 调整上游 Pipeline

### 提升 Recall

1. **减小 chunk_tokens**: 更细粒度切分 → 单块信息密度更高 → embedding 更精准
2. **增加 overlap_tokens**: 上下文更连续 → 边界信息不丢失
3. **启用 HyDE**: 在索引阶段为每个 chunk 生成假设问题，增加 query-chunk 匹配度
4. **换用更强的 embedding 模型**: 如 `text-embedding-v4` 或 `jina-embeddings-v3`

### 提升 MRR/NDCG

1. **调整 RRF k 参数**: 默认 60，增大 → 稀疏结果权重降低
2. **增加 reranker**: 在检索后加精排步骤 (Cross-Encoder)
3. **元数据过滤**: 利用 section_path / content_type 缩小检索范围

### 修复公式/表格 Chunk 召回差

1. **增加 textualization 质量**: 确保 `[FORMULA_DESC]` / `[FIGURE_DESC]` 充分描述符号含义
2. **提升 formula/token 预算**: `RAGChunkConfig` 中调整相关参数

## 最优配置传递给检索服务层

评估通过后，`optimal_retrieval_config.yaml` 是检索服务层的唯一合法输入：

```yaml
# eval_output/optimal_retrieval_config.yaml
schema_version: "1.0"
selected_at: "2026-07-16T..."
source_experiment: "dense_k10"
metrics:
  recall@5: 0.8234
  recall@10: 0.9123
  mrr: 0.7856
  ndcg@10: 0.8234
config:
  method: dense
  top_k: 10
  query_rewriting: null
  metadata_filter: null
```

检索服务层读取此配置，使用其中 `config` 部分的参数 (method, top_k, query_rewriting 等) 来初始化检索管道。

## 质量门禁规则

- **Recall@5 < 0.6**: 框架阻断，输出错误码 2。**不要强行修改阈值绕过**，应检查上游数据质量。
- **回归检测**: 若与 `historical_best` 对比，关键指标下降 >5%，报告中标记 REGRESSION 警告。

## 常见问题

### Q: 为什么 keyword 模式不需要 LLM 但 semantic/cross_chunk 需要？

keyword 模式直接提取 rag_chunks.json 中已有的 `[KEYWORDS: ...]` 字段构造查询。semantic 模式需要 LLM 阅读 chunk 内容生成自然语言问题，cross_chunk 需要 LLM 联合两个 chunk 生成多跳问题。

### Q: 评估很慢怎么办？

- 减少 `samples_per_mode` (如 20-30)
- 减少 `top_k_values` (如只用 [5, 10])
- 关闭 `llm_judge`
- 先用 keyword 模式验证流程，再加 LLM 模式

### Q: 向量库连不上？

确认 indexer config 中 `vector_store.backend` 与实际运行的向量库一致。Chroma 不需要额外服务，Qdrant 需要 `http://localhost:6333` 可用。
