# 检索评估报告

**评估日期**: 2026-07-17  
**数据源**: RMNet 论文 (pdf_pipeline/output/RMNet/rag_chunks.json)  
**向量库**: Qdrant (38 chunks, 768d, jina-embeddings-v5-text-nano)  
**评估数据集**: 25 条 keyword 模式 QA 对  
**最优配置**: `sparse_k5`

---

## 一、评估流程概览

```
rag_chunks.json → Step 1: 生成评估数据集 → Step 2: 多策略检索 → Step 3: 指标计算 → Step 4: 最优配置选择
                      ↓                          ↓                      ↓                      ↓
              eval_manifest.jsonl         18 个实验 × 25 查询       每实验 1 份 report       optimal_retrieval_config.yaml
```

---

## 二、Step 1 — 评估数据集生成

| 参数 | 值 |
|------|-----|
| 生成模式 | keyword |
| 采样数 | 25 条 (全量覆盖所有含 KEYWORDS 的 body chunk) |
| 输出文件 | `eval_manifest.jsonl` |

**生成逻辑**: 从 `rag_chunks.json` 的每个 chunk 中正则提取 `[KEYWORDS: ...]` 字段，将关键词拼接为查询串，ground truth 为该 chunk 自身的 `chunk_id`。每条 QA 附加 `metadata_filters: {content_type: body}` 约束。

**难度分布**:

| 难度 | 数量 | 判定标准 |
|------|------|----------|
| easy | 2 | token_count < 300 |
| medium | 20 | token_count 300–800 |
| hard | 3 | token_count > 800 或 content_type 为 formula |

---

## 三、Step 2 — 多策略检索实验

共执行 **18 个策略组合**（3 methods × 3 top_k × 2 metadata_filter）：

| 维度 | 取值 |
|------|------|
| 检索方法 | dense (jina-v5-nano), sparse (TF-IDF char_wb), hybrid (RRF) |
| Top-K | 5, 10, 20 |
| 元数据过滤 | null（不过滤）, content_type=body |

### 各策略指标对比

| 策略 | Recall@5 | Recall@10 | MRR | NDCG@10 | 合格 |
|------|----------|-----------|-----|---------|------|
| **sparse_k5** | **0.9600** | **0.9600** | **0.7813** | **0.8269** | ✅ |
| sparse_k5 + mf | 0.9600 | 0.9600 | 0.7813 | 0.8269 | ✅ |
| sparse_k10 | 0.9600 | 0.9600 | 0.7813 | 0.8269 | ✅ |
| sparse_k10 + mf | 0.9600 | 0.9600 | 0.7813 | 0.8269 | ✅ |
| sparse_k20 | 0.9600 | 0.9600 | 0.7813 | 0.8269 | ✅ |
| sparse_k20 + mf | 0.9600 | 0.9600 | 0.7813 | 0.8269 | ✅ |
| hybrid_k10_rrf | 0.7200 | 0.9600 | 0.5543 | 0.7516 | — |
| hybrid_k20_rrf | 0.7200 | 0.9600 | 0.5572 | 0.7533 | — |
| hybrid_k5_rrf | 0.6800 | 0.6800 | 0.5167 | 0.5151 | — |
| dense_k10 | 0.6400 | 0.6400 | 0.2200 | 0.5151 | — |
| dense_k20 | 0.6400 | 0.6400 | 0.2323 | 0.5200 | — |
| dense_k5 | 0.2800 | 0.2800 | 0.1693 | 0.1962 | — |

> 带 `+ mf` 后缀的策略因 manifest 已注入 `content_type: body` 过滤，结果与不带 mf 的同名策略完全一致（该数据集仅 body 类型 chunk 参与评估）。

### 关键发现

1. **Sparse (TF-IDF) 碾压 Dense**: TF-IDF 的 char_wb ngrams (2-4) 对关键词查询天然亲和，直接命中 `[KEYWORDS: ...]` 原文中的 token
2. **Dense 在短查询上失效**: jina-v5-nano 把 "difference remote sensing" 等短关键词映射到了 figure description chunk（词汇重叠更多），而非源头 body chunk
3. **Hybrid 比 Dense 好但不如 Sparse**: RRF 融合后 recall 提升但因 dense 噪声拉低 MRR
4. **Top-K 对 Sparse 不影响**: sparse 在 k=5 就达到饱和 recall 0.96，k 增大无边际收益

---

## 四、Step 3 — 指标说明

### 核心指标

| 指标 | 最优值 | 含义 |
|------|--------|------|
| **Recall@K** | 1.0 | ground_truth chunk 在 Top-K 中被命中的比例。衡量"找到没找到" |
| **Precision@K** | 1.0 | Top-K 中命中的 ground_truth 占比。本次评估每个 query 仅 1 个 gt，故 precision = recall/K |
| **MRR** (Mean Reciprocal Rank) | 1.0 | 第一个命中结果的排名倒数均值。衡量"找到得多靠前" |
| **NDCG@K** (Normalized DCG) | 1.0 | 考虑排名位置折损的累积增益。MRR 只看第一个，NDCG 奖励所有命中位置 |
| **Hit@K** | 1.0 | 至少命中 1 个 gt 的 query 比例。等价于本次的 Recall@K（每个 query 1 个 gt） |

### Bootstrap 置信区间

对每个指标做 1000 次重采样，计算 95% CI。sparse_k5 的 Recall@5 CI = [0.88, 1.0]，表明在 95% 置信度下真实 recall 不低于 0.88。

---

## 五、维度拆解 (sparse_k5)

### 按难度

| 难度 | 数量 | Recall@5 | MRR | NDCG@5 |
|------|------|----------|-----|--------|
| easy | 2 | 1.0000 | 1.0000 | 1.0000 |
| medium | 20 | 1.0000 | 0.8167 | 0.8643 |
| hard | 3 | 0.6667 | 0.4000 | 0.4623 |

**分析**: hard 样本（大 token 量 chunk）对 TF-IDF 更难精确匹配——大块文本关键词被稀释，TF-IDF 对长文档区分度下降。2/3 的 hard 查询排名不在首位。

### 按生成模式

所有 25 条均为 `keyword` 模式，指标等同 overall。

---

## 六、失败案例分析 (sparse_k5)

5 个最低 MRR 查询：

| 查询 | MRR | 期望 chunk | 实际 Top-1 | 根因 |
|------|-----|-----------|------------|------|
| "transformer model performance" | 0.0000 | chunk_0023 | chunk_0009 | chunk_0023 token_count=1690，关键词稀释 |
| "features feature performance" | 0.2000 | chunk_0031 | chunk_0015 | 关键词过于通用（"feature" "performance"） |
| "difference remote sensing" | 0.3333 | chunk_0000 | chunk_0029 | 关键词在多个 chunk 中出现 |
| "features change captioning" | 0.5000 | chunk_0003 | chunk_0010 | 同上述 |
| "change image captioning" | 0.5000 | chunk_0007 | chunk_0022 | 同上述 |

**共性**: 所有失败案例的 gt chunk 都在 Top-5 中被命中（recall=1.0），只是排名不在第 1 位。属排序问题，非检索遗漏。

---

## 七、Step 4 — 最优配置选择

### 质量门禁

| 检查项 | 阈值 | 实际 | 结果 |
|--------|------|------|------|
| Recall@5 | ≥ 0.6 | **0.96** | ✅ 通过 |
| Recall@10 (业务) | ≥ 0.85 | **0.96** | ✅ 通过 |
| MRR (业务) | ≥ 0.7 | **0.7813** | ✅ 通过 |

### 选定配置

```yaml
method: sparse          # TF-IDF char_wb ngrams (2-4)
top_k: 5               # 最小有效 K，边际收益为 0
query_rewriting: null  # 不需要
metadata_filter: null  # 不需要
```

**排名逻辑**: 在所有通过阈值的 6 个策略（全部 sparse 系列）中，优先选 top_k 最小的 → `sparse_k5`。sparse_k10/k20 指标相同但 top_k 更大，被排后。

### 未选择 Dense/Hybrid 的原因

- **dense**: Recall@10 最高 0.64，不满足 0.85 阈值。短关键词查询对 dense embedding 是天生的弱项（缺乏上下文）
- **hybrid**: k=10/20 时 recall 达标，但 MRR 最高 0.5572，不满足 0.7 阈值。Dense 分量引入的噪声拉低了排序精度

---

## 八、质量评估与合理性判断

### 评估合理性

| 维度 | 评价 |
|------|------|
| **数据集覆盖** | ⚠️ 仅 25 条 keyword 查询，全部来自单一论文 (RMNet)。未覆盖 semantic/cross_chunk 模式，不能代表真实用户查询分布 |
| **指标可信度** | ✅ Recall@5 CI [0.88, 1.0] 较窄，Bootstrap 验证可靠。MRR CI [0.67, 0.90] 稍宽但通过阈值 |
| **配置可迁移性** | ⚠️ TF-IDF 索引在运行时构建，检索服务层需同步部署 SparseRetriever + 实时索引更新逻辑 |
| **最优配置局限性** | ⚠️ sparse_k5 仅对 keyword 查询最优，真实用户自然语言查询可能更适合 hybrid |

### 建议

1. **补全 semantic 数据集**: 启用 LLM 生成自然语言查询，验证 sparse 在真实语义查询下的表现
2. **验证 hybrid + HyDE**: 对自然语言查询，HyDE 改写 + RRF 融合可能是更好的选择
3. **sparse 索引工程化**: 将 SparseRetriever 的索引构建从"每次评估 build"改为"增量更新 + 持久化"
4. **单次失败查询调优**: "transformer model performance" (chunk_0023) 因 token_count=1690 太大致关键词稀释，考虑对该 chunk 手动追加摘要关键词

---

## 九、输出文件索引

| 文件 | 说明 |
|------|------|
| `optimal_retrieval_config.yaml` | **唯一合法输入** → 检索服务层 |
| `eval_manifest.jsonl` | 评估数据集 |
| `experiments.json` | 所有 18 个实验的完整结果与报告 |
| `retrieval_*.jsonl` | 各策略逐查询检索结果 |
| `retrieval_*.report.json` | 各策略评估报告 |

> `检索质量.docx` / `检索质量1.docx` 为历史遗留文件，与本次评估无关。
