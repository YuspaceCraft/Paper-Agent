# retrieval — 共享检索层

离线评估与在线 API 共用的检索组件。从 `retrieval_orchestrator/retrieval_engine.py` 提取，消除评估与生产的代码重复。

## 架构

```
retrieval/
 ├── sparse.py      → SparseRetriever      — TF-IDF 稀疏检索 (sklearn)
 ├── fusion.py      → rrf_fuse / weighted_fuse — RRF + 加权融合
 ├── service.py     → RetrievalService     — 从 optimal config 加载的生产检索服务
 │                   → DenseRetriever      — 向量检索封装 (VectorStoreAdapter + EmbeddingAdapter)
 └── __init__.py    → 公开导出
```

## 已实现功能

| 功能 | 模块 | 状态 |
|------|------|------|
| TF-IDF 稀疏检索 (char_wb ngrams 2-4, max_features=10000) | sparse.py | ✅ |
| RRF 融合 (rank-based, 免归一化) | fusion.py | ✅ |
| Weighted 融合 (min-max 归一化 + 加权求和) | fusion.py | ✅ |
| DenseRetriever (VectorStoreAdapter + EmbeddingAdapter) | service.py | ✅ |
| RetrievalService.from_config() (从 optimal_retrieval_config.yaml 加载) | service.py | ✅ |
| Sparse 结果增强 (chunks_map 补全 document/metadata) | service.py | ✅ |

## 快速使用

### RetrievalService（生产检索）

```python
from retrieval import RetrievalService

svc = RetrievalService.from_config(
    optimal_config="eval_output/optimal_retrieval_config.yaml",
    indexer_config="indexer/config.yaml",
    rag_chunks="eval_output/all_rag_chunks.json",
)

results = svc.search("dual stream feature extraction", top_k=5)
# → [{chunk_id, score, document, metadata, ...}, ...]
```

### SparseRetriever（独立使用）

```python
from retrieval import SparseRetriever

sp = SparseRetriever()
sp.index(chunks)  # chunks: list of {chunk_id, content, metadata}
hits = sp.search("change detection", top_k=10)
# → [{chunk_id, score}, ...]
```

### Fusion（独立使用）

```python
from retrieval import rrf_fuse, weighted_fuse

merged = rrf_fuse(dense_hits, sparse_hits, k=60, top_k=10)
merged = weighted_fuse(dense_hits, sparse_hits, dense_weight=0.7, top_k=10)
```

## 设计约束

1. **零依赖新增**: `sparse.py` 依赖 sklearn（已在 requirements.txt），其余仅 stdlib + 项目内模块
2. **只读消费**: RetrievalService 从 optimal config 和 indexer config 读取，不修改上游数据
3. **可替换**: DenseRetriever 依赖 VectorStoreAdapter / EmbeddingAdapter ABC，切换后端不影响服务层
4. **chunks_map 补偿**: SparseRetriever 仅返回 chunk_id + score，RetrievalService 用 chunks_map 补全 document/metadata

## 依赖

- **scikit-learn** — TfidfVectorizer（sparse.py）
- **indexer** — VectorStoreAdapter / EmbeddingAdapter（service.py, via DenseRetriever）
- **PyYAML** — optimal config 解析（service.py）
