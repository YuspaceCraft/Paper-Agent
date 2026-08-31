# indexer — 知识索引与存储编排层

多粒度向量索引构建 + 增量同步。论文从 `pdf_pipeline/output/{paper}/rag_chunks.json`
进入向量库（Qdrant 生产 / Chroma 本地），并在 Redis 论文注册表（`indexer.catalog`）
记下全量元数据。

## 使用

### 全流程（CLI）

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m indexer.pipeline pdf_pipeline/output/RMNet/rag_chunks.json
```

### API

- `POST /api/index/run` — 后台建索引（`web/api/routers/index.py`）
- `POST /api/index/reconcile` — 目录对账（见下）

### 编程使用

```python
from indexer.catalog import register_indexed, register_paper, mark_indexed, patch_paper

# 原子入库收尾：注册 + 置 indexed 一次写完（推荐）
register_indexed(paper_name, metadata={"title": "..."}, page_count=12,
                 chunk_count=120, indexed_chunk_count=120)

# 仅解析（兼容流 /api/pdf/process）：indexed=false
register_paper(paper_name, metadata=meta, page_count=12, chunk_count=120)

# 对账回填（旧库修复）：以向量库实际点数为准同步 indexed 标志
from indexer.reconcile import reconcile
report = reconcile()   # 命令行: python -m indexer.reconcile / python -m web.cli reconcile
```

## 目录状态语义（方案 B）

论文对外 `state` 只分两类，由目录 `indexed` 标志（Redis `dedup:paper:{name}` 里的
bool）决定；parsed/raw 是 API 层从文件系统**派生**的诊断，不是持久终态。

| 状态 | 目录 `indexed` | 含义 |
|------|----------------|------|
| `indexed` | `true` | chunks 已进向量库，可检索（唯一持久判据） |
| `not_indexed` | `false`/缺失 | 其余一切；`detail` 派生：`parsed`（有解析产物）/`raw`（仅本地 PDF）/`""` |

**原子性**：入库（`/api/agent/ingest`）是解析+向量化一个后台任务，目录只在收尾
由 `register_indexed()` 一次写入 —— 没有「已解析未入库」的持久中间态，失败也不
残留误导性标记。

## reconcile（目录 ↔ 向量库对账）

历史缺陷（旧 schema 缺 `indexed` 键 / 旧入库路径漏置标志）会导致已入库论文
显示为未入库。对账以**向量库实际点数为准**回填：

```
POST /api/index/reconcile          # HTTP（后台任务，fixed 明细在 task result）
python -m web.cli reconcile        # CLI
python -m indexer.reconcile        # CLI（等价，输出 JSON）
```

对每篇已注册论文：向量库按 `chunk_id` 前缀（`{paper}__chunk_NNNN`）滚动计数；
标志缺失/与库不一致 → `patch_paper(name, indexed=库有无, indexed_chunk_count=实际点数)`
回填。向量库不可达时报 `{ok:false}`，不改数据。

## 关键符号

| 符号 | 位置 | 作用 |
|------|------|------|
| `IndexerPipeline.run()` | `pipeline.py` | 索引构建总入口 |
| `register_paper` / `register_indexed` / `mark_indexed` / `patch_paper` | `catalog.py` | 目录写入（按语义选用，见上） |
| `reconcile()` / `count_chunks_per_paper()` | `reconcile.py` | 目录对账回填 |
| `MultiGranularityEmbedder` | `embedder.py` | 多粒度向量化（PII 检测 + sparse keywords） |
| `ContextAssembler` | `context_assembler.py` | Small-to-Big 上下文组装 |
| `QdrantVectorStore` / `ChromaVectorStore` | `vector_store.py` | upsert/search/delete + 增量同步散表 |
| `DedupManager` | `dedup_manager.py` | 增量去重（sha256） |
| `load_config()` / `IndexerConfig` | `config.py` | 读 `indexer/config.yaml` |