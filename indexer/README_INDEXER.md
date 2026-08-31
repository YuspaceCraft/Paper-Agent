# indexer — Knowledge Indexing & Storage Orchestration

独立于上游 pdf_pipeline 的知识索引与存储编排层。以 `rag_chunks.json` 为唯一输入契约，
构建多粒度向量索引（Dense + BM25），支持断点续跑和增量同步。

## 架构

```
rag_chunks.json (read-only)
        │
        ▼
┌──────────────────────────────┐
│  1. ContextAssembler         │  Small-to-Big: prev/next 邻居拼接
│     retrieval_text /         │
│     generation_text          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. MultiGranularityEmbedder │  Dense + BM25 Sparse + HyDE(opt)
│     PII Detection → flag     │
│     Embedding API (batch)    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  3. DedupManager             │  sha256 content hash
│     new / updated / skipped  │  → Upsert 语义
│     orphaned → delete        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  4. VectorStoreAdapter       │  Chroma (本地) / Qdrant (生产)
│     upsert / search / delete │  元数据扁平化为 Filter 字段
└──────────────┬───────────────┘
               │
               ▼
         eval_manifest.jsonl    (Ragas 评估就绪)
```

## 快速开始

### 前置条件

1. pdf_pipeline 已完成 Stage 5，生成了 `rag_chunks.json`
2. Conda 环境 `demo` 已激活
3. `DASHSCOPE_API_KEY` 已配置（项目根目录 `.env` 或环境变量）

### 安装依赖

```bash
# 核心依赖
C:/Users/30811/miniconda3/envs/demo/python.exe -m pip install chromadb openai pyyaml

# 可选：Qdrant（生产级）
C:/Users/30811/miniconda3/envs/demo/python.exe -m pip install qdrant-client

# 可选：Presidio（高级 PII 检测）
C:/Users/30811/miniconda3/envs/demo/python.exe -m pip install presidio-analyzer
```

### CLI

```bash
# 使用默认配置运行
C:/Users/30811/miniconda3/envs/demo/python.exe -m indexer.pipeline pdf_pipeline/output/MV-CC/rag_chunks.json

# 指定配置文件
C:/Users/30811/miniconda3/envs/demo/python.exe -m indexer.pipeline pdf_pipeline/output/MV-CC/rag_chunks.json --config indexer/config.yaml
```

### Python API

```python
from indexer import IndexerPipeline

# 默认配置
pipeline = IndexerPipeline()
stats = pipeline.run("pdf_pipeline/output/MV-CC/rag_chunks.json")

print(f"Inserted: {stats['inserted']}, Updated: {stats['updated']}")
print(f"Skipped: {stats['skipped']}, Store: {stats['store_count']} records")

# 自定义配置
pipeline = IndexerPipeline("indexer/config.yaml")
stats = pipeline.run("pdf_pipeline/output/MV-CC/rag_chunks.json")
```

### 分步使用

```python
from indexer import (
    IndexerConfig, load_config,
    ContextAssembler, MultiGranularityEmbedder,
    ChromaVectorStore, DedupManager, export_eval_manifest,
)

config = load_config("indexer/config.yaml")

# Step 1: Context Assembly
assembler = ContextAssembler(config.context_assembly)
chunks = assembler.load_chunks("pdf_pipeline/output/MV-CC/rag_chunks.json")
assembled = assembler.assemble(chunks)

# Step 2: Embedding
embedder = MultiGranularityEmbedder(config.embedding, config.hyde, config.pii)
embedded = embedder.embed(assembled)

# Step 3: Dedup + Store
store = ChromaVectorStore()
dedup = DedupManager(store)
classification = dedup.sync(embedded)
dedup.apply(classification)

# Step 4: Eval export
export_eval_manifest(embedded, "./eval_manifest.jsonl")
```

## 配置

所有可配置参数见 [config.yaml](config.yaml)。关键配置项：

| 配置 | 默认 | 说明 |
|------|------|------|
| `embedding.model` | text-embedding-v4 | Embedding 模型 |
| `context_assembly.retrieval_max_tokens` | 512 | 向量化文本长度上限 |
| `context_assembly.generation_max_tokens` | 2048 | LLM 生成上下文长度上限 |
| `vector_store.backend` | chroma | chroma / qdrant |
| `pii.enabled` | true | 是否启用 PII 检测 |
| `hyde.enabled` | false | 是否生成假设性问题 |

## 切换 Vector Store 后端

### Chroma → Qdrant

1. 安装 Qdrant: `pip install qdrant-client`
2. 启动 Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
3. 修改 `config.yaml`:

```yaml
vector_store:
  backend: "qdrant"
  qdrant:
    url: "http://localhost:6333"
    collection_name: "rag_chunks"
```

### 适配器接口

所有后端实现 `VectorStoreAdapter` 接口：

```python
class VectorStoreAdapter(ABC):
    def upsert(self, units: list[dict]) -> int: ...
    def search(self, query_vector, filters, top_k) -> list[dict]: ...
    def delete(self, chunk_ids: list[str]) -> int: ...
    def count(self) -> int: ...
    def get_existing_hashes(self) -> dict[str, str]: ...
```

添加新后端 → 实现以上 5 个方法即可，Pipeline 业务逻辑零修改。

## 增量同步

通过 Content Hash 实现 Upsert 语义:

```
sha256(chunk_id | retrieval_text | generation_text)
```

- **相同 hash** → 跳过（零开销幂等）
- **不同 hash** → 更新向量 + 元数据
- **新增 chunk** → 插入
- **缺失 chunk** → 标记为 orphaned → 从 store 中删除

```bash
# 重复运行同一 PDF → 全量跳过
C:/Users/30811/miniconda3/envs/demo/python.exe -m indexer.pipeline pdf_pipeline/output/MV-CC/rag_chunks.json
# Output: 0 new, 0 updated, 80 skipped, 0 orphaned
```

## 索引生命周期状态机

单次 `run(rag_chunks.json)` 处理**一个 paper**，其 chunk 生命周期四态：

```
rag_chunks.json (single paper)
        │
        ▼
  sync() ── content-hash 分类
        │
        ├─ new       chunk_id 不在 store          → insert
        ├─ updated   hash 变更                     → re-embed + upsert
        ├─ skipped   hash 不变                     → 幂等跳过
        └─ orphaned  在 store、但本 paper 已移除   → delete（paper 作用域）
```

**三态语义**（新增/更新/删除）：

| 状态 | 判定 | 动作 |
|------|------|------|
| `new` | `chunk_id` 不在 `store.get_existing_hashes()` | embed + insert |
| `updated` | hash 不同 | re-embed + upsert（覆盖） |
| `skipped` | hash 相同 | 无操作（零开销幂等） |
| `orphaned` | 在 store、不在本 paper 的 current_ids、**且同 paper 作用域** | delete |

**paper 作用域（关键）**：chunk_id 由 pipeline 前缀化为 `{paper}__{chunk}`。
孤儿检测按 `__` 前缀限定在当前 paper，避免单 paper 增量索引时把**其他 paper**
的 chunk 误判为孤儿删除（见 `DedupManager.sync()`）。

**已知记账限制**：`apply()` 返回的 `inserted` 是 `new + updated` 的合计
（底层 `upsert` 不区分新增/覆盖），`updated` 恒为 0。状态机的**分类**是正确的，
仅**写入统计**不拆分二者。如需拆分，改 `VectorStoreAdapter.upsert` 返回分项计数。

## 输出文件

| 文件 | 格式 | 用途 |
|------|------|------|
| `indexer/data/chroma/` | Chroma 持久化 | 向量库本地存储 |
| `indexer/data/eval_manifest.jsonl` | JSONL | Ragas 评估输入 |
| Pipeline stdout | JSON | 结构化可观测日志 |

### eval_manifest.jsonl 格式

```json
{"query": "", "ground_truth_chunk_ids": ["chunk_0001"], "retrieval_text": "...", "generation_text": "...", "content_type": "body", "section_path": "Introduction"}
```

## 模块边界

**此模块是数据准备层，不是检索服务。**

- ✅ 向量化、索引构建、增量同步
- ✅ PII 检测、可观测性日志
- ✅ 评估数据导出
- ❌ Query Rewriting
- ❌ Reranking
- ❌ Prompt Template
- ❌ 检索 API 服务

检索层在本模块输出的向量库 + eval_manifest 之上构建。

## 论文目录 (catalog.py)

本地知识库「有哪些论文」答案的唯一来源。每个文件的元数据以结构化 JSON 存 Redis，
与向量库（Qdrant）保持一致——`indexed` 只在 chunks 真正写入 Qdrant 后才置 true。

### Redis Schema

```
dedup:paper:{name}  → JSON（完整元数据）
dedup:papers        → set（全量论文名）
dedup:hash:{sha256} → paper_name（内容去重）
dedup:doi:{doi}     → paper_name（DOI 去重）
```

`dedup:paper:{name}` 字段：

| 字段 | 说明 |
|------|------|
| `title` / `authors` / `doi` / `year` / `arxiv_id` / `filename` | 解析器元数据 |
| `content_hash` / `page_count` | 内容指纹 / 页数 |
| `chunk_count` | rag_chunks.json 的 chunk 数（解析产物） |
| `indexed` | true ⟺ chunks 已在 Qdrant |
| `indexed_chunk_count` / `indexed_at` | 实际入库 chunk 数 / 时间 |

### API

```python
from indexer.catalog import (
    register_paper, mark_indexed, list_papers,
    search_by_metadata, is_duplicate,
)

register_paper(name, metadata, page_count=3, chunk_count=42)   # 解析完成，indexed=false
mark_indexed(name, 42)                                          # 进 Qdrant 后置 indexed=true
list_papers()                                                   # 列论文（只查 Redis，不碰向量库）
search_by_metadata(author="", year="2024", keyword="")          # 元数据过滤
is_duplicate(content_hash=sha)                                  # 内容/DOI 去重
```

### 一致性契约

- **解析完成 ≠ 已入库**。`register_paper` 置 `indexed=false`；只有 `IndexerPipeline`
  写 Qdrant 成功后才由 `mark_indexed` 置 true。
- 两个入口都会同步：`python -m indexer.pipeline`（CLI）和 `POST /api/index/run`（HTTP）。
- Redis 不可用时降级 `eval_output/paper_registry.json`（JSON 冷备份）。

自检：`python -m indexer.catalog`（注册→查重→mark_indexed→list→deregister 往返）。

## 设计约束

1. **文件即契约**: rag_chunks.json 为只读输入，所有衍生数据在本模块独立持久化
2. **零状态**: 每步无状态，通过 Content Hash 保证幂等
3. **适配器模式**: 向量库/Embedding/Context Assembly 通过接口抽象
4. **失败降级**: Embedding API 失败 → 仅写 BM25 + 原文，Pipeline 不中断
5. **多粒度**: Dense + Sparse 双路索引，充分利用上游 [KEYWORDS] 和邻居链
