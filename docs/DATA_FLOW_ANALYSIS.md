# PDF 上传后数据流问题分析与解决方案

## 一、完整数据流拓扑

```
Frontend (App.tsx)
  │
  ├─[1] POST /api/pdf/process (multipart PDF)
  │     └─ _run_pipeline() 后台线程
  │          ├─ Stage 1-2: parse → raw.md + bindings.json
  │          ├─ Stage 3:   enhance → *_enhanced.txt + *_description.txt
  │          ├─ Stage 4:   enrich → final_enriched.md
  │          ├─ Stage 5:   rag-chunk → rag_chunks.json + rag_chunks.html
  │          ├─ Position:  chunk_positions.json → Redis pos:{paper}
  │          └─ Registry:  Redis dedup:* + paper_registry.json (cold backup)
  │
  ├─[2] POST /api/index/run (auto-triggered after upload)
  │     └─ IndexerPipeline.run(rag_chunks.json)
  │          ├─ ContextAssembly → IndexUnit[]
  │          ├─ Embedding → dense vectors → Qdrant/Chroma
  │          ├─ Dedup → new/updated/skipped/orphaned
  │          ├─ VectorStore → upsert + delete orphans
  │          └─ Eval Export → eval_manifest.jsonl (OVERWRITE)
  │
  ├─[3] Agent tools → /api/retrieval/search
  │     └─ RetrievalService.from_config()
  │          ├─ optimal_retrieval_config.yaml (stale after upload)
  │          ├─ all_rag_chunks.json → SparseRetriever index (NEVER GENERATED)
  │          └─ Qdrant/Chroma → DenseRetriever
  │
  ├─[4] Agent tools → /api/reader/*
  │     ├─ sections/chunks → rag_chunks.json (per-paper filesystem)
  │     ├─ abstract → raw.md (per-paper filesystem)
  │     └─ position → Redis pos:{paper} → chunk_positions.json fallback
  │
  └─[5] PDF Viewer → /api/pdf/file/{paper}
        └─ PDF from data/uploads/ or pdf_pipeline/output/{paper}/
```

## 二、模块输出对应关系问题

### 问题 1: `all_rag_chunks.json` 从未生成

**现象**:
[retrieval.py:35](retrieval/service.py#L35) 和 [retrieval.py:45](web/api/routers/retrieval.py#L35) 中 `RetrievalService` 依赖 `eval_output/all_rag_chunks.json` 构建 SparseRetriever 索引。但没有任何流水线步骤生成此文件。

每个论文独立产出 `pdf_pipeline/output/{paper}/rag_chunks.json`，chunk_id 格式为 `chunk_0001`（无 paper 前缀）。合并步骤缺失。

**影响范围**:
- `/api/retrieval/search` 的 sparse 模式完全不可用（`rag_chunks` 参数为 None 时 `SparseRetriever` 为 None）
- Agent 的 `search_literature` 工具在 sparse 策略下返回空结果
- 检索评估只能对单篇论文运行（`run_experiments` 接受单个 `rag_path`）

**根因**: 设计上 `all_rag_chunks.json` 是跨论文聚合文件，但 IndexerPipeline 只处理单篇 rag_chunks.json，没有聚合步骤。

### 问题 2: `optimal_retrieval_config.yaml` 上传后过期

**现象**:
`RetrievalService.from_config()` 在首次调用时加载 `eval_output/optimal_retrieval_config.yaml`，之后缓存在模块全局变量 `_service` 中。新论文上传并索引后：
- 向量库有新数据（Qdrant 已更新）
- 但检索策略配置（method/top_k/hybrid_mode 等）仍基于旧数据
- `_service` 全局变量永不过期，除非重启服务

**影响范围**:
- Agent 检索质量随论文数量增加而下降
- SparseRetriever 的 TF-IDF 词表不包含新论文的术语

**根因**: 检索策略离线评估 → 在线服务之间有"评估-部署"鸿沟。上传触发索引但不触发重新评估。

### 问题 3: `eval_manifest.jsonl` 被覆盖而非合并

**现象**:
[indexer/pipeline.py:216-217](indexer/pipeline.py#L216) 中 `export_eval_manifest()` 每次写入固定路径 `self.config.eval.export_path`。上传 Paper A 后上传 Paper B：
- Paper B 的 eval manifest 覆盖 Paper A 的
- 检索评估只能基于最新一篇论文的 chunks

**影响范围**:
- 离线检索评估的数据集不完整
- LLM-as-Judge 评估只在单论文子集上运行

**根因**: 每篇论文独立索引，但 manifest 是全局单文件。

### 问题 4: chunk_id 命名空间冲突

**现象**:
每篇论文的 rag_chunks.json 中 chunk_id 为 `chunk_0001`, `chunk_0002`...。如果合并到 `all_rag_chunks.json` 时不加前缀，不同论文的同名 chunk 会冲突。

**当前缓解**: `reader.py` 的 `_normalize_chunk_id()` 尝试剥离 `{paper_name}__` 前缀（[reader.py:54-65](web/api/routers/reader.py#L54)），说明设计上预期有前缀，但 rag_chunker 并未产出带前缀的 chunk_id。

**根因**: chunk_id 格式在 rag_chunker（产出）和 reader（消费）之间不一致。

## 三、数据库 / Redis / 前端数据一致性问题

### 问题 5: 论文列表的双路径不一致

**两条路径**:

| 路径 | 端点 | 数据源 | 格式 |
|------|------|--------|------|
| A | `GET /api/pdf/outputs` | 文件系统扫描 `pdf_pipeline/output/` | `{paper_name, files[], chunk_count}` |
| B | `GET /api/reader/papers` | Redis `dedup:papers` + `dedup:paper:{name}` | `{name, title, authors, arxiv_id}` |

**不一致场景**:
- Redis flush 后路径 B 返回空，但路径 A 仍列出论文（文件还在）
- 手动删除 `pdf_pipeline/output/{paper}/` 后路径 A 消失，但 Redis 仍保留 dedup 记录
- 前端 `availablePapers` 来自路径 A（纯文件名列表），Agent `list_papers` 工具来自路径 B（含元数据）

**根因**: 两个端点各有独立数据源，没有统一的数据权威方。

### 问题 6: Redis 位置数据无恢复机制

**现象**:
[pdf.py:177-183](web/api/routers/pdf.py#L177) 中 `push_to_redis_json()` 写入位置数据到 Redis，同时 [position_map.py:105-111](pdf_pipeline/position_map.py#L105) 写 `chunk_positions.json` 到磁盘。但：
- Redis 重启后位置数据全部丢失
- `chunk_positions.json` 是本地冷备份，没有被自动恢复到 Redis
- 前端调用 `/api/reader/{paper}/positions` 时走 Redis → JSON fallback，Redis 挂了每次请求都读文件

**影响范围**:
- Redis 重启后 PDF viewer 高亮跳转延迟增加（每次读 JSON 文件）
- 如果 chunk_positions.json 也被删除，PDF viewer 完全无法定位

**根因**: 缺少 Redis 数据恢复守护逻辑或启动时的 warm-up。

### 问题 7: RetrievalService 缓存与向量库不同步

**现象**:
- `web/api/routers/retrieval.py` 中 `_service` 是模块级全局变量，lazy-init 后永久缓存
- `web/api/routers/index.py` 中 `_store` 同理
- 新论文索引后向量库已更新（新向量已写入 Qdrant），但 `_service` 的 SparseRetriever 词表没有更新
- 前端调用 `/api/index/search` 可以搜到新论文（走 Qdrant），但 Agent 调用 `/api/retrieval/search` 搜不到（sparse 索引过期）

**影响范围**:
- `/api/index/search` 和 `/api/retrieval/search` 行为不一致
- Agent 和前端直接搜索的结果不同

**根因**: 两个检索入口各有独立的缓存生命周期，且都在上传完成后无 invalidation。

### 问题 8: 前端上传→索引链路缺少错误传递

**现象**:
[App.tsx:111-130](web/frontend/src/App.tsx#L111) 中上传成功后自动触发索引：
```typescript
const idx = await api.runIndexing(ragPath);
// 轮询索引状态，但失败只 console.warn
```
- 索引失败时前端无用户可见提示
- 上传成功但索引失败 → Agent 搜不到新论文，用户不知道为什么
- `_run_pipeline` 成功 → 文件产出完整，但 `_run_indexing` 可能因为 embedding API 限流失败

**根因**: 前端把上传和索引视为一个原子操作，但后端是两个独立的后台任务，缺少事务性保障。

## 四、解决方案

### 方案概览

```
┌───────────────────────────────────────────────────────────────┐
│                    三阶段修复路径                               │
├──────────────┬──────────────────────┬─────────────────────────┤
│ 阶段 1: 最小  │ 阶段 2: 数据权威统一  │ 阶段 3: 事件驱动解耦     │
│ 修复断裂链路   │ 消除双路径不一致      │ 完整生命周期管理          │
│ (~80 行)      │ (~120 行)            │ (~200 行 + 新文件)       │
└──────────────┴──────────────────────┴─────────────────────────┘
```

---

### 阶段 1: 修复断裂的数据链路（最小可行修复）

#### 1a. 新增 `all_rag_chunks.json` 聚合步骤

**位置**: `indexer/pipeline.py` 或新增 `pdf_pipeline/merge_chunks.py`

**逻辑**:
```python
# indexer/pipeline.py — IndexerPipeline.run() 末尾新增
def merge_all_chunks(output_dir: str, papers: list[str]):
    """合并所有论文的 rag_chunks.json → all_rag_chunks.json"""
    all_chunks = []
    for paper in papers:
        path = Path(f"pdf_pipeline/output/{paper}/rag_chunks.json")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for ch in data.get("chunks", []):
            # 添加 paper 命名空间避免 chunk_id 冲突
            ch["chunk_id"] = f"{paper}__{ch['chunk_id']}"
            ch.setdefault("metadata", {})["paper_name"] = paper
            all_chunks.append(ch)
    
    merged = {"chunks": all_chunks, "total_chunks": len(all_chunks), ...}
    Path(output_dir).write_text(json.dumps(merged, ...))
    return len(all_chunks)
```

**触发时机**: 每次 `IndexerPipeline.run()` 完成后自动执行。

**改动量**: ~40 行，`indexer/pipeline.py`。

#### 1b. chunk_id 统一加 paper 前缀

**位置**: `pdf_pipeline/rag_chunker.py` 或 `indexer/context_assembler.py`

**逻辑**: 在 chunk 生成阶段就直接产出 `{paper_name}__chunk_0001` 格式的 ID，而不是事后合并时再加。

**改动量**: ~10 行，`context_assembler.py` 中 `assemble()` 方法。

#### 1c. 索引完成后刷新 RetrievalService 缓存

**位置**: `web/api/routers/retrieval.py` + `web/api/routers/index.py`

**逻辑**: 在 `_run_indexing` 的 done 回调中调用 `_invalidate_retrieval_service()`:
```python
# web/api/routers/retrieval.py
def invalidate():
    global _service, _service_init_error
    _service = None
    _service_init_error = None
```

前端在索引轮询 done 后额外调用一次 `/api/retrieval/config` 触发 lazy re-init。

**改动量**: ~15 行，`retrieval.py` + `index.py`。

#### 1d. eval_manifest.jsonl 改为追加模式

**位置**: `indexer/pipeline.py`

**逻辑**: `export_eval_manifest()` 改用 append 模式 + 去重（按 chunk_id 去重）。

**改动量**: ~15 行，`pipeline.py`。

**阶段 1 总改动**: ~80 行，4 个文件。

---

### 阶段 2: 数据权威统一（消除双路径）

#### 2a. 论文元数据以 Redis 为主，文件系统为从

**问题**: `GET /api/pdf/outputs` 扫描文件系统，`GET /api/reader/papers` 读 Redis，两者可能不一致。

**方案**: 统一到 Redis 作为唯一权威源，文件系统仅用于判断"是否有产出文件"。

**逻辑**:
```python
# GET /api/pdf/outputs 改为先读 Redis，再交叉校验文件系统
async def list_outputs():
    r = _get_redis()
    if r:
        names = r.smembers("dedup:papers")
    else:
        names = _scan_filesystem()  # fallback
    
    results = []
    for name in names:
        # Redis 有元数据就用 Redis，否则 fallback 到文件扫描
        meta = json.loads(r.get(f"dedup:paper:{name}") or "{}") if r else {}
        files = _list_output_files(name)  # 文件系统扫描只拿文件列表
        results.append(PaperOutput(
            paper_name=name,
            files=files,
            chunk_count=meta.get("chunk_count", _count_chunks(name)),
        ))
    return results
```

**改动量**: ~30 行，`pdf.py`。

#### 2b. Redis 启动 warm-up：从 JSON 冷备份恢复

**位置**: 新增 `web/api/startup.py` 或在 `main.py` 的 startup event 中

**逻辑**:
```python
async def warmup_redis():
    """从 paper_registry.json + chunk_positions.json 恢复 Redis 数据"""
    r = _get_redis()
    if not r or r.dbsize() > 0:
        return  # Redis 已有数据，跳过
    
    # 恢复 dedup 索引
    if REGISTRY_PATH.exists():
        reg = json.loads(REGISTRY_PATH.read_text(...))
        for name, meta in reg.get("papers", {}).items():
            r.set(f"dedup:paper:{name}", json.dumps(meta))
            r.sadd("dedup:papers", name)
            if h := meta.get("content_hash"):
                r.set(f"dedup:hash:{h}", name)
    
    # 恢复 position map
    for paper_dir in OUTPUT_DIR.iterdir():
        pos_file = paper_dir / "chunk_positions.json"
        if pos_file.exists():
            positions = json.loads(pos_file.read_text(...))
            mapping = {k: json.dumps(v) for k, v in positions.items()}
            r.hset(f"pos:{paper_dir.name}", mapping=mapping)
```

**改动量**: ~40 行，新文件 `web/api/warmup.py` + `main.py` startup 事件。

#### 2c. 前端 `handleCitationClick` 错误处理

**问题**: [App.tsx:164-189](web/frontend/src/App.tsx#L164) 中 `api.getPositions()` 失败时整个 `try/catch` 为空（`catch {}`），用户点击引用但 PDF 不跳转，无任何反馈。

**方案**: 至少给一个 fallback 提示。

**改动量**: ~5 行，`App.tsx`。

**阶段 2 总改动**: ~120 行，4 个文件。

---

### 阶段 3: 事件驱动解耦（完整生命周期）

#### 3a. 引入轻量事件总线替代前端轮询 + 硬编码调用链

**问题**: 当前 `App.tsx` 硬编码了 upload → poll → index → poll 的线性链路。每个步骤都是前端手动触发，无法复用。

**方案**: 后端在 pipeline 完成时自动触发后续步骤，前端只需监听状态变化。

```python
# 新增 web/api/events.py — 简单的事件回调链
class PipelineEvents:
    """ponytail: in-process callback chain, no message queue needed."""
    
    _handlers: dict[str, list[callable]] = {
        "pdf.done": [],       # PDF pipeline 完成
        "index.done": [],     # 索引完成
        "eval.done": [],      # 评估完成
    }
    
    @classmethod
    def on(cls, event: str, handler):
        cls._handlers.setdefault(event, []).append(handler)
    
    @classmethod
    async def emit(cls, event: str, **data):
        for h in cls._handlers.get(event, []):
            try:
                await h(**data)
            except Exception as e:
                print(f"[EVENT] {event} handler failed: {e}")
```

**注册链**:
```python
# main.py startup
PipelineEvents.on("pdf.done", auto_index)       # PDF 完成 → 自动索引
PipelineEvents.on("index.done", merge_chunks)    # 索引完成 → 合并 all_rag_chunks
PipelineEvents.on("index.done", invalidate_retrieval)  # 索引完成 → 刷新检索缓存
PipelineEvents.on("merge.done", trigger_eval)    # 合并完成 → 可选触发评估
```

**前端简化**: 上传后只轮询一个状态端点，后端自动完成全链路。

**改动量**: ~80 行新文件 + `pdf.py`/`index.py` 改造 ~40 行 + 前端简化 ~30 行。

#### 3b. 上传-索引事务性保障

**问题**: 上传成功但索引失败 → 静默不一致。

**方案**: 在 pdf pipeline 的 done 回调中自动触发索引（后端驱动），索引失败时在 task result 中标记 `indexing_failed: true`。前端可据此展示"论文已处理但检索暂不可用"。

**改动量**: ~50 行，`pdf.py` + `index.py` + `App.tsx`。

**阶段 3 总改动**: ~200 行 + 1 新文件。

---

## 五、方案对比与推荐

| 维度 | 阶段 1 | 阶段 2 | 阶段 3 |
|------|--------|--------|--------|
| 解决问题 | 1, 2, 3, 4, 7 | 5, 6, 8 | 全部 + 架构优化 |
| 改动量 | ~80 行 | ~120 行 | ~200 行 + 1 文件 |
| 新增抽象 | 0 | 0 | 1 (事件总线) |
| 破坏性变更 | chunk_id 格式变更 | 无 | API 行为变更 |
| 风险 | 低（补全缺失功能） | 低（加固已有逻辑） | 中（引入新架构模式） |
| 何时做 | 立即 | 阶段 1 完成后 | 论文数 > 10 或检索策略变更频繁时 |

**推荐**: 先执行阶段 1 + 阶段 2 的 2a 和 2c（最关键的断裂链路 + 数据权威统一），总计 ~110 行。阶段 3 的事件总线在论文数量增长到需要频繁重新评估时再引入——当前 2-3 篇论文的规模下，手动触发评估的成本远低于维护事件系统的成本。

## 六、不做的事项

| 事项 | 理由 |
|------|------|
| 引入 Celery/RQ 等任务队列 | 当前单机场景 threading 够用，日处理 < 50 篇 |
| 数据库（PostgreSQL）替代 Redis + JSON | 当前数据结构简单（论文列表 < 1000 条），Redis + JSON 冷备份够用 |
| 前端 WebSocket 实时推送 | 轮询 1.5s 间隔对当前处理时间（~30s per paper）足够 |
| chunk_id 全局 UUID | 破坏可读性和调试体验，paper 前缀方案更直观 |
| 自动重新评估最优检索策略 | 每次上传都跑全量评估成本高（LLM API 调用多），适合作为手动触发的 `/api/eval/run` 端点 |
