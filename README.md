# Demo — 科研文献 RAG 系统 v2

从 PDF 解析到检索增强生成的完整管道，面向遥感/计算机视觉科研文献。

> **v1 已弃用** — 旧版 LangChain RAG + Agent 实现已归档至 [`v1/`](v1/)。架构从「通用文档 RAG + Agent 自主决策」重构为「PDF 原生解析 → 多粒度索引 → 可评估检索」的专业文献处理管道。

## 架构

```
data/*.pdf
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  pdf_pipeline         五阶段 PDF 预处理管道             │
│  parse → bindings → enhance → enrich → rag-chunk     │
│  输出: rag_chunks.json                                │
└──────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│  indexer              多粒度向量索引构建                │
│  chunk → embedding → vector store (Qdrant)            │
└──────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│  retrieval            共享检索层                       │
│  SparseRetriever + DenseRetriever + Fusion            │
│  RetrievalService: 从最优配置加载的生产检索服务          │
└──────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│  web/api              FastAPI 后端                    │
│  /api/pdf/*  /api/index/*  /api/retrieval/*           │
└──────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│  web/frontend         React 前端                      │
│  三栏布局: 对话列表 | Agent 聊天 | 文件浏览            │
└──────────────────────────────────────────────────────┘

验证闭环:
  retrieval_orchestrator → 离线评估 → optimal_retrieval_config.yaml
                                    → RetrievalService 自动加载
```

## 项目结构

| Directory | Purpose |
|-----------|---------|
| `pdf_pipeline/` | **PDF 预处理管道** — 五阶段 PDF→RAG 切分流水线 |
| `indexer/` | **索引编排层** — 多粒度向量索引构建 + 增量同步 |
| `retrieval/` | **共享检索层** — Sparse + Dense + Fusion + RetrievalService |
| `retrieval_orchestrator/` | **离线检索评估** — QA 生成 → 多策略检索 → 指标 → 最优配置 |
| `web/api/` | **FastAPI 后端** — 所有模块的 HTTP 接口 |
| `web/frontend/` | **React 前端** — Agent 对话 + 文件浏览三栏界面（Electron 桌面客户端） |
| `agent/` | **Agent 核心** — LangGraph 对话 agent + Docling 解析 + Chunk 可视化 |
| `skills/` | 技能注册与模板 |
| `mcphub/` | MCP Hub 服务端 |
| `mcp_simple_arxiv/` | ArXiv MCP 服务端 |
| `eval_output/` | 评估产出（manifest、报告、最优配置、论文注册表） |
| `v1/` | **已弃用** — 旧版 LangChain RAG + Agent 实现 |

## 模块

### pdf_pipeline — PDF 预处理管道

五阶段流水线将 PDF 转化为 RAG 就绪的结构化文档。详见 [pdf_pipeline/README.md](pdf_pipeline/README.md)。

| 阶段 | 模块 | 功能 |
|------|------|------|
| 1. Parse | `parser.py` | Docling 布局解析 → Markdown + 元数据 + SHA256 指纹 |
| 2. Bindings | `bindings.py` | 空间坐标提取（图/表/公式 bbox）+ 引用回溯 |
| 3. Enhance | `enhancer.py` | LLM 公式描述 + VLM 图片描述（批处理/并行） |
| 4. Enrich | `enricher.py` | 增强注入 final_enriched.md |
| 5. Chunk | `rag_chunker.py` | 结构感知切分 + BM25 关键词 + 参考文献解耦 |

输出: `pdf_pipeline/output/{paper}/rag_chunks.json`

### indexer — 索引编排层

五阶段索引管道（context→embed→dedup→store→export），Qdrant 向量存储，支持增量同步。Embedding 适配器支持 BGE/Qwen3/Jina 等模型。

### retrieval — 共享检索层

离线评估与在线 API 共用的检索组件。详见 [retrieval/README.md](retrieval/README.md)。

- `SparseRetriever` — TF-IDF 稀疏检索
- `DenseRetriever` — 向量检索（VectorStoreAdapter + EmbeddingAdapter）
- `rrf_fuse` / `weighted_fuse` — RRF + 加权融合
- `RetrievalService` — 从 `optimal_retrieval_config.yaml` 加载的生产检索服务

### retrieval_orchestrator — 离线检索评估

QA 生成 → 多策略检索（dense/sparse/hybrid）→ 指标计算 → 最优配置输出。

```bash
python -m retrieval_orchestrator evaluate --config retrieval_orchestrator/evaluation.yaml
```

质量门禁：Recall@5 < 0.6 阻断，回归检测 >5% 警告。

### web/api — FastAPI 后端

所有功能通过 HTTP 接口暴露。详见 [web/api/README.md](web/api/README.md)。

```
GET  /api/health
POST /api/pdf/process          ← PDF 上传 + 去重 + 全流程
POST /api/pdf/process-local    ← 处理磁盘上已有 PDF（download_paper 下载 / Web 上传；支持 pdf_path）
GET  /api/pdf/status/{id}
GET  /api/pdf/outputs
POST /api/index/run            ← 索引构建
POST /api/index/search         ← 向量检索
GET  /api/index/stats
POST /api/retrieval/search     ← 最优策略检索
GET  /api/retrieval/config
GET  /api/reader/{paper}/sections      ← 章节导航
GET  /api/reader/{paper}/abstract      ← 论文摘要
GET  /api/reader/{paper}/chunks/{id}/context  ← chunk 上下文
GET  /api/reader/papers               ← 元数据搜索
GET  /api/reader/local-papers         ← 本地论文三态快照 (indexed/parsed/raw)，agent 入库决策梯用
POST /api/agent/chat                 ← Agent 对话
POST /api/agent/chat/stream          ← Agent SSE 流式对话
GET  /api/agent/health               ← Agent 状态
```

> **下载与入库解耦**：`download_paper`（Agent 工具）是纯文件下载——落盘到用户指定的工作区目录（默认 `data/downloads/`）、默认按 arXiv 标题推导的论文简称命名，不触发任何解析/切分/向量化；`ingest_paper` 是独立的**原子「入库」**（解析 PDF → 写入向量库，一个后台任务），仅在用户明确要求入库/导入时才调用。两个目录 `data/uploads/`（Web 上传）与 `data/downloads/`（Agent 下载）均为 raw 文件扫描范围。

### web/frontend — React 前端

三栏布局的学术文献 RAG 界面。详见 [docs/frontend-design.md](docs/frontend-design.md)。

| 区域 | 功能 |
|------|------|
| LeftPanel（260px，可折叠） | 对话线程管理：新建/切换/删除，localStorage 持久化 |
| MainContent（flex:1） | Agent 聊天：SSE 流式输出，用户消息右对齐/系统左对齐，流式时输入禁用 |
| RightPanel（420px，可折叠） | 工作区文件浏览器：目录树 + 文件内容预览（.md 渲染为 Markdown） |

```bash
cd web/frontend
npm install
npm run dev     # → http://localhost:5173 (Vite proxy → :8000)
```

### agent — Agent 核心

**LangGraph 对话 Agent**，自主调用检索/阅读工具回答学术文献问题。详见 [agent/README.md](agent/README.md)。

| 文件 | 功能 |
|------|------|
| `graph.py` | StateGraph 构建 + InMemorySaver + `run()` 入口 |
| `nodes.py` | 8 节点 LLM 管道: understand→memory→resolve→agent→tools→synthesize→chat→clarify |
| `tools.py` | 6 个 @tool 函数 + CompositeToolProvider (Builtin/MCP/Skill) |
| `prompts.py` | System prompt 模板 |
| `state.py` | AgentState + 结构化输出模型 |
| `memory.py` | MemoryManager — buffer + summary + profile 结构化上下文组装 |
| `search_loop.py` | Search 子图 — agent ↔ tools ReAct 循环封装 |
| `resolution.py` | 确定性引用解析 — 论文/章节模糊引用的 pre-LLM 消歧 |
| `docling_parser.py` | Docling PDF 解析器 |
| `academic_chunker.py` | 学术文献切分策略 |
| `chunk_viz.py` | Chunk HTML/JSON/Streamlit 可视化 |
| `binding_export.py` | 空间绑定导出 |

## 快速开始

### 环境

```bash
# 激活 conda 环境
conda activate demo

# 或直接使用解释器路径
C:/Users/30811/miniconda3/envs/demo/python.exe -m pip install -r requirements.txt
```

### 启动后端 API

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

### 启动前端

```bash
cd web/frontend
npm install     # 首次运行
npm run dev     # → http://localhost:5173
```

Vite proxy 自动将 `/api/*` 转发到 `:8000`，无需额外 CORS 配置。

### PDF 全流程处理

```bash
# CLI
C:/Users/30811/miniconda3/envs/demo/python.exe docling_cli.py all data/paper.pdf

# API
curl -X POST http://localhost:8000/api/pdf/process -F "file=@data/paper.pdf"
```

### 检索评估

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m retrieval_orchestrator evaluate \
  --config retrieval_orchestrator/evaluation.yaml
```

## 数据流

```
PDF 上传
  → SHA256 去重检查 (paper_registry.json)
  → pdf_pipeline (5 stages)
  → rag_chunks.json
  → indexer (embedding → vector store)
  → retrieval_orchestrator (评估 → optimal config)
  → RetrievalService (生产检索)
```

## Redis 键全景

同一 Redis 实例 (`127.0.0.1:6379`)，前缀隔离：

```
dedup:hash:{sha256}     string  去重指纹索引
dedup:doi:{doi}         string  去重 DOI 索引
dedup:paper:{name}      string  论文元数据
dedup:papers            set     全量论文名集合
task:{task_id}          hash    后台任务状态 (TTL 1h)
task:list               zset    任务时间线
```

## v1 归档

旧版（LangChain RAG + Chroma + BGE + ReAct/Plan-Execute Agent + Streamlit UI）已归档至 [`v1/`](v1/)。v2 重构要点：

- PDF 原生解析（Docling）替代通用 TextSplitter
- 五阶段管道替代 ingest 单步
- 稀疏+稠密混合检索替代纯 BGE 向量检索
- 离线评估驱动配置选择替代手工调参
- FastAPI 薄封装替代 Streamlit 直调模块
