# Demo Project

## 开发准则

**FastAPI 封装原则**: 所有功能模块（pdf_pipeline、indexer 等）必须通过 FastAPI 接口暴露给前端调用，禁止前端直接 import 模块函数。模块内部保持纯 Python API，`web/api/` 层负责 HTTP 序列化、文件上传、后台任务调度。业务逻辑不得写入 API 层。

- 新功能必须先有 API endpoint，前端通过 `fetch/axios` 调用，不走 `from xxx import yyy`。
- API 层是薄封装：校验输入 → 调度后台任务 → 返回 JSON。不包含领域逻辑。
- 运行方式：`uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload`

**模块文档同步**: 对任何模块进行开发（新增/修改/删除功能）时，必须同步更新该模块的 README.md（不存在则创建）。CLI 变更需更新用法示例，API 变更需更新编程示例，新增模块需更新架构图。目标是让后来者（包括未来的自己）通过 README 即可快速了解模块当前状态。

**错误沉淀**: 遇到非 trivial 的错误（环境冲突、segfault、库不兼容等）并解决后，必须将现象、根因、解决方案写入 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。格式：`## 模块名 / 错误类型` → `现象` → `原因` → `解决`。同类错误不重复记录，已有条目覆盖的只补充新信息。

## Prompt 设计原则

所有 LLM/VLM prompt（enhancer、rag_chunker 等模块）遵循四条核心约束：

### 1. 结构化输出

- Prompt 必须指定明确的输出格式（如 `[FORMULA:id]: desc`、`[FORMULA_DESC: ...]`），禁止模型自由发挥。
- 输出格式需可机器解析（正则/JSON/固定分隔符），不能依赖自然语言后处理提取信息。
- 在 prompt 中以 `Output ONLY in this exact format` 或 `Output ONLY the description, no preamble` 收束模型行为。

### 2. 零状态与幂等性

- 每条 prompt 自包含全部必要信息，不依赖对话历史或上一条 prompt 的上下文。
- 同一输入重复调用 → 相同输出。禁止隐式状态（如 "as discussed"、"continue from above"）。
- 批处理 prompt 中每条记录的处理规则必须与单条 prompt 等价（fallback 时行为一致）。

### 3. 模型无关性

- Prompt 不写入特定模型名（如 "as qwen3.6..."）或特定模型行为假设。
- 模型名称通过 `client.chat.completions.create(model=...)` 代码层面注入，prompt 只描述任务目标。
- 避免依赖特定模型的输出偏好（如 "you tend to output shorter"），用 token 上限和示例约束替代。

### 4. 上下文感知

- 尽可能向 prompt 注入可利用的周边信息：摘要（abstract）、正文引用上下文（ref_context）、标题层级（section_path）。
- 上下文信息作为前置约束，而非可选参考。公式/图片描述必须结合其在论文中的具体角色，而非仅基于公式自身符号推导。
- 上下文长度有预算上限（abstract ≤500 chars，ref_context ≤3 entries），防止 prompt 膨胀降低遵循度。

## Environment

使用 conda 虚拟环境 `demo`，运行任何 Python 脚本或安装依赖前需先激活。

**Bash 命令中无法使用 `conda activate`**（conda 未 init），改用直接路径：

```bash
# Python 解释器
C:/Users/30811/miniconda3/envs/demo/python.exe

# pip
C:/Users/30811/miniconda3/envs/demo/python.exe -m pip

# 示例：运行脚本
C:/Users/30811/miniconda3/envs/demo/python.exe -m web.cli ingest
C:/Users/30811/miniconda3/envs/demo/python.exe agent/some_script.py
```

在终端中可手动 `conda activate demo` 后直接使用 `python` / `pip`。

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `agent/` | **LangGraph Agent 框架** — 论文问答编排（graph.py：understand→memory→resolve→react/plan→search→synthesize），也保留 docling 解析/切分/可视化工具 |
| `agent/graph.py` | **状态机入口** — `build_graph()` / `run(query, thread_id=...)`；AsyncSqliteSaver → 根目录 checkpoints.db 跨重启持久化 |
| `agent/nodes.py` + `agent/plan.py` + `agent/resolution.py` + `agent/search_loop.py` | 节点实现 — understand/memory/synthesize/chat/clarify + plan-and-execute（plan_node/executor_node）+ 论文指称消解（resolve_node）+ agent↔tools ReAct 子图 |
| `agent/providers/generic_provider.py` | **通用工具集** — read_file/list_dir/get_time/calculator/write_file/fetch_url（路径越界拒绝 + SSRF 守卫 + 权限门） |
| `agent/providers/builtin_provider.py` | **论文域工具** — search_papers/fetch_content/download_paper/ingest_paper/check_paper/check_task_status（内部走 FastAPI 链路） |
| `agent/providers/mcp_provider.py` | **MCP 工具装配** — 从 `.mcp.json`（`load_mcp_config`）加载外部 MCP 工具 |
| `agent/docling_parser.py` + `agent/academic_chunker.py` + `agent/chunk_viz.py` | **Docling PDF 解析 / 学术切分 / Chunk 可视化**（独立 CLI 链，见下） |
| `pdf_pipeline/` | **PDF 预处理管道** — 五阶段 PDF→RAG 切分流水线 |
| `indexer/` | **知识索引与存储编排层** — 多粒度向量索引构建 + 增量同步 |
| `docling_cli.py` | **CLI 工具** — 命令行 PDF 解析 & 切分（项目根目录） |
| `mcphub/` | MCP Hub 服务端实现 |
| `mcp_simple_arxiv/` | ArXiv MCP 服务端 |
| `eval_output/` | 评估产出（manifest、报告、最优配置、论文注册表） |
| `v1/` | **已弃用归档** — 旧版 LangChain RAG + Agent 实现 |
| `retrieval/` | **共享检索层** — SparseRetriever + DenseRetriever + Fusion + RetrievalService |
| `retrieval_orchestrator/` | **离线检索验证框架** — QA 生成 → 多策略检索 → 指标计算 → 最优配置选择 |
| `retrieval_orchestrator/evaluation.yaml` | 检索评估配置模板 |
| `skills/` | 技能注册与模板 |
| `web/api/` | **FastAPI 后端** — pdf_pipeline + indexer HTTP 接口 |
| `web/frontend/` | **Electron 桌面客户端** — 主进程(src/main)拉起 uvicorn 子进程 + React 渲染进程(src/renderer) |
| `web/chunk_viz_page.py` | **Streamlit 可视化页面** — chunk 交互式浏览 |

## 架构速览 / 数据流主线

**一条主链 + 两条支线**。所有路径最终落地：Qdrant(生产)/Chroma(本地) 向量库 + Redis 论文注册表 + 本地文件。

```text
PDF
 →  pdf_pipeline/parser.parse_pdf_docling      # S1 Docling 解析 → raw.md + bindings.json + page_map.json
 →  pdf_pipeline/bindings.build_bindings       # S2 空间绑定 + 引用回溯（table/formula/picture ↔ 正文引用）
 →  pdf_pipeline/enhancer.enhance_all          # S3 LLM/VLM 语义增强绑定元素
 →  pdf_pipeline/enricher.enrich_markdown      # S4 增强注入 → final_enriched.md
 →  pdf_pipeline/rag_chunker.rag_chunk_markdown# S5 RAG 切分 → rag_chunks.json + rag_chunks.html
 →
 →  indexer/pipeline.IndexerPipeline(.run)     # merge_all_chunks → ContextAssembler(Small-to-Big) → Embedder → 增量同步
 →  Qdrant/Chroma + Redis(注册表/dedup)
 →  web/api/routers/retrieval.search           # 检索：Dense(向量库) + Sparse(BM25) 融合（RRF/加权）
```

- **支线 A（离线评估 → 检索最优配置）**：`pdf_pipeline/output/*/rag_chunks.json` → `retrieval_orchestrator generate/evaluate` → `eval_output/optimal_retrieval_config.yaml`。检索服务**懒加载**读它（`web/api/routers/retrieval.py::_get_service`），配置/数据变了要 `invalidate_retrieval_service()`。
- **支线 B（Agent 问答）**：`web/api/routers/agent.py`（/chat、/chat/stream SSE）→ `agent/graph.py::run`（LangGraph 状态机）→ `agent/providers/` 工具反向调用上述各链路。

**关键数据文件**（改动/调试先确认位置）：

| 文件 | 谁读 | 角色 |
|------|------|------|
| `pdf_pipeline/output/{paper}/rag_chunks.json` | indexer | 索引构建唯一输入 |
| `pdf_pipeline/output/{paper}/final_enriched.md` | rag_chunker | RAG 切分输入（增强注入后的正文） |
| `eval_output/optimal_retrieval_config.yaml` | web/api/routers/retrieval | 检索服务唯一策略输入（离线评估产出） |
| `eval_output/all_rag_chunks.json` | 同上 | 全部论文合流 chunk（索引后重建） |
| `eval_output/paper_registry.json` | web/api/main::_warmup_redis | Redis 冷备份，启动时恢复 registry/dedup |
| `indexer/config.yaml` | indexer + retrieval | 向量库/embedding 后端选择（qdrant vs chroma / api vs local） |
| `checkpoints.db` | agent/graph | LangGraph 会话跨重启持久化 |

## 关键符号速查（动手前先查这里，省一轮全局搜索）

> 完整符号随时用 Grep `^(class|def|async def)` 按模块扫；下表是各模块最高频入口。

### pdf_pipeline（五阶段：`python -m pdf_pipeline.cli all <pdf>`）

| 符号 | 位置 | 作用 |
|------|------|------|
| `parse_pdf_docling(file, export_bindings=True)` | `pdf_pipeline/parser.py` | S1 入口 → DoclingParseResult(markdown, bindings_path) |
| `build_bindings` / `extract_bindings_from_doc` | `pdf_pipeline/bindings.py` | S2 空间绑定 + 引用回溯 |
| `enhance_all(bindings, markdown, out_dir)` | `pdf_pipeline/enhancer.py` | S3 公式/图片 LLM 增强（prompt 遵循「Prompt 设计原则」） |
| `enrich_markdown(markdown, bindings)` | `pdf_pipeline/enricher.py` | S4 → final_enriched.md |
| `rag_chunk_markdown(markdown, bindings=, page_map=, config=)` | `pdf_pipeline/rag_chunker.py` | S5 → RAGChunkingReport（含 quality_report 质检） |
| `RAGChunkConfig` | `pdf_pipeline/rag_chunker.py` | token 配置；env 覆盖 `PAPER_CHUNK_SIZE/PAPER_CHUNK_OVERLAP`（`_config.py`） |

### indexer

| 符号 | 位置 | 作用 |
|------|------|------|
| `IndexerPipeline.run()` | `indexer/pipeline.py` | 索引构建总入口；`merge_all_chunks` 合流各论文 rag_chunks.json |
| `MultiGranularityEmbedder` | `indexer/embedder.py` | 多粒度向量化（PII 检测 + sparse keywords） |
| `ContextAssembler` | `indexer/context_assembler.py` | Small-to-Big：chunk + 邻居窗口组装 retrieval/generation 上下文 |
| `VectorStoreAdapter` / `QdrantVectorStore` / `ChromaVectorStore` | `indexer/vector_store.py` | upsert/search/删除 + 增量同步散表 |
| `DedupManager` | `indexer/dedup_manager.py` | 增量同步去重（sha256 内容指纹） |
| `register_paper` / `register_indexed` / `mark_indexed` / `patch_paper` | `indexer/catalog.py` | Redis 论文注册表；原子入库用 `register_indexed` 一次写完（indexed=true），`mark_indexed` 保留给 CLI/兼容路径，`patch_paper` 供 reconcile 回填 |
| `reconcile()` / `count_chunks_per_paper()` | `indexer/reconcile.py` | 目录 ↔ 向量库对账回填（修复旧代码缺键/标志脱节）；CLI `python -m web.cli reconcile`、HTTP `POST /api/index/reconcile` |
| `load_config()` / `IndexerConfig` | `indexer/config.py` | 读 `indexer/config.yaml` |

### retrieval（共享检索层）

| 符号 | 位置 | 作用 |
|------|------|------|
| `RetrievalService.from_config(optimal_config, indexer_config, rag_chunks)` | `retrieval/service.py` | 检索服务唯一入口 |
| `DenseRetriever` / `SparseRetriever` | `retrieval/service.py` `retrieval/sparse.py` | 稠密（向量库）/ 稀疏（BM25）检索 |
| `rrf_fuse` / `weighted_fuse` | `retrieval/fusion.py` | 多路结果融合 |

### web/api（薄封装，无领域逻辑）

| 符号 | 位置 | 作用 |
|------|------|------|
| `app` + 6 个 router 挂载 | `web/api/main.py` | 入口；启动 `_warmup_redis` 冷恢复注册表 |
| `routers/pdf.py::_run_pipeline` | `web/api/routers/pdf.py` | 后台任务：五阶段 pipeline + Redis 任务状态 |
| `routers/index.py::_run_indexing` | `web/api/routers/index.py` | 后台任务：IndexerPipeline |
| `routers/retrieval.py::_get_service` | `web/api/routers/retrieval.py` | 检索服务懒加载+缓存；重新索引后须 `invalidate_retrieval_service()` |
| `routers/agent.py`（chat / chat/stream SSE） | `web/api/routers/agent.py` | Agent 对话 HTTP 接口 |
| `_task_create` / `_task_update` / `_task_get` | `web/api/routers/__init__.py` | 任务状态 store |

### agent（LangGraph 框架）

| 符号 | 位置 | 作用 |
|------|------|------|
| `build_graph()` / `get_agent()` / `run(query, thread_id=)` | `agent/graph.py` | 状态机构建与运行；同 thread_id 保持多轮会话 |
| `understand_node` `memory_node` `resolve_node` `synthesize_node` `chat_node` `clarify_node` + `route_intent` | `agent/nodes.py` `agent/resolution.py` | 各节点实现 |
| `decide_mode` / `plan_node` / `executor_node` | `agent/plan.py` | plan-and-execute（多论文/对比类走 plan，单条走 react） |
| `build_search_subgraph()` | `agent/search_loop.py` | agent↔tools ReAct 循环（step 上限 `state.max_steps` 默认 30，`AGENT_MAX_STEPS`/兼容旧名 `AGENT_MAX_ITERATIONS`；turn 上限 `max_turns` 默认 50） |
| `ensure_tools()` / `get_cached_tools()` | `agent/tools.py` | 工具装配（builtin + generic + MCP） |
| `load_mcp_config()` / `MCPProvider` | `agent/providers/mcp_provider.py` | 从 `.mcp.json` 加载外部 MCP 工具 |
| `load_limits()` / `get_limits()` | `agent/config.py` + `agent/config.yaml` | **执行约束统一配置** — 父 agent `max_steps`/`max_turns` + 各 subagent `max_steps`（`subagents.<name>`）。优先级 env `AGENT_MAX_STEPS`/`AGENT_MAX_TURNS` > yaml > 代码默认；`state.py` 默认值与 `build_subagents` 都从它取值 |
| `AgentState` | `agent/state.py` | 图状态 schema |

### retrieval_orchestrator（离线评估）

| 符号 | 位置 | 作用 |
|------|------|------|
| generate / evaluate / search / review | `retrieval_orchestrator/cli.py` | 离线工作流入口 |
| `evaluator.evaluate` | `retrieval_orchestrator/evaluator.py` | 指标（Recall/NDCG/MRR + bootstrap CI + 维度拆解） |
| `retrieval_engine.run_experiments` | `retrieval_orchestrator/retrieval_engine.py` | 策略网格实验（RRF/HyDE/hybrid） |
| `config_selector.select_optimal` | `retrieval_orchestrator/config_selector.py` | 最优配置 + 质量门禁（`QualityGateError`） |

## 已知陷阱索引（详情看 TROUBLESHOOTING.md 对应章节）

| 情境 | 陷阱 → 应对 | 详见 |
|------|-------------|------|
| 跑任何 python | bash 里 `conda activate` 无效 → 用完整路径 `C:/Users/30811/miniconda3/envs/demo/python.exe` | 环境/conda 无效 |
| import docling / sentence_transformers | **segfault (139/0xC0000005)** — 导入链与 onnxruntime 冲突；根目录 CLI 用脚本形式跑，不要 `python -m` 导入 | 环境 + indexer/sentence_transformers |
| Chroma 清理 | Windows 文件锁导致删不掉 → 先关持有进程 | indexer/Chroma 文件锁 |
| 检索不到最新论文 | `eval_output/all_rag_chunks.json` 过期或检索服务缓存未失效 → 索引后调 `invalidate_retrieval_service()` | web/api/library |
| Agent 内置工具连接失败 | `AGENT_API_BASE` 端口与后端不符 | web/frontend |
| HF 下载 / npm / electron | 国内网络卡住 → 镜像源（见 TROUBLESHOOTING） | 通用 + web/frontend |
| agent 产出空 | subagent 无 final answer → 查 synthesize 节点错误反馈 | agent |
| 写作「聊天回全文、doc 只落最后一章」 | creator 未落盘 + config 未透传 → 查 TROUBLESHOOTING「写作链路」；修复 = 落盘校验 + 串行 + 进度 synthesize | agent |

## Dependencies

安装/更新依赖：

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m pip install -r requirements.txt
```

## Docling PDF 解析 & Chunk 可视化

基于 IBM Docling 的科研论文 PDF 解析，支持章节感知切分和交互式可视化。

### 快速使用

```bash
# ⚠️ 注意：CLI 脚本在项目根目录运行，不要用 python -m 导入
# （避免 agent/__init__.py 导入链与 docling onnxruntime 冲突导致 segfault）

# 命令行：解析 + 切分 + 查看摘要
C:/Users/30811/miniconda3/envs/demo/python.exe docling_cli.py chunk data/your_paper.pdf

# 生成 HTML 可视化（在浏览器中打开查看每个 chunk）
C:/Users/30811/miniconda3/envs/demo/python.exe docling_cli.py visualize data/your_paper.pdf

# 完整流程（HTML + JSON 导出）
C:/Users/30811/miniconda3/envs/demo/python.exe docling_cli.py all data/your_paper.pdf

# Streamlit Web 界面（含 chunk 可视化）
C:/Users/30811/miniconda3/envs/demo/python.exe -m streamlit run web/chunk_viz_page.py
```

### 编程使用

```python
from agent.docling_parser import parse_pdf_docling
from agent.academic_chunker import chunk_docling_result
from agent.chunk_viz import render_chunks_html, export_chunks_json

# 解析 PDF
result = parse_pdf_docling("paper.pdf")

# 学术切分
report = chunk_docling_result(result)

# 导出 HTML
html = render_chunks_html(report.chunks)
with open("chunks.html", "w", encoding="utf-8") as f:
    f.write(html)
```

## FastAPI 后端

所有功能通过 FastAPI 接口暴露，前端通过 HTTP 调用，不直接 import 模块函数。

### 启动

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 桌面客户端（Electron）

本地客户端入口：`web/frontend/`，主进程自动拉起 uvicorn 子进程（`127.0.0.1:8001`），渲染进程通过 HTTP + SSE 调用后端，不再依赖浏览器。

```bash
cd web/frontend
npm install          # China 网络用镜像，见 TROUBLESHOOTING
npm run dev          # Vite(5173) + Electron，主进程拉起后端
npm run dist:win     # 打 Windows nsis 安装器（Python 不打包，目标机预装 demo 环境）
```

Python 后端**不打包**，从 PATH 启动 `python -m uvicorn`；桌面 app 不继承 `conda activate`，必要时设 `DEMO_PYTHON=C:/Users/30811/miniconda3/envs/demo/python.exe`。

### 开发辅助 CLI

```bash
# 查看系统状态（Redis / Qdrant / 本地文件）
C:/Users/30811/miniconda3/envs/demo/python.exe -m web.cli status

# 重置全部状态（清空 Redis + Qdrant + 本地输出 + 上传文件）
C:/Users/30811/miniconda3/envs/demo/python.exe -m web.cli reset --force
```

### API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/pdf/process` | POST | 上传 PDF，后台执行全流程（parse→enhance→enrich→rag-chunk） |
| `/api/pdf/status/{task_id}` | GET | 查询 PDF 处理任务状态 |
| `/api/pdf/outputs` | GET | 列出已处理论文 |
| `/api/pdf/outputs/{name}` | GET | 获取论文输出文件列表 |
| `/api/index/run` | POST | 后台执行索引构建（从 rag_chunks.json） |
| `/api/index/status/{task_id}` | GET | 查询索引任务状态 |
| `/api/index/search` | POST | 向量检索 |
| `/api/index/stats` | GET | 向量库统计 |
| `/api/index/reconcile` | POST | 目录 ↔ 向量库对账回填（旧数据 indexed 缺键/脱节修复；结果在 task result） |
| `/api/retrieval/search` | POST | 使用评估最优策略检索 |
| `/api/retrieval/config` | GET | 当前检索策略配置 |
| `/api/pdf/process-local` | POST | 处理本地 data/ 已有 PDF（不走上传，对应 ProcessLocalRequest） |
| `/api/agent/chat` | POST | Agent 对话（JSON 返回） |
| `/api/agent/chat/stream` | POST | Agent 对话（SSE 流式） |
| `/api/agent/health` | GET | Agent 状态 |
| `/api/reader/...` | GET | 论文阅读器 — chunk context / sections / abstract / papers 检索（全列表见 routers/reader.py docstring） |
| `/api/workspace/list`、`/api/workspace/read` | GET | workspace 文件浏览/读取 |

## 离线检索评估

评估框架验证向量库检索质量，产出评估报告和最优配置。离线运行，只读访问上游数据。

### 快速使用

```bash
# 生成评估数据集 (keyword 模式, 无需 LLM)
C:/Users/30811/miniconda3/envs/demo/python.exe -m retrieval_orchestrator generate \
  --rag-path pdf_pipeline/output/RMNet/rag_chunks.json \
  --output eval_output/eval_manifest.jsonl \
  --modes keyword --samples 50

# 完整评估流程
C:/Users/30811/miniconda3/envs/demo/python.exe -m retrieval_orchestrator evaluate \
  --config retrieval_orchestrator/evaluation.yaml

# 快速搜索测试
C:/Users/30811/miniconda3/envs/demo/python.exe -m retrieval_orchestrator search \
  --query "dual stream feature extraction" --top-k 10

# 人工审核 QA 对
C:/Users/30811/miniconda3/envs/demo/python.exe -m retrieval_orchestrator review \
  --manifest eval_output/eval_manifest.jsonl
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `eval_output/eval_manifest.jsonl` | 评估数据集 (query + ground_truth) |
| `eval_output/retrieval_*.jsonl` | 各策略检索结果 |
| `eval_output/retrieval_*.report.json` | 各策略评估报告 |
| `eval_output/optimal_retrieval_config.yaml` | 最优检索配置 (检索服务层唯一输入) |

### 质量门禁

- Recall@5 < 0.6 → 阻断性错误，禁止生成最优配置
- 回归检测：关键指标下降 >5% 标记 REGRESSION 警告
