# web/api — FastAPI 后端

HTTP 接口层，薄封装 pdf_pipeline、indexer、retrieval 模块。所有业务逻辑在对应模块内，API 层只做：校验输入 → 调度后台任务 → 返回 JSON。

## 启动

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问自动生成的交互式文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 端点总览

### Health

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |

### PDF Pipeline — `routers/pdf.py`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/pdf/process` | 上传 PDF，后台执行全流程（parse→enhance→enrich→rag-chunk）。完成后自动 `merge_all_chunks()` 刷新稀疏检索索引 `all_rag_chunks.json`，论文即刻可被 `search_papers` 检索 |
| POST | `/api/pdf/process-local` | 处理磁盘上已有 PDF（agent `download_paper` 下载或 Web 上传）。`pdf_path` 可选：工作区路径（相对/绝对，经 `resolve_workspace_path` 校验，越界 403，必须存在且 `.pdf`）；缺省依次查找 `data/uploads/{paper_name}.pdf` → `data/downloads/{paper_name}.pdf` |
| GET | `/api/pdf/status/{task_id}` | 查询 PDF 处理任务状态 |
| GET | `/api/pdf/outputs` | 列出论文。`status` 只两类：`indexed`（已入库可检索）/ `not_indexed`；派生诊断在 `detail`（`indexed` / `parsed` 有解析产物 / `raw` 仅本地 PDF / `""`）。扫描覆盖 `data/uploads/` 与 `data/downloads/` |
| GET | `/api/pdf/outputs/{name}` | 获取论文输出文件列表（同一 status/detail 语义） |

> **下载与入库解耦**：`download_paper` 是纯文件下载（默认落盘 `data/downloads/`，支持自定义目录与简称命名），不触发解析/切分/向量化；入库是**原子操作** `ingest_paper` → `/api/agent/ingest`（解析 + 向量化，一个后台任务，可传 `pdf_path` 定位到自定义目录的 PDF）。

### Indexer — `routers/index.py`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/index/run` | 后台执行索引构建（从 rag_chunks.json） |
| GET | `/api/index/status/{task_id}` | 查询索引任务状态 |
| POST | `/api/index/search` | 向量检索（dense only，embed query → search） |
| GET | `/api/index/stats` | 向量库统计（backend/collection/count） |
| POST | `/api/index/reconcile?config_path=` | **目录对账**：滚动向量库按 chunk_id 前缀统计各论文实际点数，与目录 `indexed` 标志回填对齐（修复旧代码缺键/标志与库脱节），结果含 fixed 明细，在 task result 上。CLI 等价：`python -m web.cli reconcile` / `python -m indexer.reconcile` |

### Retrieval — `routers/retrieval.py`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/retrieval/search` | 使用评估最优策略检索（dense/sparse/hybrid）。结果含 page_no |
| GET | `/api/retrieval/config` | 当前检索策略配置（method/top_k） |

### Reader — `routers/reader.py`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/reader/{paper}/chunks/{chunk_id}/context?window=2` | chunk 上下文导航（前后 N 个邻居） |
| GET | `/api/reader/{paper}/sections` | 列出论文章节及 chunk 分布 |
| GET | `/api/reader/{paper}/sections/{name}` | 获取指定章节的全部 chunk |
| GET | `/api/reader/{paper}/abstract` | 提取论文摘要 + 标题 + 作者 |
| GET | `/api/reader/papers?author=&year=&keyword=` | 按元数据搜索论文 |
| GET | `/api/reader/local-papers` | 本地产物快照（agent 决策塔用）：`state` 两类 `indexed`/`not_indexed` + 派生 `detail`（parsed/raw/indexed）+ `has_pdf`/`pdf_path`。扫描范围：Redis catalog + `pdf_pipeline/output/` 解析产物 + `data/uploads`/`data/downloads` + **`data/` 根目录裸 PDF**（location 区分） |

### Agent — `routers/agent.py`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/agent/chat` | 发送消息给科研文献助手 agent，返回回答 |
| POST | `/api/agent/chat/stream` | SSE 流式输出：token + tool_start/tool_end 步骤事件 → done |
| GET | `/api/agent/health` | agent 状态（模型名、工具数） |

Agent 自主调用检索/阅读工具，多轮对话通过 `thread_id` 保持上下文。

### Agent 后台任务 — `routers/background.py`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/agent/ingest` | **入库（原子）**：入参 `{paper_name, pdf_path, notify}`，立即返回 `task_id`。后台在**同一个任务**上推进 stage=parse（解析 PDF）→ stage=index（向量化入库），无内部子任务 |
| GET | `/api/agent/tasks` | **任务栈列表**：顶层后台任务（倒序，含 kind/paper_name/status/stage/progress/error/result/notify）。带 `parent` 的旧内部子任务被过滤隐藏 |
| GET | `/api/agent/tasks/{task_id}` | 单个后台任务状态（404 = 不存在） |
| GET | `/api/agent/tasks/stream` | **SSE 实时推送**：连接即发全量快照（`task_snapshot`），此后每个任务创建或状态变更（`_task_create`/`_task_update`）立即推 `task_update`；15s keepalive。断线由 EventSource 自动重连并重发快照补缝。事件体：`{"type": "task_snapshot"|"task_update", "task": {...TaskStatus}}` |
| POST | `/api/agent/notify/stream` | SSE：任务完成通知。入参 `{thread_id, task}`，notifier LLM 把任务事实转成 1~2 句完成/失败通知（token…done） |

> **注意**：`_run_pipeline` 会显式把 `raw.md` 与 `final_enriched.md` 写入 `pdf_pipeline/output/{paper_name}/`（enricher 默认的 assets_dir 按 PDF stem 推导，含点号文件名如 arXiv ID `2003.12462v2` 会得到 `output/2003_12462v2`，与 paper_name 目录不一致——详见 TROUBLESHOOTING「入库报 FileNotFoundError: final_enriched.md」）。

SSE 事件类型（`/api/agent/notify/stream`）：

| type | 字段 | 说明 |
|------|------|------|
| `token` | `content` | 通知文本逐 token |
| `done` | — | 流结束 |
| `error` | `message` | 出错 |

SSE 事件类型（`/api/agent/chat/stream`）：

| type | 字段 | 说明 |
|------|------|------|
| `token` | `content` | LLM token 流式增量 |
| `tool_start` | `id, name, args` | 工具开始调用（前端渲染步骤卡片） |
| `tool_end` | `id, name, status, result, execution_time` | 工具完成（`status`=success/error，`execution_time` 秒） |
| `done` | — | 流结束 |
| `error` | `message` | 出错（含 `turn_timeout`） |

`tool_start`/`tool_end` 通过 `id` 关联，前端据此渲染可折叠的工具执行步骤。

### Workspace — `routers/workspace.py`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/workspace/list?path=<相对路径>` | 列目录（目录项含 name/is_dir/size） |
| GET | `/api/workspace/read?path=<相对路径>` | 读文件内容（文本截断 ~32KB，二进制返回 is_binary 标记） |

路径安全边界复用 `agent/providers/generic_provider.py:resolve_workspace_path`（工作区根 = 项目根，越界 / symlink / `..` 穿越即 403）。供桌面客户端文件 explorer 使用。

## 去重

上传 PDF 时自动基于 Redis 进行三层去重检查（O(1) 查询）：

| 优先级 | 标识 | Redis Key | 命中后行为 |
|--------|------|-----------|-----------|
| L1 | SHA256(PDF bytes) | `dedup:hash:{sha256}` → paper_name | 直接返回已有结果，status=duplicate |
| L2 | DOI | `dedup:doi:{doi}` → paper_name | 注册表匹配后返回已有结果 |
| L3 | title + first_author | (后续实现) | — |

> **入库去重陷阱（已修复）**：`dedup:hash` 命中 ≠ 已入库——仅解析流（`/api/pdf/process`）注册论文时也会写该键，命中论文可能 `indexed=false`。`_run_ingest` 的去重快判必须叠加 `existing.indexed`：只有**确实已索引**才业务短路返回「已在库中」；`indexed=false` 命中时复用现成 `rag_chunks.json` 直接向量化收尾（`register_indexed` 置真），否则重复出入库矛盾（详 TROUBLESHOOTING「agent 入库假完成」）。

Redis key scheme：
```
dedup:hash:{sha256}  → paper_name      (指纹索引)
dedup:doi:{doi}      → paper_name      (DOI 索引)
dedup:paper:{name}   → JSON metadata   (论文详情)
dedup:papers         → set             (全量论文名列表)
```

Redis 不可用时自动降级到 `eval_output/paper_registry.json`，不阻塞上传。

> 论文目录（元数据 + 去重 + `indexed` 状态）统一由 `indexer/catalog.py` 提供，API 层只调用不实现。
> **状态语义（方案 B，两类）**：`indexed=true` 表示 chunks 已写入向量库 Qdrant → 对外 `status="indexed"`；其余一切 → `status="not_indexed"`，是否为 `parsed`（仅解析）`/ raw`（仅本地 PDF）由 `detail` 字段**从文件系统现场派生**，不作为持久终态。实际入库是**原子操作**：`register_indexed()` 一次写完目录，无「已解析未入库」中间态终值。

## 任务状态

所有后台任务（PDF 处理、索引构建、agent 复合入库）状态存储在 Redis，支持多 worker + 重启恢复：

```
task:{task_id} (hash, TTL=3600s)
  task_id, status(pending|running|done|failed), progress, error, result(JSON)
  kind, paper_name, notify, stage     # 展示字段：ingest/pdf/index 任务均带 kind+paper_name；stage=parse|index（ingest）
  parent                              # 旧复合子任务标记（兼容保留），不出现在顶层列表

task:list (sorted set, score=timestamp)
```

- `_task_list(limit)` 读 `task:list` 倒序返回顶层任务：跳过已过期 hash（顺带 `zrem` 清理 stale member）、跳过带 `parent` 的旧子任务。
- **入库 = 单个任务**（`routers/background.py::_run_ingest`）：`stage` 字段标当前阶段（parse / index），进度连续推进，无内部子任务。
- **实时推送**：`routers/__init__.py` 维护一个进程内事件总线——`_task_create`/`_task_update` 之后将任务快照广播给所有 `/api/agent/tasks/stream` 的 SSE 客户端（后台线程经 `loop.call_soon_threadsafe` 投递，无活跃客户端时快速 no-op）。前端以此为主、30s 轮询兜底。单 worker 假设（多 worker 各自广播）。
- Redis 不可用时降级为进程内存 dict（单 worker 内可用，重启丢失）。
- Redis 连接: `127.0.0.1:6379`（无认证），旧版 RESP2 协议。

## 请求/响应示例

### POST /api/pdf/process

```bash
curl -X POST http://localhost:8000/api/pdf/process \
  -F "file=@data/paper.pdf"
```

```json
{"task_id":"a1b2c3d4e5f6","paper_name":"paper","status":"pending"}
```

重复上传时返回 status=duplicate，不重新处理：

```json
{"task_id":"dup_abc123def456","paper_name":"RMNet: A Remote Sensing...","status":"duplicate"}
```

### GET /api/pdf/status/{task_id}

```json
{
  "task_id":"a1b2c3d4e5f6",
  "status":"done",
  "progress":"Complete",
  "error":null,
  "result":{"paper_name":"paper","chunk_count":42,"files":["raw.md","rag_chunks.json","rag_chunks.html"]},
  "created_at":"2026-07-20T10:00:00Z",
  "updated_at":"2026-07-20T10:00:30Z"
}
```

### POST /api/index/search

```bash
curl -X POST http://localhost:8000/api/index/search \
  -H "Content-Type: application/json" \
  -d '{"query":"remote sensing change detection","top_k":5}'
```

```json
{
  "results":[
    {"chunk_id":"RMNet_sec_4","content_type":"body","section_path":"RMNet > 4. Experiments","generation_text":"We evaluate RMNet on...","score":0.92,"metadata":{}},
    ...
  ],
  "total":5
}
```

### POST /api/retrieval/search

```bash
curl -X POST http://localhost:8002/api/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{"query":"change detection","top_k":3}'
```

```json
{
  "results":[
    {"chunk_id":"RMNet_sec_4","generation_text":"...","score":12.34,"metadata":{"content_type":"body","section_path":"RMNet > 4. Experiments"}},
    ...
  ],
  "total":3
}
```

### GET /api/retrieval/config

```json
{"status":"ready","method":"sparse","top_k":5}
```

服务未初始化时：

```json
{"status":"unavailable","error":"optimal config not found: eval_output/optimal_retrieval_config.yaml. Run evaluation first."}
```

### Reader 端点示例

```bash
# 查看论文章节
curl http://localhost:8000/api/reader/RMNet/sections

# 获取特定章节
curl "http://localhost:8000/api/reader/RMNet/sections/Loss%20Function"

# chunk 上下文导航 (前后各 2 个邻居)
curl "http://localhost:8000/api/reader/RMNet/chunks/chunk_0020/context?window=2"

# 论文摘要
curl http://localhost:8000/api/reader/RMNet/abstract

# 元数据搜索
curl "http://localhost:8000/api/reader/papers?keyword=Remote"
```

### Agent 端点示例

```bash
# Agent 问答
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the RMNet loss function?","thread_id":"session_1"}'

# 多轮对话（同一 thread_id）
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"How does it compare to MV-CC?","thread_id":"session_1"}'

# Agent 状态
curl http://localhost:8000/api/agent/health

# Agent SSE 流式问答
curl -N -X POST http://localhost:8000/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the RMNet loss function?","thread_id":"session_2"}'
```

### Workspace 端点示例

```bash
# 列出项目根目录
curl "http://localhost:8000/api/workspace/list?path=."

# 读 agent/plan.py 内容
curl "http://localhost:8000/api/workspace/read?path=agent%2Fplan.py"
```

```json
{"ok":true,"data":{"path":".","entries":[{"name":"agent","is_dir":true,"size":null},{"name":"data","is_dir":true,"size":null}]}}
```

## 请求模型

所有请求/响应的 Pydantic 模型定义在 [schemas.py](schemas.py)：

| 模型 | 字段 |
|------|------|
| `SearchRequest` | `query: str`, `top_k: int (1-100, default 10)`, `filters: dict \| None` |
| `SearchResult` | `chunk_id, content_type, section_path, generation_text, score, metadata` |
| `SearchResponse` | `results: list[SearchResult]`, `total: int` |
| `IndexRunRequest` | `rag_chunks_path: str`, `config_path: str` |
| `PaperOutput` | `paper_name, files, chunk_count, status, indexed` |
| `TaskStatus` | `task_id, status, progress, error, result, created_at, updated_at, kind, paper_name, notify, stage`（kind: ingest/pdf/index；`stage`: ingest 当前阶段 parse/index/空） |
| `AgentChatRequest` | `query: str`, `thread_id: str (default "default")` |
| `AgentChatResponse` | `answer: str`, `intent: str`, `thread_id: str`, `error: str\|None` |
| `AgentIngestRequest` | `paper_name: str`, `pdf_path: str=""`, `notify: bool=True` |
| `AgentIngestResult` | `task_id, paper_name, status` |
| `AgentNotifyRequest` | `thread_id: str (default "default")`, `task: dict` |

## 文件结构

```
web/api/
 ├── main.py              — FastAPI app 入口，注册 CORS + routers
 ├── schemas.py           — Pydantic 请求/响应模型
 ├── routers/
 │   ├── pdf.py           — PDF 处理端点
 │   ├── index.py         — 索引构建 + dense 检索端点
 │   ├── retrieval.py     — 评估最优策略检索端点
 │   ├── reader.py        — 论文阅读/导航端点
 │   ├── agent.py         — Agent 对话端点
 │   └── workspace.py     — 工作区文件浏览端点
 └── README.md            — 本文档
```
