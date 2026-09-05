# Paper-Agent — 科研文献 RAG + 多 Agent 智能系统

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) [![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)](https://langchain-ai.github.io/langgraph/) [![FastAPI](https://img.shields.io/badge/FastAPI-teal)](https://fastapi.tiangolo.com/) [![React](https://img.shields.io/badge/React-61dafb)](https://react.dev/) [![Electron](https://img.shields.io/badge/Electron-47848f)](https://www.electronjs.org/) [![Vite](https://img.shields.io/badge/Vite-646cff)](https://vitejs.dev/) [![Qdrant](https://img.shields.io/badge/Qdrant-8B5CF6)](https://qdrant.tech/)

一套面向**遥感 / 计算机视觉学术文献**的端到端科研系统：从 PDF 原生解析、多粒度向量检索，到 **LangGraph 多 Agent 编排**的问答、写作与实验，覆盖「**读 → 理解 → 写 → 跑**」完整科研流。

> ⚙️ **架构主线**：`PDF 解析 → RAG 切分 → 向量索引 → 可评估检索 → Agent 编排`，每个环节都是**可评估、可替换**的独立模块。

---

## ✨ 核心亮点

- **五阶段 PDF 原生解析管道** — Docling 布局解析 + 公式/图片 LLM 增强 + 学术结构感知切分，产出高质量 `rag_chunks.json`
- **多粒度向量索引 + 增量同步** — Small-to-Big 上下文组装、内容指纹去重（dedup）、Qdrant / Chroma 可切换
- **离线评估驱动的检索配置** — QA 生成 → 多策略检索（Dense / Sparse / 混合 RRF）→ Recall/NDCG 指标 → 自动选出最优配置并应用，Recall@5 < 0.6 阻断门禁
- **LangGraph 多 Agent 编排** — 领导-部门制 supervisor 将长任务派发到隔离子 Agent「舱」，支持监督/干预/验收；Plan-and-Execute 执行层 + 论文指称消解
- **三大领域路由** — `paper`（文献问答）/ `creation`（论文写作）/ `coding`（实验编码）按需进入专用 Agent
- **统一任务监督视图** — Agent 舱 + 实验 + 写作 + 后台任务聚合的实时进度面板，跨重启持久化
- **Electron 桌面客户端** — React 三栏工作台（对话 / 文件浏览 / 工作区），主进程自动拉起后端

---

## 🏗️ 架构总览

```text
        ┌──────────────────────────────────────────────────────────────┐
        │  pdf_pipeline    五阶段 PDF 预处理管道                          │
        │  parse → bindings → enhance → enrich → rag-chunk              │
        │                       ▼                                        │
        │  indexer          多粒度向量索引（embed → dedup → Qdrant/Chroma）│
        │                       ▼                                        │
        │  retrieval        共享检索层（Dense + Sparse + RRF 融合）       │
        └───────────────┬───────────────────────┬───────────────────────┘
                        │                       │
                 ┌──────▼─────────┐      ┌──────▼──────────────────────────┐
                 │ web/api        │      │ retrieval_orchestrator (离线评估) │
                 │ FastAPI 薄封装  │      │ QA 生成 → 多策略 → 指标 → 最优配置 │
                 └──────┬─────────┘      └──────┬──────────────────────────┘
                        │                       │ 最优配置自动加载
                        │                       ▼
        ┌───────────────▼───────────────────────▼────────────────────────┐
        │  agent  LangGraph 多 Agent 编排                                  │
        │  understand → memory → resolve → (react/plan) → search → synth  │
        │  supervisor 领导-部门制 · paper/creation/coding 领域路由          │
        └───────────────┬─────────────────────────────────────────────────┘
                        │
                 ┌──────▼──────────────┐
                 │ web/frontend        │  Electron + React 桌面工作台
                 │ 对话 · 写作 · 实验    │
                 └─────────────────────┘
```

**持久化**：Qdrant（生产）/ Chroma（本地）向量库 + Redis 论文注册表 + SQLite（LangGraph 会话 `checkpoints.db`、子 Agent 舱 `task_store.db`）。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
conda create -n demo python=3.10 -y
conda activate demo
pip install -r requirements.txt
```

### 2. 启动后端 API

```bash
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI → http://localhost:8000/docs
```

### 3. 启动桌面客户端（Electron）

```bash
cd web/frontend
npm install
npm run dev          # Vite + Electron，主进程自动拉起后端 (127.0.0.1:8001)
npm run dist:win     # 打包 Windows 安装器
```

### 4. 处理一篇 PDF

```bash
# CLI（解析 + 切分）
python docling_cli.py all data/paper.pdf

# 或 REST 上传
curl -X POST http://localhost:8000/api/pdf/process -F "file=@data/paper.pdf"
```

### 5. 离线检索评估 → 产出最优配置

```bash
python -m retrieval_orchestrator evaluate --config retrieval_orchestrator/evaluation.yaml
# 质量门禁达标后自动生成 eval_output/optimal_retrieval_config.yaml
# 生产检索服务（懒加载）自动读取该配置
```

---

## 📚 模块导读

| 模块 | 说明 |
|------|------|
| [`pdf_pipeline/`](pdf_pipeline/) | 五阶段 PDF→RAG 管道：Docling 解析 → 公式/图片增强 → 注入 → 结构感知切分 |
| [`indexer/`](indexer/) | 多粒度向量索引：Small-to-Big 上下文组装 + 去重 + Qdrant/Chroma 增量同步 |
| [`retrieval/`](retrieval/) | 共享检索层：`SparseRetriever` + `DenseRetriever` + `rrf_fuse` + `RetrievalService` |
| [`retrieval_orchestrator/`](retrieval_orchestrator/) | 离线检索验证框架：QA 生成 → 策略网格 → 指标（Recall/NDCG/MRR + bootstrap CI）→ 最优配置 |
| [`agent/`](agent/) | **多 Agent 编排核心** — 见下节 |
| [`web/api/`](web/api/) | FastAPI 薄封装：PDF/索引/检索/Agent/写作/实验/任务 全部 HTTP 化 |
| [`web/frontend/`](web/frontend/) | Electron + React 桌面工作台（对话 / 写作 / 实验 / 文件浏览） |
| [`skills/`](skills/) · [`mcphub/`](mcphub/) · [`mcp_simple_arxiv/`](mcp_simple_arxiv/) | 技能注册、MCP Hub 与 ArXiv MCP 服务端 |

> 每个模块自带独立的 `README.md`，含架构图、关键符号表与 API/CLI 用法 — 读模块前先看它的 README。

---

## 🤖 Agent 系统（agent/）

LangGraph 状态机编排的多 Agent 科研助手，核心链路：

```text
understand → memory → resolve → react / plan-and-execute → search → synthesize
```

| 能力 | 实现 |
|------|------|
| **意图路由** | `route_intent` 将请求分发到 `paper` / `creation` / `coding` 领域 |
| **领导者-部门制派发** | `agent/supervisor.py` 把长任务派发到隔离子 Agent「舱」（thread_id=task_id 状态栈），监督 / 干预 / 验收 |
| **执行层** | `react`（agent↔tools ReAct 循环）与 `plan`（Plan-and-Execute 计划-执行-验证）双模式，客户端可显式覆盖 |
| **论文指称消解** | `resolution.py` 确定性解析论文/章节的模糊引用（pre-LLM 消歧） |
| **工具汇编** | `agent/tools.py` 统一装配 Builtin + Generic + MCP 工具（`.mcp.json` 加载外部 MCP） |

**Agent 工具**（统一信封契约，路径越界 / SSRF / 权限门）：
- **论文域**：`search_papers` / `fetch_content` / `download_paper` / `ingest_paper` / `check_task_status` — 内部走 FastAPI 链路
- **通用域**：`read_file` / `write_file` / `list_dir` / `fetch_url` / `get_time` / `calculator`
- **任务监督**：`task_dispatch` / `task_progress` / `task_collect` / `task_resume` / `task_cancel` / `task_list`

---

## 🔌 API 概览（web/api）

所有功能通过 FastAPI 薄封装暴露，前端不直接 import 模块。

| 领域 | 端点 |
|------|------|
| PDF 管道 | `POST /api/pdf/process` · `/api/pdf/process-local` · `GET /api/pdf/status/{id}` · `/api/pdf/outputs` |
| 索引 | `POST /api/index/run` · `/api/index/search` · `GET /api/index/stats` · `POST /api/index/reconcile` |
| 检索 | `POST /api/retrieval/search` · `GET /api/retrieval/config` |
| 阅读器 | `GET /api/reader/{paper}/sections|abstract|chunks/{id}/context` · `GET /api/reader/papers` |
| Agent | `POST /api/agent/chat` · `/api/agent/chat/stream`（SSE）· `GET /api/agent/health` |
| 写作 / 实验 | `routers/creation.py` · `routers/experiments.py` · `routers/study.py` |
| 工作区 | `GET /api/workspace/list|read|browse` · `GET/PUT /api/settings`（可配置路径根） |
| 后台任务 | `GET /api/tasks` · `GET /api/tasks/search`（统一任务监督视图） |

---

## 🗂️ 项目结构

```
├── pdf_pipeline/          # 五阶段 PDF 预处理管道
├── indexer/               # 多粒度向量索引 + 增量同步
├── retrieval/             # 共享检索层（Dense/Sparse/Fusion）
├── retrieval_orchestrator/# 离线检索验证框架
├── agent/                 # LangGraph 多 Agent 编排（supervisor + 领域 + 工具）
├── web/
│   ├── api/               # FastAPI 后端（薄封装，纯 HTTP）
│   └── frontend/          # Electron + React 桌面客户端
├── skills/ mcphub/ mcp_simple_arxiv/   # MCP 生态
├── eval_output/           # 评估产出（manifest/报告/最优配置/论文注册表）
└── v1/                    # 已弃用的旧版 LangChain RAG 实现（归档）
```

---

## 📊 离线检索评估

QA 生成（LLM / keyword 双模式）→ 策略网格实验（RRF / HyDE / hybrid）→ 指标计算（Recall / NDCG / MRR + bootstrap 置信区间 + 维度拆解）→ 配置选择带**质量门禁**（Recall@5 < 0.6 阻断，指标回归 >5% 告警）。

更多用法见 [`retrieval_orchestrator/README.md`](retrieval_orchestrator/README.md)。

---

## 📄 License

MIT（占位 — 请按需替换）。
