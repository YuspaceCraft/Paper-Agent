# ⚠️ v1 已弃用 — 2026-07-20

本目录为旧版归档。新版 v2 重构见项目根目录 [README.md](../README.md)。

v2 重构要点：PDF 原生解析 (Docling) → 五阶段管道 → 混合检索 (TF-IDF + Dense) → 离线评估驱动配置 → FastAPI 前后端分离。

---

# LangChain RAG → Agent 科研文献助手 (v1 归档)

> 从 RAG 到 Agent 的完整进化项目 —— BGE 本地嵌入 + Cross-Encoder 精排 + 混合记忆 + ReAct/Plan-Execute Agent + 反思修正 + Streamlit Web UI

## 项目概览

一个从 RAG 到 Agent 的完整进化项目，支持多种文档格式、本地/云端双模式嵌入、二阶段检索精排、长期+短期混合记忆管理、Agent 自主决策（ReAct/Plan-Execute）、反思修正和主动澄清。

```
┌─────────────────────────────────────────────────────────────────┐
│                        📥 文档导入 (ingest)                       │
│                                                                  │
│  data/ 目录                                                      │
│  ├── *.txt *.md *.pdf *.csv *.py *.docx ...                     │
│      │                                                           │
│      ▼                                                           │
│  [loader]   多格式加载 + 错误容忍                                  │
│      │                                                           │
│      ▼                                                           │
│  [splitter] 按类型路由: 文本策略 / AST代码策略 / 表格策略 / PDF策略  │
│      │                                                           │
│      ▼                                                           │
│  [embedder] BGE bge-small-en-v1.5 本地嵌入（384维, 免费离线）         │
│      │                                                           │
│      ▼                                                           │
│  [store]    Chroma 向量数据库 → chroma_db/（磁盘持久化）            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      📤 问答查询 (query)                          │
│                                                                  │
│  用户问题: "什么是 RAG？"                                         │
│      │                                                           │
│      ▼                                                           │
│  Stage 1: Chroma 向量粗筛 → Top-20 候选                           │
│      │                                                           │
│      ▼                                                           │
│  Stage 2: BGE-reranker-base cross-encoder 精排 → Top-5              │
│      │                                                           │
│      ▼                                                           │
│  [generator] 上下文 + Prompt 模板 → LLM → 答案                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   💬 多轮对话 (interactive)                       │
│                                                                 │
│  用户问题: "这篇论文用了什么方法？"                                  │
│      │                                                           │
│      ▼                                                           │
│  [query_rewriter] 指代消解 → "BITA (2023) 用了什么方法？"           │
│      │                                                           │
│      ├─→ [STM] 最近 N 轮对话上下文                                  │
│      ├─→ [LTM] 双通道检索（关键词 + 语义向量）                       │
│      └─→ [State] 当前分析焦点 / 进度追踪                            │
│      │                                                           │
│      ▼                                                           │
│  [pipeline.conversational_query] 或 [pipeline.agent_query]       │
│      ├─ 自动摘要（>8轮或>60% token触发）                            │
│      ├─ 动态 Top-K（fact→3, review→8）                             │
│      ├─ LTM 事实注入 Prompt                                       │
│      ├─ [Agent 模式] LLM 自主决策调用工具（多步检索/反思/澄清）       │
│      └─ 答案生成 → LTM 自动提取                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   🤖 Agent 模式 (agent_query)                     │
│                                                                 │
│  用户问题                                                         │
│      │                                                           │
│      ├─→ [clarifier] 模糊检测 → 如需澄清 → 追问用户                 │
│      │                                                           │
│      ▼                                                           │
│  [agent 调度器] 复杂度判断 → 选择 Agent 类型                         │
│      │                                                           │
│      ├─── ReActAgent (简单查询)                                    │
│      │    Thought → Action(tools) → Observation → ... → Answer   │
│      │                                                           │
│      ├─── PlanExecuteAgent (复杂查询)                              │
│      │    Plan → Execute(Step1→Step2→...) → Evaluate → Answer    │
│      │                                                           │
│      └─── ReflectiveAgent (高质量需求)                             │
│           Generate → Reflect → Correct → Re-reflect → Answer     │
│      │                                                           │
│      ▼                                                           │
│  工具调用 (15 tools):                                              │
│  📚 检索 (8): search_literature | get_paper_detail | compare_papers │
│              search_long_term_memory | get_conversation_context   │
│              rewrite_query | add_to_memory | get_system_status    │
│  📁 文件 (7): list_directory | create_directory | move_file       │
│              search_files | get_file_info | organize_paper        │
│              list_paper_categories                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python >= 3.10
- 8GB+ 内存（本地模型 ~1.5GB）
- Windows / macOS / Linux

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
# 最小配置：只需 .env 文件（或直接使用默认值——本地嵌入无需 API Key）
# 如果需要 DashScope 聊天功能（LLM 生成），需配 API Key：
copy .env.example .env
# 编辑 .env，填入你的 DashScope API Key
# 获取地址: https://dashscope.console.aliyun.com/apiKey
```

**默认配置就能跑**——嵌入模型使用本地 `BGE bge-small-en-v1.5`（384维, ~100MB），精排使用 `BGE bge-reranker-base`（cross-encoder）。

### 3. 导入文档

```bash
python -m src.cli ingest
```

首次运行加载本地 BGE 模型（~100MB + ~1.1GB），之后完全离线。

### 4. 开始提问

```bash
# 方式 1: Web 界面（推荐）
# 注意：默认 headless=true（不自动打开浏览器），手动访问 http://localhost:8501
streamlit run src/app.py

# 方式 2: 命令行单次提问
python -m src.cli query "什么是 RAG？"

# 方式 3: 命令行交互式模式（默认启用 Agent 模式）
python -m src.cli interactive
# Agent 命令: /agent (切换) /agent-type react|plan|reflective /clarify <问题>

# 查看系统状态
python -m src.cli status

# 运行评测
python -m eval.cli evaluate-e2e -d evalsets/sample_e2e.jsonl
```

## 项目结构

```
Demo/
├── .env.example              # 配置模板
├── .gitignore
├── requirements.txt          # Python 依赖
├── README.md                 # 本文档
│
├── data/
│   └── knowledge.txt         # 示例知识库（AI/ML 概念）
│
├── eval/                     # 评测模块
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics_retrieval.py
│   ├── metrics_generation.py
│   ├── metrics_memory.py     # 记忆质量评测
│   ├── judge.py
│   ├── evaluate.py
│   ├── report.py
│   └── cli.py
│
├── evalsets/                 # 评测数据集
│   ├── sample_retrieval.jsonl
│   ├── sample_generation.jsonl
│   ├── sample_e2e.jsonl
│   └── sample_memory.jsonl
│
├── conversations/            # 对话持久化目录
├── long_term_memory/         # 长期记忆存储目录
│
└── src/
    ├── __init__.py
    ├── config.py             # 配置中心（含记忆/LTM/动态Top-K/Agent配置）
    ├── loader.py             # 多格式文档加载器
    ├── splitter.py           # 智能切分器（4 种策略）
    ├── embedder.py           # 本地/云端双模式嵌入 (BGE/Qwen/DashScope)
    ├── store.py              # Chroma 向量数据库
    ├── retriever.py          # 二阶段检索 + Cross-Encoder/Bi-Encoder Reranker
    ├── generator.py          # RAG 生成链（LCEL，含混合记忆链）
    ├── pipeline.py           # 流水线编排器（含conversational_query + agent_query）
    ├── app.py                # Streamlit Web 界面（文件上传 + 多轮对话 + Agent 面板）
    ├── cli.py                # 命令行接口（含 Agent 命令）
    ├── memory.py             # 短期记忆（Buffer/Window/Hybrid + 摘要）
    ├── ltm.py                # 长期记忆（双通道检索 + LLM提取 + JSON持久化）
    ├── query_rewriter.py     # 查询重写（指代消解 + 类型分类）
    ├── conversation.py       # 对话持久化（save/load/list/delete）
    ├── tools.py              # 🆕 Agent 工具集（15 个 LangChain @tool）
    ├── agent.py              # 🆕 Agent 核心（ReAct + Plan-Execute + 调度器）
    ├── reflection.py         # 🆕 反思修正模块（三维审查 + 自我修正）
    ├── clarifier.py          # 🆕 主动澄清模块（模糊检测 + 追问生成）
    ├── mcp_client.py         # 🆕 MCP 文件管理（7 内置工具 + 外部 Server 接入）
    └── mcp_filesystem_server.py  # 🆕 独立 MCP Server（stdio transport）
```

## 模块详解

---

### ① config.py — 配置中心

所有可调参数集中管理，通过 `.env` 文件或环境变量覆盖。

**核心配置项**:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_PROVIDER` | `local` | 嵌入提供方: `local` (免费离线) / `dashscope` (云端) |
| `LOCAL_EMBEDDING_MODEL` | `BGE/bge-small-en-v1.5` | 本地嵌入模型 (384维, ~100MB) |
| `RERANK_MODEL` | `BGE/bge-reranker-base` | 精排模型 (cross-encoder) |
| `LLM_MODEL` | `qwen-plus` | 对话生成模型 |
| `CHUNK_SIZE` | `500` | 文本切块大小（字符） |
| `CHUNK_OVERLAP` | `50` | 块间重叠（字符） |
| `RERANK_CANDIDATE_K` | `20` | 粗筛候选数 |
| `RERANK_FINAL_K` | `5` | 精排后最终数 |
| `MEMORY_TYPE` | `hybrid` | 记忆策略: `buffer` / `window` / `hybrid` |
| `MEMORY_WINDOW_SIZE` | `5` | 短期记忆窗口（轮数） |
| `LTM_RETRIEVAL_K` | `3` | 长期记忆检索数量 |
| `LTM_AUTO_EXTRACT` | `true` | 是否自动提取长期记忆 |
| `AUTO_SUMMARIZE_TURNS` | `8` | 触发自动摘要的轮数阈值 |
| `MAX_CONTEXT_TOKENS` | `8000` | 上下文 token 上限 |
| `FACT_QUERY_TOP_K` | `3` | 事实查询检索数 |
| `REVIEW_QUERY_TOP_K` | `8` | 综述查询检索数 |
| `AGENT_MODE` | `true` | 🆕 启用 Agent 模式 (true=自主决策, false=传统管道) |
| `AGENT_TYPE` | `react` | 🆕 Agent 类型: `react` / `plan_execute` / `reflective` |
| `AGENT_MAX_ITERATIONS` | `10` | 🆕 Agent 最大迭代步数 |
| `REFLECTION_MAX_ROUNDS` | `2` | 🆕 反思修正最大轮数 |
| `CLARIFY_ENABLED` | `true` | 🆕 是否启用主动追问澄清 |
| `PLAN_EXECUTE_COMPLEXITY_THRESHOLD` | `30` | 🆕 Plan-Execute 复杂度触发阈值 |
| `MCP_FILESYSTEM_ENABLED` | `true` | 🆕 启用内置文件系统工具（论文分类管理） |
| `MCP_SERVER_<NAME>_COMMAND` | (无) | 🆕 外部 MCP Server 启动命令 |
| `PAPERS_BASE_DIR` | `./data` | 🆕 论文文件根目录 |

---

### ② loader.py — 多格式文档加载器

**支持格式**:

| 格式 | 依赖 | 加载方式 |
|------|------|----------|
| `.txt` `.md` `.json` `.yaml` `.py` `.html` | 无 | UTF-8 读取（自动回退 GBK） |
| `.csv` | 无 | 行级解析 |
| `.pdf` | `pip install pypdf` | PyPDFLoader（每页一个 Document） |
| `.docx` | `pip install docx2txt` | 全文提取 |

**核心能力**:
- **目录递归扫描**: 传目录路径自动发现所有支持文件
- **错误容忍**: 单个文件失败不中断整体流程，最终汇总
- **向下兼容**: 传文件路径的行为不变

```python
# 加载单个文件
docs = load_documents("data/report.txt")

# 加载整个目录
docs = load_documents("data/")
```

---

### ③ splitter.py — 智能切分器（4 种策略）

根据 `file_type` metadata 自动路由到对应策略。

| 策略 | 适用格式 | 切分单元 | 原理 |
|------|----------|----------|------|
| **文本** | `.txt` `.md` `.html` `.docx` | 段落/句子 | RecursiveCharacterTextSplitter，递归找 `\n\n` → `\n` → `。` 边界 |
| **代码** | `.py` `.js` `.ts` `.java` `.go` `.rs` | 函数/类/接口 | Python 用 AST 解析；JS/Go 等用正则匹配声明边界 + 大括号计数。每个 chunk 前置 import |
| **表格** | `.csv` | 不切数据行 | 提取列定义 + 类型推断 + 行数 + 样本行，生成"表格描述"文本用于嵌入。原始数据保存在 metadata |
| **PDF** | `.pdf` | 页 + 跨页合并 | 检测"续表""接上页"等标志，合并碎片页后按段落切 |

**代码策略的 AST 示例**:

```python
# 输入: 一个 app.py 包含 2 个函数 + 1 个类
# 输出: 3 个 chunk，每个包含:
#   [Code: function hello — Say hello in app.py]
#   import os
#   def hello(): ...
```

这样搜索 "hello 函数" 就能直接命中——因为 `hello` 被写入了被嵌入的文本中。

---

### ④ embedder.py — 双模式嵌入模型

**本地模式** (默认)：

```
BGE bge-small-en-v1.5 (384维, ~100MB)
├── 文档嵌入: 原文直接编码
├── 查询嵌入: 原文编码 + 余弦相似度
└── 完全离线，不产生任何 API 费用

BGE bge-reranker-base (cross-encoder, ~1.1GB)
├── 精排: (query, doc) pair → 相关性分数
└── 精度远高于 bi-encoder 余弦相似度
```

**本地模型可选**:

| 模型 | 维度 | 大小 | 适用场景 |
|------|------|------|----------|
| `BGE/bge-small-en-v1.5` | 384 | ~100MB | 英文为主，轻量高效（当前默认） |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3GB | 英文最高精度 |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | ~1.2GB | 中英文通用, instruction-tuned |

**DashScope 模式**:

设置 `EMBEDDING_PROVIDER=dashscope`，使用阿里云 `text-embedding-v3` 云端模型。

⚠️ **切换嵌入模型后必须重建向量数据库**: `python -m src.cli ingest --rebuild`

---

### ⑤ store.py — Chroma 向量数据库

轻量级本地向量数据库，数据存储在 `chroma_db/` 目录。

```python
# 写入
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db/",
)

# 读取
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db/",
)
```

**为什么持久化？** 生成嵌入向量耗时且（如果用 API）花钱。存到磁盘后，后续查询直接从磁盘加载，又快又省。

---

### ⑥ retriever.py — 二阶段检索 + Reranker

**检索管道**:

```
用户问题
    │
    ▼
Stage 1: 向量粗筛
  Chroma 做 ANN（近似最近邻）搜索
  速度: ~5ms
  精度: 一般（独立编码 + 余弦相似度）
  返回: Top-20 候选
    │
    ▼
Stage 2: Reranker 精排
  BGE bge-reranker-base (cross-encoder) 重打分
  速度: ~50ms (对 20 个候选)
  精度: 高（query-doc 深度交叉注意力）
  返回: Top-5
```

**为什么需要 Reranker？** 向量检索用独立编码做余弦相似度，会把"词像但意不同"的文档排前面。Cross-encoder 让 query 和 doc 在模型内部深度交互（attention across tokens），精度远高于 bi-encoder 余弦相似度。

**使用方式**:

```python
# 基础检索（单阶段）
retriever = create_retriever(store)

# 二阶段检索（推荐）
retriever = create_retriever_with_rerank(store)

# metadata 过滤
retriever = create_retriever_with_rerank(
    store, filter_dict={"chunk_type": "code"}  # 只搜代码
)
```

---

### ⑦ generator.py — RAG 生成链

使用 LangChain Expression Language (LCEL) 构建。

```python
chain = (
    {
        "context": retriever | format_docs,       # 检索 + 格式化
        "question": RunnablePassthrough(),          # 转发问题
    }
    | prompt           # 填入模板
    | llm              # 调用 LLM (Qwen-Plus via DashScope)
    | StrOutputParser() # 提取纯文本
)
```

- `temperature=0`: 让 LLM 给出最确定的回答（RAG 需要忠实于文档）
- Prompt 模板: "你是科研文献助手。使用以下上下文回答问题。如果上下文中没有答案，诚实地说不知道。"

---

### ⑧ pipeline.py — 流水线编排

两条核心流水线:

| 流水线 | 路径 | 组件顺序 |
|--------|------|----------|
| `ingest()` | 写入 | loader → splitter → embedder → store |
| `query()` | 读取 | embedder → store → retriever(二阶段) → generator → LLM |

`query()` 使用 BGE bi-encoder 做 embedding、BGE cross-encoder 做 reranker，两个独立模型各自优化。

---

### ⑨ cli.py — 命令行接口

```bash
# 导入文档
python -m src.cli ingest                      # 使用默认文档
python -m src.cli ingest --file "C:\docs\"     # 指定文件/目录
python -m src.cli ingest --rebuild             # 强制重建

# 单次提问
python -m src.cli query "什么是 embedding？"

# 交互模式（多轮对话 + Agent 自主决策 + 长期/短期记忆管理）
python -m src.cli interactive
# 命令: /quit /new /history /save /load /list /ltm /state /summarize /status
#       /agent (切换Agent模式) /agent-type <react|plan|reflective> /clarify <问题> /help

# 系统状态
python -m src.cli status
```

---

### ⑩ memory.py — 短期记忆管理

三种记忆策略，可通过 `.env` 中 `MEMORY_TYPE` 切换：

| 策略 | 类 | 行为 | 适用场景 |
|------|-----|------|----------|
| `buffer` | `BufferMemory` | 保留所有历史消息 | 短对话，需完整上下文 |
| `window` | `WindowMemory` | 只保留最近 N 轮 | 长对话，控制 token 消耗 |
| `hybrid` | `HybridMemory` | STM窗口 + LTM持久化 (**默认**) | 生产环境推荐 |

**核心能力**:
- **对话摘要**: `summarize(llm)` — LLM 生成结构化摘要
- **Token 估算**: `estimate_tokens()` — 中英文混合估算
- **状态追踪**: `update_state(key, value)` — 记录当前分析焦点/进度
- **自动触发**: >8轮或>60% token 自动摘要

```python
from src.memory import create_memory
memory = create_memory("hybrid", window_size=5)
memory.add_user_message("什么是 RAG？")
memory.add_ai_message("RAG 是检索增强生成...")
```

---

### ⑪ ltm.py — 长期记忆管理

持久化的长期记忆存储，跨会话保留关键信息。

```
LongTermMemory
  ├── 写入: add(content, keywords) → JSON 持久化
  ├── 检索: retrieve(query) → 双通道混合打分
  │   ├── 关键词通道: token 重叠率 + 关键词命中加权 (0.4)
  │   └── 语义通道: 嵌入向量余弦相似度 (0.6)
  ├── 提取: extract_from_exchange(user_msg, ai_msg, llm)
  │   └── LLM 自动提取值得记住的信息 → 添加到 LTM
  └── 存储: long_term_memory/ltm.json
```

**混合打分**: `score = 0.4 × keyword_score + 0.6 × semantic_score`

```python
from src.ltm import LongTermMemory
ltm = LongTermMemory()
ltm.add("用户偏好简短的回答", keywords=["偏好", "简短"])
results = ltm.retrieve("请简短回答", top_k=3)
```

---

### ⑫ query_rewriter.py — 查询重写

```
用户: "它用的什么损失函数？"
   → 指代消解: "它" → 上一轮讨论的具体论文
   → 重写: "DenseNet (Huang et al., 2017) 用的什么损失函数？"
   → 分类: fact（事实查询）→ 动态 Top-3 检索

用户: "对比这些方法的优缺点"
   → 分类: compare（对比分析）→ 动态 Top-8 检索
```

---

### ⑬ conversation.py — 对话持久化

```python
from src.conversation import ConversationStore
store = ConversationStore()
store.save("session-1", memory)   # conversations/<uuid>.json
memory = store.load("session-1")  # 从磁盘恢复
```

---

### ⑭ app.py — Streamlit Web 界面 🆕

基于 Streamlit 的科研文献助手 Web UI，功能包括：

| 功能 | 说明 |
|------|------|
| **PDF 上传** | 拖拽上传，自动解析（章节/摘要/参考文献）→ 嵌入 → 追加到向量库 |
| **多轮对话** | 混合记忆（短期窗口 + 长期持久化）+ 指代消解 + 查询类型分类 |
| 🆕 **Agent 模式** | LLM 自主决策调用工具，支持 ReAct/Plan-Execute/Reflective 三种模式 |
| 🆕 **主动澄清** | 模糊问题自动追问（交互按钮），用户回答后合并查询 |
| **进度动画** | 分步骤显示：查询优化 → 检索 → 精排 → 生成 → 记忆更新 |
| **检索可视化** | 每条回复下可展开查看检索到的文档块、来源文件、章节 |
| **对话管理** | 新建 / 命名保存 / 加载历史 / 删除（自动保存） |
| **系统状态** | 侧边栏实时显示向量块数、已上传文件、对话轮数、模型配置 |
| 🆕 **Agent 控制** | 侧边栏 Agent 面板：模式开关 + 类型选择 + 澄清开关 + MCP 状态 |
| 🆕 **MCP 文件管理** | Agent 可操作文件系统：论文分类归档、文件夹管理、文件搜索 |

```bash
# 启动 Web 界面
# 注意：默认 headless=true（不自动打开浏览器），手动访问 http://localhost:8501
streamlit run src/app.py

# 浏览器访问 http://localhost:8501
```

**交互流程**：
1. 用户提问 → 消息立即出现在聊天界面右侧
2. `st.status` 带动画展示检索 & 生成进度
3. AI 回答渲染完毕 → 检索上下文可折叠查看
4. 多轮对话中自动指代消解（"这篇论文" → 具体标题）
5. 🆕 Agent 模式下侧边栏可切换模式/类型，澄清追问以交互按钮呈现

---

### ⑮ tools.py — Agent 工具集 🆕

封装现有 RAG 组件为 LangChain `@tool` 函数，供 Agent 调用。**15 个工具**（8 检索 + 7 文件管理）全部返回结构化字符串。

| 工具名 | 封装组件 | 功能 | 关键参数 |
|--------|---------|------|----------|
| `search_literature` | `retriever.py` | 二阶段文献检索 | `query`, `top_k` |
| `get_paper_detail` | `retriever.py` + metadata filter | 按标题/作者精确检索 | `paper_title`, `author`, `year` |
| `compare_papers` | `retriever.py` multi_type_retrieve | 多维度对比检索 | `topic`, `aspects` |
| `search_long_term_memory` | `ltm.py` | 检索长期记忆 | `query`, `top_k` |
| `get_conversation_context` | `memory.py` | 获取对话历史和状态 | `include_summary` |
| `rewrite_query` | `query_rewriter.py` | 指代消解 + 查询优化 | `query` |
| `add_to_memory` | `ltm.py` | 手动添加长期记忆 | `content`, `keywords` |
| `get_system_status` | `store.py` | 查询向量库状态 | (无) |

**📁 文件管理工具（MCP）**:

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `list_directory` | 列出目录中的论文和分类文件夹 | `path` |
| `create_directory` | 创建论文分类文件夹 | `path` |
| `move_file` | 移动文件 | `source`, `destination` |
| `search_files` | 按通配符搜索论文文件 | `directory`, `pattern` |
| `get_file_info` | 获取文件详情 | `path` |
| `organize_paper` | 将论文 PDF 按主题归入分类文件夹 | `filepath`, `category` |
| `list_paper_categories` | 列出所有分类及论文数量 | (无) |

```python
from src.tools import get_all_tools, set_agent_memory

tools = get_all_tools()        # 获取全部 15 个工具（含 MCP 文件工具）
set_agent_memory(memory)       # 设置记忆上下文供工具访问
```

---

### ⑯ agent.py — Agent 核心循环 🆕

实现三种 Agent 模式 + 自动调度器。

**ReActAgent** (`react_agent_query`):
- 基于 LangChain 1.x `create_agent`（LangGraph-based）
- Thought → Action → Observation 循环，LLM 自主决定调用工具
- 最大迭代 10 步，自动处理解析错误

**PlanExecuteAgent** (`plan_execute_query`):
- 自定义四阶段：Plan（分解子任务）→ Execute（逐步执行）→ Evaluate（评估完整性）→ Synthesize（整合答案）
- 信息不足时自动触发补充检索
- 每步根据任务描述自动选择最合适的工具

**Agent 调度器** (`agent_query`):
- 查询长度 > 30 字符 或 包含"对比/总结/分析"等关键词 → Plan-Execute
- 其他 → ReAct
- `AGENT_TYPE=reflective` 时执行 ReAct + 反思修正

```python
from src.agent import agent_query
answer = agent_query("对比 DenseNet 和 ResNet 的架构差异", memory)
```

---

### ⑰ reflection.py — 反思 & 自我修正 🆕

在 Agent 生成答案后进行批判性审查和修正。

```
生成答案 → 反思审查 → {
    满意 → 输出最终答案
    需要修正 → 修正 → 重新反思
    信息不足 → 补充检索 → 重新生成 → 重新反思
}
```

**三维审查**:
| 维度 | 评估内容 | 评分 |
|------|---------|------|
| 忠实度 | 答案是否基于检索上下文（非编造） | 0~1 |
| 完整性 | 是否覆盖了问题的所有方面 | 0~1 |
| 准确性 | 引用标注和事实是否正确 | 0~1 |

```python
from src.reflection import reflective_correct
answer = reflective_correct(question, raw_answer, memory, max_rounds=2)
```

---

### ⑱ mcp_client.py — MCP 文件管理 🆕

为 Agent 提供文件系统操作能力：论文分类归档、文件夹管理、文件搜索。同时支持接入外部 MCP Server。

**内置工具（7 个）**：

```
Agent
  │
  ├── 📚 文献检索 (8 tools): search_literature, get_paper_detail, ...
  │
  └── 📁 文件管理 (7 tools):  ← NEW
        ├── list_directory       — 列出目录内容
        ├── create_directory     — 创建分类文件夹
        ├── move_file            — 移动/重命名文件
        ├── search_files         — 文件名通配符搜索
        ├── get_file_info        — 文件元信息
        ├── organize_paper       — 论文归类到分类目录
        └── list_paper_categories— 分类概览
```

**配置 (.env)**:

```bash
# 启用内置文件系统工具（默认开启）
MCP_FILESYSTEM_ENABLED=true
# 论文文件根目录（默认 ./data）
PAPERS_BASE_DIR=./data

# 接入外部 MCP Server
MCP_SERVER_MYTOOL_COMMAND=python
MCP_SERVER_MYTOOL_ARGS=-m my_mcp_server
MCP_SERVER_MYTOOL_ENABLED=true
```

```python
from src.mcp_client import get_mcp_tools, get_mcp_status

tools = get_mcp_tools()       # 获取 7 个文件管理工具（含外部 MCP）
status = get_mcp_status()     # 查看 MCP 连接状态
```

**安全特性**:
- 所有文件操作限制在 `PAPERS_BASE_DIR` 范围内（路径穿越防护）
- 内置工具零额外进程开销（直接 LangChain Tool 包装）
- 外部 MCP Server 通过 stdio transport 通信，支持任意 MCP 协议服务

### ⑲ clarifier.py — 主动追问 & 澄清 🆕

检测用户提问的模糊性，必要时生成澄清问题。

**模糊类型**:
| 类型 | 示例 | 处理 |
|------|------|------|
| 指代模糊 | "这篇论文怎么样？" | 追问具体指哪篇 |
| 范围过宽 | "有什么论文？" | 追问关注方向 |
| 缺少参数 | "帮我分析论文" | 追问论文名 |

**交互流程**（Streamlit）：追问以按钮形式呈现，用户点击选项后自动合并查询继续 Agent 流程。

```python
from src.clarifier import clarify_if_needed, resolve_clarification
cr = clarify_if_needed("这篇论文怎么样？", memory)
if cr.needs_clarification:
    resolved = resolve_clarification(cr, "Change-Agent (2024)")
```

---

## 🧪 评测体系（Agent Evaluation）

评测模块提供了四大维度的 RAG 质量评测。

### 评测概述

```
┌─────────────────────────────────────────────────────────────────┐
│                      📊 四大评测维度                               │
│                                                                  │
│  ① 检索质量评测 (Retrieval Quality)                               │
│     Hit Rate / MRR / NDCG / Precision@K / Recall@K               │
│     → 纯算法指标，不调用 LLM                                       │
│                                                                  │
│  ② 生成质量评测 (Generation Quality)                               │
│     忠实度 / 相关性 / 正确性                                        │
│     → LLM-as-Judge                                               │
│                                                                  │
│  ③ 端到端评测 (End-to-End)                                        │
│     检索 + 生成 + 延迟追踪 → 完整报告                               │
│                                                                  │
│  ④ 记忆质量评测 (Memory Quality)  ← 新增                           │
│     提取准确性 / 检索质量 / 记忆影响 / 一致性                        │
│     → LLM-as-Judge + 算法指标                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 评测数据集格式 (JSONL)

每行一个 JSON 对象：

```json
{
  "id": "q001",
  "question": "What is RAG?",
  "relevant_sources": ["knowledge.txt"],
  "reference_answer": "RAG is a technical architecture that...",
  "metadata": {"category": "definition", "difficulty": "easy"}
}
```

| 字段 | 评测维度 | 说明 |
|------|---------|------|
| `id` | 全部 | 唯一标识 |
| `question` | 全部 | 用户问题 |
| `relevant_sources` | 检索 | 相关文档文件名列表（子串匹配） |
| `reference_answer` | 生成 | 参考答案，用于正确性对比 |
| `metadata` | 可选 | 类别/难度，用于分层统计 |

### ① 检索质量评测

**指标公式**:

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **Hit Rate@K** | Top-K 中至少命中一个相关文档的概率 | `count(hit) / total_queries` |
| **MRR@K** | 第一个相关文档排名的倒数平均 | `mean(1 / rank_of_first_relevant)` |
| **Precision@K** | Top-K 中相关文档的占比 | `mean(relevant_in_top_k / k)` |
| **Recall@K** | 相关文档被检索到的比例 | `mean(relevant_in_top_k / total_relevant)` |
| **NDCG@K** | 考虑排名位置权重的归一化增益 | `DCG / IDCG` |

**相关性判定**: 检索文档的 `metadata["filename"]` 与 ground-truth 文件名做 Unicode 规范化子串匹配。

```bash
# 运行检索评测
python -m eval.cli evaluate-retrieval -d evalsets/sample_retrieval.jsonl

# 自定义 K 值
python -m eval.cli evaluate-retrieval -d evalsets/sample_retrieval.jsonl -k 10

# 单阶段检索（跳过 Reranker，更快但精度低）
python -m eval.cli evaluate-retrieval -d evalsets/sample_retrieval.jsonl --no-rerank
```

### ② 生成质量评测（LLM-as-Judge）

使用现有 LLM（qwen-plus, temperature=0）作为自动评测裁判，评估三个维度：

```
忠实度 (Faithfulness)
  ├── 从答案中提取原子事实声明
  ├── 逐条验证是否被上下文支持
  └── 分数 = 被支持声明数 / 总声明数  [0, 1]

答案相关性 (Relevance)
  ├── 1–5 分制，评估答案是否切题
  └── 自动归一化到 [0, 1]

正确性 (Correctness)
  ├── 1–5 分制，与参考答案的语义一致性对比
  └── 自动归一化到 [0, 1]
```

**容错设计**:
- JSON 解析失败自动重试 2 次
- 正则提取 `{...}` 作为兜底
- API 限流时指数退避

```bash
# 运行生成评测
python -m eval.cli evaluate-generation -d evalsets/sample_generation.jsonl

# 跳过 Reranker 加速
python -m eval.cli evaluate-generation -d evalsets/sample_generation.jsonl --no-rerank
```

### ③ 端到端评测

完整的查询 → 检索 → 生成 → 评测管道，同时追踪每一步的延迟。

```bash
# 端到端评测 + JSON 报告
python -m eval.cli evaluate-e2e -d evalsets/sample_e2e.jsonl -k 5 -o report.json

# 快速验证（无 Reranker）
python -m eval.cli evaluate-e2e -d evalsets/sample_e2e.jsonl --no-rerank -o report.json
```

**输出示例**:

```
============================================================
  RAG 评测报告
============================================================
  数据集:       evalsets/sample_e2e.jsonl
  评测查询数:   10
  成功:         10
  失败:         0
  LLM:          qwen-plus, Rerank=True, Top-K=5

── 检索质量 ──
  Hit Rate@5:       0.9000
  MRR@5:            0.8250
  Precision@5:      0.6000
  Recall@5:         0.9000
  NDCG@5:           0.8431

── 生成质量 ──
  忠实度:           0.8549  (85.5%)
  答案相关性:        4.6 / 5  (92.0%)
  正确性:            4.6 / 5  (92.0%)

── 延迟 ──
  平均检索耗时:      4.31s
  平均生成耗时:      5.04s
  平均总耗时:        9.34s
============================================================
```

### ④ 记忆质量评测 ← 新增

覆盖长期记忆(LTM)的四大评测维度：

| 维度 | 方式 | 指标 |
|------|------|------|
| **提取准确性** | LLM-as-Judge | 每条 LTM 事实准确/部分准确/不准确 → overall_accuracy |
| **检索质量** | 算法 | Precision@K / Recall@K / MRR@K |
| **记忆影响** | LLM-as-Judge | -1(变差) / 0(无影响) / +1(改善) / +2(显著改善) |
| **一致性** | 算法（嵌入余弦相似度）| 检测语义重复 → consistency_score |

```bash
# 记忆质量评测
python -m eval.cli evaluate-memory -d evalsets/sample_memory.jsonl -o mem_report.json
```

### 所有评测命令

```bash
# 列出可用数据集
python -m eval.cli list-datasets

# 检索质量评测
python -m eval.cli evaluate-retrieval -d <dataset.jsonl> [-k 5] [--no-rerank] [-o report.json]

# 生成质量评测
python -m eval.cli evaluate-generation -d <dataset.jsonl> [--no-rerank] [-o report.json]

# 端到端评测
python -m eval.cli evaluate-e2e -d <dataset.jsonl> [-k 5] [--no-rerank] [-o report.json]

# 记忆质量评测 ← 新增
python -m eval.cli evaluate-memory -d <dataset.jsonl> [--no-rerank] [-o report.json]
```

### 评测模块结构

| 文件 | 职责 |
|------|------|
| `eval/dataset.py` | JSONL 加载 + `EvalSample` 校验 |
| `eval/metrics_retrieval.py` | 5 项纯算法检索指标 |
| `eval/metrics_generation.py` | 批量调用 Judge 并聚合 |
| `eval/metrics_memory.py` | 记忆质量 4 维度指标 ← 新增 |
| `eval/judge.py` | LLM-as-Judge（6 个评测维度 + 容错） |
| `eval/report.py` | 控制台 + JSON 报告（含记忆段落） |
| `eval/evaluate.py` | 编排器（4 条评估管道） |
| `eval/cli.py` | CLI（5 个命令） |

### 自定义评测数据集

```jsonl
{"id": "my001", "question": "...", "relevant_sources": ["your_doc.pdf"], "reference_answer": "..."}
```

- **检索评测**: 只需 `id` + `question` + `relevant_sources`
- **生成评测**: 需额外 `reference_answer`
- **端到端评测**: 全部字段

**提示**: `relevant_sources` 中的文件名需与入库时的 `metadata["filename"]` 匹配（子串匹配，大小写不敏感）。

---

## 配置指南

### 最小配置（纯本地，推荐入门）

```bash
# .env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BGE/bge-small-en-v1.5
RERANK_MODEL=BGE/bge-reranker-base
DASHSCOPE_API_KEY=你的key    # 仅 LLM 生成需要
LLM_MODEL=qwen-plus
```

### 轻量配置（省资源）

```bash
# .env — 使用 100MB 的小模型
LOCAL_EMBEDDING_MODEL=BGE/bge-small-en-v1.5   # 384维, ~100MB
RERANK_MODEL=BGE/bge-reranker-base             # cross-encoder, ~1.1GB
```

### 全云端配置

```bash
# .env
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_API_KEY=你的key
```

## 自定义文档

```bash
# 方式 1: 替换默认文档
# 把你的文件保存到 data/ 目录，然后：
python -m src.cli ingest --rebuild

# 方式 2: 指定路径（支持目录递归扫描）
python -m src.cli ingest --file "C:\my_docs\" --rebuild

# 方式 3: 添加 PDF 支持
pip install pypdf
python -m src.cli ingest --file "C:\papers\" --rebuild
```

## 常见问题

### Q: 首次运行 `ingest` 时下载模型失败

**A**: HuggingFace 在国内被墙，项目已默认配置了 `hf-mirror.com` 镜像。如果仍失败：

```bash
# 检查镜像配置
python -c "from src.config import HF_ENDPOINT; print(HF_ENDPOINT)"
# 预期输出: https://hf-mirror.com

# 手动指定镜像
# 在 .env 中添加: HF_ENDPOINT=https://hf-mirror.com
```

### Q: 提示 "未找到向量数据库"

**A**: 需要先导入文档: `python -m src.cli ingest`

### Q: 切换嵌入模型后检索结果不对

**A**: 不同模型的向量空间不兼容，必须重建: `python -m src.cli ingest --rebuild`

### Q: 内存不足

**A**: 当前已使用 `BGE bge-small-en-v1.5`（~100MB），如需更小可尝试 `sentence-transformers/all-MiniLM-L6-v2`（~80MB, 384维）

### Q: 如何调优检索质量？

**A**: 编辑 `config.py` 或 `.env:

| 参数 | 效果 |
|------|------|
| `CHUNK_SIZE` | 增大 → 更完整上下文但精度下降 |
| `CHUNK_OVERLAP` | 增大 → 块间衔接更好但存储增加 |
| `RERANK_CANDIDATE_K` | 增大 → 更多候选给精排但速度变慢 |
| `RERANK_FINAL_K` | 增大 → LLM 看到更多上下文但消耗更多 token |

## 技术栈

| 层 | 组件 | 技术 |
|----|------|------|
| 文档加载 | 多格式 Loader | LangChain Document + PyPDFLoader + docx2txt |
| 文档切分 | 多策略 Splitter | RecursiveCharacterTextSplitter + AST + 正则 |
| 文本嵌入 | 本地/云端双模式 | BGE bge-small-en-v1.5 / DashScope API |
| 向量存储 | 本地向量库 | Chroma (SQLite + HNSW) |
| 粗筛检索 | ANN 搜索 | Chroma 内置 HNSW 索引 |
| 精排 | Reranker | BGE bge-reranker-base cross-encoder 重打分 |
| Web UI | 前端界面 | Streamlit（文件上传 + 多轮对话 + Agent 面板 + 检索可视化） |
| 生成 | LLM | Qwen-Plus via DashScope OpenAI 兼容 API |
| 短期记忆 | 对话窗口 | WindowMemory + 自动摘要 + Token估算 |
| 长期记忆 | 持久化知识库 | 双通道检索(关键词+语义) + LLM自动提取 |
| 🆕 Agent 决策 | 自主推理 | LangChain 1.x create_agent (LangGraph) + 自定义 Plan-Execute |
| 🆕 工具封装 | Tool Calling | LangChain @tool 装饰器（15 个工具：8 检索 + 7 文件管理） |
| 🆕 反思修正 | 质量保证 | LLM 三维审查 + 自我修正 + 补充检索 |
| 🆕 主动澄清 | 交互优化 | 规则+LLM 双重模糊检测 + 追问生成 |
| 🆕 MCP 文件管理 | Agent 工具扩展 | MCP Client + 内置文件系统工具 + 外部 Server 接入 |
| 编排 | 流水线 | LangChain LCEL（含混合记忆链 + Agent 流水线） |
| 评测 | 四大维度 | 检索指标 + LLM-as-Judge + 端到端 + 记忆质量 |

## 下一步

1. ~~**MCP 文件管理**: Agent 操作文件系统，论文分类归档~~ ✅ 2026-06-19 已完成 (`src/mcp_client.py` `src/mcp_filesystem_server.py`)
2. **LLM 本地化**: 用 Ollama 跑 Qwen2.5/Llama3，实现全部离线
3. ~~**Web 界面**: Streamlit / Gradio 可视化 UI~~ ✅ 2026-06-19 已完成 (`src/app.py`)
4. **混合搜索**: BM25 + 向量检索 + 权重融合
5. ~~**Agent 模式**: LLM 自主决定是否需要检索~~ ✅ 2026-06-19 已完成 (ReAct + Plan-Execute + 反思 + 澄清，`src/agent.py` `src/tools.py` `src/reflection.py` `src/clarifier.py`)
6. ~~**对话记忆**: 多轮对话 + 长期/短期混合记忆~~ ✅ 已完成
7. ~~**记忆评测**: 提取准确性 / 检索质量 / 记忆影响 / 一致性~~ ✅ 已完成
8. **增量索引**: 检测文件修改时间，只对新文件做嵌入
9. ~~**评测体系**: 检索质量 + 生成质量 + 端到端评测~~ ✅ 已完成
