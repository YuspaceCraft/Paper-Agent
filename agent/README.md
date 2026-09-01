# Agent — 科研文献助手 (v5)

基于 LangGraph 的科研文献对话 agent，通过 FastAPI 后端工具检索和阅读论文。

## 架构

```
User Query
    │
    ▼
understand (Router: 3-way + confidence)
    │ intent, confidence
    ▼
memory (MemoryManager: context snapshot assembly)
    │ context_snapshot → consumed by all downstream nodes
    ▼
route_intent (pure function, no LLM)
    │
    ├── literature_search (high confidence)
    │       │
    │       ▼
    │   resolve (deterministic paper matching)
    │       │ resolved: {papers, section} + confidence levels
    │       ▼
    │   decide_mode (pure heuristic — react vs plan)
    │       │
    │       ├── react (simple query, zero regression)
    │       │       ▼
    │       │   search (subgraph: agent ↔ tools ReAct loop)
    │       │       │
    │       │       ▼
    │       │   synthesize → User
    │       │
    │       └── plan (multi-paper / comparative query)
    │               │
    │               ▼
    │           plan_node (LLM → ordered steps)
    │               │
    │               ▼
    │           executor (topological, parallel via asyncio.gather)
    │               │  each step → "tool" (parent direct) or a subagent (arxiv/ingest)
    │               ▼
    │           synthesize (merge subagent_results) → User
    │
    ├── general_chat → chat → User  (lightweight, no tools)
    │
    └── needs_clarify / low confidence → clarify → User
```

### Search subgraph (agent/search_loop.py)

```
agent (LLM + bound tools)
    │
    ▼
after_agent (pure function)
    │
    ├── tool_calls + under max_iterations → tools → agent (loop)
    ├── [FINAL_ANSWER] → exit subgraph
    └── max_iterations exhausted → exit subgraph
```

## v5 核心改进

- **SqliteSaver**: InMemorySaver → AsyncSqliteSaver，对话跨服务器重启持久化
- **真流式**: stream_mode="messages" 替代 updates + 假分块，LLM token 级别实时输出
- **子图封装**: agent ↔ tools 循环提取为 search_loop.py 子图，父图简化为 understand → memory → resolve → search → synthesize

## v6 通用底座加固（治理 / 安全 / 观测 / 调度）

对照 [docs/agent-refactor-plan-review.md](../docs/agent-refactor-plan-review.md) 裁剪版执行：

- **治理（Phase 1）**: 整轮超时 `asyncio.wait_for` + token 预算（`state.token_budget/tokens_used`）+ `max_iterations` 配置化，env `AGENT_MAX_ITERATIONS` / `AGENT_TOKEN_BUDGET` / `AGENT_TURN_TIMEOUT`。`tokens_used` 度量**当前单次 LLM 输入规模**（不跨轮累加），`token_budget` 默认 60000 = 上下文上限兜底，避免多工具任务被提前截断。
- **安全（Phase 2）**: `safety.py` PII 脱敏（邮箱/手机号/身份证/银行卡）+ 权限门（`readOnlyHint is False` 判 destructive，角色 `AGENT_USER_ROLE`），输出经 `sanitize_output` 在 HTTP 边界脱敏。
- **观测（Phase 0）**: `observability.py` contextvars `trace_id` + `count`/`timed`/`log_turn_summary`，替换裸 `print`。
- **调度（Phase 3）**: `dispatcher.py` 统一超时（兜底）/幂等重试/审计，覆盖全部工具（含 builtin，移除 `_to_langchain_tool` 直返分支）。
- **库连接守卫（v10+）**: `library_api.py` — 本地后端熔断（失败后 45s 窗口内库工具直接速断，env `AGENT_API_BREAKER_TTL`）+ 短 connect 超时（2s：127.0.0.1 本机端口没必要为死端口白等默认 10s）。后端不可达时工具快速返回 `error_type="backend_down"`，`nodes._format_error_feedback` 指示 agent 停止重试并如实汇报故障，而不是把整轮 `TURN_TIMEOUT` 烧光。react / plan executor / subagent 三条路径共用（都在 builtin_provider 汇合）；`download_paper`（外网 arXiv）不受影响。

## v7/v8 执行层结构化（Plan-and-Execute + Multi-Agent）

对照 [docs/agent-multiagent-plan.md](../docs/agent-multiagent-plan.md) 执行。不改单 agent 主干，在其上叠两层：

- **Plan 范式（Phase 7）**：`plan.py` 的 `decide_mode`（纯启发式，无 LLM：对比关键词 / ≥2 子问题 / **resolve 确证 ≥2 篇论文**）+ `plan_node`（LLM 直接输出 JSON 步骤表，`_ask_for_plan` 解析）+ `executor_node`（拓扑序执行，无依赖步骤 `asyncio.gather` 并行）。多论文对比 / 多子问题走 plan，简单查询走 react（零回归）。
- **Multi-Agent（Phase 8，Claude Code 模式）**：`subagents.py` 新增 `build_subagent`（编译子图：复用 `agent_node`/`ToolNode`/`after_agent`，仅改受限工具集 + 专属 prompt + 独立上下文）+ `as_tool`（把子图包装成父层可调工具，父层只见「任务 → 摘要」，看不到子 agent 内部正文）。**库只读工具归父 agent，2 个写/外网 subagent（arxiv/ingest）** 通过声明式配置表 `SUBAGENTS` 加入父层工具列表。

### 模式判定（代码启发式，无额外 LLM 调用）

命中任一条件 → `plan`，否则 → `react`：
- **resolve 确证 ≥ 2 篇不同论文**（`resolved.papers` 中 level ∈ {EXACT, HIGH, MEDIUM} 的 match 去重计数）。判定只看「消解层真的把名字匹配到了库里已知论文」，不看 entities 词袋 —— 任务载体/指代词（入库/下载/该论文/向量数据库…）天然匹配不到论文，自动被排除，无需黑名单；
- 含对比/多跳关键词（对比/compare/vs/versus/区别/哪个更好/综述/survey）；
- 问句含两个以上 `？` 或 `?`。

配套：`resolve_node` 同时把 `focus_papers` 和 `entities` 当作论文候选去匹配库内论文（understand 即使把论文名放进 entities —— qwen-plus 常见行为 —— 消解层也能兜住），空匹配静默丢弃，不污染 hints。

`state.mode` 落地为真实字段，plan 模式下 `synthesize` 合并 `subagent_results` 而非裸 tool messages。

### 科研 subagents

写/外网操作才隔离为 subagent，权限范围集中在 [subagents.py](subagents.py) 的 `SUBAGENTS` 声明式配置表（`SubagentSpec`：name + description + system_prompt + tools + max_iterations）。加 subagent = 加一行 config，读表即知每个 subagent 能调什么。**本地只读（`search_papers`/`fetch_content` 与文件工具）由父 agent 直接持有**——「列出/对比/浏览本地」类任务绝不经过 subagent 中转（Claude Code 模式：全能父 agent + 受限 subagent 仅做权限/上下文隔离）。

| subagent | 工具子集 | 职责 | 破坏性 |
|---|---|---|---|
| `arxiv` | `arxiv__search_papers`, `arxiv__get_paper_data`, `arxiv__get_full_paper_text`, `arxiv__list_categories`, `arxiv__update_categories` | 外部 arXiv MCP 专属入口：搜论文 / 读元数据 / 读全文 | 否（只读） |
| `ingest` | `download_paper`, `ingest_paper` | 执行**显式命令块**（`action: download \| ingest \| download_and_ingest` + `arxiv_id`/`paper_name`/`pdf_path` 等字段行）：**纯下载**（arXiv→本地）/ **入库**（解析+向量化，一个原子后台任务）/ **下载再入库**。缺 `action` 拒绝执行，绝不自行推断 | **是**（走权限门） |
| `creator`（v10） | `doc_*` 六件套 + `search_papers`, `fetch_content`, `read_file`, `list_dir` | 按自包含写作任务（doc_id/section_id/参考论文/风格）**逐章写作**：先用库工具采集对比素材（引用 [N]），再 `doc_write_section` 原子落盘。只回状态行 | 写 doc（走权限门） |
| `coder`（v10 Phase C） | `run_experiment`/`experiment_status`/`read_metrics`/`experiment_list` + `git_*` + `delegate_code_task` + `study_*` + `read_file`/`list_dir`/`search_papers`/`fetch_content` | 实验/编码执行：探索项目 → 跑实验（后台）→ 看指标 → 对照研究知识库基线 → 委托外部 coding agent 改代码 → git 提交。诚实汇报真实 metrics | 是（run/commit/delegate 走权限门） |

- **执行契约**：`executor_node` 把 plan 步骤的自由 `args`（`query`/`action`/`arxiv_id`/…）折叠进 subagent 唯一的 `task` 字段（`_subagent_task`，渲染为 `key: value` 命令块，与 react 模式同一契约），避免自由参数名触发 pydantic「missing task」校验失败。
- **调用逻辑守卫（executor 分支点）**：plan-and-execute 本身是无分支顺序执行，为避免「check_paper 已判定本地却仍无条件下载/入库」，`executor_node` 每次 batch 先执行 `check_paper` 步骤（TL 并行分批），再用 `_ingest_guard` 判定：`indexed` → 跳过同论文的下载/入库步骤；`downloaded_not_indexed` → 跳过 `download`/`download_and_ingest`（本地已有 PDF 不该再下载），`ingest` 放行（正是处理本地 PDF 的正确路径）；`absent` → 放行。论文身份用 `resolution.canonicalize` 双向包含匹配，结果写进 `subagent_results`（`skipped: true`）。
- **plan 结构化输出（直接解析模型 JSON 文本，never raise）**：`PLAN_SYSTEM` 的输出契约就是「裸 JSON 对象」。`_ask_for_plan` 直接解析 `model.ainvoke` 回复中的 JSON（剥 ```json 围栏/前后缀，`_extract_json_text` + `_parse_steps` pydantic 校验，非法步骤丢弃），同时兼容 provider 偶尔发出的 function call。**不用 `with_structured_output(method="function_calling")`** —— 它只认 OpenAI tool call，qwen-plus 按 prompt 输 JSON 文本时返回 `None`，把合法 plan 静默丢掉（历史根因：plan 连续 `plan_empty_result` → 空 fallback → executor no-op → 「抱歉，未能生成回答」）。仍空才 `_fallback_plan`：**每个已解析论文一个确定性「直接 `fetch_content`」步骤**（`target="tool"`，只读、绝不触发下载/入库副作用）；无已解析论文 → 空 plan，executor no-op，synthesize 兜底回答。观测：`plan_fallback` / `plan_empty_result` / `plan_llm_failed` 事件。
- **入库决策梯（三态检测）**：父 agent 对"入库/保存"请求**强制先调 `check_paper(term)`**（父级内置工具，只读/幂等）——Redis catalog 判 `indexed`（已入库）→ 文件系统扫 `data/uploads|downloads` + **`data/` 根目录裸 PDF** 判 `downloaded_not_indexed`（PDF 已在本地，matches 带 `pdf_path`）→ 都没有才 `absent`（才允许走 arxiv 子代理检索拿 `arxiv_id` 再 ingest）。数据源于 `GET /api/reader/local-papers`，匹配用 `resolution.match_local_state`（canonicalize 归一双侧磁盘命名清洗差异）。详见上方"工具列表"。
- **下载前 arXiv 身份核验（防张冠李戴）**：`download_paper` 下载前**必须**经 `_fetch_arxiv_title` 拿到该 arXiv ID 的真实标题，拿不到（API 不可用 / ID 无效）→ 返回 `422 unverified` 中止下载，绝不盲下载（曾出现：arxiv API 挂掉时 agent 凭记忆编造 arXiv ID，下载到与所述论文不符的 PDF）。核验通过后返回数据携带 `title`，供父层/用户核对是否与所指论文一致。配套约束（`prompts.py`「事实纪律」+ `subagents.py` ARXIV_SYSTEM）：外部源不可用时禁止凭记忆重构论文身份，不得声称未实际发生的工具调用/结果。
- **执行可视化**：`stream.py` 提供 contextvar + `asyncio.Queue` 事件通道，`plan_node`/`_run_step` 通过 `emit()` 推送 `plan`/`tool_start`/`tool_end`；`web/api/routers/agent.py` 的 `_stream()` 建队列并并发排空，让 plan 模式的分派/执行/完成全过程以 SSE 透出到前端（复用 ToolStep 卡片）。
- **层级可视化（状态树）**：subagent 内部叶子工具调用（如 `arxiv__search_papers`）默认对父层不可见。`as_tool._call` 是 subagent 边界的**唯一权威发出者**（父层 router/executor 对 subagent 工具跳过 emit，避免重复卡片）：它生成 `run_id`、`set_scope(name, run_id)` 标记当前任务，并自 emit `tool_start`/`tool_end`；`dispatcher.call` 检测到 scope 时给叶子工具事件打 `parent_id=run_id`。前端 reducer 用 `parent_id` 维护一棵真实状态树（`ToolStep.children` 嵌套，非按名字 filter），同一 subagent 一轮内多次调用也不会串层；`MessageSteps` 递归渲染。
- **工具所有权（Claude Code 模式）**：父 agent 直接持有**全部本地只读工具**——库检索/阅读（`search_papers`/`fetch_content`）+ 文件三件套（`read_file`/`list_dir`/`write_file`）+ `check_paper`（入库决策梯的确定性第一步）+ `check_task_status`（后台任务栈查询，v9）+ skills。**subagent 3 个**：`arxiv`（外网，只读边界）+ `ingest`（写操作，权限门）+ `creator`（创作域写作：doc_* 工具 + 独立写作上下文，避免章节全文进入父层消息）——只读不隔离、写/外网/长上下文才隔离，避免「列库/扫描/对比」类任务被迫经 subagent 中转导致工具错配（见下「工具列表」）。
- **Token 级流式**：LLM 节点统一走 `_stream_llm`（`model.astream` 逐 chunk `emit("token")`），`_ev_pump` 做 PII 脱敏后转发；`_msg_pump` 不重复 emit 整段 AI 消息。subagent 内部 token 由 `current_scope()` 守卫抑制，不泄漏到主层。
- **subagent vs 工具标识**：subagent 边界事件带 `kind:"subagent"`，前端 `ToolStep.kind` 据此渲染「子代理」徽标 + 主色名，与叶子工具（默认 🔧）区分。

自检：`python agent/tests/test_plan.py`（模式判定 + 拓扑序执行）、`python agent/tests/test_subagents.py`（as_tool 摘要提取 + 子图编译 + 工具子集 + destructive 门）。

## v9 三种请求状态 + 原子入库（异步任务栈）

解决「下载与入库混在一起」「入库被拆成解析/向量化两步」「入库阻塞交互」三个问题。

### 1) 请求三态（agent 必须先判定，再行动）

| 用户意图 | 命令块 action | 执行 |
|---|---|---|
| **下载**："下载…到 X 目录" / "download to ./data" | `download` | 从 arXiv 取 PDF 落盘到用户目录。纯文件，无解析 |
| **入库**："入库 / 导入 / 加入知识库 / 搜索到" | `ingest` | **一个完整操作**：解析 PDF → 写入向量库（论文变可检索）。绝不分两步 |
| **下载再入库**：两者都要 | `download_and_ingest` | 先 `download_paper` 再 `ingest_paper`，一次请求完成 |

`AGENT_SYSTEM` Workflow 第 0 步就要求贴标签；工具面（ingest subagent 的 `download_paper` + `ingest_paper`）正好一一对应这两个原子动词，不存在「把解析和向量化分开」的中间态。

### 2) ingest_paper 异步原子化：立即返回 task_id

- `ingest_paper`（原 process_paper，改名对齐职责）改 POST `/api/agent/ingest` **立即返回** `{task_id, status: "running"}`。
- 后端 `_run_ingest` 在**同一个任务**上依次执行 `_run_pipeline`（解析）→ `_run_indexing`（向量化），`stage` 字段（parse → index）与进度连续推进——不再创建内部子任务，`/api/agent/tasks` 里一条 ingest = 一次完整入库。
- agent 告知"已后台入库，完成会通知"后立即结束回合，用户可继续对话。

### 3) 进度可视化 + 完成通知

- 前端聊天区上方 `TaskCenter`：每个入库任务一行，实时显示两阶段标记 `① 解析 ⏳ → ② 向量化入库` + 当前阶段进度文本 + 不确定进度条；done → ✅ / failed → ❌（可关闭）。
- 完成/失败时前端检测任务终态 → `/api/agent/notify/stream` → `agent/notifier.py`（零状态 prompt）生成 1~2 句通知语，追加一条 assistant 气泡告知用户。
- 期间用户可随时问"入库好了吗"：父 agent 直接调用 `check_task_status(task_id)` 读任务栈回答。

### 后台入库数据流

```
用户："把 RMNet 入库"
 is_agent → check_paper("RMNet") → absent
 → arxiv subagent (拿 arxiv_id) → ingest subagent
    ├─ download_paper(arxiv_id, destination)   # 同步，秒级
    └─ ingest_paper(name, pdf_path)            # 异步：POST /api/agent/ingest → 返回 task_id
 → agent 回复："已后台入库（一个任务），完成后我会通知你"
   （回合结束）│
              ▼ 服务端 _run_ingest（在同一个 task 上推进）
   stage=parse → _run_pipeline(PDF→Markdown→RAG chunks)
   stage=index → _run_indexing(向量化入 Qdrant)
              ▼ 前端每 4s poll /api/agent/tasks → TaskCenter 阶段标记滚动
   任务 done/failed ── notify/stream ──▶ notifier LLM ──▶ "✅ RMNet 已入库…" 消息气泡
```

## v10 领域扩展（论文 / 创作 / Coding）

对照 [docs/agent-multiagent-plan.md](../docs/agent-multiagent-plan.md) 演进方向 + 一份三领域方案（
`~/.claude/plans/lexical-humming-blum.md`）。**单编排器 + 领域 subagent 保持不变**，
新增第三领域执行能力的全部 wrapper 都是既有机制（plan→executor→subagent）。

### 领域路由：paper / creation / coding

```
understand（LLM 输出 domain label）→ route_intent → resolve → domain_node → decide_mode
                                            │                          │
                                            │                          └─ react | plan（
                                            │                              creation/coding 强制 plan）
                                            └───────────（chat / clarify 不变）
```

- `UnderstandResult.domain`（state.py）: LLM 粗标（paper|creation|coding，默认 paper）。
- `route_domain`（nodes.py）: **强行为动词 rule 覆盖 LLM label**（写论文/润色/跑实验/复现/
  改代码/实验进度…）；内容词（指标/训练/实验结果）**不**算 coding 信号——"RMNet 实验部分
  用了什么指标"是 paper 问答，绝不误路由。混合/无强信号 → 回退 LLM label → 默认 paper。
- `domain_node`（graph.py 新增节点）：`resolve → domain → decide_mode`，react 路径零改动。

### 创作 Agent（creator subagent）

写作文本流：creation 域强制进 plan 通道，`plan_node` 用 `CREATION_PLAN_SYSTEM`（章节大纲，
target="creator"）→ `_creation_plan` 里 **`_ensure_writing_doc` 建 doc + 落大纲**（doc_id 注入
每个步骤 args）→ `executor_node` 逐章调 creator subagent → `doc_write_section` 落盘 + SSE
`doc_section` 事件 → `doc_export_docx` 导出。

- **章节串行**：`_creation_plan` 强制把每章 `depends_on` 链到前一章（覆盖 LLM 空依赖）。
  并行 creator 同写一份 `doc.json` 是 read-modified-write 竞争（丢章节状态），且后章
  `doc_get_state` 需要看到前章内容才有交叉一致性。
- **落盘为准（v10.1 修复）**：creator 步骤由 `executor` 做确定性校验（`_verify_creator_step` →
  `creation.verify_section_written`）：outline 中该 section `status==done` 才算产出。
  subagent 只以纯文本作答（多完整都算整章正文）→ **步骤判失败**、正文不进入
  `subagent_results`，失败自动补一次显式「必须 doc_write_section」的重试。
  此前该约束缺失，「聊天回全文、doc 只落最后一章」是真实故障（2026-09 实测）。
- **creation synthesize ≠ 正文合并**：`_synthesize_plan` 对 `domain=creation` 输出**确定性
  写作进度报告**（每章 status/字数 + doc_id，读写工作区/导出提示），不把 subagent 正文
  拼给 LLM——写作本体只落在 doc，聊天只反馈进度。
- **LangSmith 追踪**：`as_tool._call` 声明 `config: RunnableConfig` 参数并透传给
  `subgraph.ainvoke(init, config=config)`，`executor` 也把图配置传入 `tool.ainvoke(...)`，
  subagent 运行作为子 run 挂到父 trace（此前不带 config 会脱离 trace）。
- **step/turn 上限统一配置**（`agent/config.yaml` + `agent/config.py::load_limits`）：
  父 agent `max_steps`（默认 30）/ `max_turns`（默认 50）+ 各 subagent `max_steps`
  （`subagents.<name>.max_steps`，creator=12）全在配置文件里。优先级
  `env AGENT_MAX_STEPS / AGENT_MAX_TURNS` > config.yaml > 代码默认。
  `state.py` 类属性默认值、`subagents.py::build_subagents` 都从 `get_limits()` 取值。

- **数据模型**（`agent/domains/creation.py`）：`web/workspace/docs/{doc_id}/`
  - `doc.json` — outline（含每章 status）+ sections 进度
  - `{doc_id}.md` — 按 outline 顺序拼接的章节 markdown（docx 导出/阅读器读它）
  - `sections/{section_id}.md`、`exports/{doc_id}.docx`（python-docx 生成）
- **doc 工具**（仅 creator subagent 可见，不进父 agent 工具面）：`doc_create / doc_set_outline /
  doc_write_section / doc_get_state / doc_list / doc_export_docx`。返回统一 JSON 信封。
- **引用**：章节内保留 `[N]` 标记，父层 synthesize 统一 `_resolve_citations`（与 paper 域同机制）。
- **~/.demo/memory/profile.json** 扩展 `writing_style` 字段承载写作偏好（后置）。

### Coding Agent（v10 / Phase C）— 实验与编码委托

编码域：`experiments | coder` subagent + 基础工具 + MCP bridge。

- **数据模型**（[domains/coding.py](domains/coding.py)）：`web/workspace/experiments/{project}/`
  项目目录（创建即用）+ `_runs/{exp_id}/` 实验快照（state.json + run.log + metrics 快照）。
  指标解析支持 `metrics.json` / `metrics.csv`（实验结束自动归档进快照 + 写入**研究知识库**）。
- **`research_service` APIs**：`run_experiment`（后台子进程，cwd=项目目录）、
  `experiment_status`（state + log_tail + metrics）、`read_metrics`、`experiment_list`、
  `git_status/git_diff/git_log/git_commit`（cwd 限定项目，commit 走权限门）。
- **外部委托**：`delegate_code_task` — **MCP bridge 优先**（`.mcp.json` 中编码 server
  工具名带 `codex`/`delegate`/`claude_code` 前缀即接入），**CLI subprocess 兜底**
  （`AGENT_CODING_CMD` 或探测 `claude`/`codex`），无后端时返回结构化错误（不 raise）。
  prompt 零状态、模型名由后端注入（CLAUDE.md 模型无关原则）。
- **研究知识库 study**：`web/workspace/studies/{topic}/knowledge.json`（hypotheses /
  experiments / findings）。实验记录 `_study_archive` **确定性写入**（LLM 只读引用，
  防篡改），`study_context` 供创作/编码域注入对比基线。
- **编码域 plan**：`CODING_PLAN_SYSTEM`（`plan.py::_coding_plan`）→ `coder` subagent
  执行（探索→跑实验→看指标→对照基线→委托改进→git 提交）。

### 零回归保证

- domain="paper" 时 `decide_mode` / `plan_node` 行为与 v9 逐字节一致（`_ask_for_plan` 默认
  `PLAN_SYSTEM` 不变）。
- `agent/tests/test_creation.py`：route_domain 判定 + decide_mode 领域强制 plan + doc 生命周期
  （含 docx 可读回）+ 路径穿越拒绝。

自检：`python agent/tests/test_creation.py`

## 持久化

对话状态持久化到 `checkpoints.db`（项目根目录）。同一 `thread_id` 在服务器重启后继续使用。

```python
# 跨重启多轮对话
result = await run("What is RMNet?", thread_id="paper_rmnet")
# ... 服务器重启 ...
result2 = await run("How does it compare?", thread_id="paper_rmnet")
# ↑ 自动恢复上文
```

## 记忆管理 (v3.1) — 未变更

```
                    ┌─────────────────────────────┐
                    │        MemoryManager          │
                    │                              │
                    │  snapshot(state, 2000 tok)    │
                    │                              │
                    │  ┌────────────────────────┐   │
                    │  │ 摘要区 (~800 tokens)     │   │
                    │  │ LLM 压缩的更早消息         │   │
                    │  │ 惰性更新 (~每6轮一次)      │   │
                    │  ├────────────────────────┤   │
                    │  │ 缓冲區 (~1200 tokens)    │   │
                    │  │ 最近 6 条消息原文          │   │
                    │  ├────────────────────────┤   │
                    │  │ 用户画像 (JSON 文件)      │   │
                    │  │ 语言偏好、已知论文         │   │
                    │  └────────────────────────┘   │
                    └─────────────────────────────┘
```

- **统一消费**: 所有下游节点（agent/chat/clarify）从 `state["context_snapshot"]` 读取上下文，不再各自切割 `state["messages"]`
- **摘要缓存**: 超过缓冲区（6条）的消息被 LLM 压缩为摘要，缓存在 `state["summary_cache"]`，仅新消息到达时增量更新
- **跨 session 画像**: `~/.demo/memory/profile.json` 持久化用户偏好，重启不丢失

## v3 核心改进 — 保留

- **Router 三分类 + 置信度门禁**: understand 改为 literature_search / general_chat / needs_clarify，confidence < 0.5 直接走 clarify
- **上下文感知路由**: understand_node 注入最近对话上下文，解析"这段"/"它"等模糊指代
- **推理自评协议**: agent prompt 内置 3 个自评问题（Sufficiency / Next Action / Loop Check）
- **快速通道**: agent 输出 [FINAL_ANSWER] 标记 → 子图直接退出
- **工具合并**: 6→2 核心工具（search_papers + fetch_content）
- **max_iterations**: 5

## 文件

| 文件 | 说明 |
|------|------|
| `graph.py` | **父图** — 路由编排 + SqliteSaver + `run()` 入口 |
| `search_loop.py` | **搜索子图** — agent ↔ tools ReAct 循环 |
| `plan.py` | **Plan-and-Execute** — `decide_mode` + `plan_node` + `executor_node`（Phase 7） |
| `subagents.py` | **Multi-Agent 运行时** — `build_subagent` 工厂（agent+tools+synthesize 三节点）+ `as_tool`（`resolved` contextvar 跨边界注入）+ `SUBAGENTS` 配置表（arxiv/ingest，Claude Code 模式，Phase 8） |
| `stream.py` | **执行事件通道** — contextvar + `asyncio.Queue`，`emit()` 供 plan 节点推送进度事件；`set_scope`/`current_scope` 标记 subagent 作用域（{agent, id}）供层级可视化 |
| `nodes.py` | LLM 节点: understand, memory, agent, synthesize, chat, clarify + 路由函数 |
| `config.py` + `config.yaml` | **执行约束统一配置**（v10.1）— 父 agent `max_steps`/`max_turns` + 各 subagent `max_steps`；`load_limits()`/`get_limits()` 加载（env > yaml > 默认） |
| `memory.py` | **MemoryManager** — 上下文组装 + 摘要缓存 + 用户画像持久化 |
| `resolution.py` | **引用解析层** — 确定性 paper/section 匹配 + 置信度分级 |
| `tools.py` | 工具装配工厂 — providers → dispatcher → StructuredTool（`ensure_tools()` 惰性构建） |
| `providers/` | 统一工具层 — `BuiltinProvider` / `GenericProvider` / `MCPProvider` / `SkillProvider` + `CompositeToolProvider` |
| `dispatcher.py` | **统一工具调度器** — 超时兜底 + 幂等重试 + 审计（Phase 3） |
| `safety.py` | **安全层** — PII 脱敏 + 权限门（Phase 2） |
| `observability.py` | **观测层** — trace_id 结构化日志 + 计数器（Phase 0） |
| `prompts.py` | System prompt 模板（遵循 CLAUDE.md 四条约束） |
| `notifier.py` | **后台任务完成通知器**（v9）— 任务状态 dict → 1~2 句用户通知（零状态 prompt，`notify/stream` 消费） |
| `domains/creation.py` | **创作域**（v10）— DocStore（`web/workspace/docs/{doc_id}/`）+ doc 工具 + docx 导出 + `_ensure_writing_doc`（plan 建 doc） |
| `domains/coding.py` | **编码域**（v10 Phase C）— ExperimentStore（`experiments/{project}/_runs/{exp_id}/`）+ 后台实验运行/指标解析 + git 工具 + `delegate_code_task`（MCP bridge→CLI）+ study 知识库（确定性归档） |
| `state.py` | AgentState（分层设计 + 治理预算字段）+ UnderstandResult |
| `docling_parser.py` | Docling PDF 解析器（PDF → Markdown） |
| `academic_chunker.py` | 学术文献切分（章节感知） |
| `chunk_viz.py` | Chunk HTML/JSON/Streamlit 可视化 |

## 快速使用

```python
import asyncio
from agent.graph import run

async def main():
    result = await run("What is the loss function used in RMNet?")
    print(result["messages"][-1].content)

    # 多轮对话（同一 thread_id 保持上下文，跨重启持久化）
    result2 = await run(
        "How does it compare to other methods?",
        thread_id="session_1",
    )
    print(result2["messages"][-1].content)

asyncio.run(main())
```

## 流式 API

```python
# SSE 流式输出 — LLM token 实时推送
async for msg, metadata in agent.astream(
    {"messages": [HumanMessage(content=query)]},
    config=config,
    stream_mode="messages",
):
    if isinstance(msg, AIMessageChunk) and msg.content:
        print(msg.content, end="", flush=True)  # 逐 token 输出
```

## 依赖

- API 服务必须启动: `uvicorn web.api.main:app --port 8000`
- Redis 运行在 `127.0.0.1:6379`
- 模型: DashScope qwen-plus（默认），可通过 `run(model="qwen-max")` 覆盖

## 工具列表

`tools.py` 维护**两个注册表**，解决「父 agent 绑定 20 个工具但系统提示词只提 builtin+arxiv」的死重问题：

- `_BASE_TOOLS`：全部来源（builtin + generic + MCP arxiv + skills）的完整注册表，父 agent 从这里挑**全部只读工具**，subagent 按 `SUBAGENTS` 配置挑受限权子集。
- `_ALL_TOOLS`：父 agent 实际绑定的表面 = 本地只读全套（文件三件套 + `search_papers` + `fetch_content` + `check_paper` + `check_task_status`）+ skills + 2 个 subagent 工具（Claude Code 模式：只读不隔离，写/外网才隔离）。

### 父 agent 直接绑定（`_ALL_TOOLS`）

| 工具 | 来源 | 说明 |
|------|------|------|
| `search_papers` | builtin | 本地库检索；`query=""` 列出全部论文 |
| `fetch_content` | builtin | 本地论文精读：`paper_name` + `section`（空 section = 概览/章节清单） |
| `check_paper` | builtin | **入库决策梯第一步**：本地检测（Redis 已入库`indexed` → 本地产物`downloaded_not_indexed` → `absent`），只读/幂等，无网络。论文 `state` 语义两类（indexed/not_indexed），parsed/raw 由 `detail` 派生（方案 B） |
| `check_task_status` | builtin | **后台任务栈查询**：按 task_id 返回 pending/running/done/failed + progress/error/result（v9 新增，用户问"入库好了吗"时用） |
| `read_file` / `list_dir` / `write_file` | generic | 文件 explorer 三件套，桌面客户端右侧面板可见 |
| `skill__list` / `skill__load` | skills | 列/加载技能 |
| `arxiv` / `ingest` | subagents | 委派写/外网任务给 subagent，父层只见「任务 → 摘要」 |

### 仅 subagent 可见（不进 `_ALL_TOOLS`）

| 工具 | 归属 subagent | 说明 |
|------|--------------|------|
| `download_paper` / `ingest_paper` | `ingest` | 下载（纯文件动作：`destination` 指定目录 + 简称命名）+ **原子入库**（解析+向量化一个后台任务，返回 task_id，stage 连续推进，v9） |
| `arxiv__*`（5 个） | `arxiv` | 外部 arXiv MCP 透传 |
| `doc_create` / `doc_set_outline` / `doc_write_section` / `doc_get_state` / `doc_list` / `doc_export_docx` | `creator`（v10） | 创作域 doc 工具（`agent/domains/creation.py`）：建 doc / 落大纲 / 逐章原子写入（SSE `doc_section`）/ 状态查询 / 列表 / docx 导出。只进 creator subagent，父 agent 工具面不可见 |
| `run_experiment`/`experiment_status`/`experiment_list`/`read_metrics` + `git_status`/`git_diff`/`git_log`/`git_commit` + `delegate_code_task` + `study_context`/`study_add_hypothesis` | `coder`（v10 Phase C） | 编码域工具（`agent/domains/coding.py`）：后台实验运行/指标解析、项目 git 版本控制、外部 coding agent 委托（MCP bridge→CLI 兜底）、研究知识库读写。只进 coder subagent，父 agent 工具面不可见 |

### 通用工具集（generic）暴露控制

`GenericProvider.list_tools()` 用 `_EXPOSED_TOOLS` allowlist 只暴露文件三件套。
`get_time` / `calculator` / `fetch_url` 代码保留但**不注册**（研究 agent 死重，移出工具列表）。
路径 resolve 后必须落在项目根内（越界即拒），`write_file` 被 `AGENT_USER_ROLE=user` 权限门拦截。

### 出参入参契约（统一信封，P6 收敛）

信封生成/解析/截断的唯一权威是 **`agent/tool_contract.py`**：

- `ok(data)` / `err(error_type, detail, next_action="", **ctx)` 生成统一 JSON envelope；
  结构化/库工具（builtin / creation / coding）成功 → `{"ok": true, "data": {...}}`；
  失败 → `{"ok": false, "error", "error_type": "param_error|transient|permission_denied|not_found|unknown", "next", 附加字段如 available_papers}`。
  纯文本工具（`read_file`/`list_dir`/`get_time`/`calculator`/`fetch_url` 成功路径）保证「不是合法 envelope」的 UTF-8 文本。
- `parse_tool_result(content)` → ToolResult（`is_envelope`/`ok`/`data`/`text`/`error`/`error_type`/`next_action`/`extra`）是**唯一解析入口**：`build_tools_node`、`_classify_tool_error`、`_salvage_tool_content`、`plan._ingest_guard` 全部经它分流，不再各自猜格式。
- `truncate_tool_result(text, limit)` 字符级截断但保持 envelope 可解析（截在 `data` 内部），避免截断把 JSON 切成半截整体作废。
- `dispatcher` 捕获异常也归一化为此信封（不再 `raise`）。

自检：`python agent/tests/test_generic.py`（路径越界被拒 / calculator 拒绝非算术 / user 无法 write_file）、`python agent/tests/test_subagents.py`（subagent 权限子集 + as_tool 摘要提取）、`python agent/tests/test_dispatcher.py`（异常返回错误信封）、`python agent/tests/test_info_flow.py`（P6 信封生成/解析/截断 + P1 marker 过滤）。

自检：`python agent/tests/test_generic.py`（路径越界被拒 / calculator 拒绝非算术 / user 无法 write_file）、`python agent/tests/test_subagents.py`（subagent 权限子集 + as_tool 摘要提取）、`python agent/tests/test_dispatcher.py`（异常返回错误信封）。

## Graph 节点

| 节点 | 类型 | 说明 |
|------|------|------|
| `understand` | LLM + structured output | Router: 三分类 + 置信度 + 上下文感知 |
| `memory` | **纯代码 + 惰性 LLM** | 上下文快照组装（buffer + summary + profile） |
| `resolve` | **确定性代码** | Paper name 匹配 + 置信度分级 → `state["resolved"]`（checkpointed）。P2 起不再经 contextvar 传递——必要上下文由父代理按 AGENT_SYSTEM「Delegation Priority」显式写进 subagent task |
| `search` | **子图** | agent ↔ tools ReAct 循环（react 模式） |
| `plan` | LLM + structured output | 拆解复杂查询为步骤表（plan 模式） |
| `executor` | **纯代码** | 拓扑序执行 plan 步骤，无依赖并行 |
| `synthesize` | LLM | 安全网: 最终答案 |
| `chat` | LLM | 轻量对话 |
| `clarify` | LLM | 追问澄清 |

## 用户画像

手动编辑 `.demo/memory/profile.json`:

```json
{
  "preferred_language": "zh",
  "known_papers": ["Diffusion-RSCC", "RMNet"],
  "frequent_topics": ["remote sensing", "diffusion models"],
  "style_hints": "prefers concise, Chinese responses"
}
```

MemoryManager 自动注入到 context_snapshot 中供下游节点消费。

## 设计决策

### 为什么 resolve 保持为图节点而非工具
resolve_node 在 agent 之前强制执行，零 LLM token 消耗，不可跳过。降级为可选工具会引入额外往返且 LLM 可能忘记调用。

### 为什么保留 synthesize 安全网
快通道 = 模型按 Response Protocol 自评后直接作答（无 tool_calls 即终局，P1 起不再有 [FINAL_ANSWER] marker 协议）。对弱模型，synthesize 含 `_salvage_tool_content()` 内容抢救逻辑。对强模型几乎不触发。

### 为什么工具从 6 降为 2
更少的 tool definition → 更少的 token 消耗 → 更高的工具选择准确率。

### 为什么记忆层是纯代码而非 agent 自管理
Letta 的 agent 自主管理记忆模式需要可靠的 tool-calling。对 qwen-plus 太重。确定性组装 + 惰性 LLM 摘要更稳健。

### 为什么搜索循环是子图
agent ↔ tools 循环的复杂性封装在一个编译单元内，父图只关心编排。子图可独立测试，不影响父图的 understand → memory → resolve 流程。

### 为什么用 stream_mode="messages" 而非 astream_events
`messages` 模式语义更窄：只产出 (message, metadata) 元组，事件类型少、跨版本稳定性好。`astream_events` 会暴露 LangGraph 内部 Runnable 嵌套结构。

### 为什么下载与入库解耦（但入库动作由命令块显式指定）
`download_paper(arxiv_id, destination, filename)` 是**纯文件动作**：落盘到用户指定的工作区目录（`destination` 经 `resolve_workspace_path` 校验，与客户端文件树同一边界 → 文件出现在用户要求的位置），文件名默认自 arXiv 标题推导简称（冒号/破折号前的首 token，推导不出回退 arxiv_id），`filename` 可显式覆盖。它**绝不**触发 parse/chunk/index。`ingest_paper` 是独立的「入库」——**一个完整操作**：解析 PDF → 写入向量库，不拆成解析/向量化两个步骤。

- **入库意图不再依赖字面关键词**：ingest 子代理的契约是 `action: download | ingest | download_and_ingest` **命令块**（由父 agent / plan 步骤确定性填充）。缺 `action` 或无法识别 → **拒绝执行**并报"缺少明确 action"，绝不自行推断（旧版以任务串含『入库』二字才放行，漏写即静默只下载）。`download_and_ingest` 强制「先 download 再用返回的真实路径调 ingest_paper」。
- **入库前先定三分**：父 agent 对任何"入库/保存"请求第一步必须 `check_paper(term)`——本地已入库（indexed）告知即可、本地有产物（downloaded_not_indexed，detail=parsed/raw，输出有解析产物或 PDF）只 index、二者皆无（absent）才允许 arXiv 检索兜底。避免重复下载和跳过本地检测。`parsed`/`raw` 不再作为持久论文状态，由 `/api/reader/local-papers` 的 `detail` 字段现场派生。
- 两个收益：①「下载到 X」不会隐式产生向量库副作用；②工具返回**真实**的 `relative_path`/`filename`，INGEST prompt 强制 subagent 原样汇报而不得声称未执行（修复"模型谎报已按简称命名"）。`pdf_path` 参数打通"下载到自定义目录后再入库"（`/api/pdf/process-local` 支持工作区路径）。

## 信息链核查落地（INFO_FLOW_REVIEW P1–P5）

对照 Claude Code 模型核查的信息链修复，落地状态（细节见 `docs/INFO_FLOW_REVIEW.md`）：

- **P1 marker 协议已删除**：`AGENT_SYSTEM` Response Protocol 改为「回答 YES 直接写最终答案」，不再要求 `[FINAL_ANSWER]`。终止判定（`after_agent`）本就不依赖它。残留 marker 由统一正则 `[\[【]\s*final\s*[-_ ]?\s*answer\s*[\]】]`（IGNORECASE）在 `_stream_llm`（前缀缓冲 → 任意位置行过滤，容忍首 chunk `"\n"`/全角括号/分隔符变体）与 `router._strip_marker` 兜底过滤，杜绝流入 UI / checkpoints / memory 摘要。
- **P2 contextvar 通道已删除**：`resolution._resolved_ctx` 侧通道移除（不可见 / 非零状态 / 无生命周期）。`state["resolved"]` 仍是 checkpointed 官方状态；父代理按 AGENT_SYSTEM「Delegation Priority」把 Discovery Hints 的 `match` 论文名显式写进 subagent task。
- **P3 subagent 返回提取收紧**：`as_tool._call` 只取「无 tool_calls 的最后一个 AI 文本」——带 tool_calls 的 AI 消息仍是中途状态（max_steps 截断前正计划下一步），不再当答案。
- **P4 plan executor 错误恢复**：直接工具步骤失败时确定性重试一次——`transient` 原参数；`param_error` + `available_papers/sections` 按 error envelope 修正参数（同步骤内，不级联依赖步骤重跑）。`not_found`/`backend_down` 不重试，失败原因进 `subagent_results` 由 synthesize 标注。`_run_step` 把「调用成功但信封 ok=false」归一为 `ok=False`。
- **P5 SSE AIMessageChunk 分支删除**：节点内手动 `model.astream()` 不产生 graph 级 messages 流 chunk（实证），token 事件只经 `_ev_pump`（`_stream_llm` → event 队列）一条路；`_msg_pump` 只处理节点最终 AIMessage（tool_calls 边界）与 ToolMessage。

回归测试：`agent/tests/test_info_flow.py`（P1 marker + P6 契约）、`agent/tests/test_plan.py`、`agent/tests/test_subagents.py`、`agent/tests/test_loop.py`。
