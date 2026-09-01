# Agent 三领域扩展进度（v10）

> 将 agent 从「科研文献助手」扩展为「科研全流程助手」：**论文 / 创作 / Coding** 三领域。
> 完整设计方案在审批计划 `~/.claude/plans/lexical-humming-blum.md`；演进历史对照
> `agent/README.md`（v5→v10）与 `docs/agent-multiagent-plan.md`。

## 架构定案

| 决策点 | 结论 | 说明 |
|---|---|---|
| 顶层编排 | **单编排器 + 领域 subagent** | 延续 Claude Code 模式（只读归父、写/外网/长上下文才隔离），共用治理/安全/流式/记忆底座 |
| coding 委托后端 | **MCP bridge 优先**（GitHub 调研定案） | 生态主流是把 coding agent 暴露为 MCP server（claude-codex-bridge / codexmcp）；配进 `.mcp.json`，复用 `MCPProvider` 装配，**不自写 subprocess**。`coder` subagent 工具子集 = server 暴露的工具名 |
| 写作产出 | **本地 Markdown + docx** | `web/workspace/docs/{doc_id}/` 主 md + python-docx 导出；前端内置编辑器 |
| 前端形态 | **领域工作区 Tab** | 文献问答 / 论文写作 / 实验 三工作区；聊天仍是导航入口 |

## 已完成 — Phase A：创作后端（2026-08-31）

领域路由（paper/creation/coding）+ 创作数据模型 + doc 工具 + HTTP API + creator subagent。

| 模块 | 文件 | 状态 |
|---|---|---|
| 领域路由 | `UnderstandResult.domain` / `AgentState.domain`（[agent/state.py](agent/state.py)） | ✅ |
| rule 兜底路由 | `route_domain` / `domain_node`（[agent/nodes.py](agent/nodes.py)）——**强行为动词覆盖 LLM label**（写论文/润色/跑实验/复现…）；内容词（指标/训练/实验结果）不误判 coding，有回归测试防误判 | ✅ |
| 写作 plan 通道 | `decide_mode` 领域强制 plan + `_creation_plan`（[agent/plan.py](agent/plan.py)）+ `CREATION_PLAN_SYSTEM`（[agent/prompts.py](agent/prompts.py)） | ✅ |
| 创作业务模块 | `DocStore` + 6 doc 工具 + python-docx 导出 + `CreationProvider`（[agent/domains/creation.py](agent/domains/creation.py)） | ✅ |
| 写作 subagent | `CREATOR_SYSTEM` + creator（[agent/subagents.py](agent/subagents.py)）——逐章写，只回状态行 | ✅ |
| HTTP 薄封装 | [web/api/routers/creation.py](web/api/routers/creation.py)（6 端点）+ main.py 挂载 | ✅ |
| 图接线 | `resolve → domain → decide_mode`（[agent/graph.py](agent/graph.py)），react 路径零改动 | ✅ |
| Self-check | [agent/tests/test_creation.py](agent/tests/test_creation.py) | ✅ 全绿 |

**验证**：8 个 agent 测试全绿（含既有回归）；creation API 端到端通过（建 doc→大纲→章节→docx）；工具装配正确（doc 工具只进 creator，父层不含）；graph 编译 10 节点。真实 LLM 走「写综述」对话链路已跑通（见下方「写作链路连通性修复」）。

## 已完成 — Phase B：前端「论文写作」工作区（2026-08-31）

顶栏领域 Tab（文献问答/论文写作/实验）+ WriterView 写作面板 + creation API 前端封装。

| 文件 | 内容 |
|---|---|
| [TopBar.tsx](web/frontend/src/renderer/src/components/TopBar.tsx) | 领域 Tab + `Domain` 类型 |
| [WriterView.tsx](web/frontend/src/renderer/src/components/WriterView.tsx) | 三栏写作工作区：文档列表 / 章节树（✓ 徽章，5s 轮询）/ 章节 Markdown 编辑器（保存/导出 docx/字数） |
| [App.tsx](web/frontend/src/renderer/src/App.tsx) | `domain` state + main 条件渲染（write → WriterView，experiment → 占位） |
| [api.ts](web/frontend/src/renderer/src/api.ts) | creation 6 接口 + `downloadDocx`（blob 下载）+ 类型 |
| [routers/creation.py](web/api/routers/creation.py) | GET /docs/{id} 增加 `sections_content`（编辑器加载章节内容） |

**验证**：`tsc --noEmit` 0 错误；`vite build` 通过；后端 TestClient 确认 `sections_content`
返回每章内容。写作本体仍从聊天触发（plan→creator subagent→SSE `doc_section`），
本工作区轮询刷新章节进度。

## 写作链路连通性修复（2026-08-31，两个根因）

「聊天无法调用创作 subagent 写作」排查定位到 **两个独立断裂点**，均已修复并加回归测试：

1. **`PlanStep.target` Literal 漏枚举**（`agent/plan.py`）——创作 plan 的 `target="creator"`
   触发 pydantic 校验失败 → 整单计划被丢弃 → 空兜底。修复：Literal 加 `"creator"`。
   锁定：`test_parse_steps_accepts_creator_target`。
2. **`AgentState` 未声明 `domain` 字段**（`agent/state.py`）——LangGraph schema 无此 key，
   `understand_node` 返回的 `domain=creation` 被**静默丢弃**，图内恒缺失 → `plan_node`
   永远走默认 paper 分支（生成 search/fetch 步骤而非写作大纲）。之前只改了
   `UnderstandResult` 漏了 `AgentState`。修复：`AgentState` 加 `domain`/`doc_id`。
   锁定：`test_agent_state_declares_domain`。

配套强化：`CREATOR_SYSTEM` 明确「**阅读只是过程，`doc_write_section` 才是终态**」；
`CREATION_PLAN_SYSTEM` 要求 `description` 以 "Write the section" 开头（防止 subagent
只读论文不写章节）。

**验证（真实 LLM + uvicorn）**：「写一篇遥感变化检测对比的综述」→ domain=creation →
创作大纲（2 章，target=creator）→ creator subagent 逐章写作 →
`introduction | 174 words`、`method-comparison | 317 words | wrote via doc_write_section` →
doc status **done** → docx 导出（11 段 + 两个 Heading）。全链路 ~70s。产物已清理。

## 已完成 — v10.1 写作链路缺陷修复（2026-09-01）

真实场景曝光三个缺陷，全部修复并加回归测试（详见 TROUBLESHOOTING「写作链路『聊天回全文，doc 只落最后一章』」）：

| 缺陷 | 复现 | 修复 |
|---|---|---|
| creator 仅回正文不落盘 | doc 只写最后一章；聊天却回全文（`c41dd6e66ce5`：前两章 pending） | ① creator `max_steps` 5→12（防中途打满被 synthesize 兜底正文；值统一收进 `agent/config.yaml` `subagents.creator.max_steps`，env `AGENT_MAX_STEPS` 优先于文件）；② executor 对 creator 步骤**确定性落盘校验** `_verify_creator_step`→`verify_section_written`（章节 `status==done` 才产出，失败不转发正文 + 自动重试一次） |
| 并行写 doc 竞争 | —— | `_creation_plan` 强制章节串行（`depends_on` 链前章） |
| 聊天输出全文 | 合成阶段拼 subagent 正文给 LLM | `domain=creation` 的 synthesize 改输出**确定性写作进度报告**（每章 status/字数 + doc_id） |
| LangSmith 缺 creation 调用 | 日志无子代理 run，SSE 有事件 | `as_tool`/`_run_step` 透传 `config` 到 `subgraph.ainvoke`；subagent 变父 trace 子 run |

回归：`test_creation.py` 新增 2 条；8 组 agent 自检全绿。

## 已完成 — Phase C：编码域后端（2026-08-31）

实验运行/指标解析/git 版本控制/外部编码委托 + coder subagent + 研究知识库。

| 模块 | 文件 | 内容 | 状态 |
|---|---|---|---|
| 编码业务模块 | [agent/domains/coding.py](agent/domains/coding.py) | ExperimentStore（`experiments/{project}/_runs/{exp_id}/`）+ 后台实验 + 指标解析（json/csv）+ git 工具 + `delegate_code_task`（MCP bridge 优先→CLI 兜底）+ study 知识库（确定性归档） | ✅ |
| 编码 subagent | [agent/subagents.py](agent/subagents.py) | `CODER_SYSTEM` + coder（探索→跑实验→看指标→基线对比→委托改进→git 提交） | ✅ |
| 编码 plan | [agent/plan.py](agent/plan.py) | `_coding_plan` + `CODING_PLAN_SYSTEM`；`PlanStep.target` Literal 加 `coder` | ✅ |
| HTTP | [web/api/routers/experiments.py](web/api/routers/experiments.py) + [study.py](web/api/routers/study.py) + main.py | 实验 6 端点 + 知识库 2 端点（薄封装） | ✅ |
| 装配/测试 | [agent/tools.py](agent/tools.py) + [agent/tests/test_coding.py](agent/tests/test_coding.py) | CodingProvider 进 base（仅 coder 可见）+ 8+1 测试 | ✅ |

**验证**：8 组测试全绿（新增 test_coding：run_experiment 真实子进程→done→metrics→study 归档、
delegate 无后端结构化错误、路径逃逸拒绝、git commit/diff）；experiments/study API 端到端通过；
真实 agent 对话「在 experiments/demo 跑 train.py 看指标」→ domain=coding → CODING_PLAN →
coder subagent → 正确探索缺失项目并诚实汇报 next step。

## 已完成 — Phase D：前端「实验」工作区（2026-08-31）

<ExperimentView>（三 Tab 完整）：项目切换 + Run Experiment + git 只读面板 + 实验卡列表
（status 徽章/指标摘要/git_sha）+ 详情（指标表格 + sparkline + 日志自动滚动 + 3s 轮询）。
后端补 `GET /api/experiments/projects/{project}/git`。三领域 Tab（文献/写作/实验）全部就位。

## 待办（按阶段）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase A** | 创作后端：领域路由 + doc 工具 + creation API + creator subagent | ✅（2026-08-31） |
| **Phase B** | 前端「论文写作」工作区：领域 Tab、WriterView、api.ts 扩展 | ✅（2026-08-31） |
| **连通性修复** | 两个断裂点（`PlanStep.target` Literal、`AgentState.domain` schema）+ prompt 强化，真实 e2e 跑通 | ✅（2026-08-31） |
| **Phase C** | coding 后端：coder subagent、实验运行/指标解析、git 工具、delegate（MCP bridge→CLI）、`routers/experiments.py`/`study.py`、研究知识库（确定性写入） | ✅（2026-08-31） |
| **Phase D** | 前端「实验」工作区：ExperimentView（实验列表/详情/指标面板/git 面板/日志流）、三 Tab 联调 | ✅（2026-08-31） |
| 后置 | `profile.json` 结构化扩展（writing_style 等）；CLAUDE.md 关键符号/API 概览随 D 完成后更新；前端已完成进度验证 | ⬜ |

### 手动验证（桌面端）

1. `npm run dev` 起 Electron（自动拉起 uvicorn:8001）。
2. 文献问答 Tab：发「写一篇关于 XX 的简述/综述」→ 看 SSE 卡片里出现 creator subagent 调用。
3. 写作 Tab：章节树随写入逐章标 ✓ → 编辑某章 → 保存 → 导出 docx 下载可打开。

## 关键数据位置

- 创作文档：`web/workspace/docs/{doc_id}/`（doc.json + sections/*.md + 主 md + exports/*.docx）
- Redis key：`dedup:*`（论文）、`task:*`（后台任务）——doc 状态暂走文件系统，无 Redis 依赖
- SSE 事件：新增 `doc_section`（写作进度，后端 `agent/stream.py::emit`）

## 注意事项（踩坑/决策）

- **领域路由防误判**：「RMNet 实验部分用了什么指标」是 paper 问答——内容词绝不进 coding 关键词，rule 只认强行为动词（[nodes.py](agent/nodes.py) `_DOMAIN_*_STRONG`）。
- **`@tool` 包装的 StructuredTool 不是函数**：router/脚本调用 doc 工具必须 `.ainvoke({...})`（creation router 已统一）。
- **MCP 关闭时的 asyncio 清理噪音**（`cancel scope` RuntimeError）：是既有 MCPProvider 行为，非本项目引入，不影响运行。
- **coding 委托不要重写 MCP 轮子**：外部 coding server 配 `.mcp.json` 即可，复用 `load_mcp_config`。