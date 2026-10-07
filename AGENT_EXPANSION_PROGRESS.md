# Agent 三领域扩展与平台架构进度（v10 → Phase E）

> 将 agent 从「科研文献助手」扩展为「科研全流程助手」：**论文 / 创作 / Coding** 三领域。
> 完整设计方案在审批计划 `~/.claude/plans/lexical-humming-blum.md`；演进历史对照
> `agent/README.md`（v5→v10）与 `docs/agent-multiagent-plan.md`。
>
> Phase E 起同时跟踪 **Agent 平台架构**（LangGraph-first，`docs/agent-platform-architecture.md`）
> 的落地进度：运行时契约 / 工具治理 / 上下文与记忆 / 版本化 / 可观测。
>
> 评测体系（LangSmith-first，`docs/agent评测体系构建.md`）的落地状态见下文「评测体系」段落。

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

## 已完成 — 评测体系（LangSmith-first，2026-09-08；P1 嵌套/归档 + 实时进度 2026-09-15 闭合）

设计文档 [docs/agent评测体系构建.md](docs/agent评测体系构建.md)（决策：链路数据托管 LangSmith，
本地保留域事件评测层 + 归档 JSONL 兜底）；使用手册 [evaluation/README.md](evaluation/README.md)。

| 模块 | 文件 | 内容 | 状态 |
|---|---|---|---|
| 事件采集 | [evaluation/sink.py](evaluation/sink.py) + [evaluation/trace_store.py](evaluation/trace_store.py) | `log_event` 零改动透传 → asyncio.Queue → 批量 flush → `trace_store.db`（`trace_events` 宽表，每步 duration_ms + 真实 token） | ✅ |
| 显式域事件 | [evaluation/events.py](evaluation/events.py) | intent / llm_call / tool_call / retrieved_context / plan / plan_step / plan_verify / final_answer / turn_end | ✅ |
| 四类指标 | [evaluation/metrics/](evaluation/metrics) | 检索（Recall/Precision/NDCG/MRR + bootstrap CI）、LLM-as-judge（context_hit/faithfulness/intent_accuracy + 预算护栏）、工具（per-tool 成功率 / p50/p95 / error_type 归因）、任务（执行成功率 + 中断恢复率） | ✅ |
| 回归基线 | `eval_output/runs/.baseline.json` | 降幅 >5% warn、Recall@5<0.6 block；badcase 回流评测集 | ✅ |
| 评测 API | [web/api/routers/eval.py](web/api/routers/eval.py) | traces/{id} · thread/{thread_id} · runs 报告/badcases · single（阻塞）/ single/start（非阻塞）· 后台跑批 · **runs/{id}/stream 与 single/{id}/stream（SSE 实时进度）** · prune · manifest/from-trace | ✅ |
| 实时进度（过程可见） | [evaluation/live.py](evaluation/live.py) + [evaluation/runner.py](evaluation/runner.py) + [EvalPanel.tsx](web/frontend/src/renderer/src/components/EvalPanel.tsx) | 每完成一条 QA 发布 query_finished + aggregate（累计检索/任务/工具/token/成本 + ETA），SSE 推给实时进度卡；同一份聚合写进 `eval_runs` 进度行（`status=running` + `overall.progress`），跑完被报告覆盖；CLI 与 SSE 同源打印。**实时指标 == 最终报告口径**（同批纯函数） | ✅ |
| 前端评测页 | [EvalPanel.tsx](web/frontend/src/renderer/src/components/EvalPanel.tsx) / [SingleFlowView.tsx](web/frontend/src/renderer/src/components/SingleFlowView.tsx) | run 列表（带 running done/total 进度） + **实时进度卡（进度条/累计指标/最近 10 条明细）** + 报告详情 + badcase 表 + 单样例分阶段线性时间线 | ✅ |
| join 键（P0） | [agent/graph.py](agent/graph.py) / [agent/supervisor.py](agent/supervisor.py) / [evaluation/verify_smith.py](evaluation/verify_smith.py) | 本地 `trace_id` == LangSmith 根 run id（评测路径真实 run id 记进 `trace_events.run_id`）；`python -m evaluation.verify_smith` 一键断言 | ✅ |
| 嵌套透传（P1） | [agent/nodes.py](agent/nodes.py) / [agent/plan.py](agent/plan.py) / [agent/subagents.py](agent/subagents.py) / `ExecutionContext.child_config()` | 节点内 LLM/工具调用透传父 config → 并入 graph trace，游离根 run 消失（2026-09-15） | ✅ |
| run 树归档（P1） | [agent/core/trace_export.py](agent/core/trace_export.py) + [evaluation/runner.py](evaluation/runner.py) | CLI 导出 + 跑批 `EVAL_EXPORT_LANGSMITH=1` 钩子 → `eval_output/runs/<trace_id>/langsmith_runs.jsonl` + manifest（默认脱敏，`AGENT_TRACE_EXPORT_FULL=1` 才含原始 I/O） | ✅ |
| 指标切归档 + sink 退役（P1 剩余） | [evaluation/trace_store.py](evaluation/trace_store.py) + [evaluation/runner.py](evaluation/runner.py) + [agent/graph.py](agent/graph.py) | 生产在线路径不再 attach sink / 写 thread map；本地结构化事件仅评测上下文或 `AGENT_TRACE_STORE_LIVE=1` 保留；每批写 `trace_events.jsonl` 自包含归档 | ✅（2026-09-16） |
| 前端改链（P2）/ 回填闭环（P3）/ 自研解耦（P4） | [evaluation/feedback.py](evaluation/feedback.py) + [web/api/routers/eval.py](web/api/routers/eval.py) | `/api/eval` 保留本地库作为在线评测工作台，`trace_events.jsonl` 作为离线归档；新增 LangSmith feedback 回填 API；P4 明确不重写平台原语 | ✅（2026-09-16） |

**验证**：`evaluation/tests/*`（flow / runner / trace_store / **live**）自检全绿
（`test_live.py`：总线语义 + SSE 数据源 + 跑批中进度行逐条推进 + 实时指标与最终报告逐键相等 + API 路由/帧格式）；
`python -m evaluation.verify_smith "…" --mode react` 断言根 run id == trace_id；
`python -m agent.core.trace_export <trace_id>` 离线导出（网络不可达时快速失败，不影响对话与报告）。

## 已完成 — Phase E：Agent 平台架构落地（P0/P1/P2，2026-09-15）

设计文档 `docs/agent-platform-architecture.md`（LangGraph-first）的 P0/P1/P2 主体落地：
运行时契约、唯一工具入口、上下文预算、记忆生命周期、prompt/配置版本化、LangSmith 导出。
ADR 在 `docs/adr/0001–0006`，可读契约在 `docs/contracts/`，逐项验收状态见设计文档 §13。

| 模块 | 文件 | 内容 | 状态 |
|---|---|---|---|
| 运行时契约 | [agent/core/contracts.py](agent/core/contracts.py) | `ErrorType/Permission/PromptType/AgentError/ToolSpec/RetryPolicy/Budget/PromptSpec/ExecutionContext`（§4.2 全套） | ✅ |
| 每轮上下文 | [agent/core/execution_context.py](agent/core/execution_context.py) | `build_execution_context`（身份/权限/预算/prompt 绑定/工具版本）+ `contextvars` 传播 + `child_config()` | ✅ |
| 配置快照 | [agent/core/configuration.py](agent/core/configuration.py) | `ConfigurationSnapshot`（revision/hash/limits/停用工具/feature flags/prompt 版本/注册表 hash） | ✅ |
| 工具注册表 | [agent/core/tool_registry.py](agent/core/tool_registry.py) | `ToolDef → ToolSpec` 声明式元数据 + `registry_hash` | ✅ |
| 工具策略 | [agent/core/policy.py](agent/core/policy.py) | 角色权限矩阵、审批判定、必填参数校验、幂等键（摘要） | ✅ |
| 工具唯一入口 | [agent/core/tool_gateway.py](agent/core/tool_gateway.py) | §5.1 七步链：版本/schema → 权限 → 审批 → 幂等 → 并发/超时/熔断/重试 → 信封 → 脱敏审计 | ✅ |
| 调度器分层 | [agent/dispatcher.py](agent/dispatcher.py) | `call` 委托 gateway，只保留 SSE（tool_start/end）与评测事件（tool_call/retrieved_context） | ✅ |
| 上下文预算 | [agent/core/context_pack.py](agent/core/context_pack.py) | 分区预算 Context Pack（invariant/task/conversation/retrieved/memory/reserve）+ 检索去重与 MMR-lite | ✅ |
| 记忆策略 | [agent/core/memory_policy.py](agent/core/memory_policy.py) | `MemoryRecord`（source_ref/confidence/TTL/consent/revision）+ `MemoryPolicy` 过滤 | ✅ |
| Prompt 版本化 | [agent/prompt_store.py](agent/prompt_store.py) + [agent/core/prompt_registry.py](agent/core/prompt_registry.py) | 嵌套发布布局 `prompts/<domain>/<id>/<version>.yaml` + `id@version#checksum` 绑定 | ✅ |
| LangSmith 归档 | [agent/core/trace_export.py](agent/core/trace_export.py) | run 树导出 `eval_output/runs/<trace_id>/langsmith_runs.jsonl`（CLI + 评测钩子） | ✅ |
| 图入口统一 | [agent/graph.py](agent/graph.py) + [web/api/routers/agent.py](web/api/routers/agent.py) | `prepare_turn`：root run id == 本地 trace_id + 完整 metadata（三条入口同一契约） | ✅ |
| 图级工具审批 | [agent/core/approval.py](agent/core/approval.py) + [tool_gateway.py](agent/core/tool_gateway.py) + [graph.py](agent/graph.py) + [agent.py](web/api/routers/agent.py) + [MessageList.tsx](web/frontend/src/renderer/src/components/MessageList.tsx) | 副作用工具 `interrupt()` 暂停 → SSE/API 展示 → `/api/agent/resume` → checkpoint 续跑；拒绝不执行 adapter | ✅（2026-09-16） |
| Context Pack 注入 | [agent/core/context_pack.py](agent/core/context_pack.py) + [agent/memory.py](agent/memory.py) + [agent/nodes.py](agent/nodes.py) | Agent 每轮从最新 messages 重建 retrieved zone；profile 只进 memory zone；conversation 不再重复 profile | ✅（2026-09-16） |
| 长期记忆闭环 | [agent/core/memory_store.py](agent/core/memory_store.py) + [web/api/routers/memory.py](web/api/routers/memory.py) + [MemoryPanel.tsx](web/frontend/src/renderer/src/components/config/MemoryPanel.tsx) + [agent/nodes.py](agent/nodes.py) | typed JSON store；显式“记住/以后请…”写入；policy 过滤；配置中心可查看/禁用/删除 | ✅（2026-09-16） |
| Prompt canary/A-B | [agent/prompt_store.py](agent/prompt_store.py) + [prompt_registry.py](agent/core/prompt_registry.py) + [config.py](web/api/routers/config.py) + [PromptsPanel.tsx](web/frontend/src/renderer/src/components/config/PromptsPanel.tsx) | thread_id 稳定 hash 分流的 canary；版本/checksum/evaluation suite 冻结进 ExecutionContext；配置中心可调百分比 | ✅（2026-09-16） |

**行为要点**

- 每轮冻结一次元数据：`graph.run`、`/api/agent/chat`、`/api/agent/chat/stream` 共用
  `prepare_turn`，metadata 含 `request_id/actor_role/domain/graph_version/config_revision/
  config_hash/prompt_bindings/tool_versions/model_route/eval_dataset_id`。
- 工具调用只有一条路：新调用点必须用 `ExecutionContext.child_config()` 透传父 config
  （否则 LangSmith 出现游离 run）；副作用工具按幂等键重放而不是重跑。
- `memory_node` 在原有 `context_snapshot` 之外把分区预算/来源/截断理由写入
  `context_decision["pack"]`（只记元数据，不含 prompt 文本）。

**验证**：`test_tool_gateway.py`(9) / `test_context_pack.py`(7) / `test_core_contracts.py`(11)
新增全绿；`test_dispatcher/test_plan/test_loop/test_info_flow/test_subagents/test_creation/
test_coding/test_context/test_supervisor` 与 `evaluation/tests/*` 自检回归通过。

**Phase E 剩余**：无功能性剩余。`core/` 之外的物理目录不做一次性搬移；逻辑边界
已由 contracts / registry / gateway / context / memory 固化，后续仅按明确收益
增量迁移。

## 已完成 — P0 运行一致性修复（2026-09-18）

按平台改进方案的首批落地项，收敛「快照已记录但运行仍重读配置」和工具恢复语义：

| 项 | 实现 | 状态 |
|---|---|---|
| 回合运行快照 | `ExecutionContext` 冻结 `budget/plan_step_max_steps/prompt_templates/execution_id`；`graph.run`、`/chat`、`/chat/stream` 都用冻结预算启动 | ✅ |
| Prompt 冻结 | 节点经 `prompt_store.get_prompt()` 优先读取当前回合冻结模板；回合中发布新 Prompt 不再造成 trace 与实际文本错位 | ✅ |
| 审批续跑身份 | `AgentState.execution_id` 持久化，`graph.resume` 从 checkpoint 恢复原 execution_id，副作用幂等范围不跨用户回合 | ✅ |
| 幂等作用域 | 幂等键由 `thread_id + execution_id + tool@version + args` 生成；同一回合可安全重放，下一回合主动重复操作会重新执行 | ✅ |
| 工具热重载 | `reload_tools` 同时失效父图编译缓存与 supervisor 子图模板缓存；Web 路由不再长期缓存旧 compiled graph | ✅ |
| 业务错误治理 | 工具返回 `{"ok": false, ...}` 时进入统一 `AgentError` 分类、可重试判定与熔断器，不再只按“适配器未抛异常”记成功 | ✅ |

**验证**：`C:\Users\30811\miniconda3\envs\demo\python.exe -m pytest agent\tests -q`
全绿（186 passed）。

## 已完成 — P1 DAG 调度与持久任务租约（2026-09-18）

| 项 | 实现 | 状态 |
|---|---|---|
| 资源感知 DAG 调度 | `PlanStep.resource_key` + 调度器：只读步骤并发（受 `AGENT_PLAN_MAX_CONCURRENCY` 限制），写步骤按 `doc/project/paper/file` 资源键互斥，`auto` LLM 步骤保持独占 | ✅ |
| 资源推导 | 未显式提供 key 时从 `doc_id/project/paper_name/arxiv_id/path/destination/filename` 推导；工具按 ToolSpec.side_effect 判定读写 | ✅ |
| 有序结果 | 并发完成后按原计划顺序回填 `subagent_results`，保持 synthesize/verify/评测口径稳定 | ✅ |
| Worker lease | supervisor 持久化 `worker_id/lease_owner/heartbeat_at/lease_expires_at/attempt/recoverable`，后台心跳续租；进程退出/崩溃后租约过期自动标记 orphaned | ✅ |
| 重启恢复 | `resume` 在 `_graphs` 丢失时从元数据重建 worker 图；新增 `recover(task_id)` / `task_recover` 接管 orphaned checkpoint 并续跑 | ✅ |
| 单飞保护 | resume/recover 走 transition lock；单任务不会被并发回复启动两次 | ✅ |

**验证**：新增只读并发、同资源写串行、不同资源写并发、重启后 interrupt 重建、lease 过期恢复用例；
`C:\Users\30811\miniconda3\envs\demo\python.exe -m pytest agent\tests -q` 全绿（191 passed）。

## 已完成 — P2 上下文、Prompt 缓存与领域验收（2026-09-18）

| 项 | 实现 | 状态 |
|---|---|---|
| 硬 token 预算 | 新增 `token_budget.py`：模型调用前按冻结窗口、输出预留和安全比例裁剪；覆盖流式 `_stream_llm` 与非流式 `traced_ainvoke` 两条路径 | ✅ |
| tool pair 保序 | 裁剪以“AI tool_calls + 对应 ToolMessages”为不可拆分单元；最新用户消息优先保留，必要时对长工具结果做 envelope-safe 截断 | ✅ |
| Context Pack 去重 | 检索结果和近期对话已直接从 graph messages 注入，不再重复写进 system prompt；system 只保留长期记忆与旧对话摘要 | ✅ |
| Prompt 解析缓存 | `load_prompt_spec` 按 path/mtime/size 缓存且有界淘汰；版本文件变化自动失效，避免每轮重复 YAML 解析 | ✅ |
| 领域确定性验收 | 创作域以 `doc_progress` 落盘章节为准；实验域以 ExperimentStore 终态/exit code 为准；论文域要求至少一个真实步骤产出，LLM 只能在证据底线之上细化结论 | ✅ |

**验证**：`C:\Users\30811\miniconda3\envs\demo\python.exe -m pytest agent\tests evaluation\tests -q`
全绿（211 passed）。

## 已完成 — P3 模型路由、读缓存与成本看板（2026-09-18）

| 项 | 实现 | 状态 |
|---|---|---|
| 分层模型路由 | `ConfigurationSnapshot.model_routes` 冻结 router/summary/planner/agent/verify/synthesizer/chat/subagent/notifier 的任务模型；默认全走主模型，配置 `AGENT_MODEL_SMALL` 后路由、摘要、通知切小模型 | ✅ |
| 单任务覆盖 | `AGENT_MODEL_ROUTES` 支持 JSON 按任务覆盖模型；模型选择只从当回合冻结快照读取，不中途重读 env | ✅ |
| 只读工具缓存 | ToolGateway 增加短 TTL read cache（默认 `search_papers/fetch_content/arxiv` 检索类），按 thread + tool@version + canonical args 命中；写工具成功后清空缓存 | ✅ |
| 缓存观测 | `tool_call` trace 增加 `cache_hit`，dispatcher 记录 `tool_cache_hits`；命中不重复调用 adapter | ✅ |
| 成本维度 | 报告 cost 增加 `model_calls/per_model/per_node/cache`；overall 增加 cache hit rate、单成功任务成本、单成功任务 token | ✅ |

**验证**：`C:\Users\30811\miniconda3\envs\demo\python.exe -m pytest agent\tests evaluation\tests -q`
全绿（216 passed）。

## 待办（按阶段）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase A** | 创作后端：领域路由 + doc 工具 + creation API + creator subagent | ✅（2026-08-31） |
| **Phase B** | 前端「论文写作」工作区：领域 Tab、WriterView、api.ts 扩展 | ✅（2026-08-31） |
| **连通性修复** | 两个断裂点（`PlanStep.target` Literal、`AgentState.domain` schema）+ prompt 强化，真实 e2e 跑通 | ✅（2026-08-31） |
| **Phase C** | coding 后端：coder subagent、实验运行/指标解析、git 工具、delegate（MCP bridge→CLI）、`routers/experiments.py`/`study.py`、研究知识库（确定性写入） | ✅（2026-08-31） |
| **Phase D** | 前端「实验」工作区：ExperimentView（实验列表/详情/指标面板/git 面板/日志流）、三 Tab 联调 | ✅（2026-08-31） |
| **Phase E** | Agent 平台架构 P0/P1/P2：运行时契约 + 唯一工具入口（权限/审批/幂等/熔断）+ 上下文预算 Context Pack + 记忆策略 + prompt/配置版本化 + LangSmith run 树导出；ADR `docs/adr/0001–0006`、契约 `docs/contracts/` | ✅（2026-09-15） |
| Phase E 收尾 | 不执行一次性目录搬迁；保留兼容边界，按收益增量迁移 | ✅（2026-09-16） |
| **评测体系 P0/P1** | LangSmith-first 采集（join 键 + 嵌套透传 + run 树归档）+ 本地域事件评测层 + 四类指标 + 回归基线 + `/api/eval` + 前端评测页 | ✅（2026-09-08 / P1 2026-09-15） |
| **评测实时进度** | 跑批逐条发布指标（`evaluation/live.py`）+ SSE `/runs/{id}/stream`、`/single/{id}/stream` + `eval_runs` 进度行 + 前端实时进度卡 + CLI 实时打印（同源同口径） | ✅（2026-09-15） |
| 评测体系剩余 | 无功能性剩余；P2 采用本地工作台 + 离线归档边界，P3 feedback API 已接，P4 不重写平台原语 | ✅（2026-09-16） |
| 后置 | typed memory store 已可承载 writing_style 等偏好；`profile.json` 继续作为 legacy 只读适配 | ✅（2026-09-16） |

### 手动验证（桌面端）

1. `npm run dev` 起 Electron（自动拉起 uvicorn:8001）。
2. 文献问答 Tab：发「写一篇关于 XX 的简述/综述」→ 看 SSE 卡片里出现 creator subagent 调用。
3. 写作 Tab：章节树随写入逐章标 ✓ → 编辑某章 → 保存 → 导出 docx 下载可打开。

## 关键数据位置

- 创作文档：`web/workspace/docs/{doc_id}/`（doc.json + sections/*.md + 主 md + exports/*.docx）
- Redis key：`dedup:*`（论文）、`task:*`（后台任务）——doc 状态暂走文件系统，无 Redis 依赖
- SSE 事件：新增 `doc_section`（写作进度，后端 `agent/stream.py::emit`）
- 评测归档：`eval_output/runs/<run_id>/`（`run_summary.json` + `per_query.jsonl`；
  `EVAL_EXPORT_LANGSMITH=1` 时另有 `langsmith_runs.jsonl` + `langsmith_manifest.json`）
  报告表在仓库根 `trace_store.db` 的 `eval_runs` 表（`/api/eval/runs` 读它，不是读 `eval_output/`）；
  评测事件库：`trace_store.db`（`trace_events` 宽表；路径可用 `AGENT_TRACE_DB` 覆盖）
- 配置中心存储：`web/workspace/config.json`（experiment/tools/skills）；
  prompt 覆盖：`web/workspace/prompts/<ID>.yaml`（扁平）或 `prompts/<domain>/<id>/<version>.yaml`（嵌套，优先）

## 注意事项（踩坑/决策）

- **领域路由防误判**：「RMNet 实验部分用了什么指标」是 paper 问答——内容词绝不进 coding 关键词，rule 只认强行为动词（[nodes.py](agent/nodes.py) `_DOMAIN_*_STRONG`）。
- **`@tool` 包装的 StructuredTool 不是函数**：router/脚本调用 doc 工具必须 `.ainvoke({...})`（creation router 已统一）。
- **MCP 关闭时的 asyncio 清理噪音**（`cancel scope` RuntimeError）：是既有 MCPProvider 行为，非本项目引入，不影响运行。
- **coding 委托不要重写 MCP 轮子**：外部 coding server 配 `.mcp.json` 即可，复用 `load_mcp_config`。
- **新增 LLM/工具调用点必须透传父 config**：用 `ExecutionContext.child_config()`，
  否则 LangSmith 上是游离根 run（节点内直调 `model.astream`/`tool.ainvoke` 都要带 `config=`）。
- **副作用工具默认会按幂等键重放**：`ToolGateway` 用 `hash(thread_id, tool@version, args, intent)`
  记住已完成结果，中断/恢复不重跑；非幂等工具强制 1 次尝试（不做重试）。
- **审批闸门默认关闭**：`AGENT_TOOL_APPROVAL=1` 时副作用工具直接返回 `APPROVAL_REQUIRED`
  （图级 `interrupt()` 接线前不要在生产打开）。
 - **LangSmith 免费层只有 7 天保留**：结论一律读归档（`eval_output/runs/<trace_id>/langsmith_runs.jsonl`），LangSmith 只当活链路浏览器；跑批记得 `EVAL_EXPORT_LANGSMITH=1`。
 - **清 `trace_store.db` 会同时丢掉评测事件与跑批报告**：前端评测页的两个数据源都在这一个库里。
