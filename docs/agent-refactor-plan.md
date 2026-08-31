# Agent 通用层改造计划

> 目标：将 agent 从「功能可用」提升到「可治理、可观测、可审计」，通用层做厚做稳后再做领域定制。
> 原则：不推倒重来。现有 LangGraph 骨架、Provider 工具层、检索评估已成型，改造是「补短板 + 加约束」，不是重写。

> **执行状态**：本计划已按评估版 [agent-refactor-plan-review.md](agent-refactor-plan-review.md) 裁剪后执行。Phase 1/2/0/5(通用)/3 已完成，进度与产出见 review 文档「执行进度」一节。

---

## 0. 现状盘点（结论先行）

对照七大要求逐项打分，**关键文件**列出现有实现的落点。

| # | 要求 | 现状 | 关键文件 |
|---|------|------|----------|
| 1 | 稳定循环引擎（治理） | **部分**：有 ReAct 子图 + `max_iterations=5`，但无整轮超时、无 token 预算、无正式重试策略 | [search_loop.py](../agent/search_loop.py) [state.py](../agent/state.py) [nodes.py](../agent/nodes.py) |
| 2 | 分层记忆 | **部分**：工作记忆(buffer) + 短期(summary) + 用户画像(JSON) 已有；**长期记忆(向量+图谱)缺失**；画像无权限/历史 | [memory.py](../agent/memory.py) |
| 3 | 工具层 | **良**：Provider 抽象 + JSON Schema + `destructiveHint/readOnlyHint` 注解已齐；缺统一调度器(超时/重试/限流/结构化错误) | [providers/](../agent/providers/) |
| 4 | 知识管理 | **良**：RAG 五阶段 + 索引生命周期(dedup/增量) + 离线质量评估已齐；**知识图谱缺失** | [pdf_pipeline/](../pdf_pipeline/) [indexer/](../indexer/) [retrieval_orchestrator/](../retrieval_orchestrator/) |
| 5 | 规则安全层 | **缺失**：无输入过滤、输出审核、PII 脱敏、权限矩阵、合规拦截 | — |
| 6 | 全链路可观测性 | **几乎缺失**：散落 `print()`；LangSmith env 已加载但未启用；无结构化日志/指标/trace 传播 | [graph.py](../agent/graph.py) 各 `print()` |
| 7 | 人机协作 | **部分**：SSE 流式、clarify 澄清、错误分类已齐；**HITL 审批、确认请求、结构化进度缺失** | [routers/agent.py](../web/api/routers/agent.py) |

**核心判断**：真正的短板是 **#5 安全（全缺）**、**#6 可观测（几乎缺）**、**#1 治理约束（半缺）**、**#2 长期记忆/图谱（缺）**、**#7 HITL（半缺）**。工具层和知识管理只差「加固」而非「新建」。

---

## 1. 优先级原则

通用层优先，定制层靠后。排序依据：**缺失度 × 杠杆率 × 前置依赖**。

1. **观测先行**——不装仪表盘就动手术是盲改，且观测改造越晚欠债越多。
2. **治理约束**——循环的边界（超时/预算/重试）是所有上层能力可信的前提。
3. **安全层**——横切规则，必须拦在输入/输出/工具调用边界，后补会散落各调用点。
4. **工具层加固**——在 Provider 之上加调度器，把超时/重试/审计统一到一个入口。
5. **记忆系统**——补长期记忆与画像，依赖观测(埋点)与工具层(检索复用)。
6. **知识管理**——图谱是领域定制，等通用层稳了再做。
7. **HITL**——复用安全层(destructive 注解)与观测(进度事件)，收口最快。

**不做的事**（YAGNI，显式声明）：
- 不引入 OpenTelemetry / Jaeger / Prometheus 全家桶——先用 stdlib `logging` + `contextvars` trace_id + 内存计数器，确有集群需求再升级。
- 不引入 Neo4j / 图数据库——先用 SQLite(已随 checkpoints.db 存在) 存三元组。
- 不重写 Provider 抽象——已够用，只在其上加调度器。
- 不引入新依赖（Phase 5 图谱若需 jsonschema 校验，复用 LangChain 已有的 Pydantic 校验）。

---

## 2. 分阶段计划

每阶段独立可交付、可回滚，DoD 明确。

### Phase 0 — 可观测性底座（地基）

**现状**：`print()` 散落各文件；`graph.py` 加载了 `.env` 里的 LangSmith 但未启用；无 trace 传播、无结构化日志、无指标。

**改动**：
1. 新增 `agent/observability.py`：
   - `get_logger(name)` → 统一 `logging`，JSON/结构化行（含 `trace_id`, `turn_id`, `node`, `event`）。
   - `contextvars` 存 `trace_id`（复用 [builtin_provider.py](../agent/providers/builtin_provider.py) 里 cite registry 的 contextvars 模式），请求入口生成并贯穿整轮。
   - `@timed` / 计数器：记录每节点耗时、工具调用次数、LLM 调用次数、token 估算量（复用 [memory.py](../agent/memory.py) 的 `_estimate_tokens`）。
2. 替换 agent/ 与 web/api/routers/agent.py 内的 `print()` → `get_logger(...)`。
3. `graph.py` 的 `run()` / API 的 `chat()` 入口：生成 `trace_id`，结束时输出一条汇总日志（intent、迭代次数、工具数、token、耗时、错误）。
4. 暴露 `GET /api/agent/metrics`：内存计数器快照（`tools_called`, `llm_calls`, `errors`, `p50/p95_latency` 等），供前端/调试查看。

**DoD**：任意一轮对话可凭 `trace_id` 从日志还原完整执行路径（节点序列 + 工具调用 + 错误）；`/api/agent/metrics` 返回非零计数。

**依赖**：无新增（stdlib）。

---

### Phase 1 — 循环治理（引擎加约束）

**现状**：`max_iterations=5` 硬切到 synthesize；`request_timeout=120s` 只覆盖单次 LLM 调用；无整轮 token 预算；重试靠 [nodes.py](../agent/nodes.py) 里手写 `_classify_tool_error` 分类。

**改动**：
1. **整轮超时**：在 [graph.py](../agent/graph.py) 的 `run()` 与 [routers/agent.py](../web/api/routers/agent.py) 用 `asyncio.wait_for(ainvoke, timeout=...)`，超时走 synthesize 兜底（默认 300s，env `AGENT_TURN_TIMEOUT` 可调）。
2. **token 预算**：`state.py` 加 `token_budget` / `tokens_used` 字段；`agent_node` 每次调用后累加 `_estimate_tokens`，超预算 → 注入 system 提示「必须终止并输出 [FINAL_ANSWER]」。
3. **正式重试**：瞬态错误（`error_type="transient"`）用 LangGraph 内建 `RetryPolicy`/节点 `retry` 参数做**带退避重试**，替代/兜底现有手写分类；领域型错误（`param_error`/`not_found`）保留现有 `_format_error_feedback` 语义（它给 LLM 提供了可执行的恢复路径，不能丢）。
4. **循环上限可配置**：`max_iterations` 从硬编码 5 改为 env/config 注入（保留默认 5）。
5. **Plan-and-Execute（模式开关，后置）**：本阶段**只留接口位**（`state.mode`），不实现——通用 ReAct 加约束先做稳，复杂查询的 Plan 模式放 Phase 7 定制。

**DoD**：构造一个死循环式查询（反复调工具无进展），agent 在 token 预算或超时下**必然**终止并给出兜底回答，不再无限循环；瞬态失败重试有上限且带退避。

**依赖**：Phase 0（观测埋点验证约束生效）。

---

### Phase 2 — 规则安全层（横切拦截）

**现状**：完全缺失。输入直接进 prompt，输出直接回前端，工具无权限门。

**改动**（新增 `agent/safety.py`，纯代码，无新依赖）：
1. **输入过滤**：`understand_node` 之前——长度上限、基础 prompt-injection 启发式（`忽略以上指令`/`ignore previous instructions` 等关键词 + 编码绕过检测），命中则标记 `safety_flags` 并降级到 `clarify`/拒绝。
2. **输出审核**：`synthesize_node`/`chat_node` 之后——正则脱敏 PII（邮箱、手机号、身份证号、银行卡）；剥离工具内部标记；可选 LLM 审核器（开关，默认关——本阶段先正则）。
3. **权限矩阵**：新增 `agent/permissions.py`——读取 [providers/](../agent/providers/) 已存在的 `destructiveHint`/`readOnlyHint` 注解，构建 `role → 允许工具集` 映射；当前单用户先配一个默认 role（env `AGENT_USER_ROLE`），destructive 工具默认需审批（接 Phase 6 HITL）。
4. **合规拦截**：留一个 `policy_check(text) -> (allowed, reason)` 钩子，默认放行；实现一处可插拔（PII 命中即触发）。安全链做成**过滤器链**（`[input_filter, output_filter, policy_check]`），后续加规则不散落。

**DoD**：注入式输入被拦截并进入 clarify；含手机号/邮箱的回答返回前已脱敏；未授权角色无法触发 `process_paper`/`download_paper`。

**依赖**：Phase 1（拦截发生在治理边界内）。权限矩阵与 Phase 6 HITL 共用 destructive 注解。

---

### Phase 3 — 工具层加固（调度器）

**现状**：`CompositeToolProvider.call_tool` 直接路由到 provider，无统一超时/重试/审计/限流；参数校验依赖 LangChain `StructuredTool`(Pydantic)；MCP provider 内部自带一次重连，但 builtin/skill 无。

**改动**（新增 `agent/dispatcher.py`，不改 Provider 接口）：
1. 统一 `ToolDispatcher.call(name, args, ctx)`：
   - **参数校验**：按 `ToolDef.parameters` 用 Pydantic 动态校验（复用 [tools.py](../agent/tools.py) 的 `_to_langchain_tool` 逻辑），失败返回结构化 `ToolError`。
   - **超时**：`asyncio.wait_for`，按工具配置默认值（读 `annotations` 或统一 60s，`process_paper` 保留 300s）。
   - **重试**：瞬态错误退避重试（上限 2），幂等工具(`idempotentHint=True`)才重试；非幂等工具失败即返回。
   - **审计**：每次调用打结构化日志（tool、参数摘要、耗时、结果 ok/fail）→ 接入 Phase 0。
2. **结构化错误类型**：新增 `ToolError` 异常/返回值约定，替代现在的「错误用 JSON 字符串塞进 tool 结果」——保留 `{"ok": false, "error_type": ...}` 兼容格式（LLM 已依赖），但在 dispatcher 层额外暴露类型化错误给观测与重试判定。
3. `tools.py` 的 `_to_langchain_tool` 改为经 dispatcher 调用，而非直接 `call_fn`。

**DoD**：任意工具超时/参数非法都有统一、可审计、可重试的行为；观测里能看到每次工具调用的耗时与结果；现有 LLM 错误恢复语义（`param_error` → 换参数）不回归。

**依赖**：Phase 0（审计日志）、Phase 2（destructive 注解用于审批前置）。

---

### Phase 4 — 分层记忆补全（长期记忆 + 画像）

**现状**：[memory.py](../agent/memory.py) 已实现工作/短期/画像雏形；缺**长期记忆**（向量检索的事实库）与**画像权限/历史**。

**改动**：
1. **长期记忆（向量）**：复用现有向量库（Chroma/Qdrant，见 [vector_store.py](../indexer/vector_store.py)）新建一个 `memory` collection，存「对话提取的事实」(实体 → 事实断言)，每轮结束时异步写入（复用 `_estimate_tokens` 截断），轮开始时按语义相似度召回 top-k 注入 `context_snapshot`。**不做图谱**（图谱归 Phase 5）。
2. **用户画像扩展**：[memory.py](../agent/memory.py) 的 `profile.json` 增加 `role`(权限，接 Phase 2)、`interaction_history`(轻量统计：话题/论文/语言偏好)、`learned_preferences`(从 summary 增量提炼)。`save_profile` 保持现有读缓存策略。
3. **记忆写入触发**：`synthesize_node` 末尾提取本轮事实 → 写长期记忆（失败降级为跳过，不阻塞）。

**DoD**：跨会话问「我之前问过哪些论文的方法」能召回；画像在跨会话下持续累积且能被快照注入。

**依赖**：Phase 0（埋点）、Phase 3（复用向量检索路径）。此阶段是「通用层」与「定制层」的分界——长期记忆骨架通用，事实内容领域相关。

---

### Phase 5 — 知识管理补全（知识图谱 + 生命周期收口）

**现状**：RAG 管线、索引生命周期（[dedup_manager.py](../indexer/dedup_manager.py) 的增量同步）、质量评估（[retrieval_orchestrator/](../retrieval_orchestrator/)）已齐；**知识图谱缺失**。此阶段开始进入「定制层」。

**改动**：
1. **轻量图谱**：新增 `agent/knowledge_graph.py`，从 `rag_chunks.json` / `paper_registry.json` 抽取三元组（`paper — cites — paper`、`paper — uses — method`、`paper — reports — metric`），存 SQLite（`kg.db`，与 `checkpoints.db` 同根目录）。**不上图数据库**。
2. **图谱查询**：提供 `entity → 相关论文/方法` 的查询接口，供 agent 新增一个 `graph_query` 工具（或并入 `search_papers` 的补充信号）。
3. **生命周期收口**：核对 [indexer/pipeline.py](../indexer/pipeline.py) 的增量同步 + 孤儿清理已覆盖「新增/更新/删除」三态；补一份 README 说明索引生命周期状态机。
4. 同步更新 [agent/README.md](../agent/README.md) 与 [web/api/README.md](../web/api/README.md)（新增工具/端点）。

**DoD**：能回答「哪些论文用了 <某方法>」「这篇论文引用了谁」；索引删除论文后无残留 chunk。

**依赖**：Phase 0–4 稳定。图谱抽取规则遵循 CLAUDE.md 的 Prompt 设计原则（若走 LLM 抽取，输出结构化、零状态、模型无关）。

---

### Phase 6 — 人机协作（HITL）

**现状**：SSE 流式 + clarify 澄清 + 错误分类已齐；缺**审批**、**确认请求**、**结构化进度**。

**改动**：
1. **审批（HITL）**：destructive 工具（`destructiveHint=True`，即 `process_paper`/`download_paper`）触发前，用 LangGraph `interrupt()` 暂停，经 SSE 发 `{"type":"approval_request","tool":...,"args":...}` 事件；前端确认后再续跑。**审批态持久化**到 checkpointer，断线可恢复。
2. **确认请求**：费用高/耗时长的操作（下载大 PDF、全量重建索引）复用同一条审批通道，加 `reason` 字段。
3. **结构化进度**：SSE 已发 token/status 事件，补充 `progress` 事件（当前处于哪个节点、第几次工具调用/上限）——数据来自 Phase 0 的观测计数器。
4. **错误澄清增强**：`clarify_node` 在提问时附带可选答案（复用 `resolution.py` 的候选匹配），供前端渲染为按钮。

**DoD**：未确认时 destructive 工具不执行；审批请求/确认/拒绝全链路走 SSE 可见；断线重连后审批态不丢。

**依赖**：Phase 2（权限矩阵）、Phase 3（dispatcher 暴露 destructive 注解）、Phase 0（进度数据源）。

---

### Phase 7 — 定制化：Plan-and-Execute 模式（可选，后置）

在 Phase 1 预留的 `state.mode` 接口上实现复杂查询的「先规划后执行」模式：understand 后插入 `plan_node`（拆子任务），逐子任务复用 search 子图。**仅当真实出现多跳/多论文对比需求时再做**，否则保持 ReAct 单模式。

---

## 3. 执行顺序与依赖图

```
Phase 0 观测 ──► Phase 1 治理 ──► Phase 2 安全 ──► Phase 3 工具加固
      │                                        │              │
      └───────────────► Phase 4 记忆 ◄─────────┘              │
                               │                              │
                               ▼                              ▼
                        Phase 5 知识管理              Phase 6 HITL
                                                      (依赖 2+3+0)
```

- Phase 0、1 是**硬前置**，先做。
- Phase 2、3 可并行（安全层不依赖工具加固，但工具加固的审批前置依赖安全层注解，故 2 在 3 前或同时）。
- Phase 4–7 依赖通用层稳定后启动。

## 4. 通用层 vs 定制层边界

| 层 | 通用（先做厚） | 定制（后做） |
|----|--------------|-------------|
| Phase 0–3 | 观测、治理、安全、工具调度 | — |
| Phase 4 | 长期记忆骨架、画像 schema | 论文领域事实内容 |
| Phase 5 | — | 论文知识图谱、索引生命周期 |
| Phase 6 | HITL 审批机制、SSE 通道 | 具体审批规则 |
| Phase 7 | Plan 模式引擎 | 论文对比/综述类任务 |

## 5. 每个 Phase 的完成定义（汇总）

- **Phase 0**：`trace_id` 可还原整轮执行；`/api/agent/metrics` 有数据；无裸 `print()`。
- **Phase 1**：死循环必终止；超时/预算/重试三约束生效且可配。
- **Phase 2**：注入拦截 + PII 脱敏 + 权限门 + 可插拔过滤链。
- **Phase 3**：工具调用统一超时/重试/审计/参数校验，现有错误语义不回归。
- **Phase 4**：跨会话记忆可召回；画像累积。
- **Phase 5**：图谱可查询论文/方法/引用关系；索引三态收口。
- **Phase 6**：destructive 工具需审批，全链路 SSE 可见，断线恢复。

## 6. 风险与回滚

- **LangGraph 版本兼容**：`interrupt()`（HITL）与 `RetryPolicy` 需确认 [requirements.txt](../requirements.txt) 里 `langgraph>=0.2.0` 实际版本支持，Phase 1/6 前先验证。
- **回归风险**：现有错误分类（`_classify_tool_error`）与 citation 机制被 LLM 依赖，任何改造必须保留 `{"ok": false, "error_type": ...}` 与 `[CITE:N]` 兼容格式——已在各 Phase 注明「不回归」。
- **每阶段独立提交**：单阶段可回滚，不牵连后续。
