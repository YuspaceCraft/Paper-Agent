# Agent 平台架构设计（LangGraph-first）

> 状态：Active（P0–P5 功能项已闭环；仅保留按收益增量迁移的目录整理决策）  
> 决策：以 LangGraph 作为运行时编排内核，以 LangSmith 作为在线链路的唯一观测入口；保留本地评测产物作为可复现的离线事实源。  
> 范围：科研文献 Agent、创作、实验/编码三个现有域；不推倒已有检索、PDF、Web 和 MCP 能力。

## 1. 目标、原则与非目标

### 目标

1. **文档先行**：任何新能力先提交 ADR/接口契约/测试样例，再改运行时代码；实现与文档同一变更提交。
2. **可追溯**：一次请求可通过 `trace_id`、`thread_id`、`run_id`、`config_revision`、`prompt_version`、`tool_version` 定位输入、决策、调用、产出与评测结果。
3. **可复用**：跨 paper / creation / coding 域复用运行骨架、工具契约、权限、上下文、记忆、错误和评测组件；领域模块只提供节点和工具声明。
4. **可治理**：所有副作用都经过策略检查、审批、幂等和审计；所有配置、Prompt、工具与评测集都可版本化和回滚。
5. **库优先**：优先采用 LangGraph 的 `StateGraph`、checkpointer、`interrupt()`、`RetryPolicy`、`ToolNode` 与 LangSmith 原生追踪/feedback/dataset/evaluation；不自建工作流引擎、span 树或通用重试框架。

### 非目标

- 不是立即把现有目录整体迁移或替换向量库。
- 不让 LangSmith 承担业务状态、长久记忆或敏感原文的唯一存储。
- 不以“LLM 判断”替代确定性权限、配额、版本和错误策略。

## 2. 当前基线与核心问题

已有可复用基础：`agent/graph.py` 的 LangGraph 主图与 SQLite checkpoint、`dispatcher.py` 的工具统一入口、`tool_contract.py` 的结果信封、`memory.py`/`context.py`、`config_store.py`、`evaluation/` 与已有 LangSmith 启动配置。当前主图已能区分 react 和 plan，且有受限 subagent。

需要收敛的点：

| 问题 | 风险 | 本设计的收敛方式 |
|---|---|---|
| LangSmith 与本地 trace_store 都在承担“链路事实源” | 双写、口径分叉 | 在线运行树只认 LangSmith；本地只保存评测归档、badcase 和脱敏审计摘要 |
| 节点内 LLM/工具调用未保证携带父 `RunnableConfig` | LangSmith 出现游离根 run | `ExecutionContext.runnable_config` 成为所有 node / model / tool 必传依赖 |
| 配置分布在 env、YAML、JSON、代码常量 | 无法复现一次执行 | `ConfigurationSnapshot` 在 turn 开始冻结，并写入 trace metadata |
| 错误分类以字符串散落 | 重试、审批、指标和用户提示不一致 | 统一 `AgentError` 分类、可重试性、恢复动作和安全级别 |
| 记忆、对话、检索结果混入同一 prompt | token 失控、污染和不可解释 | Context Pack 分层、配额化、可审计的选择与压缩策略 |

> 上表各行的收敛进度逐项记录在 §13：config 透传（无游离 run）与工具单入口、Context Pack 与记忆策略已落地；
> 版本化部分落地（canary/A-B 未做）；评测门禁沿用 `evaluation/` 既有能力 + run 树归档。

## 3. 三层总体架构

```text
治理与演进层
  ADR / 契约 / Prompt与配置版本 / 工具目录与权限策略 / LangSmith评测与反馈 / 发布门禁
                                      │ 发布不可变快照
                                      ▼
运行时执行层
  API → ExecutionContext → LangGraph Runtime → policy → nodes/subgraphs → Tool Gateway
                         │                    │                │
                         │                    │                ├─ Context & Memory
                         │                    │                ├─ Error & Recovery
                         │                    │                └─ LangSmith Trace/Feedback
                                      ▼
基础设施层
  LLM/Embedding | Chroma/Qdrant | SQLite/Postgres/Redis | LangSmith | MCP/HTTP | 文件/PDF | 队列/对象存储
```

### 3.1 治理与演进层

- **文档与 ADR**：`docs/adr/NNNN-*.md`。每项跨模块决策说明背景、选择、替代方案、兼容性、回滚和验收指标。
- **版本中心**：Prompt、工具 schema、运行配置、模型路由、评测数据集独立版本；一次运行绑定不可变 `ConfigurationSnapshot`。
- **质量门禁**：合并前必须通过工具单测、图路由测试、契约兼容测试、评测基线比较、LangSmith trace 完整性检查；性能/成本/正确率阈值不达标不得推广。
- **变更策略**：`draft → canary → active → deprecated → retired`。Prompt/工具/策略的 canary 用 `experiment_id` 与 hash 分流，不允许覆盖活动版本。

### 3.2 运行时执行层

运行态固定为：`bootstrap → guard_input → assemble_context → route → execute graph/subgraph → verify → persist_memory → respond`。业务节点不能绕过以下横切服务：

- `ExecutionContext`：请求元数据、身份、冻结配置、LangGraph `RunnableConfig`、预算、trace 相关信息。
- `ToolGateway`：工具发现、schema 验证、权限、限流/超时、幂等、重试、结果信封、审计与埋点的唯一入口。
- `PolicyEngine`：输入/输出脱敏、角色权限、审批、资源配额、模型/工具允许列表。
- `ContextManager` 与 `MemoryManager`：分别负责“本回合可用信息包”和“跨回合事实/偏好/摘要”。
- `ErrorCenter`：标准错误分类、恢复策略、用户可见错误和指标标签。

### 3.3 基础设施层

| 能力 | 当前/建议实现 | 责任边界 |
|---|---|---|
| 工作流与暂停恢复 | LangGraph `StateGraph` + checkpointer | 图状态、`thread_id`、HITL interrupt、恢复 |
| 在线追踪 | LangSmith | run tree、token、耗时、metadata、feedback、线上排障 |
| 状态与审计 | 当前 SQLite；生产建议 Postgres | checkpoint、任务、审批、配置快照、脱敏审计索引 |
| 热态/锁/限流 | Redis | 幂等键、速率限制、短期会话缓存、分布式锁 |
| 文档与向量 | 保持 Chroma/Qdrant adapter | 检索、chunk 版本、embedding 版本、索引生命周期 |
| 外部工具 | MCP/HTTP adapters | 连接、认证、超时；不得泄漏到领域节点 |
| 归档 | JSONL/对象存储 | 评测输入输出、LangSmith run 导出、可复现报告 |

## 4. 核心运行骨架

### 4.1 LangGraph 图边界

保留 `agent/graph.py` 的主图，逐步将节点改为调用领域无关服务。建议主图：

```text
START
 → bootstrap (冻结配置、建立 ExecutionContext、根 trace metadata)
 → guard_input (验证/脱敏/预算/权限预检)
 → context (Context Pack)
 → understand
 → route
 ├─ react  → react_subgraph (agent ↔ ToolNode) ─┐
 ├─ plan   → planner → executor_subgraph → verify ┤
 ├─ chat/task/clarify                               ┤
 └─ approval interrupt → resume ────────────────────┘
 → persist (记忆候选、审计摘要、评测事件)
 → respond → END
```

- `react_subgraph`、`executor_subgraph`、受限 subagent 均使用相同 `ToolGateway`，不能直接调 provider。
- 长任务必须由 `task_id` 表达异步状态；图内只等待用户体验可接受的短步骤。
- 有副作用工具在调用前由 `interrupt()` 进入审批点；恢复时以 checkpoint 中的 idempotency key 继续，不重新执行已成功操作。
- 对瞬态错误使用 LangGraph `RetryPolicy`（指数退避、次数上限）；领域错误只转为可执行反馈，不盲目重试。

### 4.2 必备运行时模型

以下类型放入新的 `agent/core/contracts.py`，以 Pydantic v2/`Enum` 定义并生成 JSON Schema；不要在节点间传未约束的任意 dict。

```python
class ErrorType(str, Enum):
    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    POLICY = "policy"
    TOOL = "tool"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_RATE_LIMITED = "tool_rate_limited"
    MODEL = "model"
    AGENT_RUNTIME = "agent_runtime"
    GRAPH_TIMEOUT = "graph_timeout"
    CONTEXT_BUDGET = "context_budget"
    CHECKPOINT = "checkpoint"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"

class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    ADMIN = "admin"
    SECRETS = "secrets"

class PromptType(str, Enum):
    SYSTEM = "system"
    ROUTER = "router"
    PLANNER = "planner"
    EXECUTOR = "executor"
    SYNTHESIZER = "synthesizer"
    JUDGE = "judge"
    SUMMARY = "summary"
    SAFETY = "safety"

class AgentError(BaseModel):
    error_type: ErrorType
    code: str                 # 稳定机器码，如 TOOL_TIMEOUT
    message: str              # 脱敏后的内部说明
    user_message: str         # 面向用户的可操作提示
    retryable: bool
    retry_after_seconds: float | None = None
    recovery_action: str | None = None
    cause_ref: str | None = None
    tool_name: str | None = None

class ExecutionContext(BaseModel):
    request_id: str
    trace_id: str
    thread_id: str
    actor_id: str
    roles: set[str]
    permissions: set[Permission]
    config: "ConfigurationSnapshot"
    prompt_bindings: dict[PromptType, str]  # prompt_id@version
    budget: "Budget"
    # runnable_config 是运行时对象，不序列化到 checkpoint；通过工厂恢复
```

补充 `ToolSpec`（`name/version/description/input_schema/output_schema/permissions/side_effect/idempotency_scope/timeout/retry_policy/owner/tags`）、`PromptSpec`（`id/version/type/template/schema/variables/locale/status/evaluation_suite`）和 `ConfigurationSnapshot`（`revision/hash/sources/models/tool_allowlist/feature_flags/created_at`）。`AgentState` 仅保存可 checkpoint 的业务状态与上述对象的 ID/hash，绝不保存 API key、原始密钥或 runnable 对象。

## 5. 工具中心

### 5.1 唯一入口与调用顺序

`ToolGateway.invoke(spec, args, ctx)` 固定执行：

1. 检查工具版本、输入 schema 和调用方声明权限；
2. 根据 `side_effect` 和角色判定 allow / deny / approval；
3. 生成 `idempotency_key = hash(thread_id, tool@version, canonical_args, intent)`；
4. 施加并发、配额、超时、熔断和 LangGraph retry；
5. 调用 adapter，并统一为已有 `tool_contract.ok/err` 信封；
6. 记录脱敏参数摘要、耗时、错误类型、结果引用及 LangSmith metadata；
7. 将结果、错误和 retry 决策返回图节点。

### 5.2 分类和权限

| 分类 | 示例 | 默认权限 | 审批 | 幂等策略 |
|---|---|---|---|---|
| read | 检索、读文件、读论文 | READ | 否 | 结果短缓存 |
| network_read | arXiv、HTTP fetch | NETWORK + READ | 按域策略 | 请求去重 |
| write | 写章节、上传、入库 | WRITE | 是 | 业务幂等键 |
| execute | 跑实验、shell、委托 coding agent | EXECUTE | 是 | task_id + 仅一次提交 |
| admin | 配置发布、索引清理 | ADMIN | 强制 | 不自动重试 |

工具 registry 是声明式数据，不把工具名称判断散布在 prompt/node 内。现有 `providers/`、`tools.py` 和 `dispatcher.py` 可保留为 adapter 层，逐步使 `ToolDispatcher.call` 委托 `ToolGateway`。

## 6. Prompt、上下文与记忆

### 6.1 Prompt 配置中心

- 源文件：`prompts/<domain>/<prompt_id>/<version>.yaml`；模板与输入/输出 schema 一起提交。
- 发布：仅 `active` 版本可被解析；运行时绑定的 `prompt_id@version + checksum` 写入 LangSmith metadata 和离线报告。
- 变量：按 schema 校验，敏感字段先脱敏；禁止 prompt 自行拼接未知上下文。
- 评测：每个 active Prompt 必须关联最小回归集；升级以 A/B 或 canary 结果决定，不按主观输出覆盖。

### 6.2 Context Pack

`ContextManager.build(ctx, state)` 返回带来源、优先级、token 数和保留理由的 `ContextPack`：

| 区域 | 内容 | 默认预算 | 处理 |
|---|---|---:|---|
| invariant | 系统策略、权限、输出契约 | 固定上限 | 不截断，只能版本升级 |
| task | 用户当前目标、计划、审批状态 | 15% | 必保留 |
| conversation | 最近对话与压缩摘要 | 25% | 成对保留 tool call/result |
| retrieved | 检索 chunk/引用 | 45% | MMR/rerank、去重、按来源截断 |
| memory | 偏好、长期事实、工作区状态 | 10% | 置信度/TTL 过滤 |
| reserve | 工具结果与模型输出缓冲 | 5% | 不提前占用 |

预算按实际模型 context window 和 `max_output_tokens` 动态计算，不用单一字符估算决定安全边界。每次截断/压缩都写 `context_decision` 到 trace，供解释与评测。

### 6.3 记忆管理

分为四类并显式生命周期：会话工作记忆（LangGraph checkpoint）、对话摘要、用户画像、领域事实/任务记忆。每条长期记忆具备 `memory_id/type/content/source_ref/confidence/created_at/expires_at/consent/revision`；写入由 `MemoryPolicy` 判定，用户可查看、删除和禁用。不要将“检索到的全文”和“用户偏好”一起写入 profile.json。

现有 `memory.py` 的 buffer/summary/profile 是该模型的起点；`context.py` 中实验/文档的会话工作区元数据应迁入会话工作记忆，而非混入 prompt 字符串。

## 7. 异常容错中心

| 错误类别 | 例子 | 策略 | 用户面行为 | 指标 |
|---|---|---|---|---|
| validation / policy / authorization | 参数非法、无写权限 | 不重试，给修复路径 | 明确说明缺什么 | 拒绝率、误拒绝样本 |
| transient / rate_limited | 429、暂时网络失败 | 有界指数退避 | 可告知重试中 | 恢复率、重试次数 |
| tool_timeout / unavailable | MCP 挂死、后端不可达 | 熔断、fallback、停止重复调用 | 降级回答与任务状态 | P95、可用性 |
| model | 模型限流、结构解析失败 | 切备模型或安全模板；不伪造结果 | 说明能力受限 | 模型失败率 |
| graph_timeout / checkpoint | 全轮超时、持久化失败 | 安全终止；可从 checkpoint 恢复 | 返回 trace/task 参考 | 超时率、恢复率 |
| unknown | 未分类异常 | 记录 cause_ref，禁止自动副作用重试 | 通用失败提示 | 未分类率 |

错误的 `code` 是测试、告警和 UX 的稳定接口；异常堆栈只进受控审计/trace，不进模型上下文或前端。

## 8. LangSmith 可观测与全链路追溯

### 8.1 追踪契约

- 根 graph run 的 `run_id` 与本地 `trace_id` 一一对应；`thread_id` 映射为 LangSmith session/thread metadata。
- 所有 node、`model.ainvoke/astream`、`ToolGateway.invoke`、subgraph 和 subagent **必须**传入同一父 `RunnableConfig` 或从其派生的 child config，禁止裸调模型/工具。
- metadata 最少包括：`trace_id/thread_id/request_id/actor_role/domain/graph_version/config_revision/config_hash/prompt_versions/tool_versions/model_route/eval_dataset_id`。
- 输入输出先经过隐私分类；默认上传脱敏摘要、hash 和引用，只有允许的研发环境才采样完整内容。
- 在线排障查看 LangSmith；每次评测或重要发布将所需 run 树导出为 `eval_output/runs/<run_id>/langsmith_runs.jsonl`，使评测结果可重现而不依赖保留期。

### 8.2 现有迁移重点

`agent/graph.py` 已在根调用设置 `run_id`；P0 的第一项不是新建 trace 系统，而是让 `nodes._stream_llm`、`evaluation.trace_wrap.traced_ainvoke`、`dispatcher.ToolDispatcher.call` 及 subagent 路径继承该 config。完成标准：`evaluation.verify_smith` 对一个 react 和一个 plan 案例均断言只有一个根 run，LLM/工具均为该根的子孙 run。

## 9. 评测体系与质量门禁

### 9.1 指标口径

| 面向 | 指标 | 口径 |
|---|---|---|
| 工具 | 单元测试通过率、schema 合规率、调用成功率、P50/P95、重试后恢复率、副作用重复率 | 分母为实际尝试；将 policy deny 与系统失败分开 |
| 任务 | task completion rate、partial rate、人工验收率、恢复成功率、成本/任务 | 完成需达到任务验收条件，不能只以“有非空回答”判定 |
| 检索 | Recall@K、Precision@K、MRR、NDCG@K、覆盖率、引用正确率 | 用版本化 qrels，按 query 和 corpus/index/embedding 版本分组 |
| 回答 | faithfulness、answer relevance、citation correctness、结构化输出有效率 | 自动 judge + 抽样人工双轨，记录 judge prompt/model 版本 |
| 运行 | trace completeness、P95 latency、token/cost、timeout/error rate | trace completeness = 有根且关键 node/LLM/tool 父子关系完整 |

### 9.2 分层测试

1. **单元测试**：每个工具 adapter、错误映射、权限矩阵、prompt 模板 schema、context budget。
2. **契约测试**：所有工具必须产生统一 envelope；`ToolSpec` 与实际 JSON schema 一致；旧版本输入/输出兼容。
3. **图测试**：固定状态下的路由、审批 interrupt/resume、retry、checkpoint 恢复、幂等行为。
4. **集成测试**：mock LLM/MCP 的端到端 trace；不依赖真实网络即可验证全链路树。
5. **离线评测**：沿用 `evaluation/` 的 dataset/metrics/report，评测集、qrels、模型、Prompt、索引版本随报告固化。
6. **线上监控**：LangSmith feedback、采样人工评价、坏例回流；生产数据入评测集前需脱敏与审核。

建议初始发布阈值：trace completeness 100%，工具 schema 合规 100%，副作用重复 0；核心检索指标相对基线不下降超过 3%，任务完成率不下降超过 2%，P95 与单位任务成本有明确预算。阈值应按评测集规模标注置信区间，避免小样本误阻塞。

## 10. 推荐目录与现有模块映射

```text
agent/
  core/          # contracts, execution_context, errors, policy, budgets
  runtime/       # graph factory, bootstrap, graph nodes/subgraphs
  tools/         # registry, gateway, adapters (providers/MCP/builtin)
  prompts/       # registry + loader；prompt 文件可放仓库根 prompts/
  context/       # context pack, token budget, retrieval packing
  memory/        # session/profile/long-term policy and stores
  config/        # snapshot, resolver, version registry
  observability/ # LangSmith integration, audit redaction, metrics facade
  domains/       # paper, creation, coding：仅领域节点/ToolSpec/Prompt binding
evaluation/      # datasets, qrels, runners, metrics, reports, LangSmith export
docs/adr/        # 架构决策记录
docs/contracts/  # 工具、API、状态机、错误码的可读契约
```

这是逻辑边界，不要求一次移动文件。迁移顺序：`tool_contract.py` → `core/contracts`，`dispatcher.py` → `tools/gateway`，`config_store.py + config.yaml` → `config`，`memory.py/context.py` → 各自目录，`observability.py + evaluation.trace_wrap` → `observability`。旧 import 以兼容 facade 保留一个发布周期。

## 11. 分阶段实施与验收

| 阶段 | 产出 | 关键验收 |
|---|---|---|
| P0：定基线（1 周） | ADR、contracts、工具清单、错误码表、评测基线、依赖版本锁定 | 当前 react/plan 各有可复现 trace 与基线报告 |
| P1：链路闭合（1 周） | `ExecutionContext`、config 透传、LangSmith metadata/redaction、trace export | 无游离 LLM/tool run；`verify_smith` 纳入 CI |
| P2：工具治理（1–2 周） | ToolSpec/Registry/Gateway、权限矩阵、统一 timeout/retry/idempotency | 所有工具走唯一入口；副作用有审批和审计 |
| P3：上下文与记忆（1–2 周） | ContextPack、动态预算、分层 memory schema 与用户控制 | 每轮有预算决策；长会话不超窗、不丢 tool pair |
| P4：版本与发布（1 周） | Prompt/Config registry、snapshot、canary/rollback | 任意 run 可复现模型、prompt、工具、配置组合 |
| P5：质量闭环（持续） | CI 测试矩阵、离线评测、线上反馈回流、质量看板 | 指标门禁生效，badcase 能进入下一轮回归集 |

每一阶段均采用“先文档/契约/测试，后实现；先兼容 facade，后删除旧路径”。P1 和 P2 优先级最高：没有完整 trace 和工具单入口，后续记忆、计划和多 Agent 的问题仍难定位。

## 12. 首批 ADR 清单

1. `0001-langgraph-runtime-and-state-boundary`：图、state、checkpointer 与异步任务边界。
2. `0002-langsmith-tracing-and-data-redaction`：run 层级、metadata、采样与敏感数据策略。
3. `0003-tool-registry-permission-and-idempotency`：ToolSpec、权限、审批、重试和审计。
4. `0004-prompt-and-configuration-versioning`：不可变快照、发布状态与回滚。
5. `0005-context-and-memory-lifecycle`：配额、来源、TTL、同意与删除。
6. `0006-evaluation-metrics-and-release-gates`：指标定义、数据集版本、统计与质量门禁。

在这些 ADR 未批准前，不新增又一个 agent loop、工具调度器、trace store 或 prompt loader；新增能力必须复用本设计的中心组件。

以上 6 份 ADR 已按本清单落地（`docs/adr/`），状态均为 Accepted；
可读契约在 `docs/contracts/`（工具信封、错误码、状态机与 API）。

## 13. 实施状态（2026-09-16）

| 设计项 | 落地位置 | 状态 |
|---|---|---|
| §4.2 `ErrorType/Permission/PromptType/AgentError/ToolSpec` | `agent/core/contracts.py` | ✅ |
| §4.2 `ExecutionContext` + `Budget` + `PromptSpec` | `agent/core/contracts.py`、`agent/core/execution_context.py` | ✅ |
| §4.1 bootstrap 冻结配置（每轮一次） | `agent/core/configuration.py`、`agent/graph.py::prepare_turn` | ✅ |
| §4.1 turn 起点建立 ExecutionContext（`graph.run` 与 `/chat`、`/chat/stream` 同一契约） | `agent/graph.py`、`web/api/routers/agent.py` | ✅ |
| §8.1 join 键（root run id == trace_id）+ metadata 契约 | `agent/graph.py`、`agent/core/execution_context.py` | ✅ |
| §8.1 node/model/tool 透传父 `RunnableConfig` | `agent/nodes.py`、`agent/plan.py`、`agent/subagents.py`、`ExecutionContext.child_config` | ✅（新增调用点须用 `child_config`） |
| §8.1 run 树导出 `eval_output/runs/<run_id>/langsmith_runs.jsonl` | `agent/core/trace_export.py`、`evaluation/runner.py` | ✅（`EVAL_EXPORT_LANGSMITH=1` 开启） |
| §5 ToolSpec/Registry 声明式元数据 | `agent/core/tool_registry.py` | ✅ |
| §5 `ToolGateway.invoke` 七步（权限/审批/幂等/超时/熔断/重试/审计） | `agent/core/tool_gateway.py`、`agent/core/policy.py` | ✅ |
| §5 `ToolDispatcher.call` 委托唯一入口 | `agent/dispatcher.py`（保留 SSE + 评测事件） | ✅ |
| §5 图级审批 `interrupt()` / `Command(resume)` | `agent/core/approval.py`、`tool_gateway.py`、`graph.py::resume`、`web/api/routers/agent.py`、`MessageList.tsx` | ✅（API + SSE 事件 + 前端批准/拒绝卡片；无图上下文时仍 fail-closed） |
| §6.1 Prompt 版本绑定 `id@version#checksum` + 嵌套发布布局 | `agent/prompt_store.py`、`agent/core/prompt_registry.py` | ✅ |
| §6.1 canary/A-B 分流与回归集绑定 | `agent/prompt_store.py`、`prompt_registry.py`、`web/api/routers/config.py`、`PromptsPanel.tsx` | ✅（thread_id 稳定 hash 分流；prompt version/checksum/evaluation_suite 写入执行元数据；配置中心可调百分比） |
| §6.2 分区预算 Context Pack + `context_decision` | `agent/core/context_pack.py`、`agent/nodes.py::memory_node` | ✅（产出与记录） |
| §6.2 retrieved/memory 区注入节点 prompt | `agent/core/context_pack.py`、`agent/memory.py`、`agent/nodes.py::_render_agent_context` | ✅（Agent 每轮按最新 tool messages 重建 retrieved；profile 只由 memory zone 注入，避免重复） |
| §6.3 `MemoryRecord` schema + `MemoryPolicy`（置信度/TTL/consent） | `agent/core/memory_policy.py` | ✅ |
| §6.3 长期记忆写入路径 + 前端查看/删除/禁用 | `agent/core/memory_store.py`、`web/api/routers/memory.py`、`MemoryPanel.tsx`、`memory_node` 显式捕获 | ✅（typed store + policy；显式“记住/以后请…”写入；配置中心可查看/禁用/删除） |
| §7 错误分类与重试/熔断策略 | `agent/core/errors.py`、`agent/core/tool_gateway.py` | ✅ |
| §9 评测指标/报告/基线 | `evaluation/`（既有） | ✅ |
| §8/§9 在线 sink 退役 + 评测归档 | `evaluation/trace_store.py`、`evaluation/runner.py`、`agent/graph.py` | ✅（生产只走 LangSmith；`trace_store` 仅评测/live-debug；每批产出 `trace_events.jsonl`） |
| §9 评测过程可见（实时指标随执行更新） | `evaluation/live.py`（事件总线）+ `evaluation/runner.py`（逐条发布 + `eval_runs` 进度行）+ `web/api/routers/eval.py`（SSE `runs/{id}/stream`、`single/{id}/stream`）+ `EvalPanel.tsx`（实时进度卡） | ✅（2026-09-15） |
| §9 LangSmith feedback 回填 | `evaluation/feedback.py`、`web/api/routers/eval.py::create_feedback` | ✅（按 trace_id/run_id 写 feedback，供人工验收与回归采样） |
| §10 目录迁移（`core/` 之外） | 现有模块 + `agent/core/*` 兼容边界 | ✅（决策收敛：逻辑边界由契约/registry/gateway 固化；物理目录只在有明确收益时增量迁移，不做一次性搬家） |

回归证据：`agent/tests/test_tool_gateway.py`、`agent/tests/test_context_pack.py`、
`agent/tests/test_core_contracts.py`，以及既有 `test_dispatcher/test_plan/
test_supervisor/test_creation/test_coding/test_context`，评测侧
`evaluation/tests/{test_flow,test_runner,test_trace_store,test_live}.py` 自检全绿。

已知环境问题（与本设计无关）：`test_supervisor.py` 全部用例通过后进程在退出阶段
被 aiosqlite/MCP 的非 daemon 线程挂住（见 `TROUBLESHOOTING.md`）。
