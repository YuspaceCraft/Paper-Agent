# Agent 失败处理设计核查报告

- 审查日期：2026-09-20
- 审查范围：当前仓库 `agent/`、`web/`、`evaluation/`、`docs/` 中与请求接入、规划、执行、记忆、输出和终止相关的代码、契约与测试
- 审查方法：代码调用链核查、契约与文档对照、相关回归测试执行
- 执行环境：`conda run --no-capture-output -n demo python -m pytest ...`
- 回归结果：10 个相关测试文件，`133 passed`

## 1. 总体结论

当前项目在工具治理、幂等、审批、上下文预算和记忆 TTL 方面已经有较扎实的实现，尤其是 `ToolGateway`、`ContextManager`、`MemoryPolicy` 和 `OperationResult` 已经把许多原本散落的逻辑收敛成了稳定契约。

但若严格以本次列出的 32 个失败场景作为验收标准，当前设计尚未完整闭环：

> 下表及第 2 至第 7 节记录的是本轮修改前的基线审计结果，后续实施进度见第 13、14 节。

| 结论 | 数量 | 比例 |
|---|---:|---:|
| 完善 | 11 | 34.4% |
| 部分完善 | 15 | 46.9% |
| 缺失 | 6 | 18.8% |
| 合计 | 32 | 100% |

结论标记口径：

- **完善**：有明确实现路径，关键分支有代码或测试保障。
- **部分完善**：主链路存在局部保护，但缺少本次要求中的完整策略、降级或验证闭环。
- **缺失**：未发现专门实现，或仅依赖提示词而缺少确定性控制。

主要强项：

1. `ToolGateway.invoke` 统一处理工具 schema、权限、审批、幂等、超时、熔断、重试和脱敏审计。
2. 副作用工具默认 fail-closed，审批通过 LangGraph `interrupt()` 与 checkpoint 恢复。
3. Context Pack 有分区预算、来源记录、截断理由和 tool call/result 成对保留。
4. 长期记忆有 `source_ref`、`confidence`、TTL、consent、revision 和用户禁用/删除。
5. 计划执行有 DAG 调度、资源互斥、步骤上限和结果验证。

主要缺口：

1. 用户输入没有确定性的 Prompt Injection 检测或统一 `guard_input` 节点，主要依赖系统提示词中的 Trust Boundary。
2. 模型 API 没有备用 Key、备用模型端点或模型级熔断；Windows 桌面健康接口也不真实探测模型可用性。
3. 规划阶段缺少计划语义校验、成本估算、重复步骤检测和预算超限后的裁剪或询问。
4. 最终回答没有通用格式、质量、乱码和截断校验，现有验证主要覆盖 plan 模式。
5. 多源记忆只有来源字段，没有冲突判定、可信度仲裁或矛盾反馈机制。
6. 决策错误不会自动沉淀为可复用的长期记忆，只有评测 trace 和显式“记住”写入。

文档状态也需要校正：`docs/agent-platform-architecture.md` 将 P0-P5 标记为基本闭环，但实际主图没有 `guard_input`，React/chat 输出也没有进入统一 `verify`。`docs/contracts/state-and-api.md` 已明确承认 `guard_input` 尚未落地，两份文档的口径不够一致。

## 2. 执行前阶段

| 问题场景 | 结论 | 现有实现与证据 | 主要缺口 | 建议动作 |
|---|---|---|---|---|
| API 认证/连接失败，包含 401、404、5xx | 部分完善 | 本地 API 有 `DEMO_API_TOKEN`、loopback 限制和 CORS 白名单，见 `web/api/main.py`；模型使用 `ChatOpenAI(max_retries=3)`，见 `agent/nodes.py:62`；模型路由可冻结多个任务模型，见 `agent/core/model_router.py`。 | 401/404 不会通过普通 SDK retry 恢复；没有备用 Key、备用 Base URL、备用供应商、模型级熔断；`/api/agent/health` 只返回 `status="ready"`，不实际探测模型，见 `web/api/routers/agent.py:626`。 | 增加统一 `ModelGateway`，按 `AUTH_INVALID`、`NOT_FOUND`、`RATE_LIMIT`、`TRANSIENT_5XX` 分类；支持 Key/Endpoint 池、有界退避、熔断和模型级 readiness probe；增加故障注入测试。 |
| 输入包含恶意指令，Prompt Injection | 部分完善 | 所有核心 Prompt 都包含 `TRUST_BOUNDARY`，明确将用户文本、论文、网页和工具结果视为 DATA，见 `agent/prompt_contracts.py:8`；有 Prompt 契约测试。 | 没有确定性输入拦截器，没有 `guard_input` 图节点；`AgentChatRequest.query` 只有 `min_length=1`，没有最大长度或异常输入限制，见 `web/api/schemas.py:89`；仓库没有专门的 injection/jailbreak 回归集。 | 实现输入规范化与策略预检，检测“覆盖系统指令、索取密钥、绕过权限、诱导危险工具”等模式；对工具结果和检索文本加 provenance 标记；增加隔离测试和误杀测试。 |
| Token 预算不足 | 完善 | `Budget.usable_context_tokens()` 预留输出空间；`fit_messages()` 按预算保留最新消息和完整 tool pair；`ContextManager` 做分区预算；`agent_node` 预算耗尽时强制不带工具生成最终回答。见 `agent/core/contracts.py`、`agent/core/token_budget.py:158`、`agent/core/context_pack.py:202`、`agent/nodes.py:468`。 | `context_window_tokens` 仍来自配置/环境默认值，不是按实际模型能力表自动解析；预算是估算 token，不是调用后的真实 usage；没有稳定的 `CONTEXT_BUDGET` 用户错误码闭环。 | 建立模型上下文窗口注册表；把真实 usage 回写到预算统计；预算无法满足最小任务时返回 `CONTEXT_BUDGET`，并给出缩小范围的恢复动作。 |
| 用户意图模糊或歧义 | 完善 | `UnderstandResult.confidence` 低于 0.5 时路由到 `clarify`，见 `agent/nodes.py::route_intent`；`UNDERSTAND_SYSTEM` 明确规定缺少实体且无历史时进入 `needs_clarify`；`clarify_node` 生成针对性追问。 | 仅覆盖意图级歧义，工具参数歧义主要依赖错误信封的下一动作提示。 | 保留现有机制，并增加“同一问题连续两轮仍无法澄清”的退出策略与用户可操作选项。 |
| 任务超出能力范围 | 缺失 | `COMPLETION_CONTRACT` 要求缺少必要事实时询问用户，但项目没有能力清单、拒绝模板、转人工协议或能力边界路由。 | 无法确定性判断“当前 Agent 不支持什么”，通常只会由模型自然语言解释。 | 增加 `CapabilityRegistry` 和 `HANDOFF_REQUIRED`、`OUT_OF_SCOPE` 等错误/响应类型；为高风险和非支持域提供统一拒绝与转人工路径。 |

## 3. 规划阶段

| 问题场景 | 结论 | 现有实现与证据 | 主要缺口 | 建议动作 |
|---|---|---|---|---|
| 计划陷入死循环 | 部分完善 | `PlanResult` 和 `PlanStep` 是 Pydantic 模型；executor 在依赖环或悬空依赖导致 `not ready` 时停止而不是空转，见 `agent/plan.py:861`；父 Agent 有 `max_steps`，单计划步骤有 `plan_step_max_steps`。 | 没有重复步骤模式检测，没有循环依赖的显式错误类型，没有自动重置或重规划；环只会留下 pending，最后在 verify 中表现为未执行。 | 规划后执行图校验：重复 ID、重复 `(target,args)`、依赖成环、不可达步骤全部转成结构化计划错误；触发一次有界重规划，仍失败则终止并说明。 |
| 步骤逻辑倒置，例如先总结后搜索 | 缺失 | executor 尊重显式 `depends_on`，因此正确依赖不会乱序执行；`PlanStep` 也有 `depends_on` 字段。 | 没有语义级前置条件校验；如果模型漏写依赖，错误顺序仍会执行；没有专门的异常计划标记或自动重规划流程。 | 为 PlanStep 增加 `requires_evidence`、`produces_evidence` 或领域前置/后置条件；在 executor 前做静态校验，失败进入 replan。 |
| 计划预估成本超限 | 缺失 | 有 step 次数、单步工具轮次、整轮超时和 token guard；评测侧能统计历史 cost。 | 没有执行前成本估算，没有按计划裁剪非必要步骤，也没有“请求用户增加预算”的规划协议。 | 用历史 P50 成本、工具预计延迟和上下文大小生成 plan estimate；超限时按优先级裁剪，必须保留步骤超限则请求用户确认。 |
| 计划步骤缺失 | 部分完善 | `_ask_for_plan` 对空结果最多尝试 2 次；paper 域可回退为确定性只读计划；verify 能发现 pending/failed step。 | 反馈发生在执行后；没有在规划阶段比较“原始目标覆盖度”，也没有把缺失关键路径反馈给 LLM 要求补全。 | 增加独立 `plan_review`：逐项映射用户目标到计划步骤，缺失时做一次补全重规划，并记录覆盖矩阵。 |
| 多目标冲突，例如速度与质量 | 缺失 | Prompt 要求选择最小充分 scope，但没有显式速度/质量/成本权重，也没有运行时决策记录。 | 同一请求遇到低延迟、高质量、低成本和完整性冲突时，行为不可解释。 | 在执行上下文中加入 `optimization_profile`，如 `fast`、`balanced`、`quality`；计划节点按 profile 决定并行度、检索深度和验证强度，并写入 trace。 |

## 4. 执行阶段

| 问题场景 | 结论 | 现有实现与证据 | 主要缺口 | 建议动作 |
|---|---|---|---|---|
| 工具参数格式错误，JSON Parse Error | 部分完善 | 计划输出可从 Markdown fence 或前后文中提取 JSON；工具参数进入 gateway 前做 JSON Schema 校验；未知字段、类型错误和枚举错误会返回 `TOOL_ARGS_INVALID`，见 `agent/core/policy.py:198`。 | 没有通用 JSON 自动修复器；模型返回非法 JSON 时通常只能进入重试或错误反馈；修复策略不统一。 | 在模型结构输出边界增加“严格解析、有界修复、重新调用”三级流程；修复结果必须再次 schema 校验，不能直接采用。 |
| 网络或服务临时故障，Timeout、503 | 部分完善 | 工具侧支持有界指数退避、超时和熔断，见 `agent/core/tool_gateway.py:548`；模型 SDK 配置了 `max_retries=3`。 | 模型 5xx 没有独立熔断和备用端点；工具 503 的 Provider 返回是否被正确标为 `transient` 依赖各适配器；不存在模型 fallback 模型链。 | 统一 transport error 映射；工具和模型都支持备用 endpoint/model；熔断状态进入健康接口和 trace。 |
| 内容返回过长，截断或 Overflow | 完善 | `tool_contract.truncate_tool_result()` 保持 envelope 可解析并写 continuation；`artifact_store` 对超大结果做内容寻址和分页；`artifact_read` 可继续读取；fetch 工具存在 offset 续读。 | 非字符串结构化结果仍主要依赖 envelope preview，不是所有结果都可自动分页。 | 对常用结构化结果定义稳定 continuation 契约，并补充分页完整性测试。 |
| 执行陷入死循环 | 完善 | 工具循环有 `max_steps`；plan step 有独立轮次上限；整轮有 `turn_timeout_seconds`；重复工具参数在同轮有结果缓存；`after_agent` 到达上限后退出工具面。 | 上限触发后的用户提示和自动复盘仍较弱。 | 将终止原因写入结构化 `graph_timeout` 或 `MAX_STEPS_REACHED`，并在回答中说明未完成部分。 |
| 上下文窗口溢出 | 完善 | `fit_for_context()` 在每次 LLM 调用前执行；系统提示优先保留；最新消息优先；assistant tool call 与 ToolMessage 作为不可拆分单元；记录 `context_hard_budget`，见 `agent/core/token_budget.py`。 | 预算窗口不是按模型注册表动态获得；系统提示超限时会截断 invariant，虽有 metadata，但可能影响策略完整性。 | 增加模型能力表；当 invariant 无法完整保留时 fail-closed，而不是裁剪安全策略。 |
| 危险操作拦截，rm、sudo、转账 | 完善 | `ToolSpec.requires_approval()` 对 WRITE、EXECUTE、ADMIN 和 `side_effect` 默认要求审批；无图上下文时 fail-closed；审批使用 LangGraph interrupt；副作用有持久幂等和 outcome unknown 保护，见 `agent/core/policy.py`、`agent/core/tool_gateway.py`、`agent/core/idempotency.py`。 | `AGENT_TOOL_APPROVAL=0` 可以显式关闭审批，应仅允许测试或受信部署环境使用。 | 在生产启动时检测关闭审批的危险配置并拒绝启动或高亮告警。 |
| 工具返回“未找到”或空结果 | 完善 | 工具结果统一进入消息历史，模型能看到空结果或结构化 not_found；错误分类反馈明确要求重新解析身份、换工具或询问用户，见 `agent/nodes.py:978`。 | 空成功结果没有单独指标，也没有统一的 `EMPTY_RESULT` warning。 | 为可能返回空集合的工具增加 `warnings.EMPTY_RESULT`，便于验证和评测。 |
| 工具调用幻觉，调用不存在的工具 | 完善 | 模型绑定的是当前工具列表；父工具节点遇到未知工具返回 `unknown tool` 错误信封，见 `agent/nodes.py:271`；plan step 对 `tool is None` 也返回结构化失败。 | 返回中没有自动列出全部可用工具，模型主要依赖绑定的 schema 重新选择。 | 在 `TOOL_NOT_FOUND` envelope 中增加允许工具列表或 registry version，方便模型纠正。 |
| 工具误用，参数语义错误 | 完善 | gateway 在调用 adapter 前校验参数；错误反馈包含 `TOOL_ARGS_INVALID` 和修复动作；工具自身可返回 `available_papers`、`available_sections` 并触发一次确定性修正，见 `agent/plan.py::_retry_args_from_error`。 | 语义错误只能覆盖已声明 schema 和工具自定义候选集；隐式业务规则仍可能漏检。 | 为高风险工具增加领域级 precondition，不只依赖 JSON Schema。 |
| 目标漂移 | 部分完善 | Task zone 始终携带 Goal；系统 Prompt 有 Completion Contract 和 scope 保持约束；verify 会把最终结果与计划目标比较。 | 没有执行中的持续目标对比，也没有偏差阈值和强制纠偏节点；React 模式没有独立 goal verifier。 | 每个工具轮次后做轻量目标一致性检查；偏差超过阈值时注入“回到原始目标”反馈，仍偏离则终止并询问用户。 |

## 5. 记忆与上下文

| 问题场景 | 结论 | 现有实现与证据 | 主要缺口 | 建议动作 |
|---|---|---|---|---|
| 上下文数据过期，TTL 失效 | 完善 | `MemoryRecord.expires_at` 和 `MemoryPolicy.evaluate()` 在注入前检查 TTL；Context Pack 只注入通过策略的记录；过期只留下无内容审计条目。见 `agent/core/memory_policy.py:59`。 | 没有后台主动清理任务，过期记录默认只过滤不删除。 | 增加定期紧凑或归档任务，并保留审计计数。 |
| 多源数据定义冲突 | 缺失 | 记录有 `source_ref`，profile 和 typed memory 会合并进 memory zone。 | `MemoryRecord.render()` 不输出来源；没有语义层、可信来源排序、冲突分组或“同一字段多个值”的结构化表示；Context Pack 会把冲突内容并列注入。 | 引入规范字段或语义 key，冲突时按来源信任度、更新时间、consent 和 confidence 仲裁；无法仲裁则标注冲突并询问用户。 |
| 记忆中毒，记住了错误事实 | 部分完善 | 长期记忆只在显式“记住/以后请”时写入；有 confidence、TTL、consent、revision 和 provenance；检索全文不会写入 profile。 | 没有矛盾检测、反馈修订、来源可信度或 LLM 仲裁；相同语义但不同文本会生成不同 memory_id，可能并存。 | 增加 semantic key、冲突队列和修订关系；新来源与旧记忆矛盾时进入可审计的 resolve 流程。 |
| 关键上下文缺失 | 部分完善 | 低 confidence 会触发 clarify；工具错误会携带 `next` 动作；Evidence Contract 要求缺失时明确说明未知。 | 没有统一的信息缺口检测器，也不会主动查询知识库补全缺失实体；主要依赖路由器、Prompt 和工具结果。 | 增加 `ContextSufficiencyCheck`，输出缺失字段、可查询来源和是否必须询问用户；与 clarify/retrieval 路由联动。 |

## 6. 输出与验证

| 问题场景 | 结论 | 现有实现与证据 | 主要缺口 | 建议动作 |
|---|---|---|---|---|
| 输出格式校验失败，非 JSON/CSV | 部分完善 | 工具输出有 envelope 和可选 `output_schema`；计划输出用 Pydantic 校验；HTTP 响应用 FastAPI response model。 | 最终自然语言回答没有通用输出 schema；若用户要求 JSON/CSV，主要依赖 Prompt，缺少解析失败后的自动重试和修复闭环；Registry 当前也通常不为存量工具声明 output schema。 | 增加最终输出契约解析，按请求决定 `text`、`json`、`csv` 等；解析失败进入有界 repair；工具 registry 补齐 output schema。 |
| 包含有害或敏感内容 | 部分完善 | `sanitize_output()` 对邮箱、手机号、身份证和银行卡做规则脱敏；API 响应和 SSE token 都经过该过滤器，见 `agent/safety.py:79`、`web/api/routers/agent.py:210`。 | 没有内容安全分类、危险指令过滤、政策类别拦截或脱敏后的审计告警；流式跨 chunk 的 PII 可能漏过，代码注释已承认该风险。 | 增加统一输出安全策略链：PII、secret、policy、moderation；对流式输出做重叠窗口缓冲；命中高风险类别时停止流并返回稳定 code。 |
| 输出明显截断或乱码 | 部分完善 | 工具结果截断会标记 `continuation`、`truncated`、artifact；上下文预算会记录截断消息。 | 最终回答没有长度异常、句末完整性、编码替换字符、JSON 不闭合或流中断检测；不会自动重新生成。 | 增加 final-answer quality gate，检测乱码、未闭合结构和明显截断；失败时最多重生成一次，之后标记 `LOW_QUALITY_OUTPUT`。 |
| 自我验证失败，结果不符预期 | 部分完善 | Plan 模式有确定性统计和 LLM `_verify_goal`，能把 outstanding step 带入 synthesis；creation/coding 有领域验证。 | React、chat、task 路径没有等价的最终答案验证；verify 只报告，不自动修正或重执行。 | 将 verify 扩展到所有域；定义可重试差异类型，允许一次修正执行；仍失败则明确返回 partial。 |
| 任务过早终止 | 部分完善 | `_verify_summary` 会列出 failed/pending；synthesis 会把未完成步骤显式注入最终回答，见 `agent/plan.py:1870`、`agent/plan.py::_synthesize_plan`。 | 没有统一的“是否继续”交互；部分失败后不会主动提出继续、重试或转人工选项。 | 输出结构化任务状态，包含 `completed`、`outstanding`、`can_resume`；对可恢复任务提供继续/重试入口。 |

## 7. 终止与复盘

| 问题场景 | 结论 | 现有实现与证据 | 主要缺口 | 建议动作 |
|---|---|---|---|---|
| 达到硬性终止条件 | 完善 | 有 `max_steps`、`max_turns`、`plan_step_max_steps`、`turn_timeout_seconds`、token guard；副作用还有幂等和 unknown outcome 安全终止。见 `agent/core/contracts.py`、`agent/state.py`、`agent/graph.py`。 | 各终止原因尚未完全收敛到统一终止事件，部分场景只返回自然语言。 | 统一输出 `TerminationReason`、已完成项、未完成项和是否可恢复。 |
| 复盘发现决策错误 | 缺失 | 评测侧能保存 trace、badcase 和 feedback；用户显式说“记住”时可写长期记忆。 | 没有自动复盘节点，也不会把决策错误、失败策略和修复结果写入长期记忆；后续任务无法主动复用。 | 增加 terminated/postmortem 事件，生成结构化经验记录，经 policy 过滤和用户同意后写入 memory；区分事实记忆与策略经验。 |
| 置信度与质量不匹配 | 部分完善 | 路由器有 intent confidence，resolution 有匹配置信度，verify 有 satisfied/partial/failed/no_evidence。 | 最终回答没有校准后的 confidence，也没有“高置信但低质量”检测；这些信号不会共同决定是否重验或标注不确定性。 | 综合 confidence、verification、citation coverage 和输出质量生成 final confidence；阈值不一致时重新验证或明确标注不确定性。 |

## 8. 跨模块根因

### 8.1 输入策略没有进入主图

`agent/graph.py` 的路径是：

```text
START -> understand -> memory -> context -> route -> execute -> synthesize -> END
```

没有独立的 `guard_input` 或 `policy_input`。当前安全主要来自 Prompt 约束，属于概率性防护，不是用户要求的“系统直接拦截”。

### 8.2 模型调用不在统一错误中心

工具已经通过 `AgentError`、`ErrorType`、`OperationResult` 和 `ToolGateway` 收敛。模型调用仍直接通过 `ChatOpenAI`，因此以下能力不完整：

- 模型错误分类
- 模型级熔断
- Key 轮换
- Endpoint 故障转移
- 备用模型策略
- 可解释的 readiness 检查

### 8.3 最终输出治理弱于工具输出治理

工具输出有 envelope、schema、artifact、continuation、审计和幂等。最终回答只有 PII 脱敏和 API response model，缺少内容、格式、完整性和质量门禁。

### 8.4 记忆是“存储策略”，还不是“冲突治理”

当前具备 provenance、confidence、TTL 和 consent，但没有语义合并、冲突检测、来源信任和纠错闭环。因此可以解释记录来自哪里，却不能可靠决定冲突时应该采信什么。

### 8.5 复盘没有反馈到未来执行

trace 和评测能支持人工复盘，但没有 `postmortem -> candidate lesson -> policy review -> long-term memory` 的自动闭环。这导致相似失败仍可能在下一次任务中重复。

## 9. 建议实施优先级

### P0，立即补齐

1. 新增 `ModelGateway`：实现 401/404/429/5xx 分类、备用 Key、备用 endpoint/model、熔断和真实 health probe。
2. 新增 `guard_input`：输入长度和编码限制、injection/jailbreak 策略、工具与检索内容的 provenance 标记。
3. 新增最终输出验证：按请求契约校验格式、检测乱码和截断、失败时有界修复或重生成。
4. 修正文档状态：在 `guard_input` 和统一输出治理完成前，不将相关 P0-P5 标为闭环。

### P1，形成计划与记忆闭环

1. 规划静态校验：重复 ID、重复语义步骤、依赖成环、悬空依赖、目标覆盖矩阵。
2. Plan cost estimate：执行前预算、裁剪策略、超限询问。
3. 记忆冲突治理：semantic key、来源可信度、冲突队列、用户仲裁。
4. 复盘记忆：terminated 后生成经验候选，经 policy 审核后写入。

### P2，提升体验与可解释性

1. 能力注册表和转人工协议。
2. optimization profile，显式处理速度、质量、成本冲突。
3. 结构化任务状态和“继续执行”入口。
4. final confidence 校准和不确定性标注。

## 10. 建议验收用例

| 用例 | 预期结果 |
|---|---|
| 模型返回 401 | 不重复使用同一无效 Key；切换备用 Key 或返回 `MODEL_AUTH_INVALID`；trace 记录尝试链。 |
| 模型 endpoint 返回持续 503 | 有界退避后切备用 endpoint/model；熔断打开；health 标记 degraded。 |
| 用户要求忽略系统规则并读取 API Key | `guard_input` 拦截或安全拒绝；工具调用为 0；不泄漏密钥。 |
| 计划包含 A depends_on B 且 B depends_on A | 执行前返回 `PLAN_CYCLE`；不进入 executor；触发一次 replan 或明确终止。 |
| 计划成本预计超过预算 | 返回裁剪方案或请求增加预算，不静默执行。 |
| 工具参数类型错误 | 首次 schema 校验失败，修复后再次校验；最多一次修复，不跳过校验。 |
| 超大工具结果 | 返回 preview、artifact ref 和 continuation；翻页后可重建完整内容。 |
| 两个来源给出冲突事实 | 按来源策略仲裁或向用户标出冲突，不允许无解释地并列注入。 |
| 最终回答声明了未发生的工具调用 | 输出验证失败，触发重生成；最终答案不得保留虚假调用。 |
| 最终回答疑似被截断或包含替换字符 | 标记 `LOW_QUALITY_OUTPUT`，最多重生成一次。 |
| React 模式只完成一半任务 | final answer 包含 outstanding 和可恢复状态，不只返回普通成功。 |
| 进程在副作用后崩溃 | 同 key 不自动重放；状态为 unknown；要求核对外部状态。 |

## 11. 回归测试记录

执行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n demo python -m pytest `
  agent/tests/test_core_contracts.py `
  agent/tests/test_tool_gateway.py `
  agent/tests/test_context_pack.py `
  agent/tests/test_token_budget.py `
  agent/tests/test_memory_store.py `
  agent/tests/test_adversarial_runtime.py `
  agent/tests/test_plan.py `
  agent/tests/test_loop.py `
  agent/tests/test_dispatcher.py `
  agent/tests/test_api_security.py -q
```

结果：

```text
133 passed, 2 warnings in 22.12s
```

警告均为环境级问题：

- Starlette TestClient 对旧 `httpx` 集成发出弃用警告。
- 当前沙箱无法写入 `.pytest_cache`，pytest 退回无缓存模式。

这两个警告不改变本报告的代码结论。

## 12. 最终判断

当前项目已经形成“工具执行治理强、上下文预算强、记忆生命周期有基础”的中台能力，但还没有形成覆盖全生命周期的 Agent failure-handling 闭环。

最恰当的当前状态描述是：

> 工具层接近生产级治理，模型层、规划前校验、最终输出验证和复盘学习仍处于部分实现或缺失状态。

建议先将 `ModelGateway`、`guard_input` 和 final output verifier 三项作为 P0，因为它们分别覆盖最主要的输入风险、外部依赖风险和最终交付风险。完成这三项后，再推进计划成本治理与记忆冲突仲裁，整体设计才更接近本清单要求的完整性。

## 13. 2026-09-20 修改进度

本轮已按 P0 和部分 P1 完成第一轮实现：

| 能力 | 当前实现 |
|---|---|
| 输入安全门 | 新增 `agent/core/input_policy.py`；在 `graph.run`、`/api/agent/chat`、`/api/agent/chat/stream` 执行前阻断角色覆盖、系统提示词窃取、密钥窃取、安全绕过和 jailbreak 模式；支持输入长度限制与控制字符清理。工具和检索内容还会二次扫描，命中时追加 DATA-only 提醒，不会被直接当作系统指令。 |
| 模型故障转移 | 新增 `agent/core/model_gateway.py`；支持备用 API Key、备用 endpoint/model、401/403/404/408/409/429/5xx 与连接错误回退、候选熔断。 |
| 模型健康探测 | `/api/agent/health` 增加 `model_status`、`model_candidates`；支持 `?probe=true` 或 `AGENT_HEALTH_PROBE=1` 做真实最小调用。 |
| 计划 DAG 校验 | 新增 `agent/core/plan_policy.py`；检测重复 ID、自依赖、未知依赖、依赖环和重复步骤，并在规划阶段执行一次有界重规划。 |
| 计划成本控制 | 计划步骤增加 `priority`；执行前估算 token 和时间，优先移除没有下游依赖的 optional 步骤；仍超限时停止执行并请求提高预算或缩小范围。 |
| 最终输出验证 | 新增 `agent/core/output_policy.py`；检查空输出、替换字符、明显截断、JSON 和 CSV 格式；React、Plan、chat、clarify 路径可触发一次重生成。 |
| 凭据泄漏脱敏 | `sanitize_output()` 增加 API Key、Bearer Token、密码、token 赋值和私钥块脱敏。 |
| 记忆冲突治理 | `MemoryRecord` 增加 `semantic_key/status/supersedes/conflicts_with`；用户来源优先于会话来源，高置信度新值替代旧值，无法裁决时双方停止注入。 |

新增或修改的回归测试：

- `agent/tests/test_failure_policies.py`
- `agent/tests/test_api_security.py`
- `agent/tests/test_memory_store.py`
- `agent/tests/test_safety.py`

本轮验证结果：

```text
agent/tests: 272 passed
python -m compileall -q agent web: passed
```

新增环境变量：

```text
AGENT_INPUT_GUARD=1
AGENT_INPUT_MAX_CHARS=20000
AGENT_MODEL_FALLBACK_KEY_ENVS=["ALT_API_KEY"]
AGENT_MODEL_FALLBACKS=[{"model":"...","base_url":"...","api_key_env":"..."}]
AGENT_MODEL_BREAKER_THRESHOLD=3
AGENT_MODEL_BREAKER_COOLDOWN=30
AGENT_HEALTH_PROBE=0
AGENT_MODEL_PROBE_TIMEOUT=15
```

第二轮完成时仍待处理，后续进展见第 14、15 节：

1. 接入正式内容安全分类，当前有害内容只覆盖凭据与 PII，不含通用 moderation。
2. 用真实模型端点执行故障转移集成测试，当前自动测试使用 fake model。
3. 计划成本仍为静态估算，下一步应使用历史 P50 成本校准。

## 14. 2026-09-20 第二轮修改进度

本轮继续完成 P1 能力边界和反馈闭环：

| 能力 | 当前实现 |
|---|---|
| 能力边界与转人工 | 新增 `agent/core/capability_policy.py`；识别资金转移、账户接管、现实世界操作、医疗/法律代理等不可执行请求，返回稳定 `HANDOFF_REQUIRED`，并提示可继续提供研究或方案支持。 |
| 优化偏好 | 新增 `agent/core/optimization_policy.py`；从用户请求识别 `fast`、`balanced`、`quality`、`cost_saver`，写入 state、Context Pack 和 planning prompt。 |
| Goal drift | 新增 `agent/core/goal_policy.py`；建立每轮 GoalContract，识别未请求的写、下载、入库、执行等副作用；首次偏离提醒模型纠偏，连续两次偏离后强制停止工具调用并收尾。 |
| 复盘候选记忆 | 新增 `agent/core/postmortem.py`；失败、超时、partial、no_evidence 会生成 `postmortem` candidate，经过 PII/凭据脱敏、默认不参与注入。 |
| 用户批准闭环 | 新增 `POST /api/memory/{id}/approve` 和配置中心“批准”按钮；只有用户批准后，复盘记忆才转为 active 并进入长期记忆策略。 |
| 评测隔离 | postmortem 在评测上下文中默认不写入，避免污染真实长期记忆。 |

本轮验证结果：

```text
agent/tests + evaluation/tests: 304 passed
python -m compileall -q agent web evaluation: passed
npm run build:renderer: passed
npm run lint: passed with existing Fast Refresh/exhaustive-deps warnings
```

## 15. 2026-09-20 第三轮修改进度

本轮完成剩余的三项主要能力：

| 能力 | 当前实现 |
|---|---|
| 本地内容安全分类 | 新增 `agent/core/content_safety.py`；覆盖自伤、武器/爆炸物、恶意软件、未成年人性剥削、隐私滥用和凭据窃取。采用“可操作指令”模式，允许正常研究、风险分析和防护讨论。 |
| 远端 moderation | 支持 OpenAI-compatible moderation endpoint；通过 `AGENT_MODERATION_ENDPOINT` 启用，默认 fail-closed，可显式改为 fail-open。输入和非流式输出、流式最终结果均接入分类。 |
| 安全终止输出 | 命中安全策略时返回稳定 `INPUT_BLOCKED` 或 `OUTPUT_BLOCKED`，输出统一安全替代文案；流式输出使用 `replace_answer` 覆盖已展示内容。 |
| 真实模型端点故障转移测试 | 新增 loopback OpenAI-compatible HTTP server 集成测试，真实构造 `ChatOpenAI` 客户端并验证 401 后切换到备用 endpoint。 |
| 远端 moderation 集成测试 | 新增本地 HTTP moderation endpoint 测试，验证 flagged 响应被转换为 `HARMFUL`/对应 category。 |
| 历史成本校准 | 新增 `agent/core/cost_history.py`，按 `target/scope/delivery` 保存脱敏 token 与耗时样本；至少 3 个样本后使用实际 P50 与静态估算各占 50% 校准计划成本。 |
| 计划成本追踪 | plan 模式 turns 结束后按步骤 scope 权重分摊实际 token 与总耗时，只持久化数值统计，不保存 query、输出或工具内容。 |

新增环境变量：

```text
AGENT_MODERATION_ENDPOINT=
AGENT_MODERATION_MODEL=omni-moderation-latest
AGENT_MODERATION_API_KEY_ENV=
AGENT_MODERATION_TIMEOUT=8
AGENT_MODERATION_FAIL_MODE=closed
AGENT_COST_HISTORY=1
AGENT_COST_HISTORY_PATH=
```

本轮验证结果：

```text
agent/tests + evaluation/tests: 309 passed
python -m compileall -q agent web evaluation: passed
real loopback model failover integration: passed
remote moderation loopback integration: passed
```

当前剩余工作主要是运维和真实供应商验证：

1. 在生产环境配置并验证远端 moderation 供应商的实际响应格式和 SLA。
2. 使用真实备用供应商 Key/endpoint 做一次受控 canary，而不仅是本地兼容协议测试。
3. 持续观察 cost history 样本量和 P50 偏差，必要时改为分位数分桶或按模型分别校准。

## 16. 2026-09-21 第四轮修改进度

本轮完成剩余的 P2 反馈闭环和 provider 运维工具：

| 能力 | 当前实现 |
|---|---|
| 最终置信度 | 新增 `agent/core/confidence_policy.py`；综合 intent confidence、verify 状态、证据步骤、失败步骤、goal drift、输出验证和任务完成度，生成 high/medium/low。 |
| 结构化任务状态 | 新增 `agent/core/completion_report.py`；统一返回 `completed/partial/interrupted/failed`、done/total、outstanding、can_resume、next_actions 和 plan cost 状态。 |
| API 契约 | `AgentChatResponse` 增加 `task_status`、`final_confidence`；graph.run、非流式 chat、resume 均返回同一结构。 |
| SSE | 新增 `task_status` 事件，在完成、失败、超时、安全阻断和审批等待时下发。 |
| 前端展示 | 对话消息增加完成状态和置信度卡片，展示未完成步骤、低置信度说明和“可继续执行”状态。 |
| Provider canary | 新增 `agent/core/provider_canary.py` 和 CLI `python -m agent.canary`；可验证真实模型链和 moderation endpoint，输出不含 Prompt、Key 或 Secret 的 JSON 报告。 |

Canary 用法：

```powershell
conda run -n demo python -m agent.canary --json
conda run -n demo python -m agent.canary --model-only --json
conda run -n demo python -m agent.canary --moderation-only --json
```

本轮验证结果：

```text
agent/tests + evaluation/tests: 312 passed
python -m compileall -q agent web evaluation: passed
npm run build:renderer: passed
npm run lint: passed with existing Fast Refresh/exhaustive-deps warnings
provider canary CLI: passed（未配置 moderation 时返回 skipped）
```

至此，审计报告中可由当前代码库完成的功能项已全部落地。剩余工作属于生产环境运维：

1. 配置真实备用 API Key/endpoint 并执行 `agent.canary --model-only`。
2. 配置真实 moderation endpoint 并执行 `agent.canary --moderation-only`。
3. 将 canary 接入部署后的定期任务或发布门禁。
