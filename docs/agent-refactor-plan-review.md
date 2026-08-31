# 《Agent 通用层改造计划》评估与指导（修订版 v2）

> 评估对象：`docs/agent-refactor-plan.md`
> 参照系：CowAgent（成熟 Agent Harness 参考实现）
> 评估方法：逐条核对目标项目真实代码（`agent/`、`web/api/`、`indexer/` 等），再对照 CowAgent 的成熟模式给出裁剪与排序建议。
> 评估日期：2026-08-27
>
> **v2 修订说明**：项目定位已从「科研文献助手」调整为「**通用 Agent 底座**」。科研 PDF 相关的部分（向量知识库 + `search_papers`/`fetch_content`/`download_paper`/`process_paper` 等工具）只是**第一个领域的数据源 + 工具集**，不再是产品身份。本次修订据此重新划分「通用层 / 专业层」边界，并据此调整各 Phase 的分类与排序。

---

## 0. 结论先行（TL;DR）

**你的定位澄清让计划的原则从「正确」升级为「唯一正确路径」**：没有固定领域，通用层就是产品本身，「先做厚做稳通用能力、再做专业拓展」不是取舍而是唯一选择。计划的原始原则完全成立，且现在更彻底了。

但有三处**必须先修正的代码偏差**（与领域无关，照样成立），否则 Phase 2/3/6 会踩坑：

1. **`destructiveHint` 并不存在**（计划误以为「注解已齐」）。builtin 工具只标了 `readOnlyHint=False`，没有 `destructiveHint=True`。Phase 2/6 的权限矩阵和 HITL 审批读不到计划设想的字段。
2. **builtin 工具绕过了 `_to_langchain_tool`**。Phase 3 的「统一 dispatcher」改 `_to_langchain_tool` 覆盖不到 `search_papers`/`fetch_content`/`download_paper`/`process_paper` 这四条路径——它们走 `@tool` 原始函数直返。
3. **`p50/p95` 是过度设计**。单用户本地 agent，计数器 + 累计值足够，不必上分位数。

**定位调整带来的最关键新结论**：

- **知识库（文档向量索引）从「定制层」上移为「通用层」**。`indexer/` + `retrieval/` + `retrieval_orchestrator/` 是通用「文档知识库」基础设施，不是科研专用。真正科研专用的只剩两小块：学术 PDF 解析/学术切分、`paper—cites—paper` 引用图谱。
- **唯一的领域（科研）应保持「具体」，不要提前抽象**。它是通用层唯一的真实测试夹具；在第二个领域真正出现之前，不为「领域」建空抽象（不重命名 `builtin_provider`、不建 DomainProvider 基类、不做多领域路由）。这是本次修订最重要的护栏。

**裁剪后建议**：真正该做的是 **Phase 0（砍 metrics 分位数）+ Phase 1（超时/预算，最高杠杆）+ Phase 2（只做 PII 脱敏 + 权限门，砍 prompt-injection 关键词过滤器）+ Phase 3（先补 builtin 注解，再上 dispatcher）+ Phase 5 中「索引生命周期收口」这一小项**。Phase 4 长期记忆、Phase 5 引用图谱、Phase 6 HITL、Phase 7 Plan 模式全部标为「出现真实需求再做」，其中 Phase 6 是唯一技术高风险、要单独 spike 验证的。

---

## 执行进度（2026-08-27 实施记录）

按 §6 裁剪版顺序落地，每个 Phase 留一个 `assert` 级自检（`agent/tests/`、`indexer/tests/`），全部通过。

| # | 阶段 | 状态 | 产出 | 自检 |
|---|------|------|------|------|
| ① | Phase 1 超时 + token 预算 + max_iterations | ✅ 完成 | [state.py](../agent/state.py) 预算字段、[graph.py](../agent/graph.py) 整轮 `wait_for`、[nodes.py](../agent/nodes.py) 预算守卫 | `agent/tests/test_loop.py` |
| ② | Phase 2 PII 脱敏 + 权限门 | ✅ 完成 | [safety.py](../agent/safety.py)（`mask_pii` / `tool_allowed` 读 `readOnlyHint is False`）、[builtin_provider.py](../agent/providers/builtin_provider.py) 权限门、[routers/agent.py](../web/api/routers/agent.py) 脱敏 | `agent/tests/test_safety.py` |
| ③ | Phase 0 trace_id 结构化日志（砍分位数） | ✅ 完成 | [observability.py](../agent/observability.py)（contextvars `trace_id` / `count` / `timed` / `log_turn_summary`）、6 节点 `@timed` | `agent/tests/test_observability.py` |
| ④ | Phase 5 索引生命周期收口（通用） | ✅ 完成 | [dedup_manager.py](../indexer/dedup_manager.py) 孤儿 paper 作用域修复、[README_INDEXER.md](../indexer/README_INDEXER.md) 状态机 | `indexer/tests/test_dedup_manager.py` |
| ⑤ | Phase 3 工具调度器（补 builtin 覆盖） | ✅ 完成 | [dispatcher.py](../agent/dispatcher.py)、[tools.py](../agent/tools.py) 移除 `_BUILTIN_IMPLS` 直返分支、全部工具经 dispatcher | `agent/tests/test_dispatcher.py` |
| ⑥ | Phase 0 剩余（`/api/agent/metrics`） | ⏸️ 跳过 | 见 §3.3，价值低，后置或用日志聚合替代 | — |
| ⑦-⑩ | Phase 4 画像 / 5 引用图谱 / 6 HITL / 7 Plan | ⏸️ 观察期 | 各留接口位，出现真实需求再启动（§6） | — |

**关键修正落实**：
- `destructiveHint` 不存在 → 权限矩阵读 `readOnlyHint is False`（[safety.py](../agent/safety.py) `tool_allowed`）。
- builtin 直返分支（原 [tools.py:106-108](../agent/tools.py#L106-L108)）已移除，dispatcher 覆盖全部工具，含最耗时的 `process_paper`。
- `p50/p95` 分位数与 metrics 端点已砍，观测只留 `count + total + max`。

---

## 1. 现状盘点核对（逐项对着代码验过）

计划的**方向判断全部成立**（观测缺失、治理半缺、安全全缺、长期记忆缺、HITL 半缺、工具层与知识管理只差加固）。下表只列**有偏差或值得强调**的项：

| 计划声称 | 代码实况 | 结论 |
|---|---|---|
| 工具层「`destructiveHint`/`readOnlyHint` 注解已齐」 | [builtin_provider.py:589-655](../agent/providers/builtin_provider.py#L589-L655)：`search_papers`/`fetch_content` = `readOnlyHint=True`；`download_paper` = `readOnlyHint=False, idempotentHint=True`；`process_paper` = `readOnlyHint=False, idempotentHint=False`。**全文无一处 `destructiveHint=True`**。`destructiveHint` 只在 [mcp_provider.py:191](../agent/providers/mcp_provider.py#L191) 做 MCP 注解透传。 | ⚠️ **字段名对不上**。权限矩阵应改读 `readOnlyHint is False`（或给这两条补 `destructiveHint=True`）。 |
| 「`_to_langchain_tool` 改为经 dispatcher 调用」即可统一超时/重试 | [tools.py:106-108](../agent/tools.py#L106-L108)：`if td.source == "builtin" ... return _BUILTIN_IMPLS[td.name]` —— 四条 builtin 工具**直接返回 `@tool` 原始函数，不走动态包装**。 | ⚠️ **Phase 3 有覆盖盲区**。dispatcher 必须覆盖 builtin 路径，否则最耗时的 `process_paper` 拿不到统一超时/审计。 |
| 「无整轮 token 预算 / 无整轮超时」 | [state.py:74](../agent/state.py#L74) 只有 `max_iterations=5`，无 `token_budget`/`tokens_used`；[graph.py:158](../agent/graph.py#L158) `ainvoke` 无 `wait_for`。 | ✅ 成立。 |
| 「长期记忆缺失、画像无权限/历史」 | [memory.py](../agent/memory.py)：buffer(6) + summary + `profile.json`，无 role、无 interaction_history。 | ✅ 成立。 |
| 「SSE 流式 + clarify + 错误分类已齐，缺 HITL/审批/进度」 | [routers/agent.py:109-217](../web/api/routers/agent.py#L109-L217)：token 级 SSE、`status`、citation 齐；无 `approval_request`/`progress`。 | ✅ 成立。 |

**核对小结**：偏差集中在**两个字段/调用链细节**（上两处 ⚠️），属于「动手前先知道就不会翻车」的范畴，与领域无关。

---

## 2. 通用层 vs 专业层的重新划分（本次修订的核心）

定位改为「通用 Agent 底座」后，原计划 §4 的边界表需要重画。原表把「知识管理」整个归为定制层——这是错的：文档知识库是通用能力。

| 层 | 通用（做厚做稳，先做） | 专业/领域（先只保一个实例，后拓展） |
|----|------------------------|--------------------------------------|
| 观测 | `trace_id` 结构化日志、计数器 | — |
| 治理 | 整轮超时、token 预算、重试、循环上限 | — |
| 安全 | 输入过滤、PII 脱敏、权限矩阵、过滤链 | — |
| 工具 | `Provider` 抽象、统一 dispatcher、结构化错误 | `search_papers`/`fetch_content`/`download_paper`/`process_paper`（**第一个领域 provider 的实例**） |
| 记忆 | buffer/summary/画像 + 长期记忆骨架 | 领域事实内容（随领域变化） |
| 知识库 | 文档摄取、切分、索引、检索、生命周期、评估（`indexer`/`retrieval`/`retrieval_orchestrator`） | 学术 PDF 解析、学术切分策略、`paper—cites—paper` 引用图谱 |
| 人机协作 | HITL 审批机制、SSE 通道、结构化进度 | 具体审批规则（哪些操作需审批） |
| 执行模式 | Plan-and-Execute 引擎 | 多跳/多步复杂任务的 prompt 定制 |

**三条护栏（本次修订最重要的判断）**：

1. **只有一个领域时，不为「领域」建空抽象**。`builtin_provider.py` 现在叫这个名字没错，`providers/` 的 `ToolProvider` ABC + `Composite` + dispatch 已经是通用工具层的正确骨架。**不要**现在去改名 `builtin → domain`、不要建 `DomainProvider` 基类、不要做多领域路由——直到第二个领域真正出现。通用能力做厚 ≠ 提前把领域抽象成框架。
2. **科研工具和 PDF 知识库是通用层唯一的真实测试夹具**。Phase 1 的死循环、Phase 3 的超时重试、Phase 2 的权限门，都要靠这几条真实工具来验证。因此它们必须保持具体、真实、可跑，不能为了「通用化」而架空。
3. **「通用」指能力通用，不指接口虚化**。观测/治理/安全/dispatch/memory 这些是横切能力，做厚它们是对的；但它们落在**具体工具和具体数据**上才有意义。抽象的代价是没人能验证它工作。

---

## 3. 与 CowAgent 成熟模式的对照

### 3.1 已经对齐 CowAgent 的地方（保持，别动）

| CowAgent 成熟模式 | 现状 / 计划 | 评价 |
|---|---|---|
| Channel → Core → Model 解耦 | `Provider 抽象` + LangGraph Core | ✅ 方向一致 |
| 插件式工具（Skill/MCP 零代码接入） | `CompositeToolProvider` + `ToolDef.annotations` | ✅ 已是 Demo 版插件层 |
| LLM 契约「错误分类 + citation 格式不回归」 | 计划反复强调保留 `{"ok":false,"error_type":...}` 与 `[CITE:N]` | ✅ **全计划最重要纪律**，契约断裂是静默回归 |
| 分层记忆 | Phase 4 补长期记忆，明确不做图谱 | ✅ 克制 |
| 「Agent Harness」定位 | 你现在的定位（通用底座 + 领域可插拔） | ✅ 与 CowAgent 完全同构 |

### 3.2 CowAgent 有、计划缺的（值得补的最小一项）

- **可执行的自检，而非行为 DoD**。每个 DoD 都是行为描述（「死循环必终止」「注入被拦截」），但**没留下任何能跑的最小测试**。对照 Demo 自己 `CLAUDE.md` 的质量门禁文化（`indexer/tests/`、Recall@5 门禁）和 CowAgent 的成熟度来源——这是计划唯一实质性缺口。
  - 建议：每个 Phase 留 **一个** `assert` 级自检（`agent/tests/test_loop.py`），不是测试套件。Phase 1 的「死循环必终止」、Phase 2 的「PII 脱敏」「未授权拦截」最该留，因为回归 = 安全/费用事故。

### 3.3 计划该砍的（与领域无关的过度设计）

| 计划项 | 为什么该砍 |
|---|---|
| Phase 0 `p50/p95_latency` 分位数 | 分位数要存样本序列/直方图，不是计数器能算的。单用户本地，`count + total + max` 足够。 |
| Phase 2 prompt-injection 关键词过滤器 | 单用户、本地、无多租户，注入攻击面极小；关键词在自然文本里误伤率高、绕过率也高。ROI 为负。真正的安全价值是 PII 脱敏（具体）+ 权限门（具体）。 |
| Phase 4 独立 `memory` collection 存「对话事实」 | 事实抽取（每轮多一次 LLM + 异步写 + 去重 + 召回合并）是一整块工作，对单用户回报不确定。先做「画像累积」，观察真实缺口再加。 |
| Phase 0 `/api/agent/metrics` 端点 | 价值远低于 `trace_id` 日志本身。可后置或用日志聚合替代。 |

---

## 4. 逐 Phase 评估

### Phase 0 — 可观测性底座 ✅ 方向对，砍两处

- **肯定**：`contextvars` 存 `trace_id`（复用 [builtin_provider.py:53-56](../agent/providers/builtin_provider.py#L53-L56) 的 cite registry contextvars 模式）、替换 `print()`、`logging`+`contextvars` 不引全家桶，全对。
- **砍**：`p50/p95` 分位数；`/api/agent/metrics` 端点（`trace_id` 日志是第一价值，先交付它）。
- **补**：留一个自检——跑一局，断言同一 `trace_id` 贯穿节点或计数器非零。
- **DoD 修正**：`trace_id` 可还原整轮 + 无裸 `print()`；metrics 端点不作为门禁。

### Phase 1 — 循环治理 ✅ 最高杠杆，先做，可提前

- **肯定**：整轮超时（`asyncio.wait_for`）、token 预算（累加 `_estimate_tokens`）、瞬态重试用 `RetryPolicy`、`max_iterations` 配置化，全部正确且独立。
- **排序**：这是保护钱包/防死循环的最直接改动，`timeout + token_budget` 一天内可落地，可**先于完整 Phase 0 交付**（观测先行原则也站得住，任选，不阻塞）。
- **补**：留一个自检——构造只调工具无进展的查询，断言在预算/超时下**必**终止并兜底。
- **风险**：`RetryPolicy` 是较新 langgraph API，`>=0.2.0` 是宽松下限，动手前 `pip show langgraph` 确认（计划 §6 已列，强调「必须先行验证」）。

### Phase 2 — 安全层 ⚠️ 砍注入过滤，先修字段

- **砍**：prompt-injection 关键词过滤器（见 3.3）。
- **先修**：权限矩阵读 `readOnlyHint is False`（或补 `destructiveHint=True`），别读计划设想的 `destructiveHint=True`。
- **保留**：PII 正则脱敏（邮箱/手机号/身份证/银行卡）——低成本、具体、必做；过滤器链抽象合理。
- **补**：留一个自检——含手机号/邮箱的合成回答脱敏后不含原文；非授权角色无法触发 destructive 工具。

### Phase 3 — 工具层加固 ⚠️ 先补 builtin 覆盖

- **先修**：确认 dispatcher 覆盖 builtin 四条工具（[tools.py:106-108](../agent/tools.py#L106-L108) 的直返分支），否则最慢的 `process_paper` 拿不到统一超时。
- **肯定**：`ToolError` 结构化 + 保留 `{"ok":false,"error_type":...}` 兼容格式；幂等才重试（`idempotentHint`）正确。
- **简化**：`process_paper` 内部已有 `_PROCESS_PAPER_TIMEOUT`（[builtin_provider.py:555](../agent/providers/builtin_provider.py#L555)），dispatcher 层超时只做「兜底 + 审计」，别做两层 timeout 打架。
- **DoD**：现有错误恢复语义（`param_error` → 换参数）不回归，这条最关键。

### Phase 4 — 长期记忆 ⚠️ 降级为「画像累积优先」

- **先做**：画像扩展（`role` + `interaction_history` + `learned_preferences`），低成本、跨会话价值直接。
- **后置**：独立 `memory` collection 事实库，等观察到「画像 + 现有文档索引」覆盖不了的真实召回缺口再说（见 3.3）。
- **肯定**：「失败降级为跳过、不阻塞」的写入策略正确。
- **DoD 改领域中立**：别用「我之前问过哪些论文的方法」，换成「跨会话能召回用户此前提过的实体/偏好」。

### Phase 5 — 知识管理 ⚠️ 拆分：生命周期（通用，做）vs 引用图谱（领域，后置）

- **通用部分（本阶段做）**：索引生命周期收口（核对 [indexer/pipeline.py](../indexer/pipeline.py) 的「新增/更新/删除」三态 + 孤儿清理）+ 补索引生命周期状态机 README。这是通用「文档知识库」的收口，与领域无关。
- **领域部分（后置）**：`paper—cites—paper` 图谱。且难点**不在存储（SQLite 三元组足够），在引用关系的抽取**——动手前先确认引用元数据是否已存在于 `rag_chunks.json`/`paper_registry.json`，否则会卡在抽取。
- **分类修正**：原计划把「知识管理」整个归定制层，现在上移为「通用」，只有学术切分 + 引用图谱算领域。

### Phase 6 — HITL ⚠️ 全计划唯一高风险项，单独 spike

- **核心风险**：`interrupt()` 与当前 `astream(stream_mode="messages")` 流式循环**不天然兼容**。现在是「订阅 token 流」模型，HITL 需要「轮询 interrupt → 前端确认 → `Command(resume=...)` 续跑」模型，是两套驱动方式。**开工前先做最小 spike**：`interrupt()` 在 checkpointer + SSE 下能否断线恢复，验证通过再进正式计划。
- **肯定**：审批态持久化到 checkpointer、`progress` 事件复用 Phase 0 计数器、审批/确认/拒绝走同一条 SSE 通道，设计都对。
- **依赖修正**：destructive 判定同样落在「先修 `readOnlyHint=False`」。

### Phase 7 — Plan-and-Execute ✅ 通用能力，正确留接口、后置

- 完全认同「仅当真实出现多跳/多步复杂任务再做」。`state.mode` 接口位成本极低，值得留。
- **改领域中立**：这不是「论文对比/综述」专属，是通用执行模式；具体多步任务的 prompt 定制才算领域部分。

---

## 5. 最关键的风险与修正（按优先级）

1. **`langgraph` 版本**：`interrupt()`（Phase 6）与 `RetryPolicy`（Phase 1）需要较新 langgraph，`>=0.2.0` 是危险下限。**动手前 `pip show langgraph` 确认，必要时收紧下限。**
2. **LLM 契约静默回归**：`{"ok":false,"error_type":...}` + `[CITE:N]` 是全计划生命线。任何 dispatcher/审核改动都要保证原样透传。
3. **HITL 流式架构**：`interrupt()` 与 `astream` 不兼容（见 Phase 6）。最高风险，需 spike。
4. **字段名偏差**：`destructiveHint` 实际不存在，builtin 用 `readOnlyHint=False`。
5. **dispatcher 覆盖盲区**：builtin 四条工具绕过 `_to_langchain_tool`。
6. **（新，定位相关）不要为单一领域建空抽象**：通用层做厚 ≠ 把科研工具抽象成框架。在第二个领域出现前，`builtin_provider.py` 保持现状。

---

## 6. 裁剪版执行顺序（通用层做薄做对，领域保持具体）

```
第一批（高杠杆、低风险、今天就做）：
  ① Phase 1 超时 + token 预算 + max_iterations 配置化   ← 保护钱包，防死循环
  ② Phase 2 的 PII 脱敏 + 权限门（先修 destructiveHint 字段）
  ③ Phase 0 的 trace_id 结构化日志（砍 metrics 分位数）
  ④ Phase 5 的「索引生命周期收口」（通用，收口现有文档知识库）

第二批（通用层收口）：
  ⑤ Phase 3 调度器（先补 builtin 覆盖）
  ⑥ Phase 0 剩余（/api/agent/metrics 可选）

观察期（出现真实需求再启动，各留接口位）：
  ⑦ Phase 4 画像累积 → 长期记忆（视缺口）
  ⑧ Phase 5 引用图谱（先确认引用数据源）
  ⑨ Phase 6 HITL（先 spike interrupt + SSE + checkpointer）
  ⑩ Phase 7 Plan 模式（通用引擎，遇多步任务再做）
```

**顺序原则**：通用能力（超时/预算/脱敏/权限门/trace_id/知识库生命周期）做薄但做对；领域能力（科研工具、学术切分、引用图谱）保持具体，不提前抽象；第二个领域出现时，再从具体工具里抽共性。

---

## 7. 一句话行动计划

**先把「超时 + token 预算 + PII 脱敏 + 权限门（读 `readOnlyHint=False`）+ trace_id 日志 + 索引生命周期收口」这六件事做完并各留一个 `assert` 自检，科研工具和 PDF 知识库作为唯一的真实测试夹具保持具体不抽象，其余能力（长期记忆、引用图谱、HITL、Plan 模式）全部按「出现真实需求」再启动。**
