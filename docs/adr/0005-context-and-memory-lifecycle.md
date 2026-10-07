# ADR-0005：上下文与记忆生命周期

- 状态：Accepted
- 日期：2026-09-15
- 相关：`agent/core/context_pack.py`、`agent/core/memory_policy.py`、`agent/memory.py`、`agent/context.py`、`agent/state.py`

## 背景

注入模型的上下文此前是一条不断增长的字符串（`context_snapshot` + 各节点自行
拼接），风险是：token 失控、检索结果与用户偏好混在一起、无法解释「这轮为什么
截断」。会话记忆则混在 `profile.json`、`state.context`、checkpoint 与摘要缓存
里，缺统一的置信度/时效/同意（consent）语义。

## 决策

1. **分区预算的 Context Pack**（`ContextManager.build`）：`invariant（固定上限，
   不截断）/ task 15% / conversation 25% / retrieved 45% / memory 10% /
   reserve 5%（不提前占用）`。预算按模型真实 context window 与
   `max_output_tokens` 动态计算（`Budget.usable_context_tokens()`），
   不用单一字符估算决定安全边界。
2. **每个区都记录来源与理由**：`ZoneContent.trace_view()` 给出
   `budget/tokens/strategy/truncated/entries[{source_ref, tokens, kept, reason}]`，
   汇总进 `AgentState.context_decision["pack"]` 与 trace（`context_built` 事件）。
   **prompt 文本不进 decision**，只记元数据，便于解释与评测。
3. **检索区做去重 + 多样性**：先按规范化文本去重，再用 MMR-lite（`score -
   0.5 * max_jaccard`）贪心选片，超出预算的记 `kept: false` 与原因，
   不做静默丢弃。
4. **对话区复用既有装配器**：`MemoryManager.build_snapshot` 仍是 conversation 区
   的实现（tool call/result 成对保留、摘要 + 缓冲分层），避免两套拼装逻辑。
   conversation 区以 `include_profile=False` 调用，画像统一进 memory 区，避免
   同一偏好被注入两次。
5. **记忆四类分离且有生命周期**：会话工作记忆（checkpoint）、对话摘要、
   用户画像、领域事实/任务记忆。长期记忆统一为 `MemoryRecord`
   （`memory_id/type/content/source_ref/confidence/created_at/expires_at/
   consent/revision`），写入/注入前由 `MemoryPolicy`（置信度阈值、TTL、
   consent、按类型/ID 禁用）判定。
6. **画像文件保持现形**：`profile.json` 结构不变，由
   `records_from_profile()` 适配为类型化记录；**检索到的全文不得写入画像**，
   记录只存偏好/事实本身，来源放 `source_ref`。
7. **用户可控**：记录可通过 `disabled_ids/disabled_types`（后续接前端面板）
   查看、禁用与删除；被过滤的记录在 trace 里留下
   `{memory_id, type, confidence, reason}`，不含内容。

## 替代方案

- **单一 system prompt 拼接**：简单，但无法解释截断、无法按来源降级；已否决。
- **只按字符数截断**：与真实 token 偏差大，长会话容易超窗或过度截断；已否决。
- **把检索全文写进画像做「长期记忆」**：污染、隐私风险高、且画像无版本；
  已否决。

## 兼容性

- 兼容期仍保留 `context_snapshot` 供 chat/clarify 等节点使用；Agent 主循环改为
  使用 `ContextManager.build` 的 conversation + memory + retrieved 渲染结果，
  并在每个 ReAct iteration 按最新 tool messages 重建 retrieved zone。
- `AgentState.context`（对话中心化工作区绑定）继续存在，并被 task 区引用为
  「当前目标/工作区状态」，不重复写入记忆。

## 回滚

- pack 构建失败时只记 `context_pack_failed` 警告，`context_snapshot` 与旧
  `context_decision` 仍照常产出；把该调用移除即完全回到旧行为。
- `MemoryPolicy` 不写盘：回滚不涉及数据迁移。

## 验收指标

- 每轮都有预算决策（`context_decision.pack` 非空），长会话不超窗、不拆散
  tool call/result 对。
- 截断可解释：任一 `truncated: true` 的区都能列出被丢弃来源与原因。
- 记忆注入 100% 经 `MemoryPolicy`；被过滤记录可审计且不含内容。

## 后续（本 ADR 未覆盖）

- 更丰富的自动抽取/人工审核队列；当前只捕获用户显式“记住/以后请…”指令，
  避免把普通问答错误沉淀为长期事实。
