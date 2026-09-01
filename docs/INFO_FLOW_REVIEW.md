# Agent 信息传递链核查方案（参照 Claude Code 设计模型）

> 范围：`agent/` 框架内 ① subagent→parent ② tool→agent ③ agent→user 三条信息链。
> 基准：Claude Code / Claude Agent SDK 的设计原则 ——
> ① 一切流动状态放在 `messages` 里，节点间不存在隐藏 channel；
> ② subagent = 一个普通工具，入参是父代理写进 task 的纯文本，出参回到 ToolMessage；
> ③ 终止判定 = "无 tool_use 的文本响应"，不依赖任何特殊 marker；
> ④ 工具结果以原始证据进入上下文，harness 只做预算/截断，不改语义。
> 对照结论：本项目大体贴着这套模型实现，但有 3 处偏离，其中 1 处是长期污染源。

---

## 0. 结论速览

| # | 问题 | 严重度 | 结论 |
|---|------|--------|------|
| P1 | `[FINAL_ANSWER]` marker 泄漏 + 污染持久化状态 | **高（长期污染）** | 删除 prompt 中的 marker 协议；剥离逻辑改为"任意位置 + 容忍变体" |
| P2 | subagent→parent 用 contextvar 隐藏注入 `resolved`，违反零状态 | 中 | 改为显式写进 task 字符串；删除 `_resolved_ctx` 通道 |
| P3 | subagent 返回 = "最后一条非空 AI 文本"，无结构、边界脆弱 | 中 | 收紧提取规则（跳过带 tool_calls 的消息）；按 subagent 定义输出信封 |
| P4 | plan 模式的 executor 无错误恢复、无适应性（对比 react 模式） | 中 | 失败步骤复用 `_format_error_feedback` 重试一次 |
| P5 | SSE 双泵架构中 `_msg_pump` 的 AIMessageChunk 分支为死代码（已实证） | 低（误导） | 删除或加注释；tool_start/tool_end 归并到单一来源 |
| P6 | 工具结果双格式（JSON envelope / 纯文本）双份解析逻辑常驻 | 低 | 统一输出契约，逐步收敛 `_salvage_tool_content` |

**实证测试**（`langgraph 1.2.6` / `langchain-core 1.4.7`）：节点内手动 `model.astream()` 不会产生 graph 级 `stream_mode="messages"` chunk → token 只有 `_ev_pump` 一条路，**不存在双写**；但 `_stream_llm` 的前缀缓冲剥离在"首 chunk 为 `\n`"时失效，marker 原样流出（见 P1 复现）。

---

## P1 `[FINAL_ANSWER]` marker —— 不必要且正在污染

### 现状

- Prompt 协议位于 [AGENT_SYSTEM / Response Protocol](agent/prompts.py#L222-L226)，要求"输出 `[FINAL_ANSWER]` 在单独一行，然后写回答"。
- 路由 [after_agent](agent/nodes.py#L1167) 已注明 marker *不再被依赖*：`无 tool_calls + 有文本 → end`。subagent 的 Prompt 从未使用此 marker。
- 剥离点共 5 处，全部只处理"**行首、精确字面量**"：
  - [_stream_llm](agent/nodes.py#L147-L195) 前缀缓冲 `prefix.startswith("[FINAL_ANSWER]")`
  - router [_strip_marker](web/api/routers/agent.py#L71-L73)（`\[FINAL_ANSWER\]\s*\n?`，大小写敏感，但能匹配正文任意位置）

### 泄漏复现（实测）

```
chunk1: "\n"             → prefix="\n"，startswith 失败 → prefix_resolved=True
chunk2: "[FINAL_ANSWER]" → 原样 emit → 前端渲染出 marker
```

模型先行输出换行/前缀文本、全角括号 `【FINAL_ANSWER】`、大小写 `[FinalAnswer]`、或 marker 位于正文中间时，无一能剥离。

### 次生污染（最该重视的部分）

agent_node 返回的 AIMessage **带着 marker 回写 `state.messages`** → `checkpoints.db` → memory 摘要下次重算时把 marker 当正文摘要 → 后续回合的「Recent Conversation」「Prior Conversation」注入含 marker 噪声，反复造成用户观察到的现象。

### 修复方案（建议一步到位）

1. **删除 prompt 协议**：从 `AGENT_SYSTEM` 的 Response Protocol 移除 marker 要求，改用与 Claude Code 一致的规则：*当问题 1 回答 YES → 直接写最终答案，不要调用工具*。路由行为零变化（after_agent 不依赖 marker）。
2. **剥离改为兜底过滤**（防御存量）：`_strip_marker` 扩展为正则
   `re.compile(r'[\[【]\s*final\s*[-_ ]?\s*answer\s*[\]】]', re.IGNORECASE)`，
   在 router 的 `/chat` 与 `/chat/stream` 的 token 事件上统一过滤。
3. `_stream_llm` 保留前缀缓冲但改为**任意位置的行过滤**（对每个 chunk 用上述正则删除 marker 行 + 相邻空行）。
4. 评估是否清理已有 checkpoints 中的 marker 污染（一次性脚本，非必须）。

### 影响面

`AGENT_SYSTEM` 删协议不影响 subagent（其 prompt 无 marker）；只影响 react 路径的文本格式；`after_agent`/`synthesize fast path` 逻辑不动，行为等价。

---

## P2 隐藏 contextvar 注入 —— 违反零状态 & 隐性耦合

### 现状

- [resolution.py](agent/resolution.py#L35-L42) 定义模块级 `_resolved_ctx` contextvar，`resolve_node` 写入。
- [as_tool](agent/subagents.py#L128-L137) 通过 `get_resolved_ctx()` 读取，并注入 subagent 的"新 state"。
- agent_node 预检（pre-flight）和 resolve 共享同一个 contextvar。

### 问题

1. **不可见**：父代理的 model 并不知道自己的 task 会被注入 `resolved` —— 它在 prompt 里被要求"Discovery Hints 要 VERIFY 后再用"，但 subagent 那边得到的却是"trust these names, do NOT re-search"。两边对同一批 hint 的信任级别不一致，且没有任何地方把这个注入告知调用链。
2. **非零状态**：subagent 的 system 反复声明 *zero-state / 任务自包含*，实际却隐性依赖一个 task 外通道。若未来 subagent 改为独立进程/线程运行（contextvar 不跨线程传播），该通道静默失效。
3. **无生命周期管理**：contextvar 从不 reset（只有值覆盖）。在同一个 task 内跨多轮复用（如长驻后台循环反复调用 `run()`）时，旧 turn 的 `resolved` 会泄漏进下一 turn 的 subagent。

### 修复方案

删掉 `_resolved_ctx` channel，把必要上下文**显式写进 task 字符串**（parent model 自己组装），完全符合 zero-state 与 Claude Code 的"parent 负责把上下文写进任务"：

1. `as_tool` 不再调 `get_resolved_ctx()`，init_state 不再注入 `resolved`。
2. `AGENT_SYSTEM` 的「Delegation Priority」增加约束：*委托 arxiv/ingest 时，把已确认的论文名（取自 Discovery Hints 的 `match`，而非用户原话）直接写进 task*。
3. `plan_node` 已把 hints 注入 PLAN_SYSTEM prompt —— 保持，让 LLM 生成步骤时自然引用确认名。
4. 保留 `state["resolved"]`（它是 checkpointed 官方状态），只删 contextvar 侧通道。

---

## P3 subagent 返回提取 —— 无结构与边界脆弱

### 现状

[as_tool._call](agent/subagents.py#L149-L154)：

```python
for m in reversed(result.get("messages", [])):
    if getattr(m, "type", "") == "ai" and getattr(m, "content", "").strip():
        answer = str(m.content).strip()
        break
```

### 问题

- **"最后一条非空 AI 文本"不等于"最终答案"**：若 subagent 最后一条带 tool_calls（被 max_steps 截断前正在计划下一步）且 contents 非空，或中间某轮输出过探路文本后工具失败收尾，「最后一个带文本的 AI 消息」可能抓成中间态。
- **无结构**：arxiv 的 FIND 任务输出 JSON 列表、ingest/creator 输出状态行、coder 输出摘要 —— 父代理只能从一团纯文本里重新解析，没有统一的 `{ok, …}` 信封（错误路径有信封，成功路径没有）。

### 修复方案

1. **收紧提取**：只取"无 tool_calls 的 AI 消息"，且优先取**最后一个**（后续无任何 AI-with-tool_calls 的），否则进入 `_subagent_synthesize` 兜底：
   ```python
   for m in reversed(msgs):
       if m.type == "ai" and (m.content or "").strip() and not m.tool_calls:
           answer = m.content.strip(); break
   ```
2. **定义输出信封**（可选增强）：每个 SubagentSpec 增加 `output` 说明字段，要求 subagent 在最终消息里输出约定格式（arxiv 已具 JSON 契约，creator 已有状态行）——把"父代理该怎么解析"从隐式约定变成配置。

---

## P4 plan 模式 executor 缺乏错误恢复（对比 react）

### 现状

- react 模式：agent↔tools 循环每轮都经 `agent_node`，错误经 [_classify_tool_error](agent/nodes.py#L369) + [_format_error_feedback](agent/nodes.py#L408) 喂回 LLM 自适应恢复、重试、降级。
- plan 模式：[executor_node](agent/plan.py#L467-L548) 是纯确定性拓扑执行，步骤失败只记 `ok:False, error`；除 creator 落盘重试一次外，**没有任何 recovery 反馈**，失败步骤的错误解释交给 synthesize 的一句话标注。

### 问题

同一查询走 react 与走 plan（多论文/对比/写作/实验被 `decide_mode` 判定为复杂时）的容错能力差距很大：plan 的单步骤参数错误（LLM 计划的 `paper_name` 猜错）直接产生垃圾结果，没有"根据 available_papers 改参数重试"的机制。

### 修复方案（最小集）

在 `_run_step` 失败分支复用 react 的错误分类：

1. `executor_node` 对 `ok:False` 步骤执行一次 `_classify_tool_error(payload)`：
   - `param_error` 且返回里有 `available_papers/sections` → 修正 `args` 重试一次（不级联后续依赖步骤重跑）。
   - `transient` → 原参数重试一次。
   - `not_found` / `backend_down` → 不重试，标注原因进 `subagent_results`（synthesize 已有渲染）。
2. 重试仍失败时，向 synthesize 注入一条"本次哪些步骤被跳过及原因"，避免产出"成功"假象。

---

## P5 SSE 双泵中 AIMessageChunk 分支是死代码（实证）

### 实证

`agent.astream(stream_mode="messages")` 下，节点内手动 `model.astream()` 产生的 chunk **不会**出现在 graph 级 messages 流中（测试：GenericFakeChatModel 逐 token 流式，graph 级只收到节点最终返回的 AIMessage；`_stream_llm` 的 emit 是唯一 token 通道）。即 [_msg_pump 的 AIMessageChunk 分支](web/api/routers/agent.py#L189-L197) 在实践中不会触发。

### 结论

不是 bug（无重复输出），但代码表达与实际行为不符，后续维护会误读。建议：
- 删掉 AIMessageChunk token 分支，或加注释说明"chunks 经 `_stream_llm`→event 队列，此分支为框架行为未来变化时的兜底"。
- tool_start/tool_end 的来源（`_msg_pump` 处理 message 流 + `_ev_pump` 处理 event 流）保持现状即可，双方按 `tool_call_id`/`run_id` 去重阳合并已验证。

---

## P6 工具结果双格式 —— 收敛建议（低优先，可不做）

- 库工具返回 JSON envelope（`{"ok": true/false, ...}`），通用文件/数学工具返回纯文本 → `_salvage_tool_content`、`_classify_tool_error`、`build_tools_node` 三处都要兼容双格式。
- 建议：**新工具一律输出 schema 化 JSON envelope**（`{"ok": bool, "data": ..., "error": ...}`）；文本类工具保持 markdown 但 promise "not JSON"。
- 截断 `_TOOL_RESULT_MAX=8000` 维持字符级即可（语义截断的收益/成本比不值得在 MVP 做）；但注意它对 JSON envelope 的破坏性 —— 错误信封都短，实际不受影响。

---

## 落地顺序建议

| 顺序 | 事项 | 改动面 | 回归风险 |
|------|------|--------|----------|
| 1 | P1：删 marker 协议 + 过滤兜底 | `prompts.py` + `nodes.py` + `routers/agent.py` | 低（路由行为不变） |
| 2 | P3：收紧 subagent 返回提取 | `subagents.py`（几行） | 低 |
| 3 | P2：删 contextvar 通道，hints 写进 task | `subagents.py` + `resolution.py` + `AGENT_SYSTEM` 委托约束 | 中（需回归 arxiv/ingest delegate 场景） |
| 4 | P4：executor 错误恢复 | `plan.py` | 中（新增重试语义） |
| 5 | P6 / P5 | 平滑清理 | 低 |

> 关联说明：用户已打开的 `agent/config.py` / `agent/config.yaml`（step/turn/subagent 上限）与本方案无冲突，不需改动；P2 删除 contextvar 后 `get_limits()` 的 subagent 上限机制保持原样。

---

## 附：证据与文件速查

| 主题 | 位置 |
|------|------|
| after_agent 终止判定（不依赖 marker） | [agent/nodes.py](agent/nodes.py#L1154-L1185) |
| _stream_llm 前缀缓冲剥离（缺陷点） | [agent/nodes.py](agent/nodes.py#L147-L195) |
| marker prompt 协议 | [agent/prompts.py](agent/prompts.py#L222-L226) |
| router 剥离（非流式可用） | [web/api/routers/agent.py](web/api/routers/agent.py#L71-L73) |
| subagent 返回提取 | [agent/subagents.py](agent/subagents.py#L149-L154) |
| contextvar 通道定义 | [agent/resolution.py](agent/resolution.py#L35-L42) |
| as_tool 隐性注入 resolved | [agent/subagents.py](agent/subagents.py#L128-L137) |
| executor 无错误恢复 | [agent/plan.py](agent/plan.py#L467-L548) |
| 错误分类/反馈（react 持有点） | [agent/nodes.py](agent/nodes.py#L369-L457) |
| SSE 双泵 | [web/api/routers/agent.py](web/api/routers/agent.py#L163-L309) |

---

## 附2：落地记录（2026-09-01 全部完成）

| # | 状态 | 落地点 |
|---|------|--------|
| P1 | ✅ | `prompts.py`（Response Protocol 删 marker）；`nodes.py` `_FINAL_ANSWER_RE`/`_FINAL_ANSWER_LINE_RE` + `_stream_llm` 前缀缓冲改为行级剥离（容忍首 chunk `\n`/全角括号/分隔符变体）；`routers/agent.py` `_strip_marker`/`_strip_marker_segment` 统一正则；`graph.py`/`search_loop.py` 注释同步 |
| P2 | ✅ | `resolution.py` 删 `_resolved_ctx`/`get_resolved_ctx`；`subagents.py::as_tool` 不再注入 resolved；`AGENT_SYSTEM` 「Delegation Priority」加「task 写确认论文名」约束。`state["resolved"]` 保留 |
| P3 | ✅ | `subagents.py::_call` 提取改为「跳过带 tool_calls 的 AI 消息，取最后一个无 tool_calls 的文本」 |
| P4 | ✅ | `plan.py`：`_run_step` 将信封 ok=false 归一为 `ok=False`；`executor_node` 直接工具步骤确定性重试一次（`_retry_args_from_error`：transient 原参数 / param_error 按 available_papers/sections 修正）；`not_found`/`backend_down` 不重试进标注 |
| P5 | ✅ | `routers/agent.py` 删 AIMessageChunk token 分支（实证死代码），保留注释说明 token 仅走 `_ev_pump` |
| P6 | ✅（用户指定重点） | 新增 `agent/tool_contract.py`（`ok`/`err`/`parse_tool_result`/`truncate_tool_result`）；builtin/generic/creation/coding/dispatcher 信封生成全部转发；`_classify_tool_error`、`_salvage_tool_content`、`plan._ingest_guard` 三个解析点统一走 `parse_tool_result`；`build_tools_node` 截断改 `truncate_tool_result`（截断保持 envelope 可解析） |

回归：`agent/tests/test_info_flow.py` 新增 15 例（P1 marker 正则/剥离/前缀判定 + P6 信封生成/解析/截断）。全套 `pytest agent/tests`：**76 passed**；`test_observability`（全局计数跨测试串扰，单独跑通过）与 `test_coding`（缺 `tmp_root` fixture，无 conftest 提供）为**既有**失败，与本次改动无关。