# ADR-0001：LangGraph 运行内核与状态边界

- 状态：Accepted
- 日期：2026-09-15
- 相关：`agent/graph.py`、`agent/state.py`、`agent/core/execution_context.py`、`agent/supervisor.py`

## 背景

主图 `agent/graph.py` 已经承担 intent 路由、react/plan 分流、checkpoint 与
受限 subagent 派发。同时仓库里存在多种「会话记忆」载体：LangGraph
checkpoint、`trace_store.db`、内存计数器、`profile.json`。边界不清会导致
两件事同时发生：不可序列化的运行时对象被写进 state，以及业务事实被写进
与运行无关的存储。

## 决策

1. **唯一运行内核**：编排只用 LangGraph（`StateGraph`、checkpointer、
   `interrupt()`、`RetryPolicy`、`ToolNode`）。不再新增工作流引擎、span 树
   或通用重试框架。
2. **state 只放可 checkpoint 的业务事实**：`AgentState` 及其扩展字段只允许
   JSON 可序列化的业务状态与「对象的 ID/hash」。API key、原始密钥、LLM/工具
   客户端、`RunnableConfig`（含回调）一律不得进入 state。
3. **运行元数据走 `ExecutionContext`**：身份、冻结配置、预算、prompt 绑定、
   LangSmith `RunnableConfig` 由 `agent/core/execution_context.py` 用
   `contextvars` 携带（`set_current_execution_context`），每轮 bootstrap 建立、
   turn 结束重置。它不进 checkpoint，因此不会被持久化。
4. **异步任务边界由 `task_id` 表达**：长任务经
   `agent/supervisor.py` 以 `thread_id = task_id` 落同一 `checkpoints.db`，
   图内只等待用户体验可接受的短步骤。
5. **副作用前必须回到审批点**：需要人工确认的调用经 `interrupt()` 暂停，恢复
   时按 checkpoint 中的幂等键继续，不重放已成功操作（见 ADR-0003）。

## 替代方案

- **自建调度器/状态机**：可控但会重复 LangGraph 已有的 checkpoint、中断恢复、
  流式事件能力，且团队要自行维护一致性；已否决。
- **把 `ExecutionContext` 放进 state**：节点读取方便，但 checkpoint 里会出现
  不可序列化对象与凭据风险；已否决。
- **继续把 trace 当第二事实源**：本地 `trace_store` 与 LangSmith 双写会分叉
  （见 ADR-0002），已否决。

## 兼容性

- 现有节点签名 `(state, config)` 不变；新增能力以包装/委托方式接入
  （`ToolDispatcher` → `ToolGateway`、`memory_node` 增加 pack 决策）。
- 旧 import 路径保留一个发布周期；目录迁移按
  `docs/agent-platform-architecture.md` §10 逐项进行，不做一次性搬移。

## 回滚

- `ExecutionContext` 与 gateway 都是**加性**组件：把 `graph.prepare_turn` 换回
  直接构造 `config`、把 `ToolDispatcher.call` 换回直接调用 `call_fn` 即可退回
  旧路径（两者都不改 state schema 与工具返回信封）。
- checkpoint 格式未变，旧 `checkpoints.db` 可直接被回滚后的代码读取。

## 验收指标

- 图测试覆盖：路由、`interrupt`/resume、retry、checkpoint 恢复、幂等行为。
- `AgentState` 新增字段必须同时满足：可 JSON 序列化、可由 checkpoint 恢复。
- 每次 turn 的 `token/step/turn` 上限来自冻结的 `Budget`，而不是节点内重读 env。
