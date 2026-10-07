# 工具契约（Tool Contract）

本文件是工具调用链路的可读契约；实现权威在 `agent/tool_contract.py`（信封）、
`agent/core/contracts.py`（模型）、`agent/core/tool_gateway.py`（调用顺序）。

跨工具、Agent、子 Agent 与后台任务的统一业务结果以
`OperationResult` 为准，详见 `docs/contracts/operation-result.md`。
所有运行时结果使用同一个 versioned operation envelope；旧 `ok` envelope
不再作为合法输入。

## 1. 返回值信封

所有工具经 `ToolDispatcher.call` 向 LangChain 暴露统一
`OperationResult` envelope 字符串。Provider 内部仍可使用纯文本返回值，
dispatcher 会将其包装为 `data` 字符串。

```json
{"schema_version": "1.0", "outcome": "succeeded", "data": {}}
{"schema_version": "1.0", "outcome": "timed_out",
 "error": "人类可读说明", "error_type": "tool_timeout",
 "next": "可执行的下一步", "code": "TOOL_TIMEOUT", "retryable": true}
```

纯文本工具（`read_file` / `list_dir` / `get_time` / `calculator` /
`fetch_url` 成功路径）返回「不是合法信封」的 UTF-8 文本；
`parse_tool_result()` 是唯一解析入口（先试信封，再按纯文本分流）。
截断用 `truncate_tool_result()`：信封在 `data` 内部截断以保持可解析。

`AgentError.to_tool_envelope()` 保证错误侧字段稳定：
`error_type`（分类）/ `code`（机器码）/ `next`（恢复动作）/ `retryable` /
`retry_after_seconds` / `cause_ref` / `tool_name`。
**异常堆栈只进受控审计/trace，绝不进模型上下文或前端。**

## 2. ToolSpec（注册表元数据）

| 字段 | 语义 |
|---|---|
| `name` / `version` | 稳定工具名与版本；版本进 trace 与幂等键 |
| `description` | 供模型选择工具 |
| `input_schema` / `output_schema` | JSON Schema；类型/必填/枚举失败或未声明的对象字段均拒绝 |
| `permissions` | 调用方必须全部持有的 `Permission` 集合 |
| `side_effect` | 是否改变外部状态（决定审批与幂等） |
| `idempotency_scope` | `request`（可重试/可重放）/ `none`（禁止重试与重放） |
| `timeout_seconds` | 单次调用超时；缺省用 dispatcher 兜底（130s） |
| `retry_policy` | `max_attempts / backoff_seconds / backoff_multiplier / max_backoff_seconds` |
| `owner` / `tags` | 归属与检索标签 |

`ToolRegistry.registry_hash` 是对全部 spec 的稳定摘要，写入配置快照与 trace。

## 3. 调用顺序（ToolGateway.invoke）

1. 解析 spec（找不到 → `deny`，"tool is not in the registry"）。
2. `ToolPolicy.decide`：权限矩阵 → `deny`（`TOOL_NOT_PERMITTED`）；
   `requires_approval()` 且审批开启 → `approval`（`APPROVAL_REQUIRED`）。
3. 参数校验：走 JSON Schema；缺字段、类型/枚举错误、对象未知字段 →
   `TOOL_ARGS_INVALID`（不触达 adapter）。
4. 幂等键 `hash(thread_id, tool@version, canonical_args, intent)`；有副作用的
   调用先写持久 `running` 预留，成功记录可跨进程重放
   （`deduplicated: true`）；异常后的不确定结果标为
   `SIDE_EFFECT_OUTCOME_UNKNOWN`，禁止自动重试。
5. 熔断检查：冷却期内 → `TOOL_CIRCUIT_OPEN`（不触达 adapter）。
6. 并发信号量 → `asyncio.wait_for(timeout)` → 失败按 `retry_policy` 退避重试
   （仅 `timeout / transport / rate_limited`；副作用或无幂等工具强制 1 次）。
7. 脱敏审计事件 `tool_audit`（决策、尝试次数、耗时、错误类型、幂等键摘要；
   参数只记 key 列表 + sha256 摘要 + 字节数），返回
   `ToolCallOutcome(envelope, ok, error, attempts, duration_ms, decision,
   idempotency_key, deduplicated)`。

## 4. 分类与默认策略

| 分类 | 示例 | 默认权限 | 审批 | 幂等 |
|---|---|---|---|---|
| read | 检索、读文件、读论文 | READ | 否 | 结果短缓存 |
| network_read | arXiv、HTTP fetch | NETWORK + READ | 按域策略 | 请求去重 |
| write | 写章节、上传、入库 | WRITE | 是 | 业务幂等键 |
| execute | 跑实验、shell、委托 coding agent | EXECUTE | 是 | `task_id` + 仅一次提交 |
| admin | 配置发布、索引清理 | ADMIN | 强制 | 不自动重试 |

## 5. 观察层分工

- `tool_audit`（gateway）：治理视图，逐条决策与重试。
- `tool_call`（dispatcher）：指标视图（评测 `metrics/tools.py` 依赖它），
  一次逻辑调用只发一条（重试不重复计数）。
- `tool_start` / `tool_end`（dispatcher，SSE）：父/子 agent 层级可视化。
