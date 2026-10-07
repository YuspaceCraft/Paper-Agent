# 错误码与容错契约

`error_type` 是分类（`agent/core/contracts.py::ErrorType`），`code` 是**稳定机器码**
（测试、告警与 UX 的接口）。堆栈只进受控审计/trace。

## 1. 分类 → 策略 → 用户可见行为

| error_type | 例子 | 重试 | 用户可见行为 | 指标 |
|---|---|---|---|---|
| `validation` | 参数非法、schema 不符 | 否 | 说明缺什么、怎么补 | 拒绝率、误拒绝样本 |
| `authorization` | 角色缺权限 | 否 | 说明缺什么权限 | 拒绝率 |
| `policy` | 需要审批、策略禁止 | 否 | 说明需要确认 | 审批数、拒绝数 |
| `tool` | 工具内部逻辑错误 | 否 | 说明失败并可换工具 | 工具错误率 |
| `tool_timeout` | 单次调用超时 | 是（有界退避） | 可提示重试中 | 超时率、恢复率 |
| `tool_unavailable` | 后端不可达、熔断 | 是（熔断后停止） | 降级回答 + 任务状态 | P95、可用性 |
| `tool_rate_limited` | 429 | 是（等待后重试） | 可提示稍后 | 重试次数 |
| `model` | 模型限流、结构解析失败 | 切备模型 / 安全模板 | 说明能力受限，不伪造结果 | 模型失败率 |
| `subagent` | 子代理无输出或执行失败 | 否 | 说明子任务失败并可重试 | 子代理失败率 |
| `task` | 后台任务失败/取消 | 否 | 返回任务稳定错误码 | 任务失败率 |
| `protocol` | envelope/schema 不符合契约 | 否 | 说明结果协议错误 | 协议违规率 |
| `agent_runtime` | 节点未捕获异常 | 否 | 通用失败提示 | 未分类率 |
| `graph_timeout` | 整轮超时 | 否（可恢复） | 返回 trace/task 参考 | 超时率 |
| `context_budget` | 上下文预算不足 | 否 | 说明被截断/需缩小范围 | 截断次数 |
| `checkpoint` | 持久化失败 | 否（安全终止） | 可恢复提示 | 恢复率 |
| `dependency` | 下游依赖故障 | 视情况 | 降级说明 | 依赖错误率 |
| `transient` | 暂时性网络失败（兼容值） | 是 | 可提示重试中 | 恢复率 |
| `unknown` | 未分类 | **禁止自动副作用重试** | 通用失败提示 | 未分类率 |

## 2. 机器码清单（`code`）

| code | error_type | 触发点 |
|---|---|---|
| `TOOL_TIMEOUT` | tool_timeout | `tool_timeout()`；`asyncio.wait_for` 超时 |
| `TOOL_UNAVAILABLE` | tool_unavailable | 连接/OS 级失败（`tool_exception`） |
| `TOOL_CIRCUIT_OPEN` | tool_unavailable | 熔断冷却期内调用 |
| `TOOL_EXECUTION_FAILED` | tool | adapter 抛非传输类异常 |
| `TOOL_PROTOCOL_INVALID` | protocol | envelope 非布尔 `ok`、矛盾 outcome、截断 JSON |
| `TOOL_OUTPUT_INVALID` | protocol | 成功输出不满足 `output_schema` |
| `TOOL_ARGS_INVALID` | validation | JSON Schema 校验失败（缺失、类型、枚举、未知字段等） |
| `TOOL_NOT_PERMITTED` | authorization | 权限矩阵判定 deny |
| `APPROVAL_REQUIRED` | policy | 副作用工具需人工确认 |
| `APPROVAL_DENIED` | policy | 用户在审批点拒绝本次副作用操作 |
| `IDEMPOTENCY_STORE_UNAVAILABLE` | checkpoint | 无法建立持久幂等预留，副作用调用 fail-closed |
| `IDEMPOTENCY_RECORD_FAILED` | checkpoint | 副作用已执行，但成功结果未能持久记录 |
| `IDEMPOTENCY_IN_PROGRESS` | transient | 同一幂等键已有调用正在执行 |
| `SIDE_EFFECT_OUTCOME_UNKNOWN` | checkpoint | 上次副作用结果不确定，禁止自动重试 |
| `SIDE_EFFECT_PREVIOUSLY_FAILED` | validation | 上次副作用明确未执行，同一幂等意图拒绝重放 |
| `SUBAGENT_EMPTY_OUTPUT` | subagent | 子代理未生成最终结果 |
| `SUBAGENT_EXECUTION_FAILED` | subagent | 子代理执行异常 |
| `TASK_EXECUTION_FAILED` | task | 后台任务执行异常 |
| `TASK_CANCELLED` | task | 后台任务被取消 |

工具自身还可返回领域码（如 `param_error`、`not_found`、
`unsupported_operation`），由 `tool_contract.err(error_type=...)` 产出，
与上表同属一个信封。

## 3. 重试与熔断的实现位置

- 重试：`ToolSpec.effective_retry_policy()`（所有副作用工具强制 1 次；
  无副作用且非幂等的工具也强制 1 次）+ gateway 退避循环。
- 持久幂等：副作用工具先写 `running` 预留；成功写 `succeeded` 并可跨进程重放；
  异常写 `indeterminate` 并停止自动重试。存储位置由
  `AGENT_IDEMPOTENCY_DB` 配置，默认 `tool_idempotency.db`。
- 熔断：gateway `_breaker_error`/`_record_failure`，只统计传输类失败，
  阈值 `AGENT_TOOL_BREAKER_THRESHOLD`（默认 3）、冷却
  `AGENT_TOOL_BREAKER_COOLDOWN`（默认 30s）。
- 整轮超时：`agent.graph.TURN_TIMEOUT` / API `_TURN_TIMEOUT`
  （`AGENT_TURN_TIMEOUT`，默认 900s）。

## 4. 观测

- 每个错误都带 `tool_name`（可定位工具）与 `idempotency_key`（可定位调用）。
- 评测侧 `metrics/tools.py` 优先用信封里的 `parsed.error_type` 归因，
  不依赖翻译文案或异常字符串。
