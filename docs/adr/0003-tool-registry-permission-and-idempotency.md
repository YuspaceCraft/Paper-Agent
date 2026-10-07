# ADR-0003：工具注册表、权限、审批与幂等

- 状态：Accepted
- 日期：2026-09-15
- 相关：`agent/core/contracts.py`、`agent/core/tool_registry.py`、`agent/core/policy.py`、`agent/core/tool_gateway.py`、`agent/dispatcher.py`

## 背景

工具此前直接由 Provider 暴露，超时/重试/审计散在 `dispatcher.py`，「哪些工具
有副作用、需要谁授权、能否重试」只以 `annotations` 形式隐含存在，且判断逻辑
会渗进节点与 prompt。结果是：无法在调用前统一拒绝、无法保证副作用不重复执行、
错误分类与 UX 不一致。

## 决策

1. **声明式注册表**：`ToolDef` → `ToolSpec`（`agent/core/tool_registry.py`）。
   `ToolSpec` 携带 `name/version/description/input_schema/output_schema/
   permissions/side_effect/idempotency_scope/timeout_seconds/retry_policy/
   owner/tags`。工具名判断只出现在注册表与策略中，不写进 prompt/node。
2. **唯一入口**：`ToolGateway.invoke(spec, args, ctx)` 固定七步
   —— 版本与 schema 校验 → 权限/审批判定 → 幂等键 → 并发/配额/超时/熔断/重试
   → 调用 adapter → 统一信封 → 脱敏审计。`ToolDispatcher.call` 委托 gateway，
   自身只保留 SSE 与评测事件（UI/指标观察层）。
3. **权限矩阵是数据**：角色 → 权限集合（`ROLE_PERMISSIONS`）。工具声明的
   `permissions` 必须被调用方全部持有，否则 `TOOL_NOT_PERMITTED`（拒绝且
   不触达 adapter）。默认本地单用户角色 `user` 持有非 admin 全部能力。
4. **审批闸门默认关闭副作用**：`ToolPolicy.decide` 对 `requires_approval()` 的工具返回
   `approval`；graph node 会先在执行任何调用前批量调用 LangGraph
   `interrupt()`，前端/API 批准后以 `Command(resume=...)` 从 checkpoint 续跑，
   拒绝则返回 `APPROVAL_DENIED` 且不触达 adapter。无 graph 上下文时仍返回
   `APPROVAL_REQUIRED`（fail-closed）。仅隔离测试/可信自动化可显式设置
   `AGENT_TOOL_APPROVAL=0`。操作员也可用 `preapproved` 白名单。
5. **幂等键**：`idempotency_key = hash(thread_id, execution_id, tool@version,
   canonical_args, intent)`（摘要，不含原文）。每轮每个副作用
   调用都会写入 at-most-once journal；中断/崩溃后按同一 execution_id 取回已成功
   结果，不重复执行。
6. **重试限定在可复现工作**：只有 `timeout / transport / rate_limited` 类错误
   重试，且非幂等工具**强制 1 次尝试**（`effective_retry_policy`）。退避为
   有界指数（1s、2s，上限 8s）。
7. **熔断**：连续 transport 类失败达阈值（默认 3）后进入冷却（默认 30s），
   期间返回 `TOOL_CIRCUIT_OPEN` 而不再打后端；冷却结束放行一次探测。

## 替代方案

- **把权限写进 prompt 让模型自律**：不可验证、不可审计；已否决。
- **在 provider 内各自实现重试**：无法统一限流与幂等，且副作用工具容易被重放；
  已否决。
- **dispatcher 直接删除、全部改 gateway**：会丢失 SSE 层级事件与评测事件
  （`tool_call`/`retrieved_context`）；因此保留 dispatcher 为观察层。

## 兼容性

- 工具 adapter 签名不变：`call_fn(name, args) -> Any`，返回值仍由
  `tool_contract.ok/err` 封装；env 信封兼容旧消费方。
- 默认行为与旧 dispatcher 一致（幂等工具 3 次尝试、超时 130s 兜底、非幂等
  1 次），因此现网工具无需改动即可接入。

## 回滚

- 把 `ToolDispatcher.call` 指回 `self._call_fn(name, args)` 即回到旧链路；
  `ToolSpec`/`ToolRegistry` 仍是只读元数据，不影响运行。

## 验收指标

- 工具 schema 合规 100%；副作用重复执行 = 0（幂等键重放测试）。
- 所有工具调用都经唯一入口（代码检索：无 provider 直调）。
- 熔断/超时/重试决策可在 trace 的 `tool_audit` 事件中逐条复现。

## 后续（本 ADR 未覆盖）

- 工作区级别的工具开关与角色绑定：配置中心已有停用列表，尚未映射到
  `Permission` 粒度。
