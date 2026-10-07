# ADR-0002：LangSmith 在线追踪与数据脱敏

- 状态：Accepted
- 日期：2026-09-15
- 相关：`agent/graph.py`、`agent/core/execution_context.py`、`agent/core/trace_export.py`、`evaluation/verify_smith.py`

## 背景

链路事实源此前有两个：LangSmith 与本地 `trace_store.db`。两者口径不同
（本地有评测事件、LangSmith 有真实 span 树与 token），排障时需要人工对账；
同时早期节点内的 LLM/工具调用未透传父 `RunnableConfig`，LangSmith 上会出现
游离根 run，瀑布图不完整。P1 已知缺口：eval 路径的本地 `trace_id` 是确定性
qa 键，而 LangSmith 根 run 是随机 uuid，两者无法直接互查。

## 决策

1. **在线运行树只认 LangSmith**；本地 `trace_store` 只保存评测归档、badcase
   与脱敏审计摘要。
2. **join 键固定**：根 graph run 的 `run_id` 即本地 `trace_id`。线上路径二者
   同值（`prepare_turn` 生成 `run_id = uuid4()`，`trace_id = run_id.hex`）；
   评测路径本地 `trace_id` 是确定性 qa 键，故把**真实的 LangSmith run id**
   一并写入 `trace_events.run_id`（`set_thread_map(..., run_id=run_id.hex)`），
   导出时按 `trace_id` 反查。
3. **所有 node / model / tool 调用必须携带父 config**：新增调用点用
   `ExecutionContext.child_config()` 派生 child config，禁止裸调
   `model.ainvoke/astream` 或 `tool.ainvoke`。
4. **metadata 契约**（每轮根 run 必带）：
   `trace_id / thread_id / request_id / actor_id / actor_role / domain /
   graph_version / config_revision / config_hash / prompt_versions /
   prompt_bindings / tool_versions / model_route / eval_dataset_id`。
5. **默认只上传脱敏摘要**：参数只记录 key 列表、sha256 摘要与字节数
   （`agent/core/tool_gateway.py::redact_args`）；完整输入输出仅在明确允许的
   研发环境用 `AGENT_TRACE_EXPORT_FULL=1` 导出。
6. **保留期之外的复现靠导出**：`python -m agent.core.trace_export <trace_id>`
   把 run 树写进 `eval_output/runs/<trace_id>/langsmith_runs.jsonl` +
   `langsmith_manifest.json`；评测跑批用 `EVAL_EXPORT_LANGSMITH=1` 自动导出，
   失败只告警。

## 替代方案

- **只用本地 trace**：无法复现原生 span 与 token，也无法用 LangSmith 的
  dataset/feedback；已否决。
- **双写两套 run 树**：口径必然分叉，排障要先判断「哪份对」；已否决。
- **eval 阶段用 uuid5(trace_id) 强制同值**：join 键更整齐，但同一 qa 重跑会
  复用同一 run id，LangSmith 侧会合并不同批次的 run；已否决。

## 兼容性

- `verification/evaluation` 现有事件名不变；`tool_audit` 是新增事件，指标聚合
  仍只看 `tool_call`，不会重复计数。
- `verify_smith` 的断言（根 run id == 本地 trace_id）在线上路径继续成立；
  评测路径改用 `trace_events.run_id` 做映射，脚本行为不变。

## 回滚

- 关掉 LangSmith（`LANGSMITH_TRACING` 未开启）时全部功能退化为本地事件；
  导出是 best-effort，失败不影响对话与评测报告。

## 验收指标

- `trace completeness`：根 run 存在且关键 node/LLM/tool 父子关系完整；
  游离根 run 数 = 0（`verify_smith` 的 detached run 统计）。
- 每次评测/发布的 run 树可在保留期外由 `langsmith_runs.jsonl` 复现。
- 审计日志中不出现凭据、原始参数值与用户敏感原文（抽查 + 单测断言）。
