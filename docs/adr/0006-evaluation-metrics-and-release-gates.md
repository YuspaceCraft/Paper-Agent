# ADR-0006：评测指标与发布门禁

- 状态：Accepted
- 日期：2026-09-15
- 相关：`evaluation/`、`evaluation/metrics/*`、`evaluation/report.py`、`evaluation/verify_smith.py`、`docs/agent评测体系构建.md`

## 背景

评测体系（数据集 manifest、检索/工具/任务/裁判四类指标、baseline 对比、
报告落盘）已经存在，但缺少「什么情况下不允许发布」的明确阈值与 trace 完整性
前置条件：指标好而链路不可解释时，回归无法归因；trivial 的「有非空回答」
也可能被判成功。

## 决策

1. **指标分组**：任务（完成率/恢复成功率/成本）、检索（Recall@K、Precision@K、
   MRR、NDCG@K、覆盖率、引用正确率）、回答（faithfulness、answer relevance、
   citation correctness、结构化输出有效率）、运行（trace completeness、
   P95 latency、token/cost、timeout/error rate）。
2. **数据集与配置随报告固化**：评测集、qrels、模型、prompt 版本、索引/embedding
   版本写入报告元数据；`ConfigurationSnapshot` 与 `prompt_bindings` 进报告，
   保证「同一报告 = 同一配置组合」。
3. **langsmith run 树随报告归档**：`EVAL_EXPORT_LANGSMITH=1` 时把每条
   trace 的 run 树导出到 `eval_output/runs/<run_id>/langsmith_runs.jsonl`
   （见 ADR-0002），使结论在 LangSmith 保留期之后仍可复现。
4. **任务成功判定按验收条件**：任务类样例必须有确定性验收（章节落盘、
   实验产出、引用存在），不以「有非空回答」判定成功。
5. **分层测试矩阵**：单元（工具 adapter、错误映射、权限矩阵、prompt schema、
   context budget）→ 契约（统一信封、`ToolSpec` 与 JSON schema 一致、旧版本
   兼容）→ 图（路由、interrupt/resume、retry、checkpoint 恢复、幂等）→
   集成（mock LLM/MCP 的端到端 trace）→ 离线评测 → 线上反馈回流。
6. **发布阈值（初始）**：trace completeness 100%；工具 schema 合规 100%；
   副作用重复 0；核心检索指标相对 baseline 下降 ≤ 3%；任务完成率下降 ≤ 2%；
   P95 与单位任务成本有明确预算。阈值按评测集规模标注置信区间，避免小样本
   误阻塞。

## 替代方案

- **只看 LLM judge 分数**：judge 本身有版本漂移，且对检索/链路问题不敏感；
  保留 judge 作为双轨之一，不作为唯一门禁。
- **把 LangSmith 指标直接当门禁**：保留期与采样策略使其不可复现；离线报告
  才是事实源（ADR-0002）。
- **固定绝对阈值**：不同数据集/模型不可比；改为相对 baseline 的下降幅度。

## 兼容性

- 现有 `evaluation` CLI 与报告结构不变；新增字段（配置快照、prompt 绑定、
  导出清单）为加性。
- `tool_audit` 事件不参与指标聚合，因此指标口径与历史报告可比。

## 回滚

- `EVAL_EXPORT_LANGSMITH` 默认关闭，去掉该 env 即回到纯离线报告流程。
- 门禁阈值写在报告对比逻辑之外（发布 checklist），调整不需要改代码。

## 验收指标

- 每次合并都能回答：指标是否达标、是否可复现（配置/数据集/run 树齐备）。
- badcase 可回流：线上反馈与坏例进入下一轮回归集，并有脱敏与审核记录。
