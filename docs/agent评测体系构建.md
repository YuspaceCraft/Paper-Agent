# Agent 评测体系设计方案（LangSmith-first 修订版）

> **结论先行**：完整链路信息（span 树 / 瀑布图 / 每步耗时 / 真实 token / 模型与成本）**用 LangSmith 白拿，远比自建简单**。
> 本项目用 `langchain_openai.ChatOpenAI` 单一入口 + LangGraph 自动插桩，`.env` 已配 `LANGSMITH_TRACING=true`，
> 一条对话的完整嵌套链路零改动就能落在 LangSmith。因此本方案**废弃原稿的"从零自建 Trace-Span-Event 采集 + ClickHouse + 瀑布图"路线**，
> 改为：**LangSmith 为链路采集中枢 + 保留本地域事件评测层 + 归档 JSONL 兜底**（决策记录见文末）。

---

## 实施状态（2026-09-15 复核）

| 阶段 | 状态 | 落地位置 |
|---|---|---|
| P0 join 键（本地 trace_id == 根 run id） | ✅ | `agent/graph.py::prepare_turn`、`agent/supervisor.py::_run_worker`、`evaluation/verify_smith.py` |
| P1 节点内 LLM/工具嵌套透传 | ✅ | `agent/nodes.py`（`_stream_llm`、`tool.ainvoke` 带 `config=`）、`agent/plan.py`、`agent/subagents.py`、`ExecutionContext.child_config()` |
| P1 run 树归档（保留期之外可复现） | ✅ | `agent/core/trace_export.py`（CLI）+ `evaluation/runner.py`（`EVAL_EXPORT_LANGSMITH=1`）→ `eval_output/runs/<trace_id>/langsmith_runs.jsonl` |
| 实时进度（过程可见、指标随执行更新） | ✅（2026-09-15） | `evaluation/live.py`（进程内事件总线）+ `evaluation/runner.py`（逐条发布 + 进度行）+ `web/api/routers/eval.py`（SSE）+ `EvalPanel.tsx`（实时进度卡）；测试 `evaluation/tests/test_live.py` |
| P1 指标切归档 + sink 退役 | ⬜ | 四类指标仍读 `trace_store.db`（`evaluation/metrics/*`） |
| P2 前端改链 | ⬜ | `/api/eval` 仍读 `trace_store.db`（`trace_events` + `eval_runs` 表）；EvalPanel/SingleFlowView 接口不变 |
| P3 回填闭环 | ⬜ | `create_feedback` 未接；badcase 回流已可用（`POST /api/eval/manifest/from-trace`） |
| P4 自研解耦 | ⬜ | 可选，未启动 |

平台侧同批落地：`docs/agent-platform-architecture.md` §13（逐项验收表）、`docs/adr/0001–0006`、
`docs/contracts/`；里程碑对照 `AGENT_EXPANSION_PROGRESS.md`（评测体系段落 + Phase E）。

## 一、现状盘点（动手前先认清已有什么）

### 1.1 已建成的本地评测层（`evaluation/`）

- **事件采集**：`agent/observability.log_event` → `evaluation/sink`（`evaluation/sink.py attach()` 幂等挂接）→
  `trace_store.db`（SQLite 扁平事件流，`evaluation/trace_store.py`；每步 `duration_ms` + 真实 token）。
- **域事件 schema**（`trace_events` 宽表 + `payload` JSON）：
  `turn_start / intent / node_start|end|error / llm_call / tool_call / retrieved_context / plan / plan_step / plan_verify / final_answer / turn_end`。
- **四类指标**（`evaluation/metrics/`，纯函数聚合）：
  - retrieval（Recall/Precision/NDCG/MRR、bootstrap CI、badcase 分类）
  - LLM-as-judge（context_hit / faithfulness / intent_accuracy，badcase 优先抽样 + 预算护栏）
  - tools（per-tool 成功率 / p50/p95 耗时 / error_type 归因）
  - task（执行成功率 + 故障注入中断恢复率）
- **API + 前端**：`web/api/routers/eval.py`（traces/{id} · thread/{thread_id} · runs 报告/badcases · single 单样例 · manifest/from-trace · prune）+ 前端
  EvalPanel / SingleFlowView（分阶段线性时间线）。
- **回归闭环**：`eval_output/runs/.baseline.json` 回归基线（降幅 >5% warn、Recall@5<0.6 block）、badcase 回流评测集。

### 1.2 LangSmith 通道（P0 join 键 + P1 嵌套/归档已打通）

- `.env` 已置 `LANGSMITH_API_KEY` / `LANGSMITH_TRACING=true` / `LANGSMITH_SAMPLING_RATE=1.0` / `LANGSMITH_PROJECT=paper-agent`；
  `agent/graph.py:24-29` 与 `web/api/main.py:14-21` 刻意在 langchain 导入前 `load_dotenv()`，保证自动插桩回调在 import 期注册。
- **单一 LLM 入口**：全部 agent LLM 调用经由 `agent/nodes.py:63 _get_model`（`langchain_openai.ChatOpenAI`
  ，base_url=DashScope compatible-mode）→ 自动插桩全覆盖。（`agent/memory.py:108` 一处内联同型 client，同样可捕获。）
- **嵌套覆盖**：LangGraph 节点输入输出快照、StructuredTool 工具调用（参数/输出/错误）、子 agent
  （`agent/subagents.py:213-225` 已做 run config 透传，`plan.py:528` 注释即为此意）都会进同一 trace。
- **join 键已打通（P0 完成）**：`agent/graph.py::run` 用 `RunnableConfig.run_id=uuid4()` 强制顶层根 run，
  `set_trace_id(run_id.hex)` 使本地 `trace_id` 与 LangSmith 根 run **同值同键**；`agent/supervisor.py::_run_worker`
  的子 agent 舱顶层 ainvoke 同样注入新 run_id。验证脚本 `evaluation/verify_smith.py` 可一键确认。

> **P1 已闭合（2026-09-15）**：上段 P0 实测的「节点内 LLM/工具调用以 parent=None 游离」缺口已修复——
> `agent/nodes.py` 的 `_stream_llm`（`model.astream(..., config=config)`）与 `tool.ainvoke(..., config=config)`、
> `agent/plan.py`（plan / coding plan / verify 及工具分支）、`agent/subagents.py`（`traced_ainvoke` 调用点）全部
> 透传父节点 runnable config；每轮 `ExecutionContext` 用 `child_config()` 派生 child config，新调用点照此办理
> （ADR-0002）。结果：节点 / LLM / 工具 / 子 agent 并入同一 graph trace，LangSmith 瀑布树无缝；保留期之外可用
> `agent.core.trace_export` 把 run 树归档成 `langsmith_runs.jsonl` 复现。回归：`agent/tests/test_dispatcher.py`、
> `agent/tests/test_core_contracts.py`、`evaluation/tests/*`。

### 1.3 未覆盖边界（非 agent 评测对象）

raw `openai.OpenAI`（非 LangChain runnable）路径——`pdf_pipeline/enhancer.py`、`retrieval_orchestrator/*`、
`indexer/embedding_adapters.py`（API-embedding / HyDE）。这些是离线/索引链路，不参与 agent 评测；如需纳入，
后续可用 `@traceable` 显式包裹（本期不做）。

---

## 二、选型结论：LangSmith vs 自建

| 维度 | LangSmith（本方案） | 自建采集 + ClickHouse + 瀑布图（原稿） |
|---|---|---|
| 链路采集 | **零代码**（autoinstrumentation 已就位） | 逐层埋点 SDK + contextvars 透传 |
| span 树 / 瀑布 / 时间线 | **现成控制台**（检索、过滤、对比） | 自建 P1 可视化 |
| token / 成本 / 模型复现 | 原生记录 | 自建物化聚合 |
| judge 结果回填 | `client.create_feedback` 原生 | 自建评测回填机制 |
| 数据主权 | 论文/query **出网**（已决策接受） | 全本地 |
| 保留期 | 免费层约 7 天（→ 归档兜底，见 §五） | 无限制 |
| 离线/断网评测 | **不可用** | 可用 |
| 周期 | 1-3 天（含最小验证） | P0-P4 数周 |

**推荐 LangSmith-first 彻底迁移**（用户决策）：本地 SQLite 事件库退役，域事件也进 LangSmith。
因 LangGraph 自动把每个 node 的**输入输出快照**进 trace，所谓"域事件"（intent / plan / verify / 检索上下文）多数
本身就是 node output 或工具输出，**无需额外埋点**（映射见 §四）。

---

## 三、数据模型（精简版）

保留原稿 Trace-Span-Event 思想，但 **Span 树由 LangSmith Run Tree 原生承担**，不再自建：

```
thread_id（会话，多轮共享）      ──►  LangSmith session_id
  └── run_id（一次任务 = graph.run 一次顶层 ainvoke）  ＝ 本地 trace_id（同值同键）
        └── 嵌套 run 树：graph 节点(chain) → LLM(llm) / 工具(tool) / 子 agent(subgraph)
              └── token/latency/error 由 LangSmith 逐 run 记录
```

- `step_index`：不再自建（LangSmith run 树天然有序，`dotted_order` 表达因果与外层）。
- `layer`（user/system/tool）：不再自建（run_type = chain/llm/tool 原生区分；user 层信息挂 root run 的 inputs/metadata）。
- `payload_ref`（冷热分离）：LangSmith 持久化完整 inputs/outputs；本地归档才考虑裁剪（见 §五）。

**回收的原稿设计**：ID 前缀规范、`parent_span_id` 重建树、ClickHouse 物化视图 `trace_summary`、
四索引先行、TTFT/reasoning_trace 等 SQL 字段——全部不需要了。

---

## 四、域事件 → LangSmith 映射表（证明"本地事件库可替代"）

| 现有本地事件 | LangSmith 对应物 | 还需做什么 |
|---|---|---|
| `turn_start` / `turn_end` | root run（start/end + 总耗时）；turn 总 token | —（join 键 P0 已做） |
| `final_answer` | root run outputs 的 messages 里最后一个无 tool_calls 的 AIMessage | — |
| `intent` | `understand` node 的 output（`UnderstandResult`） | — |
| `llm_call`（tokens/耗时） | 每个 run_type=llm 的 run，token 在 `extra.metadata.usage` | **P1：`_stream_llm`/`traced_ainvoke` 透传 runnable config**，令其嵌套进 run 树（否则为游离根 run） |
| `tool_call`（成败/参数/结果） | 每个 run_type=tool 的 run（error 即失败） | **P1：`dispatcher` 调用 `tool.ainvoke(args, config=…)`**，同上 |
| `retrieved_context`（chunk_ids） | `search_papers`/`fetch_content` 工具 run 的 output（信封含 chunk_ids） | 指标层从 output 解析即可；稳妥起见可把 chunk_ids 显式写进工具 run 的 metadata |
| `plan` / `plan_step` | `plan`/`executor` node 的 output（`plan_progress`） | — |
| `plan_verify` | `verify` node 的 output（`verification`） | — |
| `result_used_flag`（工具结果是否被最终采纳） | 运行后分析：最终回答是否引用该工具结果 / 后续 LLM 是否延续 tool call | 仍属评测层回填（judge 启发式），同原稿 |
| 用户反馈 👍/👎/评分 | `client.create_feedback(run_id, ...)` | P3 阶段 |

**结论：除 `result_used_flag` 外，全部域事件都能从 LangSmith run 树免费重建。** 本地评测层只需保留
"指标聚合 + badcase 归因"这两块领域逻辑，数据源换到 LangSmith（或其归档）。

---

## 五、存储与保留期策略

```
Agent 运行 ─►（自动插桩）─► LangSmith SaaS（链路唯一事实源；7 天保留）
                 │
                 ▼  批量评测每轮跑完立即 list_runs 抓取
          eval_output/runs/<run_id>/langsmith_runs.jsonl   ← 本地归档（指标/回归读这里）
                 │
                 ▼
          metrics（retrieval/judge/tool/task，纯函数不变）+ .baseline.json 回归基线
```

- **LangSmith = 活链路浏览/可视化/即时分析**；**本地归档 = 评测结论的事实源**。
- 归档就是 JSONL 快照，非事件库：指标层新增一个 LangSmith-Runs-Loader，把 run 树扁平化为与原
  `trace_events` 等价的事件序列（§四映射的逆过程），四类指标纯函数**原样复用**。
- `eval_output/runs/`、`.baseline.json`、datasets registry 本就是文件并非 SQLite，**不受迁移影响**。
- 断网/抖动：采集 best-effort（SDK 异步队列 + 重试）；跑批归档失败会留下明文 gap（报告注明）。

---

## 六、采集面（最小代码清单）

| 项 | 状态 | 说明 |
|---|---|---|
| join 键（`graph.run` 注入 run_id） | ✅ P0 已做 | `agent/graph.py::run`：`config["run_id"]=uuid4()`，`set_trace_id(run_id.hex)` |
| 子 agent 舱 run_id | ✅ P0 已做 | `agent/supervisor.py::_run_worker`：`{**config, "run_id": uuid.uuid4()}` |
| 验证脚本 | ✅ P0 已做 | `evaluation/verify_smith.py`：一条对话 → `get_run` 断言同键 → 结构统计 → 控制台链接 |
| **节点内 LLM/工具嵌套透传** | ✅ P1 已做（2026-09-15） | `nodes.py` 的 `_stream_llm` / `tool.ainvoke`、`plan.py`、`subagents.py` 全量透传父节点 config；新调用点用 `ExecutionContext.child_config()` → LLM/工具 run 并入 graph trace，游离根 run 消失 |
| `retrieved_context` 显式 metadata | 🟡 指标侧已做 | `dispatcher._trace_tool` 解析信封后 `emit_retrieved_context(chunk_ids=…)`（即「指标层直接解析 output」路线）；挂 LangSmith run.metadata 未做，不影响检索指标 |
| raw-openai `@traceable` | ⬜ 可选 | 仅 pdf_pipeline/retrieval_orchestrator/indexer 需要时 |
| sink 退役 | ⬜ P1-P2 | 指标仍读 `trace_events`；归档导出已就位（`agent.core.trace_export` + 跑批 `EVAL_EXPORT_LANGSMITH=1`），切换后即可摘掉 `graph.run` 的 `_eval_sink.attach()` 与 `set_thread_map` |

---

## 七、指标与评测闭环（承接原稿 §五，数据源替换）

沿用四层指标金字塔（L1 结果 / L2 过程 / L3 组件 / L4 诊断），实现改动集中在**读取层**：

1. **规则评测**：从归档 run 树聚合——工具调用次数超限、token 预算、耗时 SLO，与现状等价。
2. **LLM-as-judge**：`evaluation/metrics/judge.py` 不变；badcase 优先抽样从 run 树选 trace，喂裁判模型。
3. **结果回填**：`client.create_feedback(run_id, key="judge_faithfulness", score=…)` 挂回 LangSmith（原稿 P3 的
   "评测结果回填"，白拿）；同时写本地归档便于回归比对。
4. **回归评测集**：`/api/eval/manifest/from-trace` 保留——改为从 LangSmith trace 挑 badcase（保留期内）或从归档挑。
5. **失败归因**：对含 error run 的 trace，按首个异常 run 的 run_type（llm/tool/chain）聚类。

### 7.1 实时（过程）指标：指标随执行更新，不再「跑完才知道」

跑批是长任务（几十条 × 分钟级），只读跑完的 `run_summary.json` 意味着第 3 条就坏掉的检索
要等到最后才暴露。因此指标链路多了一条**过程通道**，与最终报告同源、同口径：

```
runner._run_impl 每完成一条 QA
  ├─ publish(query_finished)  单条明细（category / recall@5 / mrr / 耗时 / tokens / trace_id）
  ├─ publish(aggregate)       累计指标（done/total/eta + overall + recent[-10]）
  └─ upsert eval_runs 进度行  status=running + overall.progress={done,total}
         │
         └─► evaluation/live.py::RunFeed ──► SSE /api/eval/runs/{id}/stream ──► 前端实时进度卡
                                        └─► CLI `python -m evaluation run` 逐条打印
                                        └─► 列表/报告页轮询（不依赖 SSE）
```

- **口径保证**：实时聚合与 `assemble_report` 调用同一批纯函数
  （`metrics.retrieval.aggregate` / `metrics.tools.aggregate_tools` / `report.estimate_cost`），
  所以「界面上的数」= 「报告里的数」（`evaluation/tests/test_live.py` 断言逐键相等）。
- **不阻塞主链路**：publish 吞掉一切异常、慢订阅者丢最旧事件；总线只在内存、只覆盖本进程的 run；
  非本进程（CLI / 多 worker）的 run 由 2s 轮询 `eval_runs` 行的回退路径补齐，事件同构。
- **与保留期/归档解耦**：进度通道读的是本地 `eval_runs` 行与内存事件，P1「指标切归档」落地后
  只需替换聚合的输入源，SSE/前端契约不变。

---

## 八、风险与缓解

| 风险 | 缓解 |
|---|---|
| 论文/query 出网（数据主权） | **已决策接受**；必要时索引/密钥不落 prompt 快照 |
| 免费层 7 天保留 | 每次跑批立即归档 `langsmith_runs.jsonl`，结论读归档；留档永久 |
| 断网/抖动丢 trace | SDK 异步队列 + 重试（best-effort）；报告标注 gap；本地对话本身不阻塞 |
| LangGraph/LangChain 升级断自动插桩 | CI 跑 `evaluation/verify_smith.py` 回归 |
| run_id 在子流程复用导致 lineage 串 | 顶层 ainvoke 每次新 run_id（不污染共享 config），子图嵌套由框架继承 parent |

---

## 九、实施阶段表（替换原稿 P0-P4）

| 阶段 | 周期 | 交付物 |
|---|---|---|
| **P0 打通 join 键** ✅ | 0.5 天 | `graph.run`/`supervisor` 注入 run_id；`verify_smith.py` 单条对话验证"根 run id == trace_id" |
| **P1 嵌套透传** ✅（2026-09-15） | 2-3 天 | ① 节点内 LLM/工具调用透传 runnable config（`_stream_llm`/`traced_ainvoke`/plan/subagents），run 树并入 graph trace ✅ |
| **P1 归档导出** ✅（2026-09-15） | — | ② `agent.core.trace_export` + 跑批 `EVAL_EXPORT_LANGSMITH=1` 抓 `eval_output/runs/<trace_id>/langsmith_runs.jsonl` ✅（`AGENT_TRACE_EXPORT_FULL=1` 才含原始 I/O） |
| **P1 剩余：指标切归档 + sink 退役** | 1-2 天 | ③ LangSmith-Runs-Loader 扁平化 → 四类指标复用；摘掉 `_eval_sink.attach()` 与 `set_thread_map` ⬜ |
| **P2 前端改链** | 1-2 天 | EvalPanel/SingleFlowView 数据源切到归档（或 LangSmith API）；线程语义不变——现仍读本地库 ⬜ |
| **P3 回填闭环** | 1-2 天 | `create_feedback` 挂 judge/人工标注；from-trace 从 LangSmith/归档挑 badcase 回流（端点已可用）⬜ |
| **P4 可选解耦** | — | trace 量沉淀后自研导出，摆脱 SaaS 依赖（届时才考虑 ClickHouse） |

---

## 十、踩坑清单（承接原稿，按本项目改写）

1. **join 键先行**：`RunnableConfig.run_id` 只能在顶层 ainvoke 入口注入，子内层 config 会 pop/inherit——
   拼地方会静默失效（已在 `graph.py:226` 与 `supervisor.py:220` 两处根入口做）。
2. **不要给子图/工具传 run_id**：它们应继承 parent run；只有"新的独立任务"（supervisor 舱、新一次用户请求）才新起。
3. **归档先于保留期**：指标结论只信归档 JSONL；LangSmith 只当活链路浏览器。
4. **采集绝不上热路径阻塞**：langsmith SDK 与本地 sink 都异步 best-effort；评测系统自己先过 SLO。
5. **升级回归**：LangChain/LangGraph 升级后必须跑 `verify_smith.py`（自动插桩断点很多）。
6. **原始问题收敛**：LangSmith 记录完整 prompt/工具 I/O，属敏感数据，仓库共享账号前先评估（脱敏管道仅在 P4 自研时考虑）。

---

## 决策记录（2026-09-08）

- 链路数据托管：**LangSmith SaaS**（接受出网）。
- 本地评测层：**彻底迁移**——SQLite 事件库退役，域事件进 LangSmith（映射见 §四）。
- 交付：**文档重写 + join 键最小验证**（P0 完成，代码见 §六）。

关键文件：`agent/graph.py`、`agent/supervisor.py`、`evaluation/verify_smith.py`、`evaluation/trace_store.py`、
`evaluation/metrics/`、`web/api/routers/eval.py`、`web/frontend/src/renderer/src/components/{EvalPanel,SingleFlowView}.tsx`。
