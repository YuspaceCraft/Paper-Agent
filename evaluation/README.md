# evaluation — Agent 评测体系（评估端）

> 记录 Agent 全链路信息流（用户询问 → 意图理解 → 工具调用 → 返回结果 → 最终回答），
> 每步耗时与真实 token 消耗结构化落库；四类指标：检索（Recall/Precision@K、MRR、
> NDCG@K）、LLM-as-judge（上下文命中/忠实度/意图与模式决策）、工具调用（成功率 +
> badcase 归因）、任务执行（执行成功率 + 中断恢复率）。badcase 归纳指导改进与错误定位。
>
> **链路采集（LangSmith-first，P0 已打通）**：LangChain/LangGraph 自动插桩把完整 run 树
> （节点/LLM/工具/子 agent + token + 耗时）记录到 LangSmith，本地 `trace_id` 与根 run 同值同键
> （join 键）。本目录为**评测端**：四类指标 + badcase 归因 + 回归基线。迁移路线详见
> `docs/agent评测体系构建.md`（LangSmith-first 修订版）。验证脚本 `python -m evaluation.verify_smith`。

## 任务型基准 v3（RAG + 论文操作）

当前新增 `eval_output/datasets/agent_rag_paper_ops_v3.jsonl`：4 类任务
（`rag_retrieval / paper_download / paper_read / paper_ingest`）× 3 难度
（easy/medium/hard）× 3 任务长度（short/medium/long），共 36 条。

与旧检索 manifest 的差异：

- 非检索任务显式 `evaluation.retrieval=false`，不进入 Recall/MRR 均值；
- 每条任务有 `required_tools / forbidden_tools / tool_args` 和 artifact 验收；
- 每条任务有 step/tool/token/time 预算，超预算会进入契约失败；
- 报告新增 `task_dimension`，按任务类型、难度、长度拆任务成功率、延迟与 token；
- 入库等异步副作用在 manifest 中声明 `postconditions`，需隔离 harness 执行。

详细设计见 `docs/agent任务评测数据集v3.md`，生成命令：

```bash
conda run -n demo python -m evaluation.task_dataset
```

## 架构

```
agent/observability.log_event ─sink 注册─► evaluation/sink ─► asyncio.Queue ──┐
  （node/tool/error 事件零改动透传）                        │ 批量 flush（50/1s）│
evaluation/events.py 显式事件（intent/llm_call/           ▼                    │
  retrieved_context/plan/final_answer/turn_end）      trace_store.db            │
                                                        │                     │
  evaluation/runner.py（CLI 跑批） / web/api/routers/eval.py ─┴─► 前端评测页
        │ 每完成一条 QA：query_finished + aggregate（累计指标）
        ▼
  evaluation/live.py（进程内事件总线 RunFeed）
        ├─► SSE   GET /api/eval/runs/{run_id}/stream  → EvalPanel 实时进度卡
        ├─► CLI   跑批时逐条打印累计指标（与 SSE 同源）
        └─► eval_runs 进度行（status=running + overall.progress）→ 列表/报告页轮询
```

**三个单一插桩点**（探索确认，侵入最小）：
- LLM：`nodes.py _stream_llm`（流式）外包计时 + 读真实 `response_metadata.token_usage`；
  非流式 5 个点（understand/`_ask_for_plan`/`_verify_goal`/`_subagent_synthesize`/
  `memory._summarize_with_llm`）换 `evaluation.trace_wrap.traced_ainvoke`。
- 工具：`dispatcher.ToolDispatcher.call` 唯一必经点——成功/失败现场（异常类型/堆栈/
  结果快照）＋检索类工具（search_papers/fetch_content）写 retrieved_context（chunk_ids
  供检索指标）。
- 节点：`observability.timed` 装饰器（`log_event` 自动进 trace）；resolve/context 已补。
- LangSmith：自动插桩覆盖同一批调用，run 树与本地事件 share 同一 trace_id（join 键）。

## LangSmith 链路（P0 打通，P1 嵌套已闭合）

- 全链路 run 树由 LangGraph 自动插桩录入 LangSmith（`.env` `LANGSMITH_TRACING=true`，
  `agent/graph.py:24-29` 在 langchain 导入前 load_dotenv）。节点输入输出快照、StructuredTool
  调用、子 agent（`subagents.py:213-225` run config 透传）都在同一 trace 内。
- **join 键**：`agent/graph.py::run` 与 `agent/supervisor.py::_run_worker` 顶层 ainvoke 注入
  `RunnableConfig.run_id=uuid4()`，本地 `trace_id` = 该 run 的 hex —— 本地事件与 LangSmith
  run 树**同值同键**，双向可定位。
- **评测路径的 join 键**：跑批用确定性 `trace_id = "{run_id}:{qid}"`，因此把**真实的
  LangSmith 根 run id** 记进 `trace_events.run_id`（`graph.run` 的 `set_thread_map`），
  导出/回读时按 trace_id 反查即可对上（`agent/core/trace_export.lookup_langsmith_run_id`）。
- **metadata 契约**：每轮根 run 带 `trace_id/thread_id/request_id/actor_role/domain/
  graph_version/config_revision/config_hash/prompt_bindings/tool_versions/model_route/
  eval_dataset_id`（`agent/graph.py::prepare_turn` + `agent/core/execution_context.py`）。
- **P1 嵌套已闭合**：节点内 LLM/工具调用（`_stream_llm`/`traced_ainvoke`/`tool.ainvoke`）
  已透传父 config，游离根 run 消失；**新增调用点必须用
  `ExecutionContext.child_config()` 派生 child config**（见 ADR-0002）。
- **run 树归档（保留期之外可复现）**：
  `python -m agent.core.trace_export <trace_id> [--project paper-agent]` 写出
  `eval_output/runs/<trace_id>/langsmith_runs.jsonl` + `langsmith_manifest.json`；
  跑批时设 `EVAL_EXPORT_LANGSMITH=1` 自动逐条导出（失败只告警，不影响报告）。
  默认只导出 id/名称/类型/时间/错误/**脱敏 metadata**，`AGENT_TRACE_EXPORT_FULL=1`
  才带原始 inputs/outputs（仅限允许的研发环境）。
- 验证：`python -m evaluation.verify_smith "RMNet 的 loss 函数是什么？" --mode react`
  （断言根 run id == trace_id，输出节点/LLM/工具结构统计 + 游离 run 提示 + 控制台链接）。
  注意：CLI 一次性进程收尾会被 MCP stdio 挂死（见 TROUBLESHOOTING「agent / MCP stdio 收尾」），
  脚本以 `os._exit` 绕过；完整链路建议在 uvicorn/前端 SSE 会话中观察。
- 域事件 → LangSmith 映射、归档/保留期策略、实施阶段见 `docs/agent评测体系构建.md` §四-§九。

**事件 schema**（`trace_store.trace_events` 宽表 + payload JSON）：
`trace_id/thread_id/turn_seq/seq/ts/event_type/node/duration_ms/model/intent/error/
tool/ok/source(live|eval)/run_id/parent_id/payload`。事件类型：
turn_start / intent / node_start|end|error / llm_call / tool_call /
tool_timing / tool_phase_timing / retrieved_context /
plan / plan_step / plan_verify / final_answer / turn_end（+ 其余 log_event 原样透传）。

**开关**：`AGENT_TRACE_ENABLED`（默认 1）。**归属**：eval 跑批用
`set_eval_ctx(run_id, thread_id)` 注入；线上路径 `graph.run`/router 登记
`trace_id→thread_id` 映射。

## 单样例透明流程（第一层：先看得见，再算指标）

`POST /api/eval/single` `{query, thread_id?, mode?}` 跑一条 query 的完整链路并返回
**分阶段视图**（`evaluation/flow.build_flow`）：

```
用户问题 → 意图理解（intent/confidence/entities/needs_planning）
        → 上下文/记忆 → 计划（plan_step/verify）
        → 工具执行（tool_call + 结果信封解析 ok/error_type + 检索 chunk_ids）
        → 问题回答（final_answer + verification）
每阶段标注 duration_ms 与 tokens（prompt/completion/total，估算法标记 estimated）
+ turn 级汇总：总耗时/总token/成本/工具成败/任务成败/链路 trace_id
```

前端「评测」面板默认就是单样例视图：输入问题 → 运行 → 分阶段时间线（工具信封
解析红/绿可见，工具行可折叠看 args/result，最终回答全文本）。链路数据随后可在
「评测报告」跑批中抽指标、或凭 `trace_id` 回溯归因。

```bash
# 命令行等价（拿到 flow JSON）
curl -s -X POST http://127.0.0.1:8000/api/eval/single \
  -H 'Content-Type: application/json' \
  -d '{"query":"RMNet 的 loss 函数是什么？","mode":"react"}'
```

## 实时评测（跑批过程可见，2026-09-15）

跑批是「逐条 QA、逐条出指标」的长任务：第 3 条就坏掉的检索不该等到跑完 20 条才知道。
`evaluation/live.py` 把过程变成可订阅的事件流，**指标随执行实时更新**：

```
runner.run_eval() ──publish──► RunFeed（进程内总线）──subscribe──► SSE / CLI / eval_runs 进度行
```

| 事件（`event["type"]`） | 负载 |
|---|---|
| `run_started` | run_id / dataset_id / total / started_at |
| `query_started` | index / total / qid / query |
| `query_finished` | 单条明细：category / success / mrr / recall@5 / ndcg@10 / tool_errors / run_error / duration_s / tokens / cost_usd / trace_id |
| `aggregate` | **累计指标**：done/total/elapsed_s/eta_s + overall（检索 + 任务成功率 + 工具成功率，**各自带分子分母** tool_calls / tool_failures / task_success + tokens + 成本）+ recent（最近 10 条） |
| `judge_started` / `judge_finished` | judge 采样数与结果 |
| `run_finished` / `run_failed` | 终态（含 status / duration_s / finished_at / gate / badcase_count / error） |
| `heartbeat` | 无进展时的保活帧（默认 15s，前端忽略） |

**同口径保证**：实时 `aggregate` 与最终 `run_summary.json` 复用同一批纯函数
（`metrics.retrieval.aggregate` / `metrics.tools.aggregate_tools` / `report.estimate_cost`），
界面上的数不会和报告里的数打架——`evaluation/tests/test_live.py` 直接断言两边相等。

**三条消费路径（同源）**：
1. **SSE**：`GET /api/eval/runs/{run_id}/stream`——本进程的 run 直接吃内存总线（含逐条明细）；
   非本进程（CLI / 别的 worker）退回每 2s 轮询 `eval_runs` 行，事件同构、指标粒度到「已完成条数」。
   EventSource 断线自动重连（服务端回放缓冲历史）；不存在的 run 直接 404，不静默挂死。
2. **进度行**：每条 QA 完成后把同一份聚合写进 `eval_runs`（`status=running` +
   `overall.progress={done,total}`），跑完由最终报告覆盖同一行。所以**不用 SSE 的客户端**
   （列表轮询、报告页）也能看到随执行更新的指标。`POST /api/eval/runs` 在返回前就写好这行，
   前端拿到 run_id 立刻订阅不会撞 404 竞态。
3. **CLI**：`python -m evaluation run` 订阅同一条流逐条打印；`--run-id` 可指定 run_id，
   便于跑批中途到前端订阅同一个 run 看进度。

**单样例实时视图**：`POST /api/eval/single/start`（非阻塞，返回 thread_id）配上
`GET /api/eval/single/{thread_id}/stream`（每 ~0.8s 推一次当前 flow 快照：
`single_started → single_snapshot… → single_done | single_error`）。重连只重读 trace 事件，
不会重跑该 query。

```bash
# 终端实时进度（打印的累计指标与前端 SSE 同源）
python -m evaluation run --manifest eval_output/datasets/manifest_v2_current_kb_50qa.jsonl \
  --limit 20 --run-id my-run
# 同一 run 的事件流（前端 EventSource 走的就是这条）
curl -N http://127.0.0.1:8000/api/eval/runs/my-run/stream
```

**边界约定**：publish 永不向跑批抛异常（进度通道不得影响主链路）；慢订阅者丢最旧事件、
绝不反压；内存总线只覆盖本进程的 run（跨 worker 由上面的 DB 轮询路径补齐）；
进程内只保留最近 64 个 run 的 feed（已结束的先淘汰）。

## 快速使用（批量评测）

```bash
C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation run \
  --manifest eval_output/datasets/manifest_v2_current_kb_50qa.jsonl --limit 20
C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation recovery --manifest ... --limit 3
C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation list
C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation show <run_id> --badcases
C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation qrels --manifest ... --out qrels.txt
C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation prune --older-than-days 30
# 等价入口：python -m web.cli eval run ... / show <run_id> / list
```

注意：`--limit` 默认小样本（用户决策，避免全量 50 QA 的 LLM 成本/时长）；
`--no-judge` 跳过 LLM-as-judge；`AGENT_FAULT_TOOL=download_paper` 触发故障注入
（仅评测用，不加 env 不影响生产）。

跑批前会校验 `ground_truth_ids` 是否存在于当前 `eval_output/all_rag_chunks.json`。
旧 manifest 与现行知识库 chunk ID 交集为 0 时会在调用 agent 前退出（错误码 2），
避免整批固定得到 Recall=0；需要强制执行时显式加 `--allow-stale-dataset`。

跑批过程**实时打印**逐条指标与累计聚合（与前端 SSE、最终报告同源）；
`--run-id <id>` 指定 run_id，便于跑批中途到前端「评测报告」订阅同一个 run。

## 输出

- `trace_store.db` — trace_events + eval_runs 表（评估端唯一事实源）。跑批过程中
  `eval_runs` 先是一行 `status=running` 的**进度行**（`overall.progress={done,total}`），
  跑完被最终报告覆盖同一行；前端列表/报告页因此不做 SSE 也能看到实时指标。
- **完整用时 = 启停墙钟差**：`duration_s` 由跑批启动打点 `started_at` 与收尾打点
   `finished_at` 相减得到，**不累加链路里各步骤的 duration**；逐条 QA 的墙钟样本
   另存 `query_durations_s`，并给出 `overall.answer_latency_p50_s/p95_s/max_s`。
- **终态保证（不会永远 running）**：正常收尾写 `done`；异常或收尾阶段崩溃写 `failed`
  （保留已算出的部分指标 + `overall.progress`）；取消 / 进程消失写 `interrupted`。
  长 query 期间由心跳（`EVAL_PROGRESS_HEARTBEAT_S`）刷新 `metadata.heartbeat`；
  读路径（列表 / 报告页 / `evaluation list|show`）调用 `reconcile_stale_runs`，
   把心跳过期且进程已不在的 `running` 行收敛成 `interrupted`（幂等）。
- `eval_output/runs/<run_id>/` — run_summary.json（报告全文）+ per_query.jsonl
- 同目录 `trace_events.jsonl` — 本批结构化事件归档（LangSmith / trace_store 保留期
  过后仍可复盘）。生产在线路径不写该库；需要本机在线调试时显式设
  `AGENT_TRACE_STORE_LIVE=1`。
- `eval_output/runs/.baseline.json` — 按 dataset_id 的回归基线（降幅 >5% warn，
  Recall@5<0.6 block）
- `eval_output/datasets/registry.json` — 评测集版本/来源/难度/review 状态

## API（薄封装）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/eval/traces/{trace_id}` | GET | 完整链路事件链（时间/token/工具/检索上下文） |
| `/api/eval/thread/{thread_id}` | GET | 会话各 turn 轨迹（live 与 eval 统一可见） |
| `/api/eval/runs` / `/{run_id}` / `/{run_id}/badcases` | GET | run 列表 / 报告 / 逐条 badcase 表（含 status / finished_at / duration_s / badcase_count；读时先 `reconcile_stale_runs` 收敛挂死的 running 行） |
| `/api/eval/runs` | POST | 后台评测跑批（202，result 写 run_id） |
| `/api/eval/runs/{run_id}/stream` | GET | **SSE 实时进度**：run_started → query_started/query_finished → aggregate（累计指标/token/成本/ETA）→ judge_* → run_finished｜run_failed。非本进程的 run 退回 2s 轮询 `eval_runs`，事件同构 |
| `/api/eval/single` | POST | 单样例透明评测（阻塞，返回完整 flow） |
| `/api/eval/single/start` | POST | 单样例非阻塞启动（返回 thread_id，202 语义；`reused=true` 表示复用已在跑的线程） |
| `/api/eval/single/{thread_id}/stream` | GET | SSE 单样例分阶段实时视图（每 ~0.8s 一次 flow 快照，到 single_done/single_error 结束） |
| `/api/eval/manifest/from-trace` | POST | 线上 badcase 回流评测集（source=live） |
| `/api/eval/feedback` | POST | 将人工验收 / 评分写回对应 LangSmith root run |
| `/api/eval/prune` | POST | trace 清理（live 默认 30 天，eval 永久） |

## 配置（env）

| 环境变量 | 默认 | 说明 |
|---------|------|------|
| `AGENT_TRACE_ENABLED` | 1 | trace 采集开关 |
| `AGENT_TRACE_DB` | `./trace_store.db` | trace 数据库路径 |
| `EVAL_MAX_QUERIES` / `EVAL_TURN_TIMEOUT` | 0 / 900 | 跑批条数 / 单条超时 |
| `EVAL_JUDGE_MODEL` | kimi-k2.6 | judge 模型（qwen-max 等可换） |
| `EVAL_JUDGE_SAMPLE` | 10 | judge 采样数（badcase 优先） |
| `EVAL_JUDGE_BUDGET_USD` | 0.5 | judge 预算护栏（累计即停） |
| `EVAL_EXPORT_LANGSMITH` | 空（关） | =1 时跑批结束按 trace 导出 run 树到 `eval_output/runs/<run_id>/langsmith_runs.jsonl` |
| `AGENT_TRACE_EXPORT_FULL` | 空（只导元数据） | =1 时导出含原始 inputs/outputs（仅限允许的研发环境） |
| `AGENT_FAULT_TOOL` | 空 | 故障注入工具名（恢复率测试专用） |
| `EVAL_PROGRESS_HEARTBEAT_S` | 20 | 进度行心跳刷新间隔（秒）；长 query 期间刷新 `metadata.heartbeat` |
| `EVAL_STALE_RUN_S` | 180 | `running` 行心跳过期多久后收敛为 `interrupted`（0 = 不开启收敛） |

## 指标口径（报告中标注）

- 检索 hits = trace `retrieved_context` 按 seq 序的 chunk_id，**同一 chunk 只保留
  首次出现**，顺序=工具调用序；这样多轮重复检索不会把 Recall/NDCG 推到 1 以上。
  与 retrieval_engine 静态评测（一次检索 top-k）口径不同。
- 批量评测的 `thread_id = evalqa:{run_id}:{qid}`，每个 run 使用独立 checkpoint 与
  trace；逐条指标只读取本次 `trace_id`，不会串联同 QA 的历史 run。
- **工具成功率 = 工具成功调用数 / 工具调用总次数**（分母是调用次数，不是 run 数）。
  一次逻辑调用只写**一行** `tool_call` trace；成功判定三层降级：①
  `operation.outcome` 非 `succeeded`（失败 / 超时 / 取消）→ 失败；
  ② 结果 envelope `outcome != succeeded`（HTTP 200 但业务失败）→ 失败；
  ③ 纯文本结果命中失败文案
  （`timeout` / `error` / `失败` / 异常堆栈）→ 失败。**不再「只要调用过就记成功」**。
- **任务成功率 = 达成条数 / 评测条数**。达成 = 有 final_answer ∧ 无 `run_error`
  ∧ 无工具失败 ∧（plan 模式）`verification.status == satisfied`；`status` 取
  `ok`/`satisfied`（达成）/ `degraded`（有回答但工具失败或验证未过）/ `failed`
  （无回答或中断）。**降级回答不算成功**，且一定进 badcase。
- badcase 归因优先级：`run_error` > `no_answer` > `tool_fail` > `task_fail`
  > `retrieval_fail` > `low_mrr` > `ok`——失败任务不再漏归。
- 恢复率 = 故障注入中断 → 同 thread_id 续跑产出 final_answer 且无副作用工具重复。

## 目录

```
evaluation/
  events.py / sink.py / trace_store.py / trace_wrap.py / config.py
  datasets.py        # 评测集加载/去重/版本化/注册表
  runner.py          # 跑批编排（agent_run→trace→指标→报告）
  live.py            # 跑批实时进度总线（SSE / CLI / 进度行同源的事件流）
  report.py          # 报告组装 / 成本估算 / 回归基线
  metrics/           # retrieval / tools / task / judge 四类指标
  __main__.py        # CLI：run | recovery | list | show | qrels | prune
  tests/             # 冒烟测试（assert 风格，可直接 python 运行）：test_live.py 实时进度、
                     #   test_metrics.py 指标口径、test_run_state.py 终态与计时
```

制约说明：tokens 来自 provider 真实 usage（`response_metadata.token_usage`），
取不到时 llm_call.tokens 记空，成本估算偏低；checkpoints.db 为 msgpack 不可直接
SQL 读，评测链路一律走 `aget_state(thread_id)` 或本库 trace 事件重建。
