# Agent RAG 与论文操作评测数据集 v3

## 1. 目标

v3 不是现有 50 条关键词 QA 的简单扩容，而是把评测对象从“单轮检索问答”
提升为“Agent 完成任务”。当前先覆盖四类任务：

| task_type | 中文名 | 评测对象 |
|---|---|---|
| `rag_retrieval` | RAG 检索 | 本地知识库检索、定位、跨块/跨论文综合 |
| `paper_download` | 下载论文 | arXiv 身份核验、文件落盘、路径/重名/批量与幂等 |
| `paper_read` | 论文精读/抽取精度 | 字段抽取、数值精度、证据引用、拒答与冲突核对 |
| `paper_ingest` | 论文入库 | 异步任务、解析+索引闭环、可检索性、重复与回滚 |

数据集采用完整的 `4 x 3 x 3` 平衡矩阵：

* 4 类任务
* 3 个难度：`easy / medium / hard`
* 3 个任务长度：`short / medium / long`

共 36 条。每类 9 条、每个难度 12 条、每个长度 12 条。其中 18 条有检索
`ground_truth_ids`，18 条是下载/入库等非检索任务。

## 2. 当前产物

| 文件 | 作用 |
|---|---|
| `eval_output/datasets/agent_rag_paper_ops_v3.jsonl` | 36 条可执行任务清单 |
| `evaluation/task_dataset.py` | 可重复生成、矩阵校验、GT 语料覆盖校验 |
| `evaluation/metrics/contracts.py` | 工具链、参数、答案、产物、预算的确定性契约检查 |
| `evaluation/datasets.py` | 兼容旧 manifest，并区分检索/非检索任务 |
| `evaluation/runner.py` | 逐任务执行，写入任务、契约、token、耗时与成本 |
| `evaluation/report.py` | 总体/任务类型/难度/长度的报告输出 |

重新生成：

```bash
conda run -n demo python -m evaluation.task_dataset
```

跑检索与精读子集：

```bash
conda run -n demo python -m evaluation run \
  --manifest eval_output/datasets/agent_rag_paper_ops_v3.jsonl \
  --dataset agent-rag-paperops-v3
```

下载和入库会访问外网、写入文件或改变向量库，正式跑批必须使用隔离 workspace
与隔离后端。不要直接拿生产知识库重复跑 `paper_ingest` 样本。

## 3. 每条任务包含什么

```json
{
  "schema_version": "3.0",
  "dataset_id": "agent-rag-paperops-v3",
  "id": "...",
  "query": "...",
  "task_type": "rag_retrieval",
  "difficulty_level": "easy",
  "task_length": "short",
  "ground_truth_ids": ["..."],
  "evaluation": {
    "retrieval": true,
    "task_execution": true,
    "paper_precision": false,
    "side_effects": false
  },
  "expected": {
    "required_tools": ["search_papers"],
    "forbidden_tools": [],
    "tool_args": [],
    "answer_contains": ["28.4"],
    "answer_not_contains": [],
    "artifacts": []
  },
  "budgets": {
    "timeout_s": 90,
    "max_steps": 8,
    "max_tool_calls": 3,
    "max_tokens": 12000
  },
  "metrics": {
    "retrieval": ["recall@5", "precision@5", "mrr", "ndcg@10"],
    "task": ["success", "tool_success_rate"],
    "cost": ["duration_s", "tokens_total", "cost_usd"]
  },
  "postconditions": [],
  "review": {"status": "draft", "required": true}
}
```

关键设计点：

* `evaluation.retrieval=false` 的下载/入库任务不会用空 GT 污染 Recall/MRR。
* `required_tools` 与 `forbidden_tools` 检查真实工具链，不只看最终回答。
* `tool_args` 能验证 arXiv ID、目标目录、文件名、`paper_name`、`pdf_path`。
* `artifacts` 验证文件是否真实存在、大小是否合理，以及是否是本次运行新产生。
* `budgets` 将 step、工具调用、token、墙钟时间纳入任务成功判定。
* `postconditions` 表达单次 trace 无法证明的异步状态，例如入库最终为
  `indexed`、chunk count 大于 0、检索探针命中、无重复索引。

## 4. 指标口径

### 4.1 RAG 检索

* `Recall@K`：必要证据是否被找到。
* `Precision@K`：召回上下文中相关 chunk 的占比。
* `MRR`：首个相关证据的排序位置。
* `NDCG@K`：多证据排序质量。
* `Hit@K`：至少命中一个证据的 query 比例。
* `context_precision`：LLM judge 判断上下文是否直接支持回答。

命中顺序沿用 Agent 实际工具调用顺序。同一 chunk 重复出现只保留第一次，
避免多轮重复检索人为抬高指标。

### 4.2 任务执行

* `task_success_rate`：任务达成数 / 任务数。
* `tool_success_rate`：成功工具调用 / 工具调用总数。
* `expected_tool_coverage`：必需工具实际覆盖率。
* `contract_success_rate`：确定性验收检查通过率。
* `step_count / tool_calls`：执行复杂度与绕路情况。
* `badcase category`：`run_error / no_answer / tool_fail / task_fail /
  retrieval_fail / low_mrr`。

### 4.3 论文精读/精度

* `field_coverage`：要求抽取的字段是否有值。
* `exact_numeric_match`：数字的精确匹配；需要容差时使用 tolerance。
* `citation_precision / citation_recall`：引用是否支持对应 claim。
* `unsupported_claim_rate`：无证据 claim 占比。
* `table/formula consistency`：表格/公式与原文是否一致。
* `abstention correctness`：证据不足时是否明确拒答，而不是补全幻觉。

当前 v3 的确定性契约已覆盖字段和关键数值；引用精度与无依据 claim 需要后续
接入 LLM judge 或人工复核。

### 4.4 下载论文

* `download_success`、`HTTP/API success`。
* `arxiv_id -> title` 身份核验正确率。
* `path_correctness`、`filename_correctness`。
* `artifact_exists`、`min_size_bytes`、PDF magic/header 可选校验。
* `no_accidental_ingest`：下载任务绝不能偷偷触发入库。
* `idempotency`：同 ID/同路径重复下载不损坏文件。
* 阶段耗时：`metadata_ms / download_ms / write_ms / total_ms`。

### 4.5 入库论文

* `enqueue_success`：是否拿到 `task_id`。
* `end_to_end_completion`：后台任务是否到 `done`。
* `indexed_state`：`check_paper` 最终是否为 `indexed`。
* `chunk_count`：索引 chunk 数是否大于 0。
* `post_ingest_retrieval`：入库后相同论文是否能被检索到。
* `duplicate_index_rate`：重复 `paper_name` / 重复 chunk 分组。
* `rollback_on_failure`：失败后不留下半解析、半索引状态。
* 响应延迟与完成延迟分开统计：`enqueue_s`、`queue_wait_s`、
  `parse_s`、`index_s`、`end_to_end_s`。

### 4.6 时间、token 与成本

* `duration_s`：一次任务从开始到结束的墙钟时间，不累加步骤耗时。
* `p50/p95/max`：逐条任务延迟分布。
* `prompt_tokens / completion_tokens / tokens_total`：provider 真实 usage。
* `tokens_per_successful_task`：成功任务平均 token。
* `cost_usd`、`cost_per_successful_task_usd`。
* 预算违规单独统计，避免“任务做对但成本不可接受”被隐藏。

## 5. 本次已经补全的遗漏

相对原来只考虑“检索指标、任务执行、时间、token”，v3 补齐了：

1. 任务类型与难度、长度分层，避免单一平均值掩盖局部退化。
2. 非检索任务跳过检索指标，修掉下载/入库固定 `Recall=0` 的假失败。
3. 工具链与参数级验收，区分“回答了”与“真的执行了正确动作”。
4. 文件产物与副作用验收，覆盖下载路径、文件名和意外入库。
5. 异步入库后置条件，覆盖最终 indexed、chunk count 与可检索性。
6. 幂等、重复索引、故障注入与回滚。
7. 精读精度中的数值、字段、引用、拒答与冲突证据。
8. step/tool/token/time 四类预算，以及预算违规率。
9. 按任务类型、难度、长度分组的任务指标，不只按检索维度分组。
10. 隔离 workspace / 隔离后端要求，避免评测污染生产知识库。

## 6. 还没完整解决、下一阶段必须补的

v3 是一个 36 条种子基准，适合定位问题，不足以单独作为发布置信区间。正式
版本还应继续补：

1. **样本量和统计功效**：每格 1 条只能做矩阵冒烟；建议核心格扩到 10-30
   条，总规模 300-1000，并按 bootstrap 给出置信区间。
2. **异步后置条件执行器**：当前 manifest 已声明 postconditions，但接入
   `check_paper`、任务状态轮询、chunk count、重复分组和检索探针，还需要
   独立隔离 harness。
3. **LLM/人工 judge 校准**：引用精度、无依据 claim、答案完整性不能只靠
   字符串命中；需要固定 judge prompt、人工金标、一致性/Kappa 报告。
4. **副作用清理与恢复**：入库测试必须支持独立 Qdrant collection、临时
   Redis/catalog、临时 PDF 目录，并在运行后自动销毁。
5. **网络与外部依赖稳定性**：arXiv 限流、超时和版本变化要有 recorded
   fixture/cassette，另给 live-network 分数，不能把外部抖动算成模型退化。
6. **数据泄漏与污染控制**：用于下载/精读的论文明确定义是否已存在于语料；
   同一个论文不能同时作为训练/索引输入和不可见测试目标。
7. **对抗与安全**：prompt injection、伪造 arXiv ID、恶意 PDF、超长表格、
   损坏 PDF、路径穿越、重复请求和并发入库。
8. **公平对比口径**：模型、prompt、工具版本、检索配置、并发、缓存状态必须
   固定并写入 metadata，否则不同 run 不可比较。
9. **失败任务成本**：当前成本按成功任务摊销；还应统计失败/超时任务的
   wasted tokens 与 wasted latency。
10. **人审状态**：36 条目前标记为 `draft`，正式成为门禁数据集前必须完成
    题目、GT、断言和阈值的双人复核。

## 7. 建议门禁

第一版建议把以下指标分开看，不要合成一个总分：

| 维度 | 建议初始门禁 |
|---|---|
| RAG `recall@5` | 检索子集不低于 0.80 |
| RAG `mrr` | 不低于 0.65 |
| 精读数值精确率 | 不低于 0.90 |
| 不支持 claim 率 | 不高于 0.05 |
| 下载任务成功率 | 不低于 0.95（排除外部服务故障后） |
| 意外入库率 | 必须为 0 |
| 入库端到端成功率 | 隔离环境内不低于 0.90 |
| 重复索引率 | 必须为 0 |
| 预算违规率 | 不高于 0.10 |

下载/入库测试应同时输出 `infra_failure_rate`；arXiv 或本地后端不可用时，
任务标记为 `environment_blocked`，不能直接算成 agent 能力失败。
