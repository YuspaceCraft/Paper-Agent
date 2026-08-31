# Agent Plan-and-Execute 与 Multi-Agent 改造计划

> 目标：在既有通用 Agent 底座（Phases 0/1/2/3/5 已落地）之上，补齐两块执行层能力——
> 1. **Plan 范式**：复杂查询「先规划后执行」，替代单一 ReAct 对多跳/多论文对比任务的低效重试。
> 2. **Multi-Agent**：把科研文献助手的核心能力从「一个 monolith agent + 四条工具」重构为「编排器 + 若干专注 subagent」。
>
> 原则沿用 [agent-refactor-plan.md](agent-refactor-plan.md) 与 [agent-refactor-plan-review.md](agent-refactor-plan-review.md)：不推倒重来、不做空抽象、LLM 契约（`{"ok":false,"error_type":...}` + `[CITE:N]`）不回归。

---

## 0. 现状盘点（对着当前代码核过）

| 项 | 现状 | 关键文件 |
|---|---|---|
| 执行模式 | 单一 ReAct：`understand → memory → route → resolve → search(ReAct 子图) → synthesize`，无 Plan 分支 | [graph.py](../agent/graph.py) [search_loop.py](../agent/search_loop.py) |
| `state.mode` | **未落地**（原 Phase 1 计划「留接口位」，实际 [state.py](../agent/state.py) 无 `mode` 字段） | [state.py](../agent/state.py) |
| 科研能力落点 | 全部压在一个 `AGENT_SYSTEM` prompt + 4 条 builtin 工具里（search/fetch/download/process），另有 arxiv__* MCP 工具、skill__* 技能 | [prompts.py](../agent/prompts.py) [builtin_provider.py](../agent/providers/builtin_provider.py) |
| 引用传播 | `cite_registry` 是**请求级 contextvar**，chunk 打 `[N]` 裸标记，父层 synthesize 统一解析 | [builtin_provider.py:50-100](../agent/providers/builtin_provider.py#L50-L100) |
| 工具调度 | dispatcher 已统一超时/重试/审计，覆盖含 builtin 在内全部工具 | [dispatcher.py](../agent/dispatcher.py) [tools.py](../agent/tools.py) |
| skill 机制 | 已有 skill__list/skill__load（「结构化工作流指示」），与 subagent 是两回事 | [skill_provider.py](../agent/providers/skill_provider.py) |
| 治理约束 | 整轮超时 + token 预算 + max_iterations 已齐 | [graph.py](../agent/graph.py) [state.py](../agent/state.py) |
| README | `agent/README.md` **不存在**（CLAUDE.md 要求新建并同步） | — |

**核心判断**：通用底座（治理/安全/观测/调度）已经做厚，缺的是**执行层的结构化**。当前一个 agent 身兼「找论文、精读、入库」三职，靠一个 100 行 system prompt 硬撑；复杂查询（多跳、对比、综述）退化为反复试探。Plan 与 subagent 是同一问题的两个解法——**Plan 决定拆什么步骤，subagent 决定谁把某类步骤做得好**。

---

## 1. 核心架构

不改现有单 agent 主干，在其上叠两层：

```
                 ┌──────────────────────────── 编排器（通用） ───────────────────────────┐
                 │  understand → memory → route_intent                                     │
                 │        │                                                               │
                 │        ├─ general_chat / needs_clarify ─► chat / clarify（不变）        │
                 │        └─ literature_search                                            │
                 │              │  mode 判定（代码启发式，非 LLM）                          │
                 │              ├─ mode=react  ──► 现有 search 子图（简单查询，零回归）      │
                 │              └─ mode=plan   ──► plan_node ─► executor ─► synthesize     │
                 │                                    │                                   │
                 └────────────────────────────────────┼───────────────────────────────────┘
                                                      │ 每步委派
                 ┌────────────────────────────────────▼───────────────────────────────────┐
                 │  科研 subagents（领域，各自独立上下文 + 独立 prompt + 受限工具集）          │
                 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
                 │  │paper_discovery│  │ paper_reader │  │ paper_ingest │                    │
                 │  │ search_papers │  │ fetch_content│  │download_paper│                    │
                 │  │ arxiv__search │  │ search_papers│  │process_paper │                    │
                 │  │arxiv__get_data│  │  (核对名称)   │  │  (destructive)│                   │
                 │  └──────────────┘  └──────────────┘  └──────────────┘                    │
                 └──────────────────────────────────────────────────────────────────────────┘
```

三个核心机制：

1. **subagent-as-tool**：每个 subagent 是一个**编译好的子图**（自己的 `agent_node` + `ToolNode` + 循环 + 预算），对外包装成父层可调用的一枚工具。父层只看到「传入任务 → 返回摘要字符串」，**看不到 subagent 内部的大段 PDF 文本**——这是 subagent 相对扁平工具的核心价值：上下文隔离。
2. **Plan-and-Execute**：`plan_node` 产出结构化步骤表，`executor_node` 按依赖序执行（无依赖的步骤 `asyncio.gather` 并行），结果交给 `synthesize` 合并。
3. **模式开关**：代码级启发式选 `react`/`plan`，不额外加 LLM 分类调用。简单查询走 `react`（零行为回归），复杂查询走 `plan`。

---

## 2. Phase 7 — Plan-and-Execute（Plan 范式）

**目标**：多跳/对比/综述类查询「先拆步骤再执行」，取代当前在单个 ReAct 循环里反复试探。

### 改动

1. **`state.py` 补模式字段**（补上原计划漏掉的接口位）：
   ```python
   mode: str = "react"          # "react" | "plan"
   plan: list[dict] = []        # [{id, description, target, args, depends_on}]
   plan_progress: int = 0
   subagent_results: list[dict] = []   # [{step_id, ok, output, citations}]
   ```

2. **模式判定（代码启发式，无 LLM）**：`route_intent` 之后、进 search 之前，纯函数判定 `mode`。命中任一条件 → `plan`：
   - `entities` + `focus_papers` 去重后数量 ≥ 2；
   - 含对比/多跳关键词（`对比`/`compare`/`vs`/`versus`/`区别`/`哪个更好`/`综述`/`survey`）；
   - 显式多子问题（问句含两个以上 `？` 或并列从句）。
   不命中 → `react`（现状，零回归）。

3. **`plan_node`（LLM 结构化产出）**：新增 `agent/plan.py`。输入用户 query + entities + resolved hints，产出 `PlanResult`（Pydantic，`method="function_calling"`）：
   ```python
   class PlanStep(BaseModel):
       id: str
       description: str          # 该步要回答什么
       target: str               # "paper_discovery" | "paper_reader" | "paper_ingest" | "tool"
       args: dict
       depends_on: list[str] = []
   class PlanResult(BaseModel):
       steps: list[PlanStep]
   ```
   遵循 CLAUDE.md Prompt 四原则：结构化 JSON、零状态（不含"如上"）、模型名走代码注入、上下文注入 entities/resolved（≤预算）。

4. **`executor_node`（依赖序执行，无 LLM）**：拓扑序执行 `plan`。无依赖步骤 `asyncio.gather` 并行；有依赖的等前置完成。每步调 `target` 对应能力，结果（含引用于内的字符串/对象）写入 `subagent_results`。**超时/预算复用现有字段**（每步有独立 token 预算，整轮有 `TURN_TIMEOUT`）。

5. **`synthesize_node` 扩展**：plan 模式下合并 `subagent_results`（而非裸 tool messages）成最终答案；react 模式走现有逻辑不变。

### DoD
- 「对比 RMNet 与 SRN 的 loss 设计」这类查询产出 ≥2 个步骤的 plan，按序执行，最终答案覆盖两个论文。
- 简单查询（`RMNet 的 loss 是什么`）仍走 `react`，**不产生额外 plan LLM 调用**（断言：`mode=="react"` 时无 `plan_node` 调用）。
- 自检：`agent/tests/test_plan.py` —— 给定合成 state，断言 executor 按依赖序执行、并行步骤结果都写入 `subagent_results`。

### 依赖
无新增依赖。依赖 Phase 8 的 subagent 定义 `target` 名字（本阶段可先让 `target="tool"` 走现有工具，Plan 骨架独立可用）。

---

## 3. Phase 8 — Multi-Agent（科研能力 → subagents）

**目标**：把压在 `AGENT_SYSTEM` + 4 条 builtin 工具里的科研能力，重构为 3 个专注 subagent，获得上下文隔离、prompt 专精、并行执行。

### 3.1 Subagent 运行时（通用，一个工厂）

新增 `agent/subagents.py`：

```python
def build_subagent(name, system_prompt, tools, *, max_iterations=5) -> compiled_subgraph
def as_tool(name, subgraph, description, args_model) -> BaseTool   # 包装成父层可调用工具
```

- **复用** `agent_node` / `ToolNode` / `after_agent` 的循环结构（与 [search_loop.py](../agent/search_loop.py) 同构），区别仅三点：**受限工具集**、**专属 system prompt**、**独立上下文**（subagent 用自己的 message list，父层不可见）。
- `agent_node` 现读全局 `AGENT_SYSTEM`，需小改：加一个可选 state 字段 `subagent_system`（空则回退 `AGENT_SYSTEM`）。这是最小 diff，保留现有 pre-flight / 错误分类 / token 预算守卫。
- `as_tool` 内部：`subgraph.ainvoke({task})`，取最后一条 AIMessage 作为返回串。**并发安全**：`asyncio.gather` 在同一 task 内跑多个 subagent，contextvar（cite_registry）继承共享，引用无需额外传递。

### 3.2 三个科研 subagents（领域，具体不抽象）

| subagent | 工具集 | 职责 | 输出 | 备注 |
|---|---|---|---|---|
| **`paper_discovery`** | `search_papers`, `arxiv__search_papers`, `arxiv__get_paper_data`, `arxiv__list_categories` | 给定 topic → 找相关论文（本地+外部），返回排名列表 + 摘要 + arxiv_id | 结构化 JSON | 只读，快 |
| **`paper_reader`** | `fetch_content`, `search_papers` | 给定论文 + 问题 → 精读相关章节，返回**带 `[N]` 引用的答案** | Markdown + 引用 | 只读；隔离大段正文 |
| **`paper_ingest`** | `download_paper`, `process_paper` | 给定 arxiv_id/论文名 → 下载 + 解析 + 入库 | 状态 JSON | **destructive**（`readOnlyHint=False`），30-120s，走权限门 |

**为什么是 3 个而非 2 个或 1 个**：三类能力在工具集、超时档位（discovery 秒级 / ingest 分钟级）、权限（ingest 破坏性）、上下文（reader 独占大段正文）四个维度都不同，天然是 3 个独立单元。合并回 1 个就退回 monolith prompt。这是「按能力边界拆」，不是「为抽象而抽象」。

### 3.3 父层委派

- 三枚 subagent 工具加入父层可用工具（react 模式下也可见，作为「重型工具」让简单查询也能按需委派）。
- `plan_node` 的 `target` 直接引用 subagent 名；`executor_node` 把步骤派给对应 subagent。
- **渐进迁移**：本阶段保留 4 条 builtin 工具与 subagent 工具并存。观察 subagent 路径稳定后，react 模式也可逐步改为委派 subagent（后续阶段，非本阶段硬目标）。

### 3.4 引用传播（关键决策）

当前 `[N]` 标记写进请求级 `cite_registry`（contextvar），父层 `_resolve_citations` 统一解析。subagent 场景下**沿用同一机制**：

- subagent 内部 chunk 仍打 `[N]` 裸标记（写进共享 registry），subagent 的 system prompt 要求最终答案**原样保留用到的 `[N]` 标记**（与现有 `AGENT_SYSTEM` 引用规范一致）。
- 父层 `synthesize` 对合并后的答案统一 `_resolve_citations`，行为与今天完全一致。
- **并行安全**：`asyncio.gather` 同 task 共享 contextvar，`_next_cite_id()` 计数器全局递增，`[N]` 不重复、registry 无污染。**不需要**为 subagent 单独建引用命名空间。
- 兜底：若实测发现 subagent 丢失标记（LLM 复述时漏抄），回退方案是让 subagent 返回结构化 `{answer, citations:[{paper,page,chunk_id}]}`，父层重映射。**先不加**，仅在标记丢失被观测到后再做。

### 3.5 与 skill 机制的关系

- **skill** = 注入主 agent 上下文的「结构化工作流指示」（告诉你怎么做）。
- **subagent** = 独立上下文 + 独立工具集 + 独立循环的「专职执行者」（替你做完并只回摘要）。
- 二者互补不冲突：skill 仍挂在需要它的 agent 上（如 paper_reader 内部仍可用 skill__load 加载「论文综述」工作流）。本阶段不合并、不迁移 skill。

### DoD
- 父层 tool list 含 `paper_discovery` / `paper_reader` / `paper_ingest` 三枚工具，各自仅暴露其声明工具子集（断言：`paper_reader` 拿不到 `process_paper`）。
- 一个 plan 查询能把「找论文」派给 `paper_discovery`、「精读」派给 `paper_reader`，各自独立上下文（父层消息列表里**看不到** fetch_content 的大段正文）。
- 引用不回归：subagent 产出的 `[N]` 在父层 synthesize 后被正确解析为 paper/page。
- `agent/tests/test_subagents.py`：断言 `as_tool` 包装的 subgraph 返回摘要、工具子集正确、destructive 工具经权限门。

---

## 4. 执行顺序与依赖

```
Phase 7（Plan 骨架，target="tool" 可独立跑）
      │
      ▼
Phase 8.1（subagent 运行时工厂）
      │
      ▼
Phase 8.2（3 个科研 subagents）
      │
      ▼
Phase 8.3（plan.target 指向 subagent + 父层委派）
      │
      ▼
收尾：README 同步 + 自检补齐
```

- **Phase 7 可先行独立交付**（Plan 拆步骤 → 每步走现有 search 子图），不阻塞在 subagent 上。
- Phase 8.1→8.2→8.3 是线性依赖，8.3 把两套能力焊起来。
- react 模式全程零改动，作为回归安全网。

---

## 5. DoD 汇总

- **Phase 7**：复杂查询产出多步 plan 并按序/并行执行；简单查询零新增 LLM 调用；`state.mode` 落地。
- **Phase 8**：3 个 subagent 各守其工具子集与专属 prompt；父层上下文隔离（看不到正文）；`[N]` 引用跨 subagent 边界不丢；destructive 仍走权限门。
- **横切**：`agent/README.md` 新建并同步（架构图、subagent 清单、Plan 模式说明）；每个新模块留一个 `assert` 级自检。

---

## 6. 风险与回滚

1. **`asyncio.gather` 并行 subagent + contextvar 引用**：同 task 内共享 contextvar 是当前假设，动手前用一个双-subagent 并行的最小用例验证 `[N]` 不重复、registry 完整。
2. **subagent 丢失 `[N]` 标记**：LLM 复述时可能漏抄标记 → 引用解析为空。观测到即切 §3.4 的结构化结果回退方案。
3. **Plan 质量**：plan_node 可能产出坏步骤（目标不存在 / 参数错）。executor 对 `target` 未知、参数校验失败要**结构化降级**（回退 `react` 或报错给 synthesize），不能让 plan 坏掉整轮。
4. **`agent_node` 参数化改造回归**：加 `subagent_system` 字段是唯一对既有代码的改动，必须保证字段为空时行为与今天逐字节一致（react 模式回归自检兜住）。
5. **语言模型版本**：本计划不用 `interrupt()`/`RetryPolicy`，比原 Phase 6 风险低；并行用纯 `asyncio.gather`，不依赖 langgraph 并行原语。
6. **每阶段独立提交**，单阶段可回滚。

---

## 7. 非目标（YAGNI，显式声明）

- **不做** supervisor 网络 / 多 agent 投票 / agent 间自由对话。只有一个编排器 + 一层固定的 3 个 subagent，委派是单向的。
- **不做** 通用「多领域路由」——科研仍是唯一领域，subagent 是具体实现，不抽 DomainProvider 基类（沿用 review 护栏 #1）。
- **不做** 长期记忆（Phase 4）、引用图谱（Phase 5）、HITL（Phase 6）——仍处观察期，本计划不启动；其中 `paper_ingest` 的 destructive 门沿用现有 Phase 2 权限门，HITL 审批仍后置。
- **不引入** 新依赖（langgraph 已有的子图/工具机制够用；并行用 asyncio）。
- **不迁移** skill 机制；**不删除** 4 条 builtin 工具（渐进迁移，本阶段并存）。
