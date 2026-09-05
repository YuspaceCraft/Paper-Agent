# 对话中心化工作区重构设计（conversation-centric workspace）

> 状态：**实施中**（2026-09-05 评审通过，±半日已完成 L1/L3/L4 后端 + L2 前端主体）。三处关键决策已确认：右侧工作台面板 / 围栏+信任确认 / manifest + study KB。
>
> **实施偏差记录**：
> - 围栏+信任确认：本次落地「工作目录围栏 + 后台/kill/超时 + 命令以用户权限执行」；
>   **首次运行项目的信任确认 UI 未落地**（保持现有直接运行行为），待前端实现时补。
> - 前端「文件」Tab 当前固定 root='project'（文献+写作根），未随实验绑定切换 experiments 根。
> - 全屏 WriterView/ExperimentView 不再被路由引用（由 DocPanel/ExperimentPanel 取代），文件保留作参考。
> 关联：[agent/state.py `AgentState`](agent/state.py)、[agent/subagents.py](agent/subagents.py)、[agent/workspace_config.py](agent/workspace_config.py)、[web/frontend/src/renderer/src/App.tsx](web/frontend/src/renderer/src/App.tsx)、[agent/supervisor.py](agent/supervisor.py)

## 1. 目标与痛点

文献问答、论文写作、实验三类任务当前是**各自独立的持久型窗口**（`App.tsx` 的 `domain` 切换：`write` / `experiment` 会整体替换对话区）。三处痛点：

1. **记忆在对话里，工作区在对话外**：文献问答的记忆在 agent 的 `thread_id` 状态栈，写作/实验各自独立，没有任何字段把「这个对话正在写哪个 doc / 属于哪个实验项目」写进对话状态。
2. **三类任务入口不统一**：用户需要从对话进入文献问答，再顺滑进入写作/实验，而不是跳去另一个窗口。
3. **实验的项目路径 / 沙箱 / 委托边界未定义**：项目名是随调用方乱传的字符串；运行命令无沙箱约定；主 agent 与外部 coding agent 之间没有确定的信息交换契约。

## 2. 核心分界（全文最重要一条）

- **根是全局的**：`project_path` / `experiments_path`（`settings.json`）决定「数据放哪」，是应用级偏好。
- **项目/文档绑定是对话级的**：`active_project` / `active_doc_id` 存于对话状态（`AgentState.context`），是「这场对话在做什么」。
- **项目文件夹与文档是持久工件**：实验/文档是研究资产，**长于对话**；对话是指针，不是所有权。

## 3. 已确认的三处决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 前端布局 | **右侧工作台面板**（文档/实验/文件三 Tab，绑定当前对话） | 对话永不被替换，复用 WriterView/ExperimentView 内部组件，改造集中在 App.tsx 布局 |
| 沙箱边界 | **围栏 + 信任确认**（任务执行限项目目录 + 后台/kill/超时 + 首次运行项目需信任确认） | 桌面端 + conda 现实约束；诚实标注非安全隔离 |
| 跨对话持久记忆 | **manifest + study KB** | 项目 manifest（入口/关键文件/基线）+ 已有 `study_context` 知识库作为跨对话记忆；删对话不丢研究状态 |

## 4. 分层设计

### L1 — 对话上下文状态（记忆连通）

给 `AgentState` 加结构化 `context`，由 **tool 事件自动维护**：

```python
context = {
  active_doc_id:   str | None     # doc_create / doc_write_section 触发
  active_project:  str | None     # run_experiment / delegate_code_task 触发
  study_topic:     str | None     # study_context / study_add_hypothesis 触发
  resolved_papers: [...]          # 已有 resolved 的别名扩展
  recent_experiments: [exp_id]    # 本轮会话跑过的实验（最近 N 个）
}
```

记忆连通三通道：
- **文献→写作**：creator task 生成器读 `active_doc_id` + `resolved_papers`。
- **文献→实验**：coder task 生成器带 `resolved_papers` + project manifest。
- **实验→写作（现缺口）**：creator 目前只能读论文（`search_papers`/`fetch_content`），必须加入 `read_metrics` / `study_context` 注入源，否则「实验章写进论文」无法引用真实指标。

零状态原则不冲突：记忆在父层，task 注入到子层；子 agent 依旧零状态。

### L2 — 单一入口：对话为中心 + 绑定面板

```
[左] 对话列表 | [中] 对话（永远是中心） | [右] 该对话的「工作台」面板
```

- 右面板 Tab：**文档**（绑定对话的 active doc 章节树+编辑器）、**实验**（绑定对话的 active project 运行列表+指标+日志）、**文件**（FileExplorer，root 跟随 active_project/active_doc）。
- 对话流内有**内联状态卡**（沿用 plan/tool card 的 SSE 模式）：写章节时出现「第 3 章·写作中」卡；跑实验时出现「exp_xxx·运行中」卡，可点击展开为右面板。
- 原全屏 WriterView/ExperimentView **降级为「从卡片展开」的绑定视图**，展开后不丢对话。
- 绑定是**动作的后果，不是前置弹窗**：跑实验 → 面板变那个项目；agent 写章节 → 面板变那篇文档。新建对话不弹「请选择项目」。

### L3 — 项目路径绑定语义

绑定规则（懒绑定 + 自动 + 可手动覆盖）：
1. **已有实验史**：对话历史上只碰过一个 project → 绑定它。
2. **用户点名**：请求里出现项目名，或显式工具 `set_experiment_project(project)` → 绑定它。
3. **否则**：自动建 `{experiments_root}/{主题slug}`（slug 从对话话题生成，可读、跨对话可复用，**不用 UUID**），写 project manifest。

- **已有对话（旧数据）**：绑定从历史推导；有歧义按第一条 `run_experiment` 动作懒绑定，不迁移不弹窗。
- **删除对话 ≠ 删除资产**：项目文件夹、实验结果、doc 均不随对话删除；删除仅是解绑 `active_project`（manifest 记为 orphan），运行中实验继续系统级完成。

### L4 — 委托契约与能力边界

**能力切分**：

| 能力 | 谁负责 | 形态 |
|------|--------|------|
| 决策/编排 | 主 agent | prompt + context 注入 |
| 改代码 | 外部 coding agent（`delegate_code_task`） | 编辑项目文件 + 返回 summary |
| 执行（`run_experiment`） | 平台工具（薄层） | 项目目录下后台子进程 + 日志流 + 指标解析 |
| 可视化 | 前端 | 渲染 manifest / 状态 / 指标 |

主 agent **不需要**写脚本能力，也不实现执行器——执行器是平台基础设施，写代码委托 coding agent。`delegate_code_task` 不可用时兜底：只跑 manifest 已登记命令 / 项目已有脚本，不手写文件。

**信息交换契约 = 每个项目一个 manifest**（`{project}/project.json`，机器可解析、零状态，贴合项目 Prompt 原则）：

```json
{
  "project": "RMNet-repro",
  "paper": "RMNet",
  "entry":  {"run": "python train.py", "data": "data/", "config": "config.yaml"},
  "key_files": ["train.py", "model.py", "metrics.py"],
  "metrics_schema": {"loss": {"lower_is_better": true}},
  "baseline": {"exp_id": "abc", "metrics": {}},
  "status": "draft | running | done",
  "last_run": "", "changelog": []
}
```

- `run_experiment` / `delegate_code_task` / `git_commit` **回写** manifest。
- 委托 prompt **带 manifest** → coding agent 知道入口、关键文件、结果上报格式。
- UI 渲染为快捷动作（Run / 打开关键文件 / 看基线对比）。

**沙箱（围栏 + 信任确认）**：
1. 工作目录围栏：运行/委托只在项目目录内（generic_provider 路径越界拒绝扩到项目级根）。
2. 可观测 + 可终止：后台子进程、可 kill、超时、环境钉死（`DEMO_PYTHON`）。
3. 信任边界诚实化：`run_experiment` 以用户权限执行，**非安全隔离**；首次对某项目运行需一次信任确认。
4. **改代码的 coding agent 须能行使开发者权限**（围栏内全权），否则委托无意义。防的是无意识越界与失控，不是恶意代码；不可信代码隔离（容器/云端）超出桌面端范围。

## 5. 补漏（原提问未覆盖，评审时补齐）

1. **实验→写作注入缺口**：creator 需增加 `read_metrics` / `study_context` 注入源。
2. **跨对话连续性**：新对话「接着优化项目」必须从 manifest + study KB 拿基线，不从旧对话拿。
3. **supervisor 绑定回写**：worker 跑在 `thread_id=task_id`，不修改父对话 state；`active_project` 在 **API/SSE 层按事件归因**到对话（复用 `forcedThread` 归因模式），同步/派发路径都成立。
4. **对话删除语义**：见 L3。
5. **自动创建项目命名**：slug 从对话话题生成，否则切对话产生 `conv-3842aa` 垃圾目录。
6. **SSE 中途绑定**：tool_end 事件就触发面板切换，不冻结用户钉住的视图。
7. **多项目并发**：`active_project` 是「当前」不是「全部」；manifest + exp 元数据保留全集。
8. **执行权限模型与委托自主性冲突**：改代码=受信任 coding agent（围栏内全权），执行=平台 run 工具 + 首次信任确认，两个「沙箱」概念不混。

## 6. 落地路线

1. **后端 L1+L4 最小集**：`context` state + tool 事件归因 + project manifest（读/写）+ `set_experiment_project` 工具 + creator 注入源加 metrics。
2. **后端 supervisor 归因**：派发 worker 的 tool_end 归因到 `parent_thread`。
3. **前端 L2**：`domain` 改面板 Tab，复用 WriterView/ExperimentView 组件，加内联状态卡。
4. **迁移**：旧线程懒绑定；旧项目跑任一实验时自动生成 manifest。

## 7. 未决/风险

- 主题 slug 生成需要对话标题能力（thread 目前只有 `title` 字段，无语义摘要）。
- 右面板 Tab 与现有 FileExplorer 宽度（`--right-w`）共用一套布局度量，需确认不冲突。
- 首次运行信任确认的 UI 形态（弹窗 vs 行内确认）待前端实现时定。