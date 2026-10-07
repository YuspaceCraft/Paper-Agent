# 状态机与 API 契约

## 1. 主图状态机（`agent/graph.py::build_graph`）

```text
START → understand → memory → context → route_intent
  ├─ resolve → domain → decide_mode
  │     ├─ react → search(subgraph: agent ↔ tools) → synthesize → END
  │     └─ plan  → plan → executor → verify → synthesize → END
  ├─ chat    → chat    → END
  ├─ clarify → clarify → END
  └─ task    → task    → END
```

- 运行态骨架（设计文档 §3.2）：`bootstrap → guard_input → assemble_context →
  route → execute(sub)graph → verify → persist_memory → respond`。
  现状：bootstrap = `prepare_turn`（冻结配置 + ExecutionContext），
  assemble_context = `memory` + `context`，verify = `verify`，
  persist/respond = `synthesize` + checkpoint 写入。
- **每轮元数据**：`prepare_turn` 生成 `run_id`（LangSmith 根 run），
  `trace_id = run_id.hex`，并写入 metadata（见 ADR-0002）。
- **副作用审批**：需要人工确认的调用由 `ToolPolicy` 判定为 `approval`
  → graph node 内 `interrupt()` → SSE `approval_required` / `/api/agent/resume`
  → `Command(resume={"approved": bool})` 从 checkpoint 续跑；拒绝返回
  `APPROVAL_DENIED`，不执行 adapter。
- **受限 subagent**：`agent/supervisor.py` 以 `thread_id = task_id` 派发，
  与主图共用 `ToolGateway` 与 checkpointer；`request_review` → `gate` 节点
  `interrupt()` → `task_resume` 续跑。

### AgentState 允许/禁止

| 允许（可 checkpoint 的业务状态） | 禁止 |
|---|---|
| `messages`、`intent`、`entities`、`plan`、`verification` | API key / 原始密钥 |
| `context`（工作区绑定）、`context_decision`（预算元数据） | LLM/工具客户端、`RunnableConfig` |
| `active_tasks`、`doc_id`、`tool_result_cache` | 原始异常堆栈、未脱敏的检索全文 |

## 2. HTTP API（FastAPI，`web/api/routers/`）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/agent/chat` | 非流式一轮对话（冻结元数据 + `trace_id` 登记） |
| POST | `/api/agent/chat/stream` | SSE：`token` / `tool_start` / `tool_end` / `plan` / `doc_section` / `tasks` / `approval_required` / `done` / `error` |
| POST | `/api/agent/resume` | 回复副作用工具审批，`Command(resume={"approved": bool})` 从 checkpoint 续跑 |
| GET | `/api/agent/health` | 就绪状态、模型、工具数 |
| GET/PUT | `/api/config/tools`、`/experiment`、`/mcp`、`/skills`、`/limits` | 配置中心读写（工具开关/实验/ MCP / Skills / 限额） |
| GET/PUT | `/api/config/prompts` | Prompt active/canary 版本、percent 与 evaluation suite |
| GET/POST/PUT/DELETE | `/api/memory`、`/api/memory/{id}`、`/api/memory/policy` | typed 长期记忆查看/新增/删除/禁用 |
| POST | `/api/pdf/process`、`/process-local` | 入库（异步后台任务） |
| GET | `/api/pdf/tasks`、`/tasks/{id}`、`/tasks/stream` | 后台任务状态与流式通知 |
| POST | `/api/index/run`、`/reconcile`、`/search` | 索引与检索 |
| GET/POST | `/api/creation/docs...`（含 `export-docx`） | 写作文档与导出 |
| GET/POST | `/api/experiments/projects`、`/run`、`/{exp_id}/metrics`、`/{exp_id}/logs`、`/{project}/git`、`/{project}/manifest` | 实验运行、指标、日志、git 与 manifest |
| GET/POST | `/api/study/context`、`/hypotheses` | 研究知识库 |
| GET | `/api/reader/papers`、`/{paper}/sections`、`/{paper}/sections/{name}`、`/{paper}/abstract`、`/{paper}/chunks/{chunk}/context` | 论文阅读器 |
| GET | `/api/eval/traces/{trace_id}`、`/thread/{thread_id}`、`/runs`、`/runs/{run_id}`、`/runs/{run_id}/badcases`；POST `/single`、`/runs`、`/prune`、`/manifest/from-trace` | 评测与 trace 查询 |
| GET | `/api/tasks/search`、`/api/workspace/*` | 任务检索、工作区文件浏览 |

约定：

- 长任务一律返回 `task_id`（202/200），不阻塞请求；状态经 `/api/tasks` 或 SSE。
- `/api/agent/chat*` 的响应或 SSE `error` 事件只含**脱敏**文案与稳定 `code`。
- 所有写操作在 `web/api/routers` 内做薄封装，业务逻辑在 `agent/domains/*`。
