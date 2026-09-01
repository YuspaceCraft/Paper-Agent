# agent/domains — 领域子模块

三领域扩展（v10）的纯 Python 业务层，遵循 CLAUDE.md FastAPI 封装原则：
业务逻辑在此，`web/api/routers/*` 只做 HTTP 薄包装。

`agent/` 主包负责编排（graph/nodes/plan/subagents），domains 提供领域能力。

## creation（创作域，v10/Phase A）

- **DocStore**：`web/workspace/docs/{doc_id}/`（doc.json + sections/*.md + 拼接主 md + exports/*.docx）
- **doc 工具**（仅 creator subagent 可见）：`doc_create / doc_set_outline /
  doc_write_section / doc_get_state / doc_list / doc_export_docx`
- **入口**：`_ensure_writing_doc(title, outline, plan_steps) → doc_id`（plan_node 建 doc + 大纲）
- **docx 导出**：`_render_docx`（python-docx 最小 md→docx：`#/##/###` → Heading，`-` → List Bullet）
- 写入进度经 `stream.emit("doc_section", ...)` 透出到前端（章节树打勾）

工具表与用法见 `agent/README.md`「v10 领域扩展」与 `web/api/README.md`「创作（Creation）」。

## coding（编码域，v10 / Phase C 已实现）

- **ExperimentStore**：`web/workspace/experiments/{project}/_runs/{exp_id}/`
  （state.json + run.log + metrics 快照）；`run_experiment` 后台子进程（cwd=project 目录），
  结束自动归档指标进快照 + 确定性写入研究知识库。
- **外部委托** `delegate_code_task`：**MCP bridge 优先**（`.mcp.json` 编码 server 工具名
  带 `codex`/`delegate`/`claude_code` 前缀即接入），**CLI subprocess 兜底**（`AGENT_CODING_CMD`
  或探测 `claude`/`codex`）；无后端返回结构化错误（不 raise）。
- **研究知识库**：`web/workspace/studies/{topic}/knowledge.json`；实验记录由代码
  `_study_archive` 确定性写入，LLM 只读引用（防篡改）。
- agent/README「Coding Agent」节 + web/api/README「实验/研究知识库」节。

## coding 连接外部 coding agent（MCP bridge 接入指引）

1. 安装现成 coding MCP server（如 `claude-codex-bridge`、`codexmcp`、`claude-code-mcp`）。
2. 配进 `.mcp.json`（复用 `load_mcp_config`），server 暴露的工具名若含
   `codex`/`delegate`/`claude_code` 前缀，`delegate_code_task` 会自动优先走它。
3. 无 server 时设置 `AGENT_CODING_CMD=claude`（或 `codex`）走 CLI 兜底。
   两者都缺 → `delegate_code_task` 返回结构化错误，coder 明确告知用户。