# ADR-0004：Prompt 与配置版本化

- 状态：Accepted
- 日期：2026-09-15
- 相关：`agent/prompt_store.py`、`agent/core/prompt_registry.py`、`agent/core/configuration.py`、`agent/config.yaml`、`web/api/routers/config.py`

## 背景

Prompt 文本在 `agent/prompts.py` 常量里，运行上限在 `agent/config.yaml` 与 env
之间，工具开关在 `web/workspace/config.json`。一次执行事后无法回答「当时用的
是哪个 prompt、哪份配置」——同一问题在不同时间跑出不同结果时无从归因。

## 决策

1. **一次运行绑定不可变快照**：`ConfigurationSnapshot`（revision / config_hash /
   model / limits / disabled_tools / feature_flags / prompt_versions /
   tool_registry_hash）在 turn 开始时冻结，并写入 LangSmith metadata
   （`agent/core/configuration.py`、`agent/graph.py::prepare_turn`）。
2. **Prompt 版本标识**：每个 prompt 绑定为 `prompt_id@version#checksum`
   （`PromptSpec.binding`），`checksum = sha256(template)[:12]`。运行期
   `ExecutionContext.prompt_bindings` 记录 type → binding，进 trace 与离线报告。
3. **发布契约**：源文件 `prompts/<domain>/<prompt_id>/<version>.yaml`，字段
   `id/version/status/template` 必需，`type/schema/variables/locale/
   evaluation_suite` 可选；`active` 是默认版本，`canary` 仅按配置百分比分流。
   扁平布局
   `web/workspace/prompts/<ID>.yaml`（配置中心当前用法）继续被支持，
   两者并存时**嵌套布局优先**。
4. **失败降级**：文件缺失、id 不匹配、非 active、模板为空 → 记录
   `prompt_fallback` 审计事件并回退内置常量，**绝不阻断请求**。
5. **变量契约**：模板变量由 `variables` 声明，`PromptSpec.render()` 拒绝未声明
   变量；敏感字段先脱敏再入 prompt，禁止 prompt 自行拼接未知上下文。
6. **发布状态机**：`draft → canary → active → deprecated → retired`；升级以
   回归集结果决定，不允许覆盖既有版本号（改内容必须递增版本）。
7. **确定性分流**：canary 规则驻留配置中心（`prompts.canary`），按
   `hash(thread_id, prompt_id, version) % 100` 分流；同一会话固定命中同一版本。
   生效版本的 `binding/checksum/evaluation_suite` 一并冻结进 ExecutionContext。

## 替代方案

- **只保留内置常量**：无法灰度、无法回滚单条 prompt；已否决。
- **把 prompt 存数据库**：本地单用户场景引入迁移/备份负担，且不利于 code
  review；已否决。
- **用 hash 代替 version**：hash 能定位内容但无法表达「同一逻辑版本的第 2 次
  修订」，也无法表达 draft/deprecated；因此版本 + checksum 同时保留。

## 兼容性

- `get_prompt(prompt_id, fallback)` 行为不变；新增可选 `unit_id/config`
  参数用于 canary 分流，不传时仍解析 active。`resolve_prompt`/
  `load_prompt_spec` 为新增 API。
- `active_prompt_versions()` 保留（legacy 视图）；新增
  `active_prompt_bindings()` 报告真正生效的内容。

## 回滚

- 删除/改名 workspace 下的 prompt YAML、关闭 canary 规则，或把 status 改为
  draft 即刻回到 active/内置版本；无需改代码、无需重启后清理缓存。
- `ConfigurationSnapshot` 是只读视图，回滚不影响任何运行配置。

## 验收指标

- 任意 run 可由 metadata 复现「模型 + prompt 版本 + 工具版本 + 配置 hash」组合。
- 每个 active prompt 关联最小回归集（`evaluation_suite`），升级有 A/B 或
  canary 依据。
- prompt 校验失败率（`prompt_fallback`）作为配置健康度指标，异常升高即告警。
