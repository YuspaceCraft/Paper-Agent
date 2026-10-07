# Prompt 配置中心

运行时按以下顺序解析活动版本，任一环节缺失/非法都回退到 `agent/prompts.py` 的内置版本，
且不会中断请求：

1. **嵌套发布布局（优先）**：`prompts/<domain>/<PROMPT_CONSTANT>/<version>.yaml`
   （设计文档 §6.1 / ADR-0004；`AGENT_PROMPT_DIR` 可覆盖根目录）。同名多版本时取
   `status: active` 的那份。
2. **扁平工作区布局**：`web/workspace/prompts/<PROMPT_CONSTANT>.yaml`
   （配置中心当前用法）。

每次解析出的绑定是 `id@version#checksum`（`checksum = sha256(template)[:12]`），写进
`ExecutionContext.prompt_bindings` → LangSmith metadata 与离线报告，用于回答
「这一轮到底用了哪份 prompt」。

## 文件契约

```yaml
id: CHAT_SYSTEM
version: 2026.09.15-1
status: active
template: |
  You are a concise research assistant.
```

可选字段（嵌套布局推荐全部填写；`PromptSpec` 会解析校验）：

```yaml
type: system            # system/router/planner/executor/synthesizer/judge/summary/safety
variables: [question]   # 模板变量白名单；出现未声明变量 → render() 报错
schema:                 # 输入契约（JSON Schema 片段）
  type: object
  required: [question]
locale: zh
evaluation_suite: chat-regression   # 该 prompt 必须关联的最小回归集
```

- `id` 必须等于文件名（例如 `CHAT_SYSTEM.yaml`）。
- `version` 为不可变发布标识；改内容必须递增版本，不能覆盖既有版本。
- 只有 `status: active` 生效；`draft`、`deprecated` 不会影响线上运行。
- `template` 必须是非空字符串，并保留该 Prompt 原有的 `.format(...)` 变量（如 `{question}`）。
- 嵌套布局的 `<version>` 必须与文件内 `version` 一致；解析失败会记录 `prompt_fallback`
  审计事件（可作配置健康度指标）。

## 治理规则

1. 提交 YAML 前，先在评测集运行对应节点的回归测试。
2. 发布后查看 LangSmith 的 `prompt_versions` / `prompt_bindings` metadata 与质量指标；
   内容 hash（checksum）可定位实际模板。
3. 发生回归时将状态改为非 `active` 或移除该文件，即刻回退内置 Prompt。
4. 不在模板、trace metadata 或配置文件中写入 API key、令牌或用户敏感信息。

> 状态：`draft → canary → active → deprecated → retired`。canary/A-B 按
> `experiment_id` + hash 分流尚未实现（设计文档 §13 剩余项），当前以「改状态即回滚」
> 作为最小可用发布手段。
