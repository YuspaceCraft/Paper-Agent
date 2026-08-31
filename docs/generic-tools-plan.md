# 通用工具集补全计划（v1，待审）

> 定位依据：项目已从「科研文献助手」调整为「通用 Agent 底座」（见 [agent-refactor-plan-review.md](agent-refactor-plan-review.md)）。
> 现状：通用层骨架（`Provider` 抽象 / `dispatcher` / `safety` 权限门）已就位，但**工具实例全是科研领域**（4 builtin + 1 MCP arxiv + 1 skill paper-review），通用工具集这一层为空。
> 本文只做规划，不落地。审完后再动手。

---

## 1. 判断

- **有骨架、没实例**：`dispatcher` 统一超时/重试/审计、`safety` 权限门都建好了，但没有一个「横切底座」该有的通用工具（文件 / 网络 / 系统能力）塞进去。
- **科研工具保留为「第一个领域 provider 实例」**，不改名、不拆、不提前抽象（守住 review 护栏）。
- **通用工具 ≠ 领域抽象**：文件/网络/系统能力是横切底座，新建一个 provider 承载是合理的，不算「为单一领域建空抽象」。

## 2. 目标工具清单

分三批。落点统一新建 `agent/providers/generic_provider.py`（`GenericProvider`），在 `ensure_tools()` 的 providers 列表插入（builtin 之后、mcp 之前）。

### 第一批：只读 + 零/轻依赖（先做）

| 工具 | 签名 | 描述（LLM 可见） | annotations | 依赖 | 备注 |
|---|---|---|---|---|---|
| `read_file` | `(path: str)` | 读取工作区内的文本文件内容 | `readOnlyHint=True, idempotentHint=True` | stdlib | 路径 resolve 后必须在允许根目录内 |
| `list_dir` | `(path: str = ".")` | 列出目录下的文件/子目录 | `readOnlyHint=True, idempotentHint=True` | stdlib | 同上路径约束 |
| `get_time` | `()` | 返回当前日期时间（本地时区） | `readOnlyHint=True, idempotentHint=False` | stdlib | 无参，走空 args 模型 |
| `calculator` | `(expr: str)` | 安全求值算术表达式 | `readOnlyHint=True, idempotentHint=True` | stdlib `ast` | `ast` 白名单求值，禁裸 `eval` |

### 第二批：破坏性 + 通用网络（权限门自动覆盖）

| 工具 | 签名 | 描述（LLM 可见） | annotations | 依赖 | 备注 |
|---|---|---|---|---|---|
| `write_file` | `(path: str, content: str)` | 写入文本文件（覆盖） | `readOnlyHint=False, idempotentHint=True` | stdlib | 破坏性，`AGENT_USER_ROLE=user` 时被 `tool_allowed` 拦截 |
| `fetch_url` | `(url: str)` | 抓取 URL 的 HTML/正文 | `readOnlyHint=True, idempotentHint=True` | `httpx`（已有） | 只读网络，需限超时 + 限制返回长度 |
| `web_search` | `(query: str, top_k: int = 5)` | 通用网络搜索 | `readOnlyHint=True, idempotentHint=True` | **外部搜索源，待定** | 见 §4 |

### 第三批：观察期（出现真实需求再上）

| 工具 | 签名 | 说明 |
|---|---|---|
| `run_shell` | `(command: str)` | 命令执行。`readOnlyHint=False`，高危，需单独定沙箱策略 |

## 3. 权限矩阵

复用 [safety.py](../agent/safety.py) `tool_allowed`（读 `readOnlyHint is False` → destructive），零新增代码：

| 角色 | 只读工具（read_file/list_dir/get_time/calculator/fetch_url/web_search） | write_file | run_shell |
|---|---|---|---|
| `admin`（默认） | ✅ | ✅ | ✅ |
| `user` | ✅ | ❌ 拦截 → `permission_denied` | ❌ |

## 4. 待你拍板的开放项（动手前必须定）

1. **`web_search` 搜索源**：项目没有通用搜索 API。选项 —— Tavily（付费 key）/ DuckDuckGo 免费抓取 / DashScope 插件 / 先只做 `fetch_url` 跳过 `web_search`。**这条不定，第二批就做不了 web_search。**
2. **`run_shell` 是否进规划**：威力大但攻击面也大。单用户本地默认 `admin`，权限门形同虚设。建议保持观察期，除非明确要。
3. **文件工具的工作区根目录**：`read_file`/`write_file`/`list_dir` 允许访问哪个根？建议「项目根 + `data/`」，禁止越界。要更严（只 `data/`）还是更宽（整个家目录）？

## 5. 分期与自检约定

- **第一批**（只读，今天可落地）：`read_file` / `list_dir` / `get_time` / `calculator`。每个留一个 `assert` 自检（`agent/tests/test_generic.py`），核心断言：路径越界被拒、`calculator` 拒绝非算术输入。
- **第二批**（依赖开放项定了才做）：`write_file` / `fetch_url` / `web_search`。自检断言：`user` 角色无法触发 `write_file`（复用 Phase 2 权限门逻辑）。
- **第三批**：观察期，不落地。

## 6. 附：现有科研工具「粒度」问题（不默认改）

`search_papers`（列全部 + 语义搜）、`fetch_content`（概览 + 深读）各是「一个工具两模式」。通用底座倾向单一职责可组合，但这与 v5「6→2 合并省 token」的设计相反。**不默认拆**，留作独立决策；若拆，收益是更清晰，代价是 tool definition 变多、token 涨。
