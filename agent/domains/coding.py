"""
coding.py — 编码域（v10 / Phase C）。

职责（遵循 CLAUDE.md FastAPI 封装原则：本模块 = 纯 Python 业务层，
`web/api/routers/experiments.py`/`study.py` 只做 HTTP 薄包装）：

- **ExperimentStore**：实验记录 `web/workspace/experiments/{project}/runs/{exp_id}/`
  （state.json + run.log + metrics 文件），后台子进程跑命令。
- **指标解析**：`runs/{exp_id}/` 下的 `metrics.json` / `metrics.csv` → 规范化 dict。
- **git 工具**：`git_status/diff/log/commit`（cwd 限定 experiments/{project}）。
- **delegate_code_task**：外部 coding agent 委托——**MCP bridge 优先，CLI subprocess
  兜底**（GitHub 调研定案 + 开箱可用原则）。后端可插拔：优先读 `.mcp.json` 中
  coding server 工具（见 `_coding_mcp_tool()`），不可用时走 `AGENT_CODING_CMD`
  /探测 `claude`/`codex` 头less CLI。prompt **不写死模型名**（模型由外部 CLI/Server 注入）。
- **研究知识库 study**：`web/workspace/studies/{topic}/knowledge.json`——实验记录由
  **确定性代码**在 exp 结束自动追加（agent 只读引用，防 LLM 篡改事实）。

安全：project 名白名单字符；路径全部经 `resolve_workspace_path` 越界判定并限定在
experiments/ 内；`run_experiment`/`git_commit`/`delegate_code_task` 走权限门
（CodingProvider destructive）。
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from langchain_core.tools import tool

from ..providers import ToolDef, ToolProvider
from ..providers.generic_provider import resolve_workspace_path
from ..safety import tool_allowed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / "web" / "workspace" / "experiments"
STUDY_ROOT = PROJECT_ROOT / "web" / "workspace" / "studies"

_SLUG_RE = __import__("re").compile(r"[^a-zA-Z0-9_.-]+")


def _ok(data: dict | list) -> str:
    from agent.tool_contract import ok as _ok_contract
    return _ok_contract(data)


def _err(error: str, error_type: str = "param_error", **ctx) -> str:
    from agent.tool_contract import err as _err_contract
    return _err_contract(error_type, error,
                         next_action="Fix the arguments and retry.", **ctx)


# ---- path safety ----

def _safe_project(name: str) -> str:
    return _SLUG_RE.sub("_", (name or "").strip()) or "default"


def _project_dir(project: str) -> Path:
    """experiments 项目目录（安全 slug）。路径必须落在 EXPERIMENTS_ROOT 内。"""
    d = (EXPERIMENTS_ROOT / _safe_project(project)).resolve()
    try:
        d.relative_to(EXPERIMENTS_ROOT.resolve())
    except ValueError:
        raise PermissionError(f"project escapes experiments root: {project}")
    return d


def _exp_dir(exp_id: str) -> Path:
    # exp_id 是生成的 hex uuid，天然安全
    return EXPERIMENTS_ROOT / "_runs" / exp_id


_RUNS = EXPERIMENTS_ROOT / "_runs"


def _state_path(exp_id: str) -> Path:
    return _exp_dir(exp_id) / "state.json"


def _load_exp(exp_id: str) -> dict | None:
    p = _state_path(exp_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _load_projects() -> list[str]:
    if not EXPERIMENTS_ROOT.exists():
        return []
    return sorted(
        p.name for p in EXPERIMENTS_ROOT.iterdir()
        if p.is_dir() and p.name != "_runs"
    )


def _list_experiments(project: str | None = None) -> list[dict]:
    out: list[dict] = []
    if not _RUNS.exists():
        return out
    for d in sorted(_RUNS.iterdir(), reverse=True):
        st = _load_exp(d.name)
        if not st:
            continue
        if project and _safe_project(st.get("project", "")) != _safe_project(project):
            continue
        out.append(_public_exp(st))
    return out


def _public_exp(st: dict) -> dict:
    return {
        "exp_id": st["exp_id"],
        "project": st.get("project", ""),
        "name": st.get("name", ""),
        "command": st.get("command", ""),
        "status": st.get("status", "unknown"),
        "exit_code": st.get("exit_code"),
        "git_sha": st.get("git_sha", ""),
        "metrics": _summarize(st.get("metrics", {})),
        "created_at": st.get("created_at", ""),
        "finished_at": st.get("finished_at", ""),
    }


def _summarize(metrics: dict) -> dict:
    vals = {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float))}
    return {k: round(v, 6) if isinstance(v, float) else v
            for k, v in dict(sorted(vals.items(), key=lambda kv: str(kv[0]))).items()}


# ---- metrics parsing ----

def _parse_metrics(exp: dict) -> dict:
    """解析 runs/{exp_id}/ 下 metrics.json / metrics.csv → 规范化 dict。

    tfevents（tensorboard 事件）不在 MVP 解析范围——文档注明，需要时引入
    tensorboard 库。返回合并后 dict（json 优先，csv 追加）。
    """
    merged: dict = {}
    d = _exp_dir(exp["exp_id"])

    jp = d / "metrics.json"
    if jp.exists():
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
        except (ValueError, OSError):
            pass

    cp = d / "metrics.csv"
    if cp.exists():
        try:
            lines = [ln.strip() for ln in cp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                for row in lines[1:]:
                    cells = [c.strip() for c in row.split(",")]
                    if len(cells) == len(headers):
                        for k, v in zip(headers, cells):
                            try:
                                merged[k] = float(v)
                            except ValueError:
                                merged[k] = v
        except (ValueError, OSError):
            pass
    return merged


# ---- run_experiment（后台子进程） ----

_bg_tasks: set[asyncio.Task] = set()


async def _watch(exp: dict, proc: asyncio.subprocess.Process) -> None:
    """后台监听子进程：日志尾部实时落盘 → 结束更新 state → 知识库归档。"""
    from ..observability import log_event

    logf = _exp_dir(exp["exp_id"]) / "run.log"
    try:
        if proc.stdout is not None:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                with open(logf, "ab") as f:
                    f.write(chunk)
    except Exception:
        pass
    rc = await proc.wait()
    state = _load_exp(exp["exp_id"]) or exp
    state.update({
        "status": "done" if rc == 0 else "failed",
        "exit_code": rc,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    # 子进程 cwd = project 目录：把命令写出的指标文件归档进本次实验快照
    _archive_metrics(state)
    state["metrics"] = _parse_metrics(state)
    _save_state(state)
    _study_archive(state)
    log_event("experiment_finished", node="coding", exp_id=state["exp_id"],
              status=state["status"], exit_code=rc)


def _archive_metrics(state: dict) -> None:
    """project 根下的 metrics.json / metrics.csv → runs/{exp_id}/（实验快照自包含）。"""
    import shutil as _sh
    src = _project_dir(state.get("project", "default"))
    dst = _exp_dir(state["exp_id"])
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("metrics.json", "metrics.csv"):
        p = src / fn
        if p.exists():
            try:
                _sh.copy2(p, dst / fn)
            except OSError:
                pass


def _save_state(state: dict) -> None:
    d = _exp_dir(state["exp_id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


async def _spawn(exp: dict) -> None:
    cmd = exp.get("command", "")
    cwd = _project_dir(exp.get("project", "default"))
    cwd.mkdir(parents=True, exist_ok=True)
    state = _load_exp(exp["exp_id"]) or exp
    state["status"] = "running"
    _save_state(state)

    # shell=True：命令是用户显式提供的（形如 `python train.py` / `bash run.sh`），
    # 它在用户自己的实验目录里运行——与 Web 上传 run 一致的可信边界。
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:
        state = _load_exp(exp["exp_id"]) or exp
        state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                      "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        _save_state(state)
        return

    task = asyncio.create_task(_watch(state, proc))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# ---- study knowledge base（确定性写入 / 只读引用） ----

def _study_path(topic: str) -> Path:
    return STUDY_ROOT / _SLUG_RE.sub("_", (topic or "general").strip()) / "knowledge.json"


def load_study(topic: str) -> dict:
    p = _study_path(topic)
    if not p.exists():
        return {"topic": _SLUG_RE.sub("_", (topic or "general").strip()),
                "hypotheses": [], "experiments": [], "findings": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {**data, "topic": _SLUG_RE.sub("_", (topic or "general").strip())}
    except (ValueError, OSError):
        return {"topic": topic, "hypotheses": [], "experiments": [], "findings": []}


def _save_study(topic: str, data: dict) -> None:
    p = _study_path(topic)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _study_archive(exp: dict) -> None:
    """实验结束 → 确定性归档到研究知识库（agent 不直接写，防 LLM 篡改事实）。"""
    topic = _SLUG_RE.sub("_", (exp.get("project", "general") or "general").strip())
    study = load_study(topic)
    rec = {
        "exp_id": exp["exp_id"],
        "name": exp.get("name", ""),
        "status": exp.get("status", ""),
        "git_sha": exp.get("git_sha", ""),
        "command": exp.get("command", "")[:300],
        "metric_summary": _summarize(exp.get("metrics", {})),
        "finished_at": exp.get("finished_at", ""),
    }
    study.setdefault("experiments", []).append(rec)
    # 裁剪：单 topic 保留最近 100 条
    study["experiments"] = study["experiments"][-100:]
    _save_study(topic, study)


def _git_sha(project: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_project_dir(project)), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


# ---- 工具实现（agent 可调用，返回统一 JSON 信封） ----

@tool
async def run_experiment(project: str, command: str, name: str = "") -> str:
    """Run a command as a background experiment inside experiments/{project}.
    The project folder receives web/workspace/experiments/{project} (created if
    missing). Logs stream to runs/<exp_id>/run.log; a metrics.json/metrics.csv
    written by the command is parsed automatically on completion. Returns
    {"ok": true, "data": {exp_id, status: "running"}} — poll experiment_status."""
    exp_id = uuid.uuid4().hex[:10]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    exp = {
        "exp_id": exp_id,
        "project": _safe_project(project),
        "name": name or os.path.basename(command)[:60],
        "command": command,
        "status": "pending",
        "exit_code": None,
        "git_sha": _git_sha(project),
        "metrics": {},
        "created_at": now,
        "finished_at": "",
    }
    _save_state(exp)
    await _spawn(exp)
    return _ok({"exp_id": exp_id, "status": "running", "project": exp["project"]})


@tool
async def experiment_status(exp_id: str) -> str:
    """Return an experiment's current state — status, exit code, git_sha, metrics
    summary and the last ~10k chars of its log tail."""
    st = _load_exp(exp_id)
    if not st:
        return _err(f"experiment '{exp_id}' not found (list via experiment_list)",
                    error_type="param_error")
    logf = _exp_dir(exp_id) / "run.log"
    tail = ""
    if logf.exists():
        tail = logf.read_text(encoding="utf-8", errors="replace")[-10_000:]
    return _ok({**(_public_exp(st)), "log_tail": tail})


@tool
async def experiment_list(project: str = "") -> str:
    """List experiments (optional project filter), newest first."""
    return _ok({"experiments": _list_experiments(project or None)})


@tool
async def read_metrics(exp_id: str, metric_key: str = "") -> str:
    """Read an experiment's parsed metrics (from metrics.json / metrics.csv in
    its run folder). Empty metric_key returns all numeric metrics."""
    st = _load_exp(exp_id)
    if not st:
        return _err(f"experiment '{exp_id}' not found", error_type="param_error")
    metrics = _parse_metrics(st) or st.get("metrics", {})
    if metric_key:
        if metric_key not in metrics:
            return _err(f"metric '{metric_key}' not found in experiment", error_type="param_error",
                        available=list(metrics.keys())[:20])
        return _ok({"metric": metric_key, "value": metrics[metric_key]})
    return _ok({"metrics": _summarize(metrics)})


@tool
async def git_status(project: str) -> str:
    """Show the experiment project's working-tree status (files changed since HEAD)."""
    d = _project_dir(_safe_project(project))
    if not (d / ".git").exists():
        return _err(f"{project} is not a git repo (git init first)", error_type="param_error")
    return _git(d, ["git", "-C", str(d), "status", "--short"])


@tool
async def git_diff(project: str) -> str:
    """Show uncommitted diffs in the experiment project."""
    d = _project_dir(_safe_project(project))
    if not (d / ".git").exists():
        return _err(f"{project} is not a git repo", error_type="param_error")
    return _git(d, ["git", "-C", str(d), "diff"])


@tool
async def git_log(project: str, n: int = 10) -> str:
    """Show the experiment project's recent commit history."""
    d = _project_dir(_safe_project(project))
    if not (d / ".git").exists():
        return _err(f"{project} is not a git repo", error_type="param_error")
    return _git(d, ["git", "-C", str(d), "log", "--oneline", "-n", str(min(max(n, 1), 50))])


@tool
async def git_commit(project: str, message: str) -> str:
    """Commit all current changes in the experiment project (destructive, gated)."""
    d = _project_dir(_safe_project(project))
    if not (d / ".git").exists():
        return _err(f"{project} is not a git repo (git init first)", error_type="param_error")
    add = subprocess.run(["git", "-C", str(d), "add", "-A"],
                         capture_output=True, text=True, timeout=30)
    if add.returncode != 0:
        return _err(add.stderr[:500], error_type="transient")
    cm = subprocess.run(["git", "-C", str(d), "commit", "-m", message],
                        capture_output=True, text=True, timeout=30)
    if cm.returncode != 0:
        return _err(cm.stderr[:500], error_type="transient")
    return _ok({"sha": _git_sha(project), "message": message})


def _git(d: Path, cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return _ok({"output": (out.stdout or out.stderr)[:6000]}) if out.returncode == 0 \
            else _err((out.stderr or out.stdout)[:600], error_type="transient")
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}", error_type="transient")


# ---- delegate_code_task（外部 coding agent 委托） ----

_CODING_CMDS = ("claude", "codex")


@tool
async def delegate_code_task(project: str, prompt: str, timeout: int = 600) -> str:
    """Delegate a coding task to an EXTERNAL coding agent (MCP bridge first,
    CLI subprocess fallback), running inside web/workspace/experiments/{project}.

    The delegate edits/writes files directly in the project folder and returns
    a summary — the parent then inspects changes via git_diff. External model
    choice is injected by the backend; the prompt itself must be self-contained.
    Returns {"ok": true, "data": {backend, output, changed_files}}."""
    cwd = _project_dir(project)
    cwd.mkdir(parents=True, exist_ok=True)

    # 1) MCP bridge：.mcp.json 中出现 coding 委托工具（如 codex__exec）则优先
    mcp = await _coding_mcp_tool()
    if mcp is not None:
        try:
            out = await mcp.ainvoke({"prompt": prompt, "cwd": str(cwd)})
            return _ok({"backend": "mcp", "output": str(out)[:10000], "changed_files": _changed_files(cwd)})
        except Exception as exc:
            return _err(f"mcp delegate failed: {type(exc).__name__}: {exc}", error_type="transient")

    # 2) CLI subprocess fallback（开箱可用；AGENT_CODING_CMD 覆盖后端）——等待/依赖本机头less CLI
    exe = os.environ.get("AGENT_CODING_CMD", "")
    if not exe:
        for c in _CODING_CMDS:
            if shutil.which(c):
                exe = c
                break
    if not exe:
        return _err(
            "no coding backend: set AGENT_CODING_CMD (e.g. 'claude') or add a coding "
            "MCP server to .mcp.json",
            error_type="unknown", next="Install a headless coding CLI and retry.")
    args = shlex.split(exe)
    name = os.path.basename(args[0]).lower()
    if "codex" in name:
        cli_args = [*args, "exec", prompt, "--json"]
    else:
        cli_args = [*args, "-p", prompt]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cli_args, cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out_b, _ = await asyncio.wait_for(proc.communicate(),
                                          timeout=min(max(timeout, 60), 3600))
    except asyncio.TimeoutError:
        proc.kill()
        return _err(f"delegate timed out after {timeout}s", error_type="transient",
                    partial=out_b[-2000:] if "out_b" in dir() else "")
    except Exception as exc:
        return _err(f"delegate launch failed: {type(exc).__name__}: {exc}",
                    error_type="transient")
    rc = proc.returncode
    text = out_b.decode("utf-8", errors="replace")
    payload = {"backend": name or "cli", "exit_code": rc, "output": text[:10000]}
    if rc != 0:
        return _err(text[:1000] or "delegate failed", error_type="transient")
    if text.strip().startswith("{"):
        try:
            data = json.loads(text)
            payload["parsed"] = data if isinstance(data, dict) else None
        except ValueError:
            pass
    payload["changed_files"] = _changed_files(cwd)
    return _ok(payload)


async def _coding_mcp_tool():
    """探测 coding 委托 MCP 工具（.mcp.json）。命名约定：工具名包含
    `exec`+`codex` / `delegate` / `code_exec` 视为编码委托。返回 BaseTool 或 None。"""
    try:
        from ..tools import get_base_tools
        for t in get_base_tools():
            n = t.name
            if ("codex" in n or "claude_code" in n or "delegate" in n) and n.startswith(
                    ("codex", "claude", "delegate")):
                return t
    except Exception:
        pass
    return None


def _changed_files(cwd: Path, limit: int = 30) -> list[str]:
    """git 未提交变更文件清单（无 git 仓库时回退目录快照）。"""
    if (cwd / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(cwd), "status", "--porcelain"],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return [ln[3:].strip() for ln in out.stdout.splitlines() if ln.strip()][:limit]
        except Exception:
            pass
    try:
        return sorted(str(p.relative_to(cwd)) for p in cwd.rglob("*")
                      if p.is_file() and ".git" not in p.parts)[:limit]
    except Exception:
        return []


@tool
async def study_context(topic: str) -> str:
    """Read the study knowledge base for a topic: hypotheses, historical
    experiment records (with metric summaries + git sha), and findings.
    Injected into writing/experiment context for comparison baselines."""
    study = load_study(topic)
    exp_summary = []
    for e in study.get("experiments", [])[-8:]:
        exp_summary.append(
            f"- {e.get('name', '')} [{e.get('status', '')}] metrics={e.get('metric_summary', {})}"
        )
    return _ok({
        "topic": study.get("topic", topic),
        "hypotheses": study.get("hypotheses", []),
        "recent_experiments": exp_summary,
        "findings": study.get("findings", []),
    })


@tool
async def study_add_hypothesis(topic: str, hypothesis: str) -> str:
    """Record a research hypothesis (append-only) into the study knowledge base."""
    study = load_study(topic)
    study.setdefault("hypotheses", []).append({
        "text": hypothesis, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _save_study(topic, study)
    return _ok({"topic": study.get("topic"), "n_hypotheses": len(study["hypotheses"])})


# ---- ToolDef + Provider ----

CODING_TOOLDEFS = [
    ToolDef(name="run_experiment", source="builtin", annotations={"readOnlyHint": False},
            description=(
                "Run a command as a background experiment inside experiments/{project}. "
                "Logs to runs/<exp_id>/run.log; metrics.json/metrics.csv written by the "
                "command are parsed on completion. Returns {exp_id, status:'running'} — "
                "then poll experiment_status. For '复现/跑一下实验/训练' requests."),
            parameters={"type": "object",
                        "properties": {
                            "project": {"type": "string",
                                        "description": "Experiment project folder name under experiments/."},
                            "command": {"type": "string",
                                        "description": "Shell command to run (e.g. 'python train.py')"},
                            "name": {"type": "string", "description": "Optional experiment label.", "default": ""}},
                        "required": ["project", "command"]}),
    ToolDef(name="experiment_status", source="builtin", annotations={"readOnlyHint": True},
            description="Return an experiment's state: status / exit_code / git_sha / metrics / log tail.",
            parameters={"type": "object",
                        "properties": {"exp_id": {"type": "string"}},
                        "required": ["exp_id"]}),
    ToolDef(name="experiment_list", source="builtin", annotations={"readOnlyHint": True},
            description="List experiments (optional project filter), newest first.",
            parameters={"type": "object",
                        "properties": {"project": {"type": "string", "default": ""}}}),
    ToolDef(name="read_metrics", source="builtin", annotations={"readOnlyHint": True},
            description="Read parsed metrics of an experiment (metrics.json/metrics.csv).",
            parameters={"type": "object",
                        "properties": {
                            "exp_id": {"type": "string"},
                            "metric_key": {"type": "string", "default": ""}},
                        "required": ["exp_id"]}),
    ToolDef(name="git_status", source="builtin", annotations={"readOnlyHint": True},
            description="Show experiment project working-tree status.",
            parameters={"type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"]}),
    ToolDef(name="git_diff", source="builtin", annotations={"readOnlyHint": True},
            description="Show uncommitted diffs of the experiment project.",
            parameters={"type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"]}),
    ToolDef(name="git_log", source="builtin", annotations={"readOnlyHint": True},
            description="Show recent commit history of the experiment project.",
            parameters={"type": "object",
                        "properties": {"project": {"type": "string"},
                                       "n": {"type": "integer", "default": 10}},
                        "required": ["project"]}),
    ToolDef(name="git_commit", source="builtin", annotations={"readOnlyHint": False},
            description="Commit all changes in the experiment project (destructive, gated).",
            parameters={"type": "object",
                        "properties": {"project": {"type": "string"},
                                       "message": {"type": "string"}},
                        "required": ["project", "message"]}),
    ToolDef(name="delegate_code_task", source="builtin", annotations={"readOnlyHint": False},
            description=(
                "Delegate a coding task to an EXTERNAL coding agent (MCP bridge first, "
                "headless CLI fallback: claude -p / codex exec). It works inside "
                "web/workspace/experiments/{project}, edits files there, returns a "
                "summary + changed_files. Use for implement/tune/bugfix/refactor jobs."),
            parameters={"type": "object",
                        "properties": {
                            "project": {"type": "string"},
                            "prompt": {"type": "string",
                                       "description": "Self-contained task; mention files, constraints, expected JSON"},
                            "timeout": {"type": "integer", "default": 600,
                                        "description": "Seconds (60–3600)."}},
                        "required": ["project", "prompt"]}),
    ToolDef(name="study_context", source="builtin", annotations={"readOnlyHint": True},
            description="Read the study knowledge base for a topic (hypotheses, recent experiment records with metrics, findings) — comparison baseline for writing/optimization.",
            parameters={"type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"]}),
    ToolDef(name="study_add_hypothesis", source="builtin", annotations={"readOnlyHint": False},
            description="Record a research hypothesis into the study knowledge base (append-only).",
            parameters={"type": "object",
                        "properties": {"topic": {"type": "string"},
                                       "hypothesis": {"type": "string"}},
                        "required": ["topic", "hypothesis"]}),
]

_FUNC_MAP = {
    "run_experiment": run_experiment,
    "experiment_status": experiment_status,
    "experiment_list": experiment_list,
    "read_metrics": read_metrics,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "git_commit": git_commit,
    "delegate_code_task": delegate_code_task,
    "study_context": study_context,
    "study_add_hypothesis": study_add_hypothesis,
}


class CodingProvider(ToolProvider):
    """编码域工具提供商（coder subagent 专用子集；外部编码委托走统一 delegate）。"""

    name = "coding"

    def __init__(self):
        self._tool_map = {td.name: td for td in CODING_TOOLDEFS}

    async def list_tools(self) -> list[ToolDef]:
        return list(self._tool_map.values())

    async def call_tool(self, name: str, arguments: dict):
        if name not in _FUNC_MAP:
            raise KeyError(f"Coding tool '{name}' not found")
        td = self._tool_map[name]
        if td is not None and not tool_allowed(td.annotations):
            return _err(
                f"Action '{name}' is not authorized for the current role.",
                error_type="permission_denied")
        fn = _FUNC_MAP[name]
        return await fn.ainvoke(arguments)