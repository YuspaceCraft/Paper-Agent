"""
experiments.py — 实验域 HTTP 薄封装（v10 / Phase C）。

序列化 + 后台任务触发，领域逻辑在 agent/domains/coding.py（ExperimentStore）。
前端（实验工作区，Phase D）走这里；agent 工具直接调模块。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.domains.coding import (
    _exp_dir,
    _load_exp,
    _load_projects,
    _list_experiments,
    _parse_metrics,
    run_experiment,
)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class RunExperimentBody(BaseModel):
    project: str = Field(min_length=1, max_length=100)
    command: str = Field(min_length=1, max_length=2000)
    name: str = Field(default="", max_length=200)


@router.get("/projects")
async def list_projects():
    return {"projects": _load_projects()}


@router.get("/projects/{project}/git")
async def project_git(project: str, kind: str = "diff"):
    """项目级 git 只读面板（Phase D）：kind ∈ diff | log | status。

    agent 侧已有同名工具（coding.py），此处为薄封装保持一致。
    """
    from agent.domains.coding import git_diff, git_log, git_status

    fn = {"diff": git_diff, "log": git_log, "status": git_status}.get(kind)
    if fn is None:
        raise HTTPException(400, "kind must be diff | log | status")
    args = {"project": project}
    if kind == "log":
        args["n"] = 10
    result = json.loads(await fn.ainvoke(args))
    if not result.get("ok"):
        raise HTTPException(
            404 if result.get("error_type") == "param_error" else 502,
            result.get("error", "git failed"))
    return {"kind": kind, "output": result["data"]["output"]}


@router.get("")
async def list_experiments(project: str = ""):
    return {"experiments": _list_experiments(project or None)}


@router.get("/{project}/manifest")
async def project_manifest(project: str):
    """项目 manifest（project.json，委托信息交换契约）+ 近期实验。

    对话中心化重构 L4：前端实验面板用 manifest 渲染入口/key_files/baseline；
    coding agent 侧同样读这个文件。
    """
    from agent.domains.coding import experiment_project_state
    result = json.loads(await experiment_project_state.ainvoke({"project": project}))
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "manifest read failed"))
    return result["data"]


@router.post("/run", status_code=202)
async def run(body: RunExperimentBody):
    result = json.loads(await run_experiment.ainvoke({
        "project": body.project, "command": body.command, "name": body.name}))
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "start failed"))
    return result["data"]


@router.get("/{exp_id}")
async def get_exp(exp_id: str):
    st = _load_exp(exp_id)
    if st is None:
        raise HTTPException(404, f"experiment '{exp_id}' not found")
    logf = _exp_dir(exp_id) / "run.log"
    tail = ""
    if logf.exists():
        tail = logf.read_text(encoding="utf-8", errors="replace")[-10_000:]
    out = {
        "exp_id": st["exp_id"], "project": st.get("project", ""),
        "name": st.get("name", ""), "command": st.get("command", ""),
        "status": st.get("status", ""), "exit_code": st.get("exit_code"),
        "git_sha": st.get("git_sha", ""),
        "metrics": st.get("metrics", {}),
        "created_at": st.get("created_at", ""), "finished_at": st.get("finished_at", ""),
        "log_tail": tail,
    }
    return out


@router.get("/{exp_id}/metrics")
async def get_metrics(exp_id: str):
    st = _load_exp(exp_id)
    if st is None:
        raise HTTPException(404, f"experiment '{exp_id}' not found")
    return {"exp_id": exp_id, "metrics": _parse_metrics(st)}


@router.get("/{exp_id}/logs")
async def get_logs(exp_id: str):
    st = _load_exp(exp_id)
    if st is None:
        raise HTTPException(404, f"experiment '{exp_id}' not found")
    logf = _exp_dir(exp_id) / "run.log"
    content = logf.read_text(encoding="utf-8", errors="replace") if logf.exists() else ""
    return {"exp_id": exp_id, "log": content[-20_000:]}