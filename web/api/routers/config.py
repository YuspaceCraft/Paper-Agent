"""
config.py — 配置中心 HTTP 薄封装（工具 / 实验 / MCP / Skills 清单与持久化）。

读/写逻辑归属：
- agent/config_store.py    → 实验 + 工具开关 + skills 停用的键值持久化（web/workspace/config.json）
- agent/workspace_config.py→ 工作区路径（/api/settings 已是该模块的薄封装，路径不改在这里）
- agent/providers/mcp_provider.py  → .mcp.json 读写 + probe
- agent/providers/skill_provider.py→ skills 清单扫描
- agent/tools.py           → 工具重载（reload_tools）
- agent/config.py          → 生效的执行上限（只读展示）

本层只做：校验输入 → 组装 JSON → 调用上述模块 → 返回。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent import config_store, workspace_config
from agent.config import get_limits
from agent.providers.mcp_provider import (
    default_config_path,
    probe_server,
    read_mcp_config_raw,
    write_mcp_config,
)
from agent.providers.skill_provider import discover_skills
from agent.subagents import SUBAGENTS
from agent.tools import PARENT_NAMES, ensure_tools, get_base_tools, reload_tools

router = APIRouter(prefix="/api/config", tags=["config"])

_AGENT_KEYS = ("parent",) + tuple(s.name for s in SUBAGENTS)

# ---- 工具清单 ----

def _agent_label(name: str) -> str:
    return {
        "parent": "父 Agent",
        "arxiv": "论文检索（arxiv）",
        "ingest": "论文入库（ingest）",
        "creator": "章节写作（creator）",
        "coder": "实验编码（coder）",
    }.get(name, name)


def _effective_max_steps(agent: str, limits) -> int:
    """展示用的步数上限：config_store 持久化的覆盖值优先，否则生效配置（config.yaml）。"""
    stored = config_store.get_max_steps_display().get(agent)
    if stored:
        return stored
    if agent == "parent":
        return limits.max_steps
    sub = limits.subagents.get(agent)
    return sub.max_steps if sub else 5


def _build_tools_inventory() -> dict:
    """组装按 agent 分组的工具清单（父 + 各 subagent）。"""
    base_map = {t.name: t for t in get_base_tools()}
    disabled = config_store.get_disabled_tools()
    limits = get_limits()

    spec_names: dict[str, list[str]] = {s.name: s.tools for s in SUBAGENTS}
    agent_tool_names: dict[str, list[str]] = {"parent": sorted(PARENT_NAMES)}
    for name, names in spec_names.items():
        agent_tool_names[name] = names

    agents: dict = {}
    for agent, names in agent_tool_names.items():
        blocked = set(disabled.get(agent, []) or [])
        tools = []
        for tname in names:
            bt = base_map.get(tname)
            tools.append({
                "name": tname,
                "description": getattr(bt, "description", "") if bt else "",
                "source": getattr(bt, "source", "") if bt else "builtin",
                "loaded": bt is not None,
                "enabled": tname not in blocked,
            })
        agents[agent] = {
            "label": _agent_label(agent),
            "max_steps": _effective_max_steps(agent, limits),
            "tools": tools,
        }
    return {"agents": agents}


@router.get("/tools")
async def get_tools():
    try:
        await ensure_tools()
    except Exception as e:  # MCP 连接失败不影响工具清单展示
        raise HTTPException(502, f"tools not ready: {e}")
    return _build_tools_inventory()


class UpdateToolsBody(BaseModel):
    """disabled: {agent: [工具名]}；max_steps: {agent: int} 可选（展示用覆盖存储，缺省保留现值）。"""
    disabled: dict[str, list[str]] = {}
    max_steps: dict[str, int] | None = None


@router.put("/tools")
async def update_tools(body: UpdateToolsBody):
    if unk := [k for k in body.disabled if k not in _AGENT_KEYS]:
        raise HTTPException(400, f"unknown agent keys: {unk}")
    patch: dict = {"disabled": {k: list(dict.fromkeys(v)) for k, v in body.disabled.items()}}
    if body.max_steps is not None:
        patch["max_steps"] = body.max_steps   # 未传 → 保留既有展示覆盖值
    config_store.set_many("tools", patch)
    try:
        await reload_tools()
    except Exception as e:
        raise HTTPException(502, f"tools reload failed: {e}")
    return {"ok": True, "agents": _build_tools_inventory()["agents"]}


# ---- 实验配置 ----

@router.get("/experiment")
async def get_experiment():
    exp = config_store.get_ns("experiment")
    return {
        "paths": {
            "project_path": workspace_config.get_project_path(),
            "project_root": str(workspace_config.get_project_root()),
            "experiments_path": str(workspace_config.get_experiments_path()),
            "writing_dir": str(workspace_config.get_docs_dir()),
        },
        "delegate_prefer": exp.get("delegate_prefer", "mcp"),
        "delegate_timeout": exp.get("delegate_timeout", 600),
        "auto_git_commit": bool(exp.get("auto_git_commit", False)),
        "manifest_auto_update": bool(exp.get("manifest_auto_update", True)),
    }


class UpdateExperimentBody(BaseModel):
    delegate_prefer: str | None = None
    delegate_timeout: int | None = None
    auto_git_commit: bool | None = None
    manifest_auto_update: bool | None = None


@router.put("/experiment")
async def update_experiment(body: UpdateExperimentBody):
    patch: dict = {}
    if body.delegate_prefer is not None:
        if body.delegate_prefer not in ("mcp", "cli"):
            raise HTTPException(400, "delegate_prefer must be 'mcp' or 'cli'")
        patch["delegate_prefer"] = body.delegate_prefer
    if body.delegate_timeout is not None:
        if body.delegate_timeout < 30 or body.delegate_timeout > 36000:
            raise HTTPException(400, "delegate_timeout must be in [30, 36000]")
        patch["delegate_timeout"] = body.delegate_timeout
    if body.auto_git_commit is not None:
        patch["auto_git_commit"] = body.auto_git_commit
    if body.manifest_auto_update is not None:
        patch["manifest_auto_update"] = body.manifest_auto_update
    if patch:
        config_store.set_many("experiment", patch)
    return await _experiment_payload()


async def _experiment_payload():
    exp = config_store.get_ns("experiment")
    return {
        "paths": {
            "project_path": workspace_config.get_project_path(),
            "experiments_path": str(workspace_config.get_experiments_path()),
            "writing_dir": str(workspace_config.get_docs_dir()),
        },
        "delegate_prefer": exp["delegate_prefer"],
        "delegate_timeout": exp["delegate_timeout"],
        "auto_git_commit": exp["auto_git_commit"],
        "manifest_auto_update": exp["manifest_auto_update"],
    }


# ---- MCP 配置 ----

@router.get("/mcp")
async def get_mcp():
    try:
        await ensure_tools()  # 保证 base registry 有当前工具（status 的 tool 计数才有意义）
    except Exception:
        pass
    raw = read_mcp_config_raw()
    servers_raw = raw.get("mcpServers", {})
    base_map = {t.name for t in get_base_tools()}

    servers = []
    for name in servers_raw:
        cfg = servers_raw[name] if isinstance(servers_raw[name], dict) else {}
        tools = sum(1 for t in base_map if t.startswith(f"{name}__"))
        servers.append({
            "name": name,
            **cfg,  # 透传 transport/command/args/url/headers/env/disabled 等原始字段
            "status": "ok" if tools else "unknown",
            "tools": tools,
        })
    return {
        "exists": bool(servers_raw),
        "path": str(default_config_path()),
        "servers": servers,
    }


class UpdateMcpBody(BaseModel):
    """servers: [{name, ...原始字段}] —— 全量替换 mcpServers。"""
    servers: list[dict] = []


@router.put("/mcp")
async def update_mcp(body: UpdateMcpBody):
    _DERIVED = {"status", "tools"}  # 只读衍生字段，不落盘
    servers: dict = {}
    for entry in body.servers:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise HTTPException(400, "each server entry needs a name")
        srv = {k: v for k, v in entry.items() if k not in _DERIVED and k != "name"}
        if "command" not in srv and "url" not in srv:
            raise HTTPException(400, f"server '{entry['name']}' needs command or url")
        if srv.get("disabled") is not None:
            srv["disabled"] = bool(srv["disabled"])
        servers[entry["name"]] = srv
    write_mcp_config(servers)
    try:
        await reload_tools()
    except Exception as e:
        raise HTTPException(502, f"tools reload failed: {e}")
    return {"ok": True, "servers": get_mcp()["servers"]}


class TestMcpBody(BaseModel):
    name: str


@router.post("/mcp/test")
async def test_mcp(body: TestMcpBody):
    if not body.name.strip():
        raise HTTPException(400, "name required")
    name = body.name.strip()
    # 已在当前工具表加载（ensure_tools 连接成功过）→ 免再起新进程，即时返回。
    # 冷启动 mcp server（如本文 arxiv）可能耗时数十秒，仅对未加载/新编辑做真实试连。
    live = sum(1 for t in get_base_tools() if t.name.startswith(f"{name}__"))
    if live > 0:
        return {"name": name, "ok": True, "tool_count": live, "error": None, "reused": True}
    try:
        return await probe_server(name, timeout=60.0)
    except Exception as e:
        raise HTTPException(502, f"probe failed: {e}")


# ---- Skills 配置 ----

@router.get("/skills")
async def get_skills():
    skills = discover_skills()
    blocked = set(config_store.get_disabled_skills())
    return {
        "skills": [
            {
                "name": name,
                "description": info.get("description", ""),
                "path": info.get("path", ""),
                "resources": info.get("resources", []),
                "enabled": name not in blocked,
            }
            for name, info in sorted(skills.items())
        ]
    }


class UpdateSkillsBody(BaseModel):
    disabled: list[str] = []


@router.put("/skills")
async def update_skills(body: UpdateSkillsBody):
    config_store.set_many("skills", {"disabled": list(dict.fromkeys(body.disabled))})
    try:
        await reload_tools()
    except Exception as e:
        raise HTTPException(502, f"tools reload failed: {e}")
    return {"ok": True, "skills": (await _skills_payload())["skills"]}


async def _skills_payload():
    skills = discover_skills()
    blocked = set(config_store.get_disabled_skills())
    return {
        "skills": [
            {
                "name": name,
                "description": info.get("description", ""),
                "path": info.get("path", ""),
                "resources": info.get("resources", []),
                "enabled": name not in blocked,
            }
            for name, info in sorted(skills.items())
        ]
    }


# ---- 生效执行上限（只读展示） ----

@router.get("/limits")
async def get_limits_endpoint():
    limits = get_limits()
    return {
        "max_steps": limits.max_steps,
        "max_turns": limits.max_turns,
        "plan_step_max_steps": limits.plan_step_max_steps,
        "subagents": {name: s.max_steps for name, s in limits.subagents.items()},
    }