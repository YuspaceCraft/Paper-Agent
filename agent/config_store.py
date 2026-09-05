"""
config_store.py — 配置中心键值存储（工作区路径之外的其余配置）。

与 workspace_config.py 分工：
- workspace_config 负责「可配置工作区根」（project_path / experiments_path / study_root），
  持久化在 web/workspace/settings.json。
- 本模块负责其余运行时配置，持久化在 web/workspace/config.json：
    experiment: {delegate_prefer, delegate_timeout, auto_git_commit, manifest_auto_update}
    tools:      {disabled: {parent, arxiv, ingest, creator, coder}, max_steps}
    skills:     {disabled: [...]}

写盘原子性同 workspace_config（临时文件 + os.replace，防写到一半崩溃留下半截）。
本模块面向 agent/domains、web/api/routers/config 共用；测试可用 set_override 接缝。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# 代码根（= 仓库根）。
_CODE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _CODE_ROOT / "web" / "workspace" / "config.json"

# 默认值（未显式配置时的回退）。
_DEFAULTS: dict = {
    "experiment": {
        "delegate_prefer": "mcp",   # mcp | cli —— 外部 coding agent 委托通道
        "delegate_timeout": 600,    # 秒 —— delegate_code_task 默认超时（V2 消费）
        "auto_git_commit": False,   # 实验后自动 git commit（V2 消费）
        "manifest_auto_update": True,  # 跑实验自动维护 project.json changelog（V2 消费）
    },
    "tools": {
        "disabled": {"parent": [], "arxiv": [], "ingest": [], "creator": [], "coder": []},
        "max_steps": {},            # {agent: int} —— 仅展示/持久化（步骤上限权威在 agent/config.yaml）
    },
    "skills": {
        "disabled": [],
    },
}

# 测试接缝：getter 先查 override（key = 完整键路径，如 "tools.disabled"）。
_OVERRIDES: dict[str, object] = {}


# ---- 设置读写 ----

def load_config() -> dict:
    """读 config.json；缺失/损坏 → {}（全部走默认回退）。"""
    try:
        raw = CONFIG_PATH.read_text(encoding="utf-8")
        cfg = json.loads(raw)
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    """原子写 config.json（临时文件 + os.replace）。"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _merge_defaults(cfg: dict) -> dict:
    """返回 cfg 与默认值合并后的视图（不改写 cfg 本身；缺命名空间补默认）。"""
    out = dict(_DEFAULTS)
    for ns, sub in _DEFAULTS.items():
        user_ns = cfg.get(ns)
        if isinstance(user_ns, dict):
            out[ns] = {**sub, **user_ns}
    return out


# ---- 命名空间读写（配置中心 API 用） ----

def get(ns: str, key: str, default=None):
    """读取单个配置键；未配置 → 默认值（首次 get 前 _DEFAULTS 已合并）。"""
    ov = _OVERRIDES.get(f"{ns}.{key}")
    if ov is not None:
        return ov
    return _merge_defaults(load_config()).get(ns, {}).get(key, default)


def set(ns: str, key: str, value) -> None:
    """更新单个配置键并原子落盘。"""
    cfg = load_config()
    cfg_ns = cfg.setdefault(ns, {})
    if not isinstance(cfg_ns, dict):
        cfg_ns = {}
        cfg[ns] = cfg_ns
    cfg_ns[key] = value
    save_config(cfg)


def set_many(ns: str, patch: dict) -> None:
    """批量更新某命名空间下多个键并原子落盘。"""
    cfg = load_config()
    cfg_ns = cfg.setdefault(ns, {})
    if not isinstance(cfg_ns, dict):
        cfg_ns = {}
        cfg[ns] = cfg_ns
    cfg_ns.update(patch)
    save_config(cfg)


def get_ns(ns: str) -> dict:
    """读取整个命名空间（合并默认值）。"""
    return _merge_defaults(load_config()).get(ns, {})


# ---- 类型化 getter（消费端用） ----

def get_delegate_prefer() -> str:
    """外部 coding agent 委托通道：mcp（MCP bridge 优先）| cli（CLI subprocess 优先）。"""
    v = get("experiment", "delegate_prefer", "mcp")
    return v if v in ("mcp", "cli") else "mcp"


def get_delegate_timeout() -> int:
    v = get("experiment", "delegate_timeout", 600)
    return v if isinstance(v, int) and v > 0 else 600


def get_disabled_tools() -> dict[str, list[str]]:
    """按 agent（parent/arxiv/ingest/creator/coder）的停用工具名集合。"""
    raw = get("tools", "disabled", {})
    if not isinstance(raw, dict):
        return {}
    return {k: [t for t in v if isinstance(t, str)] if isinstance(v, list) else []
            for k, v in raw.items()}


def get_max_steps_display() -> dict[str, int]:
    """工具配置面板持久化的 max_steps 展示值（生效接线在 V2；权威在 agent/config.yaml）。"""
    raw = get("tools", "max_steps", {})
    return {k: v for k, v in raw.items() if isinstance(v, int) and v > 0} if isinstance(raw, dict) else {}


def get_disabled_skills() -> list[str]:
    raw = get("skills", "disabled", [])
    return [s for s in raw if isinstance(s, str)] if isinstance(raw, list) else []


# ---- 测试接缝 ----

def set_override(key: str, value) -> None:
    _OVERRIDES[key] = value


def clear_overrides() -> None:
    _OVERRIDES.clear()