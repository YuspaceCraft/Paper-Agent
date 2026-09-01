"""
config.py — agent 执行约束 YAML 配置加载
=========================================

统一管理 step / turn 上限,替代散落在代码与 env 里的硬编码:
  - 父 agent 步数上限历史来源: env AGENT_MAX_STEPS / AGENT_MAX_ITERATIONS(state.py 默认 30)
  - subagent 步数上限历史来源: SubagentSpec.max_steps 硬编码(subagents.py SUBAGENTS 表)
  - turn 上限历史来源: env AGENT_MAX_TURNS(state.py 默认 50)

优先级: 显式 env(AGENT_MAX_STEPS / AGENT_MAX_TURNS) > agent/config.yaml > 代码默认值。

用法: `from .config import load_limits; limits = load_limits()`。
与 indexer/config.py 同构(YAML + dataclass + 缺失回退),无 pydantic 依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class SubagentLimits:
    max_steps: int = 5  # step 粒度: 单个 subagent 一轮内最多执行的工具轮次


@dataclass
class AgentLimits:
    """agent 执行粒度上限(max_steps=step 粒度, max_turns=turn 粒度)。"""
    max_steps: int = 30
    max_turns: int = 50
    subagents: dict[str, SubagentLimits] = field(default_factory=dict)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _resolve(v, default: int) -> int:
    return v if isinstance(v, int) and v > 0 else default


def load_limits(path: str = "") -> AgentLimits:
    """从 agent/config.yaml 加载执行上限,缺失值使用默认值;文件/YAML 缺失 → 全默认。"""
    config = AgentLimits()
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if yaml is None or not p.exists():
        return config
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return config
    if not isinstance(raw, dict):
        return config

    config.max_steps = _resolve(raw.get("max_steps"), config.max_steps)
    config.max_turns = _resolve(raw.get("max_turns"), config.max_turns)

    subs = raw.get("subagents")
    if isinstance(subs, dict):
        for name, v in subs.items():
            if isinstance(v, dict):
                ms = _resolve(v.get("max_steps"), 5)
                config.subagents[str(name)] = SubagentLimits(max_steps=ms)
    return config


# 模块级缓存: AgentState 类属性默认值在 import 时求值,复用避免重复读盘
_limits_cache: AgentLimits | None = None


def get_limits() -> AgentLimits:
    global _limits_cache
    if _limits_cache is None:
        _limits_cache = load_limits()
    return _limits_cache