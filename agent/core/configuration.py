"""Frozen, non-secret configuration metadata for one agent turn.

Configuration sources may change while a server is running. A trace must keep
the effective values it used, but must never receive an environment dump or
credentials. This module is deliberately read-only and has no LangGraph
dependency so it can also be used by evaluation and CLI code.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ConfigurationSnapshot(BaseModel):
    revision: str
    config_hash: str
    created_at: str
    model: str
    model_routes: dict[str, str] = Field(default_factory=dict)
    limits: dict[str, int] = Field(default_factory=dict)
    disabled_tools: dict[str, list[str]] = Field(default_factory=dict)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    prompt_evaluation_suites: dict[str, str] = Field(default_factory=dict)
    tool_registry_hash: str = ""

    def trace_metadata(self) -> dict:
        """Small JSON-safe subset suitable for LangSmith run metadata."""
        return {
            "config_revision": self.revision,
            "config_hash": self.config_hash,
            "model_route": self.model,
            "model_routes": self.model_routes,
            "runtime_limits": self.limits,
            "disabled_tools": self.disabled_tools,
            "feature_flags": self.feature_flags,
            "prompt_versions": self.prompt_versions,
            "prompt_evaluation_suites": self.prompt_evaluation_suites,
            "tool_registry_hash": self.tool_registry_hash,
        }


def build_configuration_snapshot(
    *, model: str | None = None, unit_id: str = "",
    require_initialized_tools: bool = False,
) -> ConfigurationSnapshot:
    """Read effective non-secret settings and return an immutable value object."""
    from ..config import get_limits
    from ..config_store import get_disabled_tools
    from .prompt_registry import (
        active_prompt_evaluation_suites,
        active_prompt_versions,
    )
    from ..prompt_store import get_prompt_overrides
    try:
        from ..tools import get_tool_registry, tools_initialized
    except Exception:
        if require_initialized_tools:
            raise RuntimeError("tool registry module is unavailable")
        tool_registry_hash = "unavailable"
    else:
        if require_initialized_tools and not tools_initialized():
            raise RuntimeError(
                "tool registry must be initialized before freezing a turn"
            )
        try:
            tool_registry_hash = get_tool_registry().registry_hash
        except Exception:
            if require_initialized_tools:
                raise
            tool_registry_hash = "unavailable"

    limits = get_limits()
    main_model = model or os.getenv("LLM_MODEL", "qwen-plus")
    from .model_router import resolve_model_routes

    model_routes = resolve_model_routes(
        main_model,
        small_model=os.getenv("AGENT_MODEL_SMALL", ""),
    )
    prompt_versions = active_prompt_versions(unit_id)
    prompt_versions.update(get_prompt_overrides())
    values = {
        "model": main_model,
        "model_routes": model_routes,
        "limits": {
            "max_steps": int(limits.max_steps),
            "max_turns": int(limits.max_turns),
            "plan_step_max_steps": int(limits.plan_step_max_steps),
            "turn_timeout_seconds": int(float(os.getenv("AGENT_TURN_TIMEOUT", "900"))),
        },
        "disabled_tools": get_disabled_tools(),
        "feature_flags": {
            "trace_enabled": os.getenv("AGENT_TRACE_ENABLED", "1") != "0",
            "langsmith_tracing": os.getenv("LANGSMITH_TRACING", "").lower()
            in {"1", "true", "yes"},
            "input_guard": os.getenv("AGENT_INPUT_GUARD", "1").lower()
            not in {"0", "false", "no", "off"},
            "model_fallback_configured": bool(
                (os.getenv("AGENT_MODEL_FALLBACKS", "") or "").strip()
                or (os.getenv("AGENT_MODEL_FALLBACK_KEY_ENVS", "") or "").strip()
            ),
            "postmortem_capture": os.getenv("AGENT_POSTMORTEM", "1").lower()
            not in {"0", "false", "no", "off"},
            "remote_moderation": bool(
                (os.getenv("AGENT_MODERATION_ENDPOINT", "") or "").strip()
            ),
            "cost_history": os.getenv(
                "AGENT_COST_HISTORY", "1"
            ).lower() not in {"0", "false", "no", "off"},
        },
        "prompt_versions": prompt_versions,
        "prompt_evaluation_suites": active_prompt_evaluation_suites(unit_id),
        "tool_registry_hash": tool_registry_hash,
    }
    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return ConfigurationSnapshot(
        revision=f"runtime-{digest}",
        config_hash=digest,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **values,
    )
