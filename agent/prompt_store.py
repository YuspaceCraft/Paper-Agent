"""Versioned prompt configuration with safe fallback to bundled prompts.

Two published layouts are accepted, in this order:

1. ``prompts/<domain>/<prompt_id>/<version>.yaml`` — the contract in the
   platform design (ADR-0004); the file whose ``status`` is ``active`` wins.
2. ``<prompt_id>.yaml`` — the flat workspace layout used by the prompt
   configuration centre.

Required fields are ``id``, ``version``, ``status: active`` and ``template``.
Invalid, draft, or missing files never interrupt an agent turn: the caller
receives the bundled prompt and an audit event is emitted.
"""

from __future__ import annotations

import os
import hashlib
import json
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .observability import log_event


_SPEC_CACHE: "OrderedDict[tuple[str, int, int, str], object]" = OrderedDict()
_SPEC_CACHE_MAX = 256


# Evaluation-only deterministic override.  The context variable is scoped to
# the asyncio task running one evaluation turn, so parallel A/B samples never
# leak versions into each other or into the interactive chat path.
_prompt_overrides: ContextVar[dict[str, str]] = ContextVar(
    "prompt_overrides", default={},
)


def set_prompt_overrides(overrides: dict[str, str] | None) -> None:
    cleaned = {
        str(key): str(value)
        for key, value in (overrides or {}).items()
        if str(key).strip() and str(value).strip()
    }
    _prompt_overrides.set(cleaned)


def clear_prompt_overrides() -> None:
    _prompt_overrides.set({})


def get_prompt_overrides() -> dict[str, str]:
    return dict(_prompt_overrides.get())


def _prompt_dir() -> Path:
    override = os.getenv("AGENT_PROMPT_DIR", "")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "web" / "workspace" / "prompts"


def _unit_id(*, unit_id: str = "", config=None) -> str:
    if unit_id:
        return str(unit_id)
    if config is not None:
        configurable = getattr(config, "configurable", None)
        if configurable is None and isinstance(config, dict):
            configurable = config.get("configurable")
        if isinstance(configurable, dict):
            return str(configurable.get("thread_id") or configurable.get("run_id") or "")
    return ""


def _canary_bucket(unit_id: str, prompt_id: str, version: str) -> int:
    payload = f"{unit_id or 'anonymous'}|{prompt_id}|{version}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100


def _canary_rules() -> dict:
    rules: dict = {}
    try:
        from .config_store import get

        raw = get("prompts", "canary", {})
        if isinstance(raw, dict):
            rules.update({str(k): v for k, v in raw.items() if isinstance(v, dict)})
    except Exception:
        pass
    env = os.getenv("AGENT_PROMPT_CANARY", "").strip()
    if env:
        try:
            parsed = json.loads(env)
            if isinstance(parsed, dict):
                rules.update({str(k): v for k, v in parsed.items() if isinstance(v, dict)})
        except ValueError:
            pass
    return rules


def canary_rules() -> dict:
    """Public read view for the prompt configuration centre."""
    return _canary_rules()


def get_prompt(prompt_id: str, fallback: str, *, unit_id: str = "",
               config=None) -> str:
    """Resolve the active override; retain fallback on every validation error."""
    try:
        from .core.execution_context import get_current_execution_context

        ctx = get_current_execution_context()
        if ctx is not None:
            frozen = ctx.prompt_templates.get(prompt_id)
            if frozen is not None:
                return frozen
    except Exception:  # noqa: BLE001 — bootstrap/eval paths may have no turn ctx
        pass
    return resolve_prompt(
        prompt_id, fallback,
        unit_id=_unit_id(unit_id=unit_id, config=config),
    )


def resolve_prompt(prompt_id: str, fallback: str, *, unit_id: str = "") -> str:
    """Same contract as :func:`get_prompt`, phrased for new call sites."""
    spec = load_prompt_spec(prompt_id, unit_id=unit_id)
    if spec is None:
        return fallback
    log_event("prompt_resolved", node="prompt_store", prompt_id=prompt_id,
              version=spec.version, checksum=spec.checksum, source="workspace",
              status=spec.status)
    return spec.template


def load_prompt_spec(prompt_id: str, *, unit_id: str = ""):
    """Return the active :class:`PromptSpec`, or ``None`` when unusable.

    Every validation failure logs an audit event and yields ``None`` so the
    caller falls back to the bundled prompt instead of failing the turn.
    """
    if yaml is None:
        return None
    specs = [
        spec for path in _candidate_paths(prompt_id)
        if (spec := _load_spec(path, prompt_id)) is not None
    ]

    override = _prompt_overrides.get().get(prompt_id)
    if override:
        selected = next((spec for spec in specs if spec.version == override), None)
        if selected is not None:
            log_event("prompt_override_selected", node="prompt_store",
                      prompt_id=prompt_id, version=selected.version,
                      checksum=selected.checksum, unit_id=unit_id)
            return selected

    active = next((spec for spec in specs if spec.status == "active"), None)
    if active is None:
        return None

    rule = _canary_rules().get(prompt_id)
    if isinstance(rule, dict):
        version = str(rule.get("version") or "")
        try:
            percent = float(rule.get("percent") or 0)
        except (TypeError, ValueError):
            percent = 0
        canary = next(
            (spec for spec in specs
             if spec.status == "canary" and spec.version == version),
            None,
        )
        if canary is not None and _canary_bucket(unit_id, prompt_id, version) < percent:
            log_event("prompt_canary_selected", node="prompt_store",
                      prompt_id=prompt_id, version=version, percent=percent,
                      unit_id=unit_id)
            return canary
    return active


def list_prompt_specs(prompt_id: str) -> list:
    """All published active/canary versions for one prompt id."""
    specs = [
        spec for path in _candidate_paths(prompt_id)
        if (spec := _load_spec(path, prompt_id)) is not None
    ]
    return sorted(specs, key=lambda item: item.version, reverse=True)


def _candidate_paths(prompt_id: str) -> list[Path]:
    """Active version files for ``prompt_id``: nested layout first, then flat."""
    root = _prompt_dir()
    paths: list[Path] = []
    try:
        for path in sorted(root.glob(f"*/{prompt_id}/*.yaml"), reverse=True):
            paths.append(path)
    except OSError:
        pass
    paths.append(root / f"{prompt_id}.yaml")
    return paths


def _load_spec(path: Path, prompt_id: str):
    from .core.contracts import PromptSpec, PromptType

    if not path.is_file():
        return None
    try:
        stat = path.stat()
        cache_key = (
            str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size),
            str(prompt_id),
        )
    except OSError:
        cache_key = ()
    if cache_key:
        cached = _SPEC_CACHE.get(cache_key)
        if cached is not None:
            _SPEC_CACHE.move_to_end(cache_key)
            return cached
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("document must be a mapping")
        if raw.get("id") != prompt_id:
            raise ValueError("id does not match filename")
        status = str(raw.get("status") or "")
        if status not in {"active", "canary"}:
            raise ValueError("status is not active or canary")
        if not str(raw.get("version", "")).strip():
            raise ValueError("version is required")
        template = raw.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError("template is required")
        try:
            prompt_type = PromptType(str(raw.get("type") or PromptType.SYSTEM.value))
        except ValueError:
            prompt_type = PromptType.SYSTEM
        spec = PromptSpec(
            id=prompt_id,
            version=str(raw["version"]),
            type=prompt_type,
            template=template,
            schema=raw.get("schema") or {},
            variables=[str(v) for v in (raw.get("variables") or [])],
            locale=str(raw.get("locale") or "en"),
            status=status,
            evaluation_suite=str(raw.get("evaluation_suite") or ""),
        )
        if cache_key:
            _SPEC_CACHE[cache_key] = spec
            _SPEC_CACHE.move_to_end(cache_key)
            while len(_SPEC_CACHE) > _SPEC_CACHE_MAX:
                _SPEC_CACHE.popitem(last=False)
        return spec
    except Exception as exc:  # prompt config must never make the agent unavailable
        if cache_key:
            _SPEC_CACHE.pop(cache_key, None)
        log_event("prompt_fallback", node="prompt_store", level="warning",
                  prompt_id=prompt_id, reason=type(exc).__name__)
        return None
