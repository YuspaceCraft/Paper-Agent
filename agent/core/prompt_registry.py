"""Version identifiers for the current Python-based prompt catalogue.

This is an adapter, not a second prompt loader. Prompt text remains in
``agent.prompts`` until YAML-backed prompt publishing is introduced; traces can
already reproduce exactly which content was active through content hashes.
"""

from __future__ import annotations

import hashlib

from .contracts import PromptType


_PROMPT_NAMES: dict[PromptType, str] = {
    PromptType.SYSTEM: "AGENT_SYSTEM",
    PromptType.ROUTER: "UNDERSTAND_SYSTEM",
    PromptType.PLANNER: "PLAN_SYSTEM",
    PromptType.EXECUTOR: "STEP_EXEC_SYSTEM",
    PromptType.SYNTHESIZER: "SYNTHESIZE_SYSTEM",
    PromptType.JUDGE: "VERIFY_SYSTEM",
    PromptType.SUMMARY: "MEMORY_SUMMARY",
    PromptType.SAFETY: "SAFETY_SYSTEM",
}

_AUX_PROMPT_NAMES = (
    "CHAT_SYSTEM",
    "CLARIFY_SYSTEM",
    "TASK_SYSTEM",
    "NOTIFY_SYSTEM",
    "ARXIV_SYSTEM",
    "INGEST_SYSTEM",
    "CREATOR_SYSTEM",
    "CODER_SYSTEM",
)


def prompt_ids() -> list[str]:
    """Stable prompt ids managed by the configuration centre."""
    return [*_PROMPT_NAMES.values(), *_AUX_PROMPT_NAMES]


def _aux_prompt_text(constant: str) -> str | None:
    """Resolve auxiliary prompts without importing every module eagerly."""
    from .. import prompts

    text = getattr(prompts, constant, None)
    if isinstance(text, str) and text:
        return text
    if constant in {
        "ARXIV_SYSTEM", "INGEST_SYSTEM", "CREATOR_SYSTEM", "CODER_SYSTEM",
    }:
        from .. import subagents

        text = getattr(subagents, constant, None)
        if isinstance(text, str) and text:
            return text
    return None


def active_prompt_versions(unit_id: str = "") -> dict[str, str]:
    """Return stable ``legacy:<constant>@sha256`` IDs for available prompts."""
    return active_prompt_snapshot(unit_id)["bindings"]


def active_prompt_snapshot(unit_id: str = "") -> dict[str, dict[str, str]]:
    """Resolve the effective prompt text, bindings and suites exactly once.

    The returned template map is the source used by every node for the rest of
    the turn. This prevents a prompt publish between bootstrap and execution
    from making the traced binding disagree with the text sent to the model.
    """
    from .. import prompts
    from ..prompt_store import load_prompt_spec, resolve_prompt

    bindings: dict[str, str] = {}
    suites: dict[str, str] = {}
    templates: dict[str, str] = {}

    def _record(constant: str, type_key: str = "") -> None:
        bundled = getattr(prompts, constant, None)
        if constant in _AUX_PROMPT_NAMES:
            bundled = _aux_prompt_text(constant)
        if not isinstance(bundled, str) or not bundled:
            return

        spec = load_prompt_spec(constant, unit_id=unit_id)
        if spec is not None:
            binding = spec.binding
            text = spec.template
            suite = spec.evaluation_suite
        else:
            text = resolve_prompt(constant, bundled, unit_id=unit_id) or bundled
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            binding = f"{constant}@bundled#{digest}"
            suite = ""

        keys = [constant]
        if type_key:
            keys.append(type_key)
        for key in keys:
            bindings[key] = binding
            templates[key] = text
            if suite:
                suites[key] = suite

    for prompt_type, constant in _PROMPT_NAMES.items():
        _record(constant, prompt_type.value)
    for constant in _AUX_PROMPT_NAMES:
        _record(constant)

    return {
        "bindings": bindings,
        "evaluation_suites": suites,
        "templates": templates,
    }


def active_prompt_bindings(unit_id: str = "") -> dict[str, str]:
    """Prompt-type → ``id@version#checksum`` for the content actually in force."""
    return active_prompt_snapshot(unit_id)["bindings"]


def active_prompt_evaluation_suites(unit_id: str = "") -> dict[str, str]:
    """Prompt type → regression suite for the actually resolved variant."""
    return active_prompt_snapshot(unit_id)["evaluation_suites"]
