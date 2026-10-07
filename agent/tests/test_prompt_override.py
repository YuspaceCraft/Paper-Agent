from __future__ import annotations

from pathlib import Path

from agent.prompt_store import (
    clear_prompt_overrides,
    resolve_prompt,
    set_prompt_overrides,
)


def _write_prompt(root: Path, prompt_id: str, version: str, status: str, text: str) -> None:
    path = root / "paper" / prompt_id / f"{version}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            f"id: {prompt_id}",
            f"version: {version}",
            f"status: {status}",
            f"template: {text}",
        ]),
        encoding="utf-8",
    )


def test_eval_prompt_override_pins_version(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PROMPT_DIR", str(tmp_path))
    _write_prompt(tmp_path, "AGENT_SYSTEM", "v1", "active", "active prompt")
    _write_prompt(tmp_path, "AGENT_SYSTEM", "v2", "canary", "candidate prompt")

    set_prompt_overrides({"AGENT_SYSTEM": "v2"})
    try:
        assert resolve_prompt("AGENT_SYSTEM", "fallback") == "candidate prompt"
    finally:
        clear_prompt_overrides()

    assert resolve_prompt("AGENT_SYSTEM", "fallback") == "active prompt"
