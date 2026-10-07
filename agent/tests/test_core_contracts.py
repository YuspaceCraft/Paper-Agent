"""Core contract checks — no LLM, network or external service required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.core import AgentError, ErrorType, Permission, ToolSpec
from agent.core import build_configuration_snapshot
from agent.core.tool_registry import ToolRegistry
from agent.providers import ToolDef


def test_error_envelope_is_compatible():
    error = AgentError(
        error_type=ErrorType.TOOL_TIMEOUT,
        code="TOOL_TIMEOUT",
        message="safe internal message",
        user_message="工具超时。",
        retryable=True,
        tool_name="search_papers",
    )
    payload = error.to_tool_envelope()
    assert '"error_type": "tool_timeout"' in payload
    assert '"code": "TOOL_TIMEOUT"' in payload
    assert "safe internal message" not in payload


def test_tool_spec_marks_side_effects_for_approval():
    readonly = ToolSpec(name="search")
    write = ToolSpec(name="ingest", permissions={Permission.WRITE})
    assert readonly.requires_approval() is False
    assert write.requires_approval() is True


def test_configuration_snapshot_is_stable_and_has_no_secret():
    first = build_configuration_snapshot(model="test-model")
    second = build_configuration_snapshot(model="test-model")
    assert first.model == "test-model"
    assert first.config_hash == second.config_hash
    metadata = first.trace_metadata()
    assert metadata["config_hash"] == first.config_hash
    assert "api_key" not in str(metadata).lower()


def test_legacy_tooldefs_have_governance_specs():
    registry = ToolRegistry.from_tooldefs([
        ToolDef("search", "", {"type": "object"}, "builtin",
                {"readOnlyHint": True, "idempotentHint": True}),
        ToolDef("ingest", "", {"type": "object"}, "builtin",
                {"readOnlyHint": False}),
    ])
    assert registry.get("search").side_effect is False
    assert registry.get("ingest").requires_approval() is True
    assert len(registry.registry_hash) == 16


def test_context_decision_contains_no_prompt_payload():
    from agent.memory import MemoryManager

    snapshot, decision = MemoryManager().build_snapshot_with_decision(
        {"messages": [], "summary_cache": ""}, max_tokens=100
    )
    assert decision["max_tokens"] == 100
    assert decision["estimated_tokens"] >= 0
    assert "snapshot" not in decision


def test_tool_metrics_use_structured_error_type():
    from evaluation.metrics.tools import aggregate_tools

    report = aggregate_tools([{
        "event_type": "tool_call", "tool": "search_papers", "node": "dispatcher",
        "outcome": "timed_out", "duration_ms": 100.0, "error": "TOOL_TIMEOUT",
        "payload": {"parsed": {"error_type": "tool_timeout"}},
    }])
    assert report["per_tool"]["search_papers"]["error_types"] == {"tool_timeout": 1}


def test_active_prompt_override_and_invalid_fallback(tmp_path, monkeypatch):
    from agent.prompt_store import get_prompt

    monkeypatch.setenv("AGENT_PROMPT_DIR", str(tmp_path))
    (tmp_path / "CHAT_SYSTEM.yaml").write_text(
        "id: CHAT_SYSTEM\nversion: 2026.09.15\nstatus: active\ntemplate: custom chat\n",
        encoding="utf-8",
    )
    assert get_prompt("CHAT_SYSTEM", "fallback") == "custom chat"
    (tmp_path / "CHAT_SYSTEM.yaml").write_text("id: WRONG\n", encoding="utf-8")
    assert get_prompt("CHAT_SYSTEM", "fallback") == "fallback"


def test_published_prompt_layout_binds_id_version_and_checksum(tmp_path, monkeypatch):
    """``prompts/<domain>/<id>/<version>.yaml``: only the active version applies."""
    from agent.prompt_store import load_prompt_spec, get_prompt

    monkeypatch.setenv("AGENT_PROMPT_DIR", str(tmp_path))
    version_dir = tmp_path / "paper" / "CHAT_SYSTEM"
    version_dir.mkdir(parents=True)
    (version_dir / "2026.09.15-2.yaml").write_text(
        "id: CHAT_SYSTEM\nversion: 2026.09.15-2\nstatus: draft\n"
        "type: system\nvariables: [question]\ntemplate: draft text\n",
        encoding="utf-8",
    )
    # Only a draft exists → the bundled prompt stays in force.
    assert get_prompt("CHAT_SYSTEM", "bundled") == "bundled"

    (version_dir / "2026.09.15-3.yaml").write_text(
        "id: CHAT_SYSTEM\nversion: 2026.09.15-3\nstatus: active\n"
        "type: system\nvariables: [question]\n"
        "schema: {type: object}\nevaluation_suite: chat-regression\n"
        "template: 'Answer {question}'\n",
        encoding="utf-8",
    )
    spec = load_prompt_spec("CHAT_SYSTEM")
    assert spec.version == "2026.09.15-3"
    assert spec.binding.startswith("CHAT_SYSTEM@2026.09.15-3#")
    assert spec.render(question="why?") == "Answer why?"
    assert spec.evaluation_suite == "chat-regression"
    try:
        spec.render(unknown="x")
    except ValueError as exc:
        assert "undeclared prompt variables" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("undeclared variables must be rejected")


def test_prompt_canary_is_deterministic_and_falls_back_to_active(tmp_path, monkeypatch):
    from agent import config_store
    from agent.prompt_store import get_prompt

    monkeypatch.setenv("AGENT_PROMPT_DIR", str(tmp_path))
    version_dir = tmp_path / "paper" / "CHAT_SYSTEM"
    version_dir.mkdir(parents=True)
    (version_dir / "2026.09.15-1.yaml").write_text(
        "id: CHAT_SYSTEM\nversion: 2026.09.15-1\nstatus: active\n"
        "type: system\ntemplate: active text\n",
        encoding="utf-8",
    )
    (version_dir / "2026.09.15-2.yaml").write_text(
        "id: CHAT_SYSTEM\nversion: 2026.09.15-2\nstatus: canary\n"
        "type: system\ntemplate: canary text\n"
        "evaluation_suite: chat-canary\n",
        encoding="utf-8",
    )
    config_store.set_override("prompts.canary", {
        "CHAT_SYSTEM": {"version": "2026.09.15-2", "percent": 100},
    })
    try:
        assert get_prompt("CHAT_SYSTEM", "bundled", unit_id="thread-a") == "canary text"
        from agent.core.prompt_registry import active_prompt_bindings
        assert active_prompt_bindings("thread-a")["CHAT_SYSTEM"].startswith(
            "CHAT_SYSTEM@2026.09.15-2#"
        )
    finally:
        config_store.clear_overrides()
    assert get_prompt("CHAT_SYSTEM", "bundled", unit_id="thread-a") == "active text"


def test_turn_prompt_snapshot_is_frozen_after_publish(tmp_path, monkeypatch):
    from agent.core import build_execution_context
    from agent.core.execution_context import (
        reset_current_execution_context,
        set_current_execution_context,
    )
    from agent.prompt_store import get_prompt

    monkeypatch.setenv("AGENT_PROMPT_DIR", str(tmp_path))
    version_dir = tmp_path / "paper" / "CHAT_SYSTEM"
    version_dir.mkdir(parents=True)
    prompt_file = version_dir / "v1.yaml"
    prompt_file.write_text(
        "id: CHAT_SYSTEM\nversion: v1\nstatus: active\n"
        "type: system\ntemplate: frozen text\n",
        encoding="utf-8",
    )

    ctx = build_execution_context(thread_id="frozen-prompt")
    prompt_file.write_text(
        "id: CHAT_SYSTEM\nversion: v2\nstatus: active\n"
        "type: system\ntemplate: newly published text\n",
        encoding="utf-8",
    )

    token = set_current_execution_context(ctx)
    try:
        assert get_prompt("CHAT_SYSTEM", "bundled") == "frozen text"
        assert ctx.prompt_templates["CHAT_SYSTEM"] == "frozen text"
    finally:
        reset_current_execution_context(token)


def test_prompt_spec_cache_invalidates_when_file_changes(tmp_path, monkeypatch):
    from agent.prompt_store import load_prompt_spec

    monkeypatch.setenv("AGENT_PROMPT_DIR", str(tmp_path))
    prompt_dir = tmp_path / "paper" / "CHAT_SYSTEM"
    prompt_dir.mkdir(parents=True)
    prompt_file = prompt_dir / "v1.yaml"
    prompt_file.write_text(
        "id: CHAT_SYSTEM\nversion: v1\nstatus: active\n"
        "type: system\ntemplate: first text\n",
        encoding="utf-8",
    )
    first = load_prompt_spec("CHAT_SYSTEM", unit_id="cache-test")
    assert first is not None and first.template == "first text"

    prompt_file.write_text(
        "id: CHAT_SYSTEM\nversion: v1\nstatus: active\n"
        "type: system\ntemplate: second text with a different size\n",
        encoding="utf-8",
    )
    second = load_prompt_spec("CHAT_SYSTEM", unit_id="cache-test")
    assert second is not None
    assert second.template == "second text with a different size"


def test_runtime_state_comes_from_frozen_budget(monkeypatch):
    from agent.core import build_execution_context

    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    monkeypatch.setenv("AGENT_MAX_TURNS", "11")
    monkeypatch.setenv("AGENT_TOKEN_BUDGET", "12345")
    monkeypatch.setenv("AGENT_PLAN_STEP_MAX_STEPS", "4")

    ctx = build_execution_context(thread_id="budget-freeze")
    assert ctx.runtime_state() == {
        "execution_id": ctx.execution_id,
        "max_steps": 7,
        "max_turns": 11,
        "token_budget": 12345,
    }
    assert ctx.budget.plan_step_max_steps == 4


def test_execution_context_metadata_contract():
    from agent.core import build_execution_context

    ctx = build_execution_context(thread_id="t-meta", model="test-model",
                                  runnable_config={"configurable": {"thread_id": "t-meta"}})
    # The live RunnableConfig must never be serialised into a checkpoint.
    assert "runnable_config" not in ctx.model_dump()
    assert "_runnable_config" not in ctx.model_dump()
    metadata = ctx.trace_metadata()
    for key in ("trace_id", "execution_id", "thread_id", "request_id",
                "actor_role", "domain",
                "graph_version", "config_revision", "config_hash",
                "prompt_versions", "tool_versions", "model_route", "model_routes",
                "eval_dataset_id", "prompt_bindings",
                "prompt_evaluation_suites"):
        assert key in metadata, key
    assert metadata["model_route"] == "test-model"
    child = ctx.child_config(metadata={"extra": True})
    assert child["configurable"]["thread_id"] == "t-meta"
    assert child["metadata"]["trace_id"] == ctx.trace_id


def test_model_routes_freeze_small_and_explicit_overrides(monkeypatch):
    from agent.core.model_router import resolve_model_routes
    from agent.core import build_execution_context

    routes = resolve_model_routes(
        "main-model",
        small_model="small-model",
        explicit={"planner": "planner-model"},
    )
    assert routes["router"] == "small-model"
    assert routes["summary"] == "small-model"
    assert routes["planner"] == "planner-model"
    assert routes["agent"] == "main-model"

    monkeypatch.setenv("LLM_MODEL", "main-model")
    monkeypatch.setenv("AGENT_MODEL_SMALL", "small-model")
    ctx = build_execution_context(thread_id="route-freeze")
    assert ctx.model_for("router") == "small-model"
    assert ctx.model_for("agent") == "main-model"


def test_model_factory_applies_small_route_without_turn_context(monkeypatch):
    import agent.nodes as nodes

    captured = {}

    class FakeChat:
        def __init__(self, model, **kwargs):
            captured["model"] = model

    monkeypatch.setenv("LLM_MODEL", "main-model")
    monkeypatch.setenv("AGENT_MODEL_SMALL", "small-model")
    monkeypatch.setattr(nodes, "ChatOpenAI", FakeChat)

    nodes._get_model({"configurable": {}}, task="router")
    assert captured["model"] == "small-model"


def test_turn_entrypoint_freezes_langsmith_join_keys(monkeypatch):
    from agent import tools
    from agent.graph import prepare_turn

    monkeypatch.setattr(
        tools, "_tool_registry",
        ToolRegistry([ToolSpec(name="probe_tool", version="9")]),
    )
    monkeypatch.setattr(tools, "_built", True)
    config, ctx = prepare_turn(thread_id="t-turn", mode="plan")
    assert ctx.trace_id == config["run_id"].hex
    assert config["metadata"]["trace_id"] == ctx.trace_id
    assert config["metadata"]["requested_mode"] == "plan"
    assert config["metadata"]["graph_version"] not in ("", "unknown")
    assert ctx.tool_versions == {"probe_tool": "9"}
    assert config["metadata"]["tool_versions"] == {"probe_tool": "9"}
    assert len(config["metadata"]["tool_registry_hash"]) == 16
    assert ctx.budget.max_steps > 0 and ctx.budget.usable_context_tokens() > 0


def test_turn_entrypoint_requires_initialized_tools(monkeypatch):
    from agent import tools
    from agent.graph import prepare_turn

    monkeypatch.setattr(tools, "_built", False)
    monkeypatch.setattr(tools, "_tool_registry", None)
    with pytest.raises(RuntimeError, match="tool registry"):
        prepare_turn(thread_id="t-missing-tools")


def test_memory_record_schema_round_trips():
    from agent.core import MemoryRecord, MemoryType, MemoryPolicy

    record = MemoryRecord(memory_id="m-1", type=MemoryType.WORKSPACE,
                          content="project demo", source_ref="manifest.json",
                          confidence=0.8, consent=True)
    assert MemoryPolicy().evaluate(record)[0] is True
    assert "content" not in record.trace_view()


def test_prompt_catalogue_has_shared_contracts():
    from agent import prompts
    from agent.core.prompt_registry import active_prompt_versions

    names = [
        "UNDERSTAND_SYSTEM", "AGENT_SYSTEM", "PLAN_SYSTEM",
        "STEP_EXEC_SYSTEM", "CREATION_PLAN_SYSTEM", "CODING_PLAN_SYSTEM",
        "VERIFY_SYSTEM", "SYNTHESIZE_SYSTEM", "CHAT_SYSTEM", "TASK_SYSTEM",
        "CLARIFY_SYSTEM", "MEMORY_SUMMARY", "NOTIFY_SYSTEM",
    ]
    for name in names:
        text = getattr(prompts, name)
        assert "## Trust boundary" in text, name

    versions = active_prompt_versions("prompt-contract-test")
    assert "VERIFY_SYSTEM" in versions
    assert "MEMORY_SUMMARY" in versions
    assert "NOTIFY_SYSTEM" in versions
    assert "ARXIV_SYSTEM" in versions
    assert "INGEST_SYSTEM" in versions
    assert "CREATOR_SYSTEM" in versions
    assert "CODER_SYSTEM" in versions


def test_prompt_catalogue_logic_boundaries():
    from agent import prompts

    assert "plus a final synthesis step" not in prompts.PLAN_SYSTEM
    assert "suggest task_list" not in prompts.TASK_SYSTEM
    assert "search_papers() or check_paper()" in prompts.AGENT_SYSTEM
    assert "broad topic first" in prompts.AGENT_SYSTEM
    assert 'Emit only "coder" targets' in prompts.CODING_PLAN_SYSTEM
    assert "topic-only" in prompts.STEP_EXEC_SYSTEM


if __name__ == "__main__":
    test_error_envelope_is_compatible()
    test_tool_spec_marks_side_effects_for_approval()
    test_configuration_snapshot_is_stable_and_has_no_secret()
    test_legacy_tooldefs_have_governance_specs()
    test_context_decision_contains_no_prompt_payload()
    test_tool_metrics_use_structured_error_type()
    test_execution_context_metadata_contract()
    test_turn_entrypoint_freezes_langsmith_join_keys()
    test_memory_record_schema_round_trips()
    test_prompt_catalogue_has_shared_contracts()
    test_prompt_catalogue_logic_boundaries()
    print("Core contract self-check OK")
