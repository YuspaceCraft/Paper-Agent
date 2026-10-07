"""ToolGateway / ToolPolicy self-check — no LLM, no network, no backend.

Run: python agent/tests/test_tool_gateway.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.core.contracts import ErrorType, ExecutionContext, ToolSpec, Permission
from agent.core.policy import ToolPolicy, idempotency_key
from agent.core.tool_gateway import ToolGateway, redact_args
from agent.core.tool_registry import ToolRegistry
from agent.core.result_utils import tool_ui_status


def _ctx(**kwargs):
    from agent.core import build_execution_context

    return build_execution_context(thread_id="t-gateway", **kwargs)


def _gateway(call_fn, specs, **kwargs):
    # Most gateway tests exercise retry/idempotency mechanics directly, not the
    # graph interrupt. Production now fails closed unless explicitly overridden.
    kwargs.setdefault("policy", ToolPolicy(enforce_approval=False))
    return ToolGateway(call_fn, ToolRegistry(specs), **kwargs)


def test_tool_ui_status_preserves_partial_and_skipped():
    assert tool_ui_status("succeeded") == "success"
    assert tool_ui_status("partial") == "partial"
    assert tool_ui_status("skipped") == "skipped"
    assert tool_ui_status("interrupted") == "interrupted"
    assert tool_ui_status("timed_out") == "error"


def test_read_tool_is_allowed_and_audited():
    events = []
    from agent import observability

    observability.register_trace_sink(events.append)
    try:
        async def call(name, args):
            return '{"schema_version":"1.0","outcome":"succeeded"}'

        gw = _gateway(call, [ToolSpec(name="search_papers")])
        out = asyncio.run(gw.invoke("search_papers", {"query": "rmnet"},
                                    ctx=_ctx()))
    finally:
        observability.unregister_trace_sink(events.append)

    assert out.outcome.value == "succeeded" and out.decision.action == "allow"
    assert json.loads(out.envelope)["outcome"] == "succeeded"
    audits = [e for e in events if e.get("event") == "tool_audit"]
    assert audits and audits[-1]["decision"] == "allow"
    assert audits[-1]["tool_version"] == "1"


def test_protected_args_never_reach_the_audit_stream():
    events = []
    from agent import observability

    observability.register_trace_sink(events.append)
    try:
        async def call(name, args):
            return "ok"

        gw = _gateway(call, [ToolSpec(name="write_file")])
        asyncio.run(gw.invoke("write_file", {"token": "sk-secret-value"},
                              ctx=_ctx()))
    finally:
        observability.unregister_trace_sink(events.append)

    blob = str(events)
    assert "sk-secret-value" not in blob
    assert redact_args({"a": 1})["arg_keys"] == ["a"]
    preview = redact_args({
        "command": "python train.py",
        "token": "sk-secret-value",
    })
    assert preview["arg_preview"]["command"] == "python train.py"
    assert preview["arg_preview"]["token"] == "[redacted]"


def test_side_effects_fail_closed_by_default(monkeypatch):
    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return "ran"

    monkeypatch.delenv("AGENT_TOOL_APPROVAL", raising=False)
    gw = ToolGateway(
        call,
        ToolRegistry([
            ToolSpec(
                name="run_experiment",
                permissions={Permission.EXECUTE},
                side_effect=True,
            ),
        ]),
    )
    out = asyncio.run(gw.invoke(
        "run_experiment", {"command": "echo unsafe"}, ctx=_ctx(),
    ))
    assert out.outcome.value == "interrupted"
    assert out.error is not None
    assert out.error.code == "APPROVAL_REQUIRED"
    assert calls["n"] == 0


def test_tool_node_does_not_swallow_graph_interrupt():
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool
    from langgraph.errors import GraphInterrupt

    from agent.nodes import build_tools_node

    async def pause() -> str:
        raise GraphInterrupt(())

    tool = StructuredTool.from_function(
        coroutine=pause, name="write_file", description="pause",
    )
    state = {
        "messages": [AIMessage(content="", tool_calls=[{
            "name": "write_file", "args": {}, "id": "approval-1",
        }])],
    }
    try:
        asyncio.run(build_tools_node([tool])(state))
    except GraphInterrupt:
        pass
    else:
        raise AssertionError("GraphInterrupt must propagate to LangGraph")


def test_missing_permission_is_denied_before_the_adapter_runs():
    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return "ran"

    ctx = _ctx()
    ctx.permissions = {Permission.READ}
    gw = _gateway(call, [ToolSpec(name="ingest", permissions={Permission.WRITE},
                                  side_effect=True)])
    out = asyncio.run(gw.invoke("ingest", {}, ctx=ctx))
    assert out.outcome.value == "failed" and out.decision.action == "deny"
    assert calls["n"] == 0
    assert '"code": "TOOL_NOT_PERMITTED"' in out.envelope


def test_approval_gate_blocks_side_effects_when_enforced():
    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return "ran"

    gw = _gateway(call, [ToolSpec(name="ingest", permissions={Permission.WRITE},
                                  side_effect=True)],
                  policy=ToolPolicy(enforce_approval=True))
    out = asyncio.run(gw.invoke("ingest", {}, ctx=_ctx()))
    assert out.outcome.value == "interrupted" and out.decision.action == "approval"
    assert calls["n"] == 0
    assert '"code": "APPROVAL_REQUIRED"' in out.envelope

    # Pre-approved tools (operator/UI decision) pass the same gate.
    gw_pre = _gateway(call, [ToolSpec(name="ingest", permissions={Permission.WRITE},
                                      side_effect=True)],
                      policy=ToolPolicy(enforce_approval=True,
                                        preapproved={"ingest"}))
    assert (
        asyncio.run(gw_pre.invoke("ingest", {}, ctx=_ctx())).outcome.value
        == "succeeded"
    )
    assert calls["n"] == 1


def test_missing_required_arguments_fail_validation():
    async def call(name, args):
        return "ran"

    spec = ToolSpec(name="search_papers",
                    input_schema={"required": ["query"]})
    gw = _gateway(call, [spec])
    out = asyncio.run(gw.invoke("search_papers", {}, ctx=_ctx()))
    assert out.outcome.value == "failed" and out.error.code == "TOOL_ARGS_INVALID"


def test_argument_schema_rejects_wrong_type_and_unknown_fields():
    async def call(name, args):
        return "ran"

    spec = ToolSpec(
        name="search_papers",
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        },
    )
    gw = _gateway(call, [spec])
    wrong_type = asyncio.run(gw.invoke(
        "search_papers", {"limit": "many"}, ctx=_ctx(),
    ))
    extra = asyncio.run(gw.invoke(
        "search_papers", {"limit": 1, "unexpected": True}, ctx=_ctx(),
    ))
    assert wrong_type.error.code == "TOOL_ARGS_INVALID"
    assert extra.error.code == "TOOL_ARGS_INVALID"
    assert "unexpected" in extra.error.message


def test_side_effect_result_is_replayed_instead_of_rerun():
    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return f"run-{calls['n']}"

    gw = _gateway(call, [ToolSpec(name="run_experiment", side_effect=True,
                                  idempotency_scope="task")])
    ctx = _ctx()
    first = asyncio.run(gw.invoke("run_experiment", {"project": "demo"}, ctx=ctx))
    second = asyncio.run(gw.invoke("run_experiment", {"project": "demo"}, ctx=ctx))
    assert calls["n"] == 1
    assert first.envelope == second.envelope == "run-1"
    assert second.deduplicated is True


def test_read_cache_hits_within_ttl_and_expires(monkeypatch):
    import agent.core.tool_gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "READ_CACHE_TTL", 10.0)
    calls = {"n": 0}
    now = {"t": 100.0}

    async def call(name, args):
        calls["n"] += 1
        return f"result-{calls['n']}"

    gw = _gateway(
        call,
        [ToolSpec(name="search_papers")],
        clock=lambda: now["t"],
    )
    ctx = _ctx()
    first = asyncio.run(gw.invoke("search_papers", {"query": "x"}, ctx=ctx))
    second = asyncio.run(gw.invoke("search_papers", {"query": "x"}, ctx=ctx))
    assert first.envelope == second.envelope == "result-1"
    assert calls["n"] == 1
    assert second.cache_hit is True

    now["t"] += 11
    third = asyncio.run(gw.invoke("search_papers", {"query": "x"}, ctx=ctx))
    assert third.envelope == "result-2"
    assert calls["n"] == 2


def test_read_cache_is_cleared_after_a_write(monkeypatch):
    import agent.core.tool_gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "READ_CACHE_TTL", 60.0)
    calls = {"search_papers": 0, "write_file": 0}

    async def call(name, args):
        calls[name] += 1
        return f"{name}-{calls[name]}"

    gw = _gateway(call, [
        ToolSpec(name="search_papers"),
        ToolSpec(name="write_file", side_effect=True,
                 permissions={Permission.WRITE}),
    ])
    ctx = _ctx()
    asyncio.run(gw.invoke("search_papers", {"query": "x"}, ctx=ctx))
    asyncio.run(gw.invoke("search_papers", {"query": "x"}, ctx=ctx))
    assert calls["search_papers"] == 1

    asyncio.run(gw.invoke("write_file", {"path": "x"}, ctx=ctx))
    asyncio.run(gw.invoke("search_papers", {"query": "x"}, ctx=ctx))
    assert calls["search_papers"] == 2


def test_durable_idempotency_replays_across_gateway_restart(tmp_path):
    from agent.core.idempotency import SQLiteIdempotencyStore

    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return f"durable-run-{calls['n']}"

    spec = ToolSpec(name="run_experiment", side_effect=True,
                    idempotency_scope="task")
    store_path = tmp_path / "idempotency.db"
    first = _gateway(
        call, [spec],
        idempotency_store=SQLiteIdempotencyStore(store_path),
    )
    second = _gateway(
        call, [spec],
        idempotency_store=SQLiteIdempotencyStore(store_path),
    )
    ctx = _ctx()
    one = asyncio.run(first.invoke(
        "run_experiment", {"project": "demo"}, ctx=ctx,
    ))
    two = asyncio.run(second.invoke(
        "run_experiment", {"project": "demo"}, ctx=ctx,
    ))
    assert calls["n"] == 1
    assert one.envelope == two.envelope == "durable-run-1"
    assert two.deduplicated is True


def test_indeterminate_side_effect_is_not_retried_after_restart(tmp_path):
    from agent.core.idempotency import SQLiteIdempotencyStore

    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        raise RuntimeError("crashed after the side effect may have started")

    spec = ToolSpec(name="run_experiment", side_effect=True,
                    idempotency_scope="task")
    store_path = tmp_path / "idempotency.db"
    first = _gateway(
        call, [spec],
        idempotency_store=SQLiteIdempotencyStore(store_path),
    )
    second = _gateway(
        call, [spec],
        idempotency_store=SQLiteIdempotencyStore(store_path),
    )
    ctx = _ctx()
    one = asyncio.run(first.invoke(
        "run_experiment", {"project": "demo"}, ctx=ctx,
    ))
    two = asyncio.run(second.invoke(
        "run_experiment", {"project": "demo"}, ctx=ctx,
    ))
    assert one.outcome.value == "failed" and one.error.code == "TOOL_EXECUTION_FAILED"
    assert two.outcome.value == "failed" and two.error.code == "SIDE_EFFECT_OUTCOME_UNKNOWN"
    assert calls["n"] == 1


def test_concurrent_side_effect_same_key_is_rejected():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        started.set()
        await release.wait()
        return "completed"

    spec = ToolSpec(name="run_experiment", side_effect=True,
                    idempotency_scope="task")
    gw = _gateway(call, [spec])
    ctx = _ctx()

    async def _run():
        first = asyncio.create_task(gw.invoke(
            "run_experiment", {"project": "demo"}, ctx=ctx,
        ))
        await started.wait()
        second = await gw.invoke(
            "run_experiment", {"project": "demo"}, ctx=ctx,
        )
        release.set()
        return await first, second

    first, second = asyncio.run(_run())
    assert first.outcome.value == "succeeded" and first.envelope == "completed"
    assert second.outcome.value == "failed"
    assert second.error.code == "IDEMPOTENCY_IN_PROGRESS"
    assert calls["n"] == 1


def test_cancelled_side_effect_becomes_indeterminate_not_running():
    started = asyncio.Event()
    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        started.set()
        await asyncio.Event().wait()

    spec = ToolSpec(name="ingest", side_effect=True,
                    idempotency_scope="task")
    gw = _gateway(call, [spec])
    ctx = _ctx()

    async def _run():
        task = asyncio.create_task(gw.invoke("ingest", {"path": "x"}, ctx=ctx))
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        again = await gw.invoke("ingest", {"path": "x"}, ctx=ctx)
        return again

    again = asyncio.run(_run())
    assert again.error.code == "SIDE_EFFECT_OUTCOME_UNKNOWN"
    assert calls["n"] == 1


def test_stale_running_idempotency_is_recovered():
    from agent.core.idempotency import SQLiteIdempotencyStore

    async def _run(tmp_path):
        store = SQLiteIdempotencyStore(
            tmp_path / "idempotency.db", stale_after_seconds=0.01,
        )
        first = await store.reserve(
            "key", tool_name="ingest", tool_version="1",
        )
        assert first.acquired is True
        await asyncio.sleep(0.02)
        second = await store.reserve(
            "key", tool_name="ingest", tool_version="1",
        )
        assert second.acquired is False
        assert second.record is not None
        assert second.record.status == "indeterminate"

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_run(Path(tmp)))


def test_retry_only_for_idempotent_tools():
    attempts = {"n": 0}

    async def flaky(name, args):
        attempts["n"] += 1
        await asyncio.Event().wait()  # always times out

    async def no_sleep(*a, **k):
        return None

    import unittest.mock as mock

    idempotent = _gateway(flaky, [ToolSpec(name="search_papers")],
                          timeout_resolver=lambda name: 0.02)
    with mock.patch("asyncio.sleep", new=no_sleep):
        out = asyncio.run(idempotent.invoke("search_papers", {}, ctx=_ctx()))
    assert attempts["n"] == 3, attempts["n"]
    assert out.error.code == "TOOL_TIMEOUT" and out.attempts == 3

    attempts["n"] = 0
    effectful = _gateway(flaky, [ToolSpec(name="ingest", side_effect=True,
                                          idempotency_scope="task")],
                         timeout_resolver=lambda name: 0.02)
    with mock.patch("asyncio.sleep", new=no_sleep):
        asyncio.run(effectful.invoke("ingest", {}, ctx=_ctx()))
    assert attempts["n"] == 1, attempts["n"]


def test_business_error_envelope_is_retried_and_reported_as_failure():
    attempts = {"n": 0}

    async def flaky(name, args):
        attempts["n"] += 1
        return (
            '{"outcome": "failed", "error": "temporary backend failure", '
            '"error_type": "transient", "retryable": true}'
        )

    async def no_sleep(*a, **k):
        return None

    import unittest.mock as mock

    gw = _gateway(flaky, [ToolSpec(name="search_papers")])
    with mock.patch("asyncio.sleep", new=no_sleep):
        out = asyncio.run(gw.invoke("search_papers", {}, ctx=_ctx()))
    assert attempts["n"] == 3
    assert out.outcome.value == "failed"
    assert out.error is not None
    assert out.error.error_type == ErrorType.TRANSIENT


def test_business_errors_open_the_circuit_breaker():
    attempts = {"n": 0}

    async def unavailable(name, args):
        attempts["n"] += 1
        return (
            '{"outcome": "failed", "error": "backend unavailable", '
            '"error_type": "backend_down", "retryable": true}'
        )

    async def no_sleep(*a, **k):
        return None

    import unittest.mock as mock

    gw = _gateway(
        unavailable,
        [ToolSpec(name="search_papers")],
        breaker_threshold=1,
        breaker_cooldown=60.0,
    )
    with mock.patch("asyncio.sleep", new=no_sleep):
        first = asyncio.run(gw.invoke("search_papers", {}, ctx=_ctx()))
        before = attempts["n"]
        blocked = asyncio.run(gw.invoke("search_papers", {}, ctx=_ctx()))
    assert first.outcome.value == "failed"
    assert before == 3
    assert attempts["n"] == before
    assert blocked.error.code == "TOOL_CIRCUIT_OPEN"


def test_circuit_breaker_short_circuits_a_wedged_tool():
    attempts = {"n": 0}

    async def wedged(name, args):
        attempts["n"] += 1
        raise ConnectionError("backend down")

    async def no_sleep(*a, **k):
        return None

    import unittest.mock as mock

    gw = _gateway(wedged, [ToolSpec(name="search_papers")],
                  timeout_resolver=lambda name: 1.0,
                  breaker_threshold=2, breaker_cooldown=60.0)
    with mock.patch("asyncio.sleep", new=no_sleep):
        for _ in range(2):
            outcome = asyncio.run(gw.invoke("search_papers", {}, ctx=_ctx()))
            assert outcome.error.code == "TOOL_UNAVAILABLE"
    before = attempts["n"]
    blocked = asyncio.run(gw.invoke("search_papers", {}, ctx=_ctx()))
    assert attempts["n"] == before, "breaker must not call the adapter"
    assert blocked.error.code == "TOOL_CIRCUIT_OPEN"
    assert blocked.error.retry_after_seconds > 0


def test_idempotency_key_is_stable_and_content_free():
    spec = ToolSpec(name="run_experiment")
    first = idempotency_key(thread_id="t", spec=spec,
                            args={"project": "demo"}, intent="coding")
    second = idempotency_key(thread_id="t", spec=spec,
                             args={"project": "demo"}, intent="coding")
    assert first == second and len(first) == 32
    assert "demo" not in first
    assert idempotency_key(thread_id="t", spec=spec, args={"project": "other"},
                           intent="coding") != first


def test_idempotency_key_is_scoped_to_one_turn():
    spec = ToolSpec(name="run_experiment")
    first = idempotency_key(
        thread_id="t", execution_id="turn-1", spec=spec,
        args={"project": "demo"},
    )
    second = idempotency_key(
        thread_id="t", execution_id="turn-2", spec=spec,
        args={"project": "demo"},
    )
    assert first != second


def test_intentional_repeat_in_a_new_turn_executes_again():
    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return f"run-{calls['n']}"

    spec = ToolSpec(
        name="run_experiment", side_effect=True, idempotency_scope="task",
    )
    gw = _gateway(call, [spec])
    first_ctx = _ctx(request_id="turn-1")
    second_ctx = _ctx(request_id="turn-2")

    first = asyncio.run(gw.invoke(
        "run_experiment", {"project": "demo"}, ctx=first_ctx,
    ))
    second = asyncio.run(gw.invoke(
        "run_experiment", {"project": "demo"}, ctx=second_ctx,
    ))
    assert calls["n"] == 2
    assert first.envelope == "run-1"
    assert second.envelope == "run-2"


def test_graph_approval_interrupts_and_resumes(monkeypatch):
    """A policy approval decision becomes a real LangGraph interrupt."""
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, StateGraph
    from langgraph.types import Command

    from agent.core.approval import pending_tool_approval, tool_approval_scope
    from agent.dispatcher import ToolDispatcher
    from agent.providers import ToolDef

    monkeypatch.setenv("AGENT_TOOL_APPROVAL", "1")

    class State(TypedDict):
        output: str

    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return '{"outcome": "succeeded", "data": {"written": true}}'

    dispatcher = ToolDispatcher(call, [
        ToolDef("write_file", "write", {"type": "object"}, "builtin",
                {"readOnlyHint": False}),
    ])

    async def node(state, config):
        with tool_approval_scope(config):
            result = await dispatcher.call("write_file", {"path": "x.txt"})
        return {"output": str(result)}

    graph = StateGraph(State)
    graph.add_node("write", node)
    graph.add_edge(START, "write")
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-ok"}}

    first = asyncio.run(app.ainvoke({"output": ""}, config=config))
    approval = pending_tool_approval(first)
    assert approval and approval["tool"] == "write_file"
    assert calls["n"] == 0

    resumed = asyncio.run(app.ainvoke(
        Command(resume={"approved": True}), config=config,
    ))
    assert calls["n"] == 1
    assert '"written": true' in resumed["output"]


def test_graph_approval_denial_stops_side_effect(monkeypatch):
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, StateGraph
    from langgraph.types import Command

    from agent.core.approval import tool_approval_scope
    from agent.dispatcher import ToolDispatcher
    from agent.providers import ToolDef

    monkeypatch.setenv("AGENT_TOOL_APPROVAL", "1")

    class State(TypedDict):
        output: str

    calls = {"n": 0}

    async def call(name, args):
        calls["n"] += 1
        return "should-not-run"

    dispatcher = ToolDispatcher(call, [
        ToolDef("write_file", "write", {"type": "object"}, "builtin",
                {"readOnlyHint": False}),
    ])

    async def node(state, config):
        with tool_approval_scope(config):
            result = await dispatcher.call("write_file", {"path": "x.txt"})
        return {"output": str(result)}

    graph = StateGraph(State)
    graph.add_node("write", node)
    graph.add_edge(START, "write")
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-deny"}}

    asyncio.run(app.ainvoke({"output": ""}, config=config))
    resumed = asyncio.run(app.ainvoke(
        Command(resume={"approved": False}), config=config,
    ))
    assert calls["n"] == 0
    assert '"code": "APPROVAL_DENIED"' in resumed["output"]


def test_parallel_side_effects_approve_as_one_batch_without_replay(monkeypatch):
    from typing import TypedDict

    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, StateGraph
    from langgraph.types import Command

    from agent.core.approval import pending_tool_approval
    from agent.dispatcher import ToolDispatcher
    from agent.nodes import build_tools_node
    from agent.providers import ToolDef
    from agent.tools import _to_langchain_tool

    monkeypatch.setenv("AGENT_TOOL_APPROVAL", "1")

    calls: list[str] = []

    async def call(name, args):
        calls.append(name)
        return f'{{"outcome":"succeeded","data":{{"name":"{name}"}}}}'

    defs = [
        ToolDef("write_a", "w", {"type": "object", "properties": {}}, "builtin",
                {"readOnlyHint": False}),
        ToolDef("write_b", "w", {"type": "object", "properties": {}}, "builtin",
                {"readOnlyHint": False}),
    ]
    dispatcher = ToolDispatcher(call, defs)
    tools = [_to_langchain_tool(td, dispatcher.call) for td in defs]

    class State(TypedDict):
        messages: list
        tool_result_cache: dict

    def entry(_state):
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "write_a", "args": {}, "id": "a"},
            {"name": "write_b", "args": {}, "id": "b"},
        ])]}

    graph = StateGraph(State)
    graph.add_node("entry", entry)
    graph.add_node("tools", build_tools_node(tools))
    graph.add_edge(START, "entry")
    graph.add_edge("entry", "tools")
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "batch-approval"}}

    first = asyncio.run(app.ainvoke(
        {"messages": [], "tool_result_cache": {}}, config=config,
    ))
    approval = pending_tool_approval(first)
    assert approval is not None
    assert approval["tool"] == "batch"
    assert len(approval["calls"]) == 2
    assert calls == []

    resumed = asyncio.run(app.ainvoke(
        Command(resume={"approved": True}), config=config,
    ))
    assert pending_tool_approval(resumed) is None
    assert calls == ["write_a", "write_b"]


if __name__ == "__main__":
    test_read_tool_is_allowed_and_audited()
    test_protected_args_never_reach_the_audit_stream()
    test_missing_permission_is_denied_before_the_adapter_runs()
    test_approval_gate_blocks_side_effects_when_enforced()
    test_missing_required_arguments_fail_validation()
    test_side_effect_result_is_replayed_instead_of_rerun()
    test_retry_only_for_idempotent_tools()
    test_circuit_breaker_short_circuits_a_wedged_tool()
    test_idempotency_key_is_stable_and_content_free()
    print("ToolGateway self-check OK")
