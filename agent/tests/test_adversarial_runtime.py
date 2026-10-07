"""Adversarial runtime regressions for token, memory and tool governance."""

from __future__ import annotations

import asyncio
import json
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from agent.core.context_pack import ContextManager, ContextZone
from agent.core.contracts import Budget, ToolSpec
from agent.core.memory_policy import MemoryRecord, MemoryType
from agent.core.memory_store import list_records, upsert_record
from agent.core.tool_gateway import ToolGateway
from agent.core.tool_registry import ToolRegistry
from agent.dispatcher import ToolDispatcher
from agent.memory import MemoryManager, _estimate_tokens, _truncate_to_tokens
from agent.nodes import build_tools_node
from agent.providers import ToolDef


def test_cjk_truncation_never_exceeds_token_budget():
    text = "遥感变化检测" * 2000
    for limit in (1, 2, 5, 100):
        truncated = _truncate_to_tokens(text, limit)
        assert _estimate_tokens(truncated) <= limit


def test_context_pack_charges_actual_invariant_size_and_stays_bounded():
    from agent.core.execution_context import build_execution_context

    ctx = build_execution_context(thread_id="adversarial-context")
    ctx.budget = Budget(context_window_tokens=8000, max_output_tokens=1000)
    invariant = "策略约束" * 500
    pack = ContextManager().build(
        ctx,
        {
            "messages": [
                HumanMessage(content="问题" * 1500),
            ],
        },
        invariant=invariant,
        retrieved=[
            {"text": "证据" * 1500, "source_ref": f"p{i}", "score": 1.0}
            for i in range(8)
        ],
        memory_records=[
            MemoryRecord(memory_id=f"m{i}", content="偏好" * 200)
            for i in range(8)
        ],
    )

    invariant_zone = pack.zone(ContextZone.INVARIANT)
    assert invariant_zone is not None
    assert invariant_zone.tokens <= invariant_zone.budget
    assert pack.decision["estimated_tokens"] <= pack.decision["max_tokens"]
    assert pack.decision["budget_exceeded"] is False


def test_malformed_retrieval_score_does_not_break_context_pack():
    pack = ContextManager().build(
        None,
        {"messages": []},
        retrieved=[{
            "text": "valid evidence",
            "source_ref": "p1",
            "score": "not-a-number",
        }],
    )
    assert "valid evidence" in pack.zone(ContextZone.RETRIEVED).content


def test_summary_refresh_is_batched_instead_of_recomputed_every_turn():
    manager = MemoryManager()
    messages = [HumanMessage(content=f"m{i}") for i in range(13)]
    state = {
        "messages": messages,
        "summary_cache": "existing summary",
        "summary_through_seq": len(messages) - manager.BUFFER_SIZE,
    }

    state["messages"] = [*messages, HumanMessage(content="one new turn")]
    assert manager.needs_summary_update(state) is False

    state["messages"] = [
        *messages,
        *[HumanMessage(content=f"new-{i}") for i in range(6)],
    ]
    assert manager.needs_summary_update(state) is True


def test_small_budget_keeps_newest_conversation_not_oldest():
    manager = MemoryManager()
    messages = [
        HumanMessage(content="旧问题" * 200),
        HumanMessage(content="新问题"),
    ]
    rendered = manager._format_buffer(messages, max_chars=20)
    assert "新问题" in rendered
    assert "旧问题" not in rendered
    assert len(rendered) <= 20


def test_snapshot_token_budget_is_exact_for_cjk():
    manager = MemoryManager()
    messages = [
        HumanMessage(content="旧问题" * 500),
        HumanMessage(content="新问题" * 500),
    ]
    snapshot = manager.build_snapshot({"messages": messages}, max_tokens=120)
    assert _estimate_tokens(snapshot) <= 120


def test_dispatcher_redacts_named_secrets_from_trace():
    from agent import observability

    events: list[dict] = []
    observability.register_trace_sink(events.append)
    try:
        async def call(name, args):
            return '{"outcome": "succeeded"}'

        dispatcher = ToolDispatcher(call, [
            ToolDef(
                "write_file",
                "write",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "token": {"type": "string"},
                    },
                },
                "builtin",
                {"readOnlyHint": False, "idempotentHint": False},
            ),
        ])
        asyncio.run(dispatcher.call(
            "write_file", {"path": "x.txt", "token": "sk-super-secret"},
        ))
    finally:
        observability.unregister_trace_sink(events.append)

    tool_events = [event for event in events if event.get("event") == "tool_call"]
    assert len(tool_events) == 1
    assert "sk-super-secret" not in json.dumps(tool_events, ensure_ascii=False)
    assert "[redacted]" in str(tool_events[0].get("args"))


def test_oversized_memory_is_normalized_before_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    base = "x" * 1000
    first = upsert_record(content=base + "AAA", memory_type=MemoryType.PREFERENCE)
    second = upsert_record(content=base + "BBB", memory_type=MemoryType.PREFERENCE)

    records = list_records()
    assert first.memory_id == second.memory_id
    assert len(records) == 1
    assert len(records[0].content) == 1000


def test_gateway_decision_history_is_bounded(monkeypatch):
    monkeypatch.setattr("agent.core.tool_gateway.log_event", lambda *a, **k: None)

    async def call(name, args):
        return "ok"

    gateway = ToolGateway(
        call,
        ToolRegistry([ToolSpec(name="read")]),
        max_concurrency=4,
    )

    async def invoke_many():
        for _ in range(520):
            await gateway.invoke("read", {})

    asyncio.run(invoke_many())
    assert len(gateway.decisions) == 512
    assert gateway.decisions_total == 520


def test_concurrent_duplicate_tool_calls_execute_once():
    calls = 0

    async def echo(value: str) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return json.dumps({"outcome": "succeeded", "data": {"value": value}})

    tool = StructuredTool.from_function(
        coroutine=echo, name="echo", description="echo",
    )
    state = {
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"value": "x"}, "id": "c1"},
            {"name": "echo", "args": {"value": "x"}, "id": "c2"},
        ])],
    }
    result = asyncio.run(build_tools_node([tool])(state))

    assert calls == 1
    assert len(result["messages"]) == 2
    assert any("重复调用" in message.content for message in result["messages"])


def test_new_turn_resets_tool_result_cache():
    from agent.nodes import _detect_follow_up

    state = {
        "messages": [
            HumanMessage(content="first question"),
            AIMessage(content="first answer"),
            HumanMessage(content="继续"),
        ],
        "intent": "literature_search",
        "tool_result_cache": {
            "write_file|{\"path\":\"x\"}": {"content": "old", "count": 1},
        },
    }
    update = _detect_follow_up(state)
    assert update is not None
    assert update["tool_result_cache"] == {}


def test_task_registry_queries_independent_sources_concurrently(monkeypatch):
    from agent import task_registry

    active = 0
    max_active = 0

    def source(kind: str):
        async def query():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [{
                "task_id": kind,
                "kind": kind,
                "title": kind,
                "status": "done",
                "created_at": "2026-01-01",
            }]
        return query

    monkeypatch.setattr(task_registry, "_dispatched", source("dispatched"))
    monkeypatch.setattr(task_registry, "_experiments", source("experiment"))
    monkeypatch.setattr(task_registry, "_docs", source("doc"))
    monkeypatch.setattr(task_registry, "_redis_tasks", source("pipeline"))

    tasks = asyncio.run(task_registry.list_tasks())
    assert len(tasks) == 4
    assert max_active == 4


def test_study_topic_and_project_dot_cannot_escape_roots(
    monkeypatch, tmp_path,
):
    from agent import workspace_config
    from agent.domains import coding

    experiments = tmp_path / "experiments"
    studies = tmp_path / "studies"
    experiments.mkdir()
    studies.mkdir()
    monkeypatch.setattr(
        workspace_config, "get_experiments_path", lambda: experiments,
    )
    monkeypatch.setattr(workspace_config, "get_study_root", lambda: studies)

    try:
        coding._project_dir(".")
    except PermissionError:
        pass
    else:
        raise AssertionError("project '.' must not resolve to experiments root")

    for topic in (".", "..", "../../outside"):
        path = coding._study_path(topic)
        assert path.resolve().is_relative_to(studies.resolve())
        assert path.parent != studies.resolve()


def test_workspace_and_generic_file_reads_are_bounded(monkeypatch, tmp_path):
    from agent.providers import generic_provider
    from web.api.routers import workspace as workspace_router

    root = tmp_path / "project"
    root.mkdir()
    payload = b"x" * 200_000
    (root / "large.txt").write_bytes(payload)
    monkeypatch.setattr(workspace_router, "get_project_root", lambda: root)
    monkeypatch.setattr(generic_provider, "get_project_root", lambda: root)

    api_result = asyncio.run(workspace_router.read_workspace(
        "large.txt", root="project",
    ))
    api_data = api_result["data"]
    assert api_data["truncated"] is True
    assert api_data["size"] == len(payload)
    assert len(api_data["content"].encode("utf-8")) < len(payload)

    tool_result = asyncio.run(generic_provider._read_file(str(root / "large.txt")))
    assert "truncated" in tool_result
    assert len(tool_result.encode("utf-8")) < len(payload)


def test_invalid_outline_does_not_silently_succeed(monkeypatch, tmp_path):
    from agent import workspace_config
    from agent.domains.creation import doc_create, doc_set_outline

    monkeypatch.setattr(
        workspace_config, "get_docs_dir", lambda: tmp_path / "docs",
    )
    created = json.loads(asyncio.run(doc_create.ainvoke({"title": "test"})))
    doc_id = created["data"]["doc_id"]
    result = json.loads(asyncio.run(doc_set_outline.ainvoke({
        "doc_id": doc_id, "outline": '{"not": "a list"}',
    })))
    assert result["outcome"] == "failed"
    assert result["error_type"] == "param_error"


def test_calculator_rejects_expensive_exponents():
    from agent.providers.generic_provider import _calculator

    result = asyncio.run(_calculator("2 ** 10000"))
    assert "too large" in result


def test_experiment_concurrency_limit_has_no_startup_race(
    monkeypatch, tmp_path,
):
    from agent import workspace_config
    from agent.domains import coding

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    monkeypatch.setattr(
        workspace_config, "get_experiments_path", lambda: experiments,
    )
    monkeypatch.setattr(coding, "_MAX_RUNNING_EXPERIMENTS", 1)
    monkeypatch.setattr(coding, "_starting_experiments", 0)
    monkeypatch.setattr(coding, "_bg_tasks", set())

    async def slow_spawn(exp):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(coding, "_spawn", slow_spawn)

    async def run_pair():
        return await asyncio.gather(
            coding.run_experiment.ainvoke({
                "project": "demo", "command": "echo first",
            }),
            coding.run_experiment.ainvoke({
                "project": "demo", "command": "echo second",
            }),
        )

    first, second = asyncio.run(run_pair())
    payloads = [json.loads(first), json.loads(second)]
    assert sum(
        1 for payload in payloads if payload["outcome"] == "succeeded"
    ) == 1
    rejected = next(
        payload for payload in payloads
        if payload["outcome"] == "failed"
    )
    assert rejected["error_type"] == "transient"


def test_notify_stream_disconnect_cancels_generation(monkeypatch):
    from agent import notifier
    from agent.stream import emit
    from web.api.routers.background import notify_stream
    from web.api.schemas import AgentNotifyRequest

    cancelled = asyncio.Event()

    async def fake_notify(task):
        emit({"type": "token", "content": "hello"})
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(notifier, "stream_task_notify", fake_notify)

    async def exercise():
        response = await notify_stream(AgentNotifyRequest(
            task={"task_id": "t1", "status": "done"},
        ))
        iterator = response.body_iterator
        first = await anext(iterator)
        assert "hello" in first
        await iterator.aclose()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert cancelled.is_set()

    asyncio.run(exercise())


def test_fallback_task_store_is_bounded_and_expires(monkeypatch):
    import web.api.routers as routers_init

    now = time.time()
    fallback = {
        f"old-{i}": {
            "task_id": f"old-{i}",
            "created_at": str(now - 4000),
            "updated_at": str(now - 4000),
        }
        for i in range(10)
    }
    fallback.update({
        f"new-{i}": {
            "task_id": f"new-{i}",
            "created_at": str(now - i),
            "updated_at": str(now - i),
        }
        for i in range(501)
    })
    monkeypatch.setattr(routers_init, "_FALLBACK", fallback)
    monkeypatch.setattr(routers_init, "_FALLBACK_MAX", 500)

    routers_init._fallback_prune()
    assert len(fallback) == 500
    assert all(not task_id.startswith("old-") for task_id in fallback)


def test_bounded_task_event_queue_drops_oldest(monkeypatch):
    import web.api.routers as routers_init

    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    routers_init._put_task_event(queue, "one")
    routers_init._put_task_event(queue, "two")
    routers_init._put_task_event(queue, "three")
    assert [queue.get_nowait(), queue.get_nowait()] == ["two", "three"]


def test_memory_rejects_invalid_expiry(monkeypatch, tmp_path):
    from agent.core.memory_store import upsert_record

    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    try:
        upsert_record(content="temporary", expires_at="not-a-date")
    except ValueError as exc:
        assert "ISO-8601" in str(exc)
    else:
        raise AssertionError("invalid expiry must not become permanent memory")
