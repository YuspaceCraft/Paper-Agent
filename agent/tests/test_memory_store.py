"""Long-term memory store self-check — no network or LLM."""

from agent.core.context_pack import ContextManager, ContextZone
from agent.core.memory_policy import MemoryType
from agent.core.memory_store import (
    approve_record,
    capture_explicit_memory,
    delete_record,
    get_policy,
    list_records,
    update_policy,
    upsert_record,
)
from agent.core.postmortem import record_turn_postmortem


def test_memory_crud_policy_and_context_injection(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    record = upsert_record(
        content="以后请使用中文回答",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.9,
    )
    assert [item.memory_id for item in list_records()] == [record.memory_id]

    pack = ContextManager().build(
        None, {"messages": []}, memory_records=list_records(),
    )
    assert "以后请使用中文回答" in pack.zone(ContextZone.MEMORY).content

    update_policy(disabled_ids=[record.memory_id])
    assert record.memory_id in get_policy().disabled_ids
    pack_disabled = ContextManager().build(None, {"messages": []})
    entries = pack_disabled.zone(ContextZone.MEMORY).entries
    assert any(
        item.get("memory_id") == record.memory_id
        and item.get("kept") is False
        and item.get("reason") == "disabled by the user"
        for item in entries
    )

    assert delete_record(record.memory_id) is True
    assert list_records() == []


def test_explicit_memory_capture_does_not_store_ordinary_queries(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    assert capture_explicit_memory("遥感论文有哪些？") is None
    record = capture_explicit_memory("记住：我偏好简洁回答")
    assert record is not None
    assert record.type == MemoryType.PREFERENCE
    assert "简洁回答" in record.content


def test_conflicting_semantic_keys_are_not_injected(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    first = upsert_record(
        content="preferred language: zh",
        memory_type=MemoryType.PREFERENCE,
        source_ref="user",
        confidence=0.9,
    )
    second = upsert_record(
        content="preferred language: en",
        memory_type=MemoryType.PREFERENCE,
        source_ref="user",
        confidence=0.9,
    )

    assert second.status == "conflict"
    records = {item.memory_id: item for item in list_records()}
    assert records[first.memory_id].status == "conflict"
    pack = ContextManager().build(
        None, {"messages": []}, memory_records=list_records(),
    )
    assert "preferred language" not in pack.zone(ContextZone.MEMORY).content.lower()


def test_lower_trust_conflict_is_superseded_by_user_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    upsert_record(
        content="preferred language: zh",
        memory_type=MemoryType.PREFERENCE,
        source_ref="conversation",
        confidence=0.9,
    )
    user_record = upsert_record(
        content="preferred language: en",
        memory_type=MemoryType.PREFERENCE,
        source_ref="user",
        confidence=0.9,
    )

    assert user_record.status == "active"
    records = list_records()
    assert sum(item.status == "superseded" for item in records) == 1


def test_postmortem_stays_candidate_until_user_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    record = record_turn_postmortem(
        query="compare two papers",
        status="timeout",
        error="turn_timeout",
        trace_id="trace-1",
    )
    assert record is not None
    assert record.status == "candidate"
    assert record.consent is False

    before = ContextManager().build(
        None, {"messages": []}, memory_records=list_records(),
    )
    assert "compare two papers" not in before.zone(ContextZone.MEMORY).content

    approved = approve_record(record.memory_id)
    assert approved is not None and approved.status == "active"
    after = ContextManager().build(
        None, {"messages": []}, memory_records=list_records(),
    )
    assert "compare two papers" in after.zone(ContextZone.MEMORY).content
