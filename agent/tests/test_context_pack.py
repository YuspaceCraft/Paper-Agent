"""Context Pack budget/policy self-check — no LLM, no network.

Run: python agent/tests/test_context_pack.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.core.context_pack import ContextManager, ContextZone
from agent.core.contracts import Budget
from agent.core.memory_policy import MemoryPolicy, MemoryRecord, MemoryType
from agent.core.memory_policy import records_from_profile


def _small_window_ctx(context_window: int = 8000, max_output: int = 1000):
    from agent.core import build_execution_context

    ctx = build_execution_context(thread_id="t-pack")
    ctx.budget = Budget(context_window_tokens=context_window,
                        max_output_tokens=max_output)
    return ctx


def test_zone_budgets_follow_the_model_window():
    cm = ContextManager()
    budgets = cm.zone_budgets(usable_tokens=20000)
    assert budgets[ContextZone.INVARIANT] == 1500
    assert budgets[ContextZone.RESERVE] == int(18500 * 0.05)
    assert budgets[ContextZone.RETRIEVED] > budgets[ContextZone.TASK]
    # Invariant + reserve + the four flexible zones cover the usable window.
    assert sum(budgets.values()) == 20000


def test_pack_records_truncation_without_leaking_prompt_text():
    cm = ContextManager()
    pack = cm.build(
        None,
        {"messages": [], "intent": "literature_search"},
        invariant="SYSTEM POLICY " * 50,
        retrieved=[{"text": "chunk-big " * 200, "source_ref": "p1"}],
        memory_records=[MemoryRecord(memory_id="m1", content="prefers English")],
    )
    decision = pack.decision
    assert decision["max_tokens"] > 0
    assert "SYSTEM POLICY" not in str(decision)
    assert "chunk-big" not in str(decision)
    assert decision["zones"]["retrieved"]["entries"][0]["source_ref"] == "p1"
    assert pack.zone(ContextZone.MEMORY).tokens > 0


def test_retrieved_duplicates_are_dropped_and_diversity_is_kept():
    cm = ContextManager()
    pack = cm.build(None, {"messages": []}, retrieved=[
        {"text": "same chunk", "source_ref": "p1", "score": 0.9},
        {"text": "same   chunk", "source_ref": "p1-dup", "score": 0.8},
        {"text": "different evidence", "source_ref": "p2", "score": 0.5},
    ])
    zone = pack.zone(ContextZone.RETRIEVED)
    assert zone.content.count("same chunk") == 1
    assert "different evidence" in zone.content
    dropped = [e for e in zone.entries if e.get("kept") is False]
    assert dropped and "duplicate" in dropped[0]["reason"]


def test_retrieved_zone_stops_at_its_budget():
    cm = ContextManager()
    pack = cm.build(_small_window_ctx(), {"messages": []}, retrieved=[
        {"text": f"evidence {i} " * 400, "source_ref": f"p{i}"} for i in range(20)
    ])
    zone = pack.zone(ContextZone.RETRIEVED)
    assert zone.tokens <= zone.budget
    assert zone.truncated is True
    assert any(e.get("kept") is False for e in zone.entries)


def test_task_zone_carries_the_workspace_binding():
    cm = ContextManager()
    pack = cm.build(None, {
        "messages": [],
        "intent": "coding",
        "context": {"active_project": "demo", "recent_experiments": ["exp-1"]},
    })
    content = pack.zone(ContextZone.TASK).content
    assert "Active experiment project: demo" in content
    assert "exp-1" in content


def test_memory_policy_filters_by_confidence_consent_and_ttl():
    policy = MemoryPolicy(min_confidence=0.5, max_age_days=30,
                          require_consent=True)
    records = [
        MemoryRecord(memory_id="ok", content="keep me", confidence=0.9),
        MemoryRecord(memory_id="low", content="weak", confidence=0.2),
        MemoryRecord(memory_id="noconsent", content="private", consent=False),
        MemoryRecord(memory_id="expired", content="old",
                     expires_at="2000-01-01T00:00:00+00:00"),
    ]
    kept, dropped = policy.filter(records)
    assert [r.memory_id for r in kept] == ["ok"]
    reasons = {d["memory_id"]: d["reason"] for d in dropped}
    assert "confidence" in reasons["low"]
    assert reasons["noconsent"] == "no consent recorded"
    assert reasons["expired"] == "expired"
    # Audit entries never carry memory content.
    assert "private" not in str(dropped)


def test_profile_records_are_typed_and_trace_view_is_metadata_only():
    records = records_from_profile({"preferred_language": "zh",
                                    "known_papers": ["RMNet", "SAM"]})
    types = {r.type for r in records}
    assert MemoryType.PREFERENCE in types and MemoryType.LONG_TERM_FACT in types
    view = records[0].trace_view()
    assert "content" not in view and view["memory_id"].startswith("profile:")


def test_context_pack_resolves_cross_source_memory_conflicts():
    cm = ContextManager()
    pack = cm.build(
        None,
        {"messages": []},
        memory_records=[
            MemoryRecord(
                memory_id="profile-language",
                content="preferred language: zh",
                source_ref="profile.json",
                confidence=1.0,
                semantic_key="preference:preferred language",
            ),
            MemoryRecord(
                memory_id="user-language",
                content="preferred language: en",
                source_ref="user",
                confidence=0.95,
                semantic_key="preference:preferred language",
            ),
        ],
    )

    zone = pack.zone(ContextZone.MEMORY)
    assert "preferred language: en" in zone.content
    assert "preferred language: zh" not in zone.content
    assert any(
        item.get("memory_id") == "profile-language"
        and item.get("kept") is False
        for item in zone.entries
    )


def test_recent_retrieval_tool_message_populates_retrieved_zone():
    from langchain_core.messages import ToolMessage

    cm = ContextManager()
    payload = json.dumps({
        "outcome": "succeeded",
        "data": {
            "results": [{
                "chunk_id": "paper__chunk_0001",
                "text": "retrieved evidence about remote sensing",
                "score": 0.9,
            }],
        },
    })
    pack = cm.build(None, {
        "messages": [ToolMessage(
            content=payload, tool_call_id="call-1", name="search_papers",
        )],
    }, memory_records=[])
    zone = pack.zone(ContextZone.RETRIEVED)
    assert "retrieved evidence" in zone.content
    assert zone.entries[0]["source_ref"] == "paper__chunk_0001"


def test_context_pack_keeps_profile_in_memory_zone(monkeypatch):
    from agent import memory

    monkeypatch.setattr(memory, "load_profile",
                        lambda: {"preferred_language": "zh"})
    cm = ContextManager()
    pack = cm.build(None, {"messages": []})
    conversation = pack.zone(ContextZone.CONVERSATION).content.lower()
    memory_zone = pack.zone(ContextZone.MEMORY).content.lower()
    assert "preferred language" not in conversation
    assert "preferred language" in memory_zone


if __name__ == "__main__":
    test_zone_budgets_follow_the_model_window()
    test_pack_records_truncation_without_leaking_prompt_text()
    test_retrieved_duplicates_are_dropped_and_diversity_is_kept()
    test_retrieved_zone_stops_at_its_budget()
    test_task_zone_carries_the_workspace_binding()
    test_memory_policy_filters_by_confidence_consent_and_ttl()
    test_profile_records_are_typed_and_trace_view_is_metadata_only()
    test_recent_retrieval_tool_message_populates_retrieved_zone()
    print("Context Pack self-check OK")
