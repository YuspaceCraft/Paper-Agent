"""Content-addressed artifact storage and pagination regressions."""

from __future__ import annotations

from agent.core.artifact_store import metadata, persist_text, read_page


def test_artifact_store_round_trip_and_paging(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    content = "".join(f"{i:05d}\n" for i in range(2500))

    ref = persist_text(
        content,
        tool_name="fetch_content",
        execution_id="exec-1",
        thread_id="thread-1",
        threshold=100,
    )
    assert ref is not None
    assert ref["artifact_id"]
    assert ref["tool_name"] == "fetch_content"
    assert metadata(ref["artifact_id"])["sha256"] == ref["sha256"]

    first = read_page(ref["artifact_id"], offset=0, max_chars=6000)
    assert first is not None
    assert first["text"].startswith("00000")
    assert first["next_offset"] == 6000
    assert first["eof"] is False

    second = read_page(
        ref["artifact_id"], offset=first["next_offset"], max_chars=6000,
    )
    assert second is not None
    assert second["text"]
    assert second["offset"] == 6000


def test_artifact_store_skips_small_and_invalid_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    assert persist_text("small", threshold=100) is None
    assert metadata("../escape") is None
    assert read_page("not-an-id") is None


def test_artifact_store_is_content_addressed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    first = persist_text("same payload", threshold=1)
    second = persist_text("same payload", threshold=1)
    assert first is not None and second is not None
    assert first["artifact_id"] == second["artifact_id"]
    assert first["sha256"] == second["sha256"]
