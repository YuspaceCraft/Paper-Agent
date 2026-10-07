from __future__ import annotations

from pathlib import Path

from evaluation.artifacts import persist_if_large
from evaluation.sink import clear_eval_ctx, set_eval_ctx


def test_large_eval_result_is_content_addressed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_TOOL_ARTIFACT_DIR", str(tmp_path))
    set_eval_ctx(thread_id="artifact-test", source="single")
    try:
        raw = "x" * 4100
        meta = persist_if_large("search_papers", raw)
        assert meta["result_stored"] is True
        assert meta["result_bytes"] == 4100
        assert len(meta["result_sha256"]) == 16
        target = Path(meta["result_ref"])
        if not target.is_absolute():
            target = Path.cwd() / target
        assert target.read_text(encoding="utf-8") == raw
    finally:
        clear_eval_ctx()


def test_small_result_stays_inline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_TOOL_ARTIFACT_DIR", str(tmp_path))
    set_eval_ctx(thread_id="artifact-small", source="single")
    try:
        meta = persist_if_large("search_papers", "small")
        assert meta["result_stored"] is False
        assert "result_ref" not in meta
    finally:
        clear_eval_ctx()
