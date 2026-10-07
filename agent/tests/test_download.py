"""test_download.py — download/process decoupling + short-name naming self-check.

Covers the pure helpers behind download_paper (naming derivation, sanitize) and
the ToolDef/function signature contract. No network, no backend — assert-based.

Run: python agent/tests/test_download.py
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.providers.builtin_provider import (
    _sanitize_dl_name,
    _short_name_from_title,
    _resolve_stem,
    BUILTIN_TOOLDEFS,
)
from agent.nodes import _download_preflight_hint


def test_sanitize_dl_name():
    assert _sanitize_dl_name("RMNet") == "RMNet"
    assert _sanitize_dl_name(" Some  Name ") == "Some_Name"
    assert _sanitize_dl_name("a:b<cd>ef|") == "abcdef"
    assert len(_sanitize_dl_name("X" * 200)) <= 80
    assert _sanitize_dl_name("").startswith("paper_"), "empty must fall back to a stem"


def test_short_name_from_title():
    assert _short_name_from_title(
        "RMNet: Re-parameterizing Multi-resolution Networks for Change Detection"
    ) == "RMNet"
    assert _short_name_from_title(
        "Diffusion-RSCC: Diffusion Probabilistic Model for Change Captioning "
        "in Remote Sensing Images"
    ) == "Diffusion-RSCC"
    assert _short_name_from_title(
        "Transformers — A Rapid Survey"
    ) == "Transformers"
    # No clean leading identifier → None (caller falls back to arxiv_id)
    assert _short_name_from_title("A Survey of Change Detection") is None
    assert _short_name_from_title("Attention Is All You Need") is None
    assert _short_name_from_title("") is None


def test_resolve_stem_priority():
    title = "RMNet: Re-parameterizing Multi-resolution Networks for Change Detection"
    # Explicit filename wins
    assert _resolve_stem("2305.03195", title, "My_Paper") == "My_Paper"
    # Empty filename → title-derived short name
    assert _resolve_stem("2305.03195", title, "") == "RMNet"
    # No title / no short name → arxiv_id
    assert _resolve_stem("2305.03195", None, "") == "2305.03195"
    assert _resolve_stem("2305.03195", "A Survey of X", "") == "2305.03195"


def test_tooldef_matches_function_signature():
    """LLM-visible ToolDef params must equal the @tool function's kwargs
    (calls go through fn.ainvoke(arguments); extra='forbid' rejects unknowns)."""
    by_name = {t.name: t for t in BUILTIN_TOOLDEFS}
    dl = by_name["download_paper"].parameters
    assert "destination" in dl["properties"], "download_paper must expose destination"
    assert "filename" in dl["properties"], "download_paper must expose filename"
    assert dl["properties"]["destination"].get("default") == "./data/downloads"
    assert "required" in dl and dl["required"] == ["arxiv_id"]

    pp = by_name["ingest_paper"].parameters
    assert "pdf_path" in pp["properties"], "ingest_paper must expose pdf_path"


def test_download_preflight_routes_topic_only_to_arxiv():
    topic = _download_preflight_hint([])
    assert "Do NOT call search_papers() or check_paper()" in topic
    assert "arxiv subagent directly" in topic

    named = _download_preflight_hint(["RMNet"])
    assert "check_paper" in named
    assert "state is 'absent'" in named


def test_failed_arxiv_then_local_search_skips_redundant_check(monkeypatch):
    import agent.nodes as nodes
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    captured = {}

    async def fake_llm(model, messages, *, emit_tokens=True, config=None):
        captured["messages"] = messages
        return AIMessage(content="already indexed")

    monkeypatch.setattr(nodes, "_get_bound_model", lambda *a, **k: object())
    monkeypatch.setattr(nodes, "_stream_llm", fake_llm)

    arxiv_failure = json.dumps({
        "schema_version": "1.0",
        "outcome": "failed",
        "error": "arXiv API HTTP error: 406 Not Acceptable",
        "error_type": "subagent",
        "code": "arxiv_api_unavailable",
        "next": "Fall back to the local library.",
    })
    search_success = json.dumps({
        "schema_version": "1.0",
        "outcome": "succeeded",
        "data": {
            "results": [{"paper": "llm_medqa"}],
            "papers": ["llm_medqa"],
        },
    })
    state = {
        "messages": [
            HumanMessage(content="帮我下载一篇agent方向的论文到本地"),
            AIMessage(content="", tool_calls=[{
                "name": "arxiv",
                "args": {"task": "find an agent paper"},
                "id": "call-arxiv",
            }]),
            ToolMessage(
                content=arxiv_failure,
                tool_call_id="call-arxiv",
                name="arxiv",
            ),
            AIMessage(content="", tool_calls=[{
                "name": "search_papers",
                "args": {"query": "agent"},
                "id": "call-search",
            }]),
            ToolMessage(
                content=search_success,
                tool_call_id="call-search",
                name="search_papers",
            ),
        ],
        "iteration": 2,
        "intent": "literature_search",
        "focus_papers": [],
    }

    result = asyncio.run(nodes.agent_node(state, {}))
    assert result["messages"][0].content == "already indexed"
    prompt_text = "\n".join(
        str(getattr(message, "content", ""))
        for message in captured["messages"]
    )
    assert "Do NOT call check_paper" in prompt_text
    assert "already local" in prompt_text


def test_download_records_phase_timings():
    import agent.providers.builtin_provider as bp
    from agent import workspace_config as wc

    class FakeResponse:
        status_code = 200
        content = b"%PDF-1.4 fake"

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            return FakeResponse()

    async def fake_title(_arxiv_id):
        return "RMNet: A Timed Download"

    original_title = bp._fetch_arxiv_title
    original_client = bp.httpx.AsyncClient
    bp._fetch_arxiv_title = fake_title
    bp.httpx.AsyncClient = lambda *args, **kwargs: FakeClient()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wc.set_override("project_root", Path(tmp))
            try:
                raw = asyncio.run(bp.download_paper.ainvoke({
                    "arxiv_id": "2301.00001v1",
                    "destination": ".",
                    "filename": "timed",
                }))
            finally:
                wc.clear_overrides()
    finally:
        bp._fetch_arxiv_title = original_title
        bp.httpx.AsyncClient = original_client

    payload = json.loads(raw)
    timings = payload["data"]["timings_ms"]
    assert set(timings) == {"metadata_ms", "download_ms", "write_ms", "total_ms"}
    assert all(value >= 0 for value in timings.values())


if __name__ == "__main__":
    test_sanitize_dl_name()
    test_short_name_from_title()
    test_resolve_stem_priority()
    test_tooldef_matches_function_signature()
    test_download_preflight_routes_topic_only_to_arxiv()
    test_failed_arxiv_then_local_search_skips_redundant_check()
    test_download_records_phase_timings()
    print("download/process decoupling self-check OK")
