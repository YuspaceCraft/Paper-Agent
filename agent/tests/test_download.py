"""test_download.py — download/process decoupling + short-name naming self-check.

Covers the pure helpers behind download_paper (naming derivation, sanitize) and
the ToolDef/function signature contract. No network, no backend — assert-based.

Run: python agent/tests/test_download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.providers.builtin_provider import (
    _sanitize_dl_name,
    _short_name_from_title,
    _resolve_stem,
    BUILTIN_TOOLDEFS,
)


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


if __name__ == "__main__":
    test_sanitize_dl_name()
    test_short_name_from_title()
    test_resolve_stem_priority()
    test_tooldef_matches_function_signature()
    print("download/process decoupling self-check OK")