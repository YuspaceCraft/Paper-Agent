"""Tri-state local check self-check — canonicalize + match_local_state + ToolDef.

Run: python agent/tests/test_check_paper.py
ponytail: assert-based, no framework, no LLM/backend calls (matches are pure).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.resolution import canonicalize, match_local_state  # noqa: E402


def _snapshot(*papers):
    return list(papers)


def _paper(name, state="raw", location="downloads", pdf_path="", has_pdf=True):
    # detail 与生产保持一致（reader.py 派生：indexed/parsed/raw/""）
    detail = {"indexed": "indexed", "parsed": "parsed", "raw": "raw"}.get(state, "")
    return {"paper_name": name, "state": state, "detail": detail,
            "location": location, "pdf_path": pdf_path, "has_pdf": has_pdf}


def test_canonicalize_strips_noise():
    # CJK + brackets stripped; spaces/dashes/underscores collapse to nothing
    assert canonicalize("RMNet") == "rmnet"
    assert canonicalize("RM-Net") == "rmnet"
    assert canonicalize("RM_Net") == "rmnet"
    assert canonicalize("Diffusion-RSCC (模型)") == "diffusionrscc"
    assert canonicalize("") == ""


def test_canonicalize_bridges_disk_cleaning_pipelines():
    # pdf.py strips brackets/CJK: _sanitize_dl_name keeps them; and the output dir
    # is truncated at ~80 chars. canonicalize must make the shared alias contained:
    # "Diffusion-RSCC" + "(模型)" → "diffusionrscc" in BOTH raw_stem and dir_name.
    raw_stem = "Diffusion-RSCC_ Diffusion Probabilistic Model for Change Captioning in Remote Sensing Images"
    output_dir = "Diffusion-RSCC_Diffusion_Probabilistic_Model_for_Change_Captioning_in_Remote"
    for side in (raw_stem, output_dir):
        assert "diffusionrscc" in canonicalize(side), side
    assert canonicalize("Diffusion-RSCC (模型)") == "diffusionrscc"


def test_exact_indexed():
    snapshot = _snapshot(_paper("RMNet", state="indexed", location="catalog"))
    out = match_local_state("RMNet", snapshot)
    assert out["state"] == "indexed"
    assert out["matches"][0]["paper_name"] == "RMNet"
    assert out["matches"][0]["has_pdf"] is True


def test_containment_raw():
    snapshot = _snapshot(
        _paper("SRN: Stability Representation Network", state="raw",
               pdf_path="data/downloads/SRN.pdf"),
    )
    out = match_local_state("SRN", snapshot)
    assert out["state"] == "downloaded_not_indexed"
    assert out["matches"][0]["pdf_path"] == "data/downloads/SRN.pdf"


def test_indexed_beats_local_file():
    # Same term matches both an indexed paper and a raw file → indexed wins.
    snapshot = _snapshot(
        _paper("RMNet", state="raw", location="downloads"),
        _paper("RMNet", state="indexed", location="catalog"),
    )
    out = match_local_state("RMNet", snapshot)
    assert out["state"] == "indexed"
    assert len(out["matches"]) == 2
    assert out["matches"][0]["state"] == "indexed", "indexed ranked first"


def test_absent():
    snapshot = _snapshot(_paper("SRN", state="raw"))
    out = match_local_state("RMNet", snapshot)
    assert out["state"] == "absent"
    assert out["matches"] == []


def test_absent_on_empty_snapshot_or_term():
    assert match_local_state("X", [])["state"] == "absent"
    assert match_local_state("", [])["state"] == "absent"


def test_short_term_no_noisy_containment():
    # "cv" must NOT match "change captioning" via containment (≥4 chars guard)
    snapshot = _snapshot(_paper("Change Captioning Model", state="raw"))
    out = match_local_state("cv", snapshot)
    assert out["state"] == "absent"


def test_parsed_entry_counts_as_downloaded_not_indexed():
    snapshot = _snapshot(_paper("RSCC", state="parsed", location="output", has_pdf=False))
    out = match_local_state("RSCC", snapshot)
    assert out["state"] == "downloaded_not_indexed"
    assert out["matches"][0]["has_pdf"] is False


def test_tooldef_contract():
    from agent.providers.builtin_provider import BUILTIN_TOOLDEFS
    by_name = {t.name: t for t in BUILTIN_TOOLDEFS}
    td = by_name.get("check_paper")
    assert td is not None, "check_paper ToolDef must exist"
    props = td.parameters["properties"]
    assert "term" in props and props["term"].get("default") == ""
    # readOnly 是 safety 权限门的前置条件：非 readOnly 会被当 destructive 直接 403
    assert td.annotations.get("readOnlyHint") is True
    assert td.annotations.get("idempotentHint") is True


def test_check_paper_function_signature():
    from agent.providers.builtin_provider import check_paper, BuiltinProvider
    p = BuiltinProvider()
    assert "check_paper" in p._func_map
    assert list(check_paper.args) == ["term"], "term is the only kwarg"


if __name__ == "__main__":
    test_canonicalize_strips_noise()
    test_canonicalize_bridges_disk_cleaning_pipelines()
    test_exact_indexed()
    test_containment_raw()
    test_indexed_beats_local_file()
    test_absent()
    test_absent_on_empty_snapshot_or_term()
    test_short_term_no_noisy_containment()
    test_parsed_entry_counts_as_downloaded_not_indexed()
    test_tooldef_contract()
    test_check_paper_function_signature()
    print("check_paper tri-state self-check OK")