"""
test_info_flow.py — INFO_FLOW_REVIEW 落地回归测试（P1 / P6）。

P6：统一工具输出信封契约（tool_contract.py）——生成/解析/截断单点权威。
P1：legacy [FINAL_ANSWER] marker 的兜底过滤（nodes.py 正则助手）。
"""

import json

from agent import tool_contract as tc
from agent.nodes import (
    _FINAL_ANSWER_RE, _FINAL_ANSWER_LINE_RE,
    _may_be_marker_prefix, _strip_lead_marker,
)


# ---- P6: envelope 生成 ----

def test_ok_envelope_shape():
    raw = tc.success({"papers": [{"arxiv_id": "2301.07093"}]})
    data = json.loads(raw)
    assert data["schema_version"] == "1.0"
    assert data["outcome"] == "succeeded"
    assert "ok" not in data
    assert data["data"] == {"papers": [{"arxiv_id": "2301.07093"}]}


def test_err_envelope_shape():
    raw = tc.failure("param_error", "paper not found",
                     "Pick from available_papers.", available_papers=["RMNet"])
    data = json.loads(raw)
    assert data["outcome"] == "failed"
    assert "ok" not in data
    assert data["error_type"] == "param_error"
    assert data["next"]
    assert data["available_papers"] == ["RMNet"]


# ---- P6: 解析 ----

def test_parse_envelope_success():
    r = tc.parse_tool_result('{"outcome": "succeeded", "data": {"count": 3}}')
    assert r.is_envelope and r.outcome == "succeeded"
    assert r.data == {"count": 3}
    assert not r.text


def test_parse_envelope_error():
    r = tc.parse_tool_result(
        '{"outcome": "failed", "error": "nope", "error_type": "not_found", '
        '"next": "try again", "available_papers": ["RMNet"]}'
    )
    assert r.is_envelope and r.outcome == "failed"
    assert r.error == "nope"
    assert r.error_type == "not_found"
    assert r.next_action == "try again"
    assert r.extra["available_papers"] == ["RMNet"]


def test_parse_plain_text():
    r = tc.parse_tool_result("## Method (RMNet)\nbody text")
    assert not r.is_envelope and r.outcome == "succeeded"
    assert r.text.startswith("## Method")


def test_parse_json_without_ok_is_text():
    # 文件内容恰好是合法 JSON 但无顶层 outcome → 仍按文本处理
    r = tc.parse_tool_result('{"foo": 1, "bar": 2}')
    assert not r.is_envelope and r.outcome == "succeeded"
    assert r.text == '{"foo": 1, "bar": 2}'


def test_parse_nested_ok_json_is_still_text():
    raw = '{"foo": {"ok": false}}'
    r = tc.parse_tool_result(raw)
    assert not r.is_envelope and r.outcome == "succeeded"
    assert r.text == raw


def test_parse_rejects_legacy_ok_envelope():
    r = tc.parse_tool_result('{"ok": "false", "error": "bad"}')
    assert r.is_envelope and r.outcome == "failed"
    assert r.code == "TOOL_PROTOCOL_INVALID"
    assert "legacy" in r.protocol_error


def test_parse_rejects_malformed_envelope_instead_of_success_text():
    r = tc.parse_tool_result('{"ok": false')
    assert r.is_envelope and r.outcome == "failed"
    assert r.code == "TOOL_PROTOCOL_INVALID"


# ---- P6: 截断保持 envelope 可解析 ----

def test_truncate_envelope_stays_parseable():
    long_text = "x" * 300
    payload = {"outcome": "succeeded", "data": {"paper_name": "RMNet", "chunks": [
        {"content": long_text}, {"content": long_text}, {"content": long_text},
    ]}}
    raw = json.dumps(payload, ensure_ascii=False)
    cut = tc.truncate_tool_result(raw, 600)
    assert len(cut) <= 600
    data = json.loads(cut)  # 必须仍可解析
    assert data["outcome"] == "succeeded"
    assert data["meta"]["truncated"] is True
    assert data["data"]["preview"]
    assert data["continuation"]["shown_chars"] > 0


def test_large_string_envelope_keeps_content_and_continuation():
    """Regression: oversized fetch_content envelopes must not become empty."""
    body = "".join(f"line {i}: evidence\n" for i in range(1200))
    raw = json.dumps({
        "schema_version": "1.0",
        "outcome": "succeeded",
        "operation_id": "fetch-1",
        "kind": "tool",
        "data": body,
        "meta": {"tool_name": "fetch_content"},
    }, ensure_ascii=False)

    attached = tc.attach_artifact(raw, {"artifact_id": "b" * 32})
    cut = tc.truncate_tool_result(attached, 8000)
    data = json.loads(cut)
    assert len(cut) <= 8000
    assert data["outcome"] == "succeeded"
    assert data["data"].startswith("line 0: evidence")
    assert data["artifacts"][0]["artifact_id"] == "b" * 32
    assert data["continuation"]["next_offset"] == len(data["data"])
    assert data["continuation"]["artifact_id"] == "b" * 32
    assert data["continuation"]["payload_chars"] == len(body)
    assert "envelope dropped" not in cut


def test_truncate_plain_text():
    long = "y" * 500
    cut = tc.truncate_tool_result(long, 200)
    assert len(cut) <= 200
    assert "truncated" in cut


def test_content_paging_returns_explicit_next_offset():
    text = "".join(f"{i:05d}\n" for i in range(2500))
    raw = json.dumps({
        "schema_version": "1.0",
        "outcome": "succeeded",
        "data": text,
    })
    attached = tc.attach_artifact(raw, {"artifact_id": "a" * 32})
    first = json.loads(tc.paginate_tool_result(
        attached, offset=0, max_chars=6000,
    ))
    assert first["data"].startswith("00000")
    assert first["continuation"]["next_offset"] == 6000
    assert first["artifacts"][0]["artifact_id"] == "a" * 32

    second = json.loads(tc.paginate_tool_result(
        attached, offset=6000, max_chars=6000,
    ))
    assert second["data"]
    assert second["continuation"]["next_offset"] == 12000


def test_truncate_small_result_untouched():
    raw = '{"outcome": "succeeded", "data": {"n": 1}}'
    assert tc.truncate_tool_result(raw, 8000) == raw


# ---- P1: marker 兜底过滤 ----

def test_marker_regex_variants():
    for marker in ("[FINAL_ANSWER]", "【FINAL_ANSWER】", "[final_answer]",
                   "[ Final Answer ]", "[FINAL-ANSWER]"):
        assert _FINAL_ANSWER_RE.search(marker), marker
        assert _FINAL_ANSWER_LINE_RE.search(marker + "\n"), marker


def test_strip_lead_marker_with_preceding_newline():
    # 复现文档缺陷: 首 chunk 为 "\n" 时旧逻辑失效
    in_ = "\n[FINAL_ANSWER]\nanswer here"
    out = _strip_lead_marker(in_)
    assert out == "answer here"


def test_strip_lead_marker_full_width_and_colon():
    out = _strip_lead_marker("【FINAL_ANSWER】:body")
    assert out == "body"


def test_may_be_marker_prefix():
    assert _may_be_marker_prefix("")
    assert _may_be_marker_prefix("  \n[")
    assert _may_be_marker_prefix("\n[fI")
    assert not _may_be_marker_prefix("  [1] citation")
    assert not _may_be_marker_prefix("normal text")
