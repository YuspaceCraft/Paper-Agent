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
    raw = tc.ok({"papers": [{"arxiv_id": "2301.07093"}]})
    data = json.loads(raw)
    assert data == {"ok": True, "data": {"papers": [{"arxiv_id": "2301.07093"}]}}


def test_err_envelope_shape():
    raw = tc.err("param_error", "paper not found",
                 "Pick from available_papers.", available_papers=["RMNet"])
    data = json.loads(raw)
    assert data["ok"] is False
    assert data["error_type"] == "param_error"
    assert data["next"]
    assert data["available_papers"] == ["RMNet"]


# ---- P6: 解析 ----

def test_parse_envelope_success():
    r = tc.parse_tool_result('{"ok": true, "data": {"count": 3}}')
    assert r.is_envelope and r.ok
    assert r.data == {"count": 3}
    assert not r.text


def test_parse_envelope_error():
    r = tc.parse_tool_result(
        '{"ok": false, "error": "nope", "error_type": "not_found", '
        '"next": "try again", "available_papers": ["RMNet"]}'
    )
    assert r.is_envelope and not r.ok
    assert r.error == "nope"
    assert r.error_type == "not_found"
    assert r.next_action == "try again"
    assert r.extra["available_papers"] == ["RMNet"]


def test_parse_plain_text():
    r = tc.parse_tool_result("## Method (RMNet)\nbody text")
    assert not r.is_envelope and r.ok
    assert r.text.startswith("## Method")


def test_parse_json_without_ok_is_text():
    # 文件内容恰好是合法 JSON 但无 "ok" 键 → 仍按文本处理
    r = tc.parse_tool_result('{"foo": 1, "bar": 2}')
    assert not r.is_envelope and r.ok
    assert r.text == '{"foo": 1, "bar": 2}'


# ---- P6: 截断保持 envelope 可解析 ----

def test_truncate_envelope_stays_parseable():
    long_text = "x" * 300
    payload = {"ok": True, "data": {"paper_name": "RMNet", "chunks": [
        {"content": long_text}, {"content": long_text}, {"content": long_text},
    ]}}
    raw = json.dumps(payload, ensure_ascii=False)
    cut = tc.truncate_tool_result(raw, 600)
    assert len(cut) <= 600
    data = json.loads(cut)  # 必须仍可解析
    assert data["ok"] is True


def test_truncate_plain_text():
    long = "y" * 500
    cut = tc.truncate_tool_result(long, 200)
    assert len(cut) <= 200 + 64
    assert "truncated" in cut


def test_truncate_small_result_untouched():
    raw = '{"ok": true, "data": {"n": 1}}'
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