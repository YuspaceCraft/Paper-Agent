"""M-单样例 self-check — flow.build_flow 阶段聚合（mock 事件，零成本）。

Run:  C:/Users/30811/miniconda3/envs/demo/python.exe evaluation/tests/test_flow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.flow import build_flow


def _ev(event_type, seq, node="", duration_ms=None, payload=None, tool=None,
        model=None, outcome=None, intent=None):
    return {
        "seq": seq, "event_type": event_type, "node": node,
        "duration_ms": duration_ms, "payload": payload or {},
        "tool": tool, "model": model, "outcome": outcome, "intent": intent,
    }


def _sample() -> list[dict]:
    """模拟 react 模式一轮完整链路（backend down 工具返回 err 信封）。"""
    return [
        _ev("turn_start", 1, "graph"),
        _ev("node_start", 2, "understand"),
        _ev("intent", 3, "understand",
            payload={"confidence": 0.9, "entities": ["RMNet"],
                     "focus_papers": [], "needs_planning": False, "domain": "paper"},
            intent="literature_search"),
        _ev("llm_call", 4, node="", model="qwen-plus",
            payload={"mode": "invoke", "tokens": {"prompt_tokens": 100,
                                                   "completion_tokens": 20,
                                                   "total_tokens": 120}}),
        _ev("node_end", 5, "understand", duration_ms=2100.0),
        _ev("node_start", 6, "agent"),
        _ev("llm_call", 7, node="", model="qwen-plus",
            payload={"mode": "stream", "tokens": {"prompt_tokens": 800,
                                                   "completion_tokens": 10,
                                                   "total_tokens": 810}}),
        _ev("tool_call", 8, "dispatcher", duration_ms=120.0, tool="search_papers",
            outcome="failed",
            payload={"args": '{"query": "RMNet loss"}', "result_summary": "..text..",
                     "parsed": {"is_envelope": True, "outcome": "failed",
                                "error_type": "backend_down", "error": "unavailable"}}),
        _ev("retrieved_context", 9, "dispatcher", tool="search_papers", outcome="succeeded",
            payload={"query": "RMNet loss", "chunk_ids": [], "snapshot": ""}),
        _ev("node_end", 10, "agent", duration_ms=1500.0),
        _ev("node_start", 11, "agent"),
        _ev("llm_call", 12, node="", model="qwen-plus",
            payload={"mode": "stream", "tokens": {"prompt_tokens": 500,
                                                   "completion_tokens": 40,
                                                   "total_tokens": 540}}),
        _ev("node_end", 13, "agent", duration_ms=900.0),
        _ev("final_answer", 14, "synthesize", payload={"answer": "RMNet 的 loss 是 MSE。"}),
        _ev("turn_end", 15, "graph", duration_ms=8000.0,
            payload={"status": "ok", "query": "RMNet 的 loss？"}),
    ]


def run_all() -> None:
    flow = build_flow(_sample(), query="RMNet 的 loss？")

    # query 透传
    assert flow["query"] == "RMNet 的 loss？", flow["query"]

    keys = [s["key"] for s in flow["stages"]]
    assert keys == ["intent", "tools", "answer"], keys

    # 阶段 token 归属
    intent_tok = flow["stages"][0]["tokens"]
    assert intent_tok["total"] == 120, intent_tok          # understand llm
    answer_tok = flow["stages"][2]["tokens"]
    assert answer_tok["total"] == 540, answer_tok          # 收尾 llm → 回答
    tools_tok = flow["stages"][1]["tokens"]
    assert tools_tok["total"] == 810, tools_tok            # react 环内推理 → 工具

    # 工具结果信封解析可见
    tool_item = next(it for s in flow["stages"] for it in s["items"]
                     if it.get("type") == "tool")
    assert tool_item["parsed"]["error_type"] == "backend_down", tool_item
    assert tool_item["outcome"] == "failed"

    # 总时长以 turn_end 时延为准
    assert flow["metrics"]["total_duration_ms"] == 8000.0

    # 汇总
    assert flow["metrics"]["llm_calls"] == 3
    assert flow["metrics"]["tool_calls"] == 1
    assert flow["metrics"]["tool_failures"] == 1
    assert flow["metrics"]["task_success"] is False
    assert flow["metrics"]["task_status"] == "degraded"
    assert flow["metrics"]["intent"] == "literature_search"
    assert flow["metrics"]["mode"] == ""
    assert [event["event_type"] for event in flow["events"]][:2] == [
        "turn_start", "node_start",
    ]

    # 阶段耗时来自 node 计时（understand 2.1s + agent 推理两段）
    assert flow["stages"][0]["duration_ms"] == 2100.0

    print("flow 阶段聚合 self-check OK")


if __name__ == "__main__":
    run_all()
