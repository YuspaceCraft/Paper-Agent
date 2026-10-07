"""指标口径 self-check —— 工具成功率 / 任务成功率 / badcase 归因（零成本）。

Run:  C:/Users/30811/miniconda3/envs/demo/python.exe evaluation/tests/test_metrics.py

覆盖三个历史缺陷：
1. 工具返回失败 envelope（HTTP 200 + outcome=failed）时成功率仍为 1；
2. 纯文本工具的降级文案（"Timeout: ..."）被当成成功；
3. 任务「有兜底回答但工具全挂 / 执行超时」被算成成功，且不进 badcase。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics.retrieval import aggregate, classify_badcase, per_query_metrics
from evaluation.metrics.task import task_success
from evaluation.metrics.tools import aggregate_tools, call_error
from evaluation.report import cache_metrics, estimate_cost, timing_stats


def _tool(outcome="succeeded", tool="search_papers", duration_ms=10.0, parsed=None,
          result_summary="", error=""):
    return {
        "event_type": "tool_call", "node": "dispatcher", "tool": tool,
        "outcome": outcome, "duration_ms": duration_ms, "error": error or None,
        "payload": {"parsed": parsed or {}, "result_summary": result_summary,
                    "error": error or None},
    }


def _answer_event(answer="answer OK", mode=""):
    return {"event_type": "final_answer", "node": "synthesize",
            "payload": {"answer": answer, "mode": mode}}


def test_envelope_failure_counts_as_tool_failure():
    """库工具 HTTP 200 但信封 ok=false → 失败（历史缺陷：成功率恒为 1）。"""
    events = [
            _tool(outcome="failed", tool="search_papers",
                  parsed={"is_envelope": True, "outcome": "failed",
                          "error_type": "backend_down"}),
            _tool(outcome="succeeded", tool="ingest_paper",
                  parsed={"is_envelope": True, "outcome": "succeeded"}),
    ]
    agg = aggregate_tools(events)
    assert agg["overall"]["calls"] == 2, agg
    assert agg["overall"]["succeeded"] == 1, agg
    assert agg["overall"]["success_rate"] == 0.5, agg
    assert agg["per_tool"]["search_papers"]["success_rate"] == 0.0, agg
    assert agg["per_tool"]["search_papers"]["error_types"] == {
        "backend_down": 1}, agg
    # badcase 归因带上信封 error_type
    assert agg["badcases"][0]["error"] == "backend_down", agg["badcases"]


def test_text_failure_counts_as_tool_failure():
    """纯文本工具的降级文案（非 envelope）也按失败计。"""
    assert call_error(_tool(result_summary="Timeout: PDF conversion exceeded 120 secs")) == "timeout"
    assert call_error(_tool(result_summary="Error: paper not found")) == "error"
    assert call_error(_tool(result_summary="## RMNET: results")) == ""
    agg = aggregate_tools([
        _tool(result_summary="Timeout: PDF conversion exceeded 120 secs"),
        _tool(result_summary="Found 3 total results."),
    ])
    assert agg["overall"]["success_rate"] == 0.5, agg
    assert agg["overall"]["errors"] == 1, agg


def test_duplicate_hits_do_not_inflate_retrieval_metrics():
    qm = per_query_metrics(
        ["c1", "c1", "c2", "c2", "c3"],
        ["c1", "c2", "c3"],
    )
    assert qm["recall@5"] == 1.0, qm
    assert qm["precision@5"] == 0.6, qm
    assert qm["ndcg@5"] == 1.0, qm
    assert qm["mrr"] == 1.0, qm

    empty = aggregate([])
    assert empty["query_count"] == 0, empty
    assert empty["overall"]["recall@5"] == 0.0, empty
    assert empty["overall"]["mrr"] == 0.0, empty


def test_tool_failure_dominates_conflicting_success_outcome():
    event = _tool(
        outcome="succeeded",
        parsed={"is_envelope": True, "outcome": "failed",
                "error_type": "backend_down"},
    )
    assert call_error(event) == "backend_down"


def test_transport_failure_and_no_calls():
    """网关层失败（超时/熔断）照旧算失败；完全没有工具调用时率是 0（形状稳定）。"""
    agg = aggregate_tools([_tool(outcome="timed_out", error="TOOL_TIMEOUT",
                                     parsed={"error_type": "tool_timeout"})])
    assert agg["overall"]["success_rate"] == 0.0, agg
    empty = aggregate_tools([])
    assert empty["overall"] == {"calls": 0, "succeeded": 0, "success_rate": 0.0,
                                "p50_ms": 0.0, "p95_ms": 0.0, "errors": 0}, empty


def test_task_success_requires_clean_run():
    """有回答 ≠ 任务达成：工具失败 / run_error 都不算成功。"""
    ok_events = [_tool(parsed={"is_envelope": True, "outcome": "succeeded"}),
                 _answer_event()]
    good = task_success(ok_events)
    assert good["success"] is True and good["status"] == "ok", good

    degraded = task_success([
            _tool(outcome="failed",
                  parsed={"is_envelope": True, "outcome": "failed",
                          "error_type": "backend_down"}),
        _answer_event("工具暂时不可用，以下是降级回答。")])
    assert degraded["success"] is False, degraded
    assert degraded["status"] == "degraded", degraded
    assert degraded["tool_failures"] == 1, degraded
    assert "failed" in degraded["task_error"], degraded

    timed_out = task_success([_tool(), _answer_event()], run_error="turn_timeout")
    assert timed_out["success"] is False, timed_out
    assert timed_out["task_error"] == "turn_timeout", timed_out

    no_answer = task_success([_tool()])
    assert no_answer["success"] is False and no_answer["status"] == "failed"


def test_task_success_plan_mode_needs_verification():
    def _events(vstatus):
        return [_tool(), _answer_event(mode="plan"),
                {"event_type": "plan_verify", "node": "verify",
                 "payload": {"verification": {"status": vstatus}}}]

    assert task_success(_events("satisfied"))["success"] is True
    partial = task_success(_events("partial"))
    assert partial["success"] is False and partial["status"] == "degraded"


def test_task_success_uses_latest_answer_and_verification():
    events = [
        _answer_event("stale answer", mode="plan"),
        {"event_type": "plan_verify", "node": "verify",
         "payload": {"verification": {"status": "partial"}}},
        _answer_event("final answer", mode="plan"),
        {"event_type": "plan_verify", "node": "verify",
         "payload": {"verification": {"status": "satisfied"}}},
    ]
    result = task_success(events)
    assert result["success"] is True, result
    assert result["answer_length"] == len("final answer"), result


def test_badcase_classification_covers_task_level_failures():
    base = {"answer": "一个回答", "recall@5": 0.5, "mrr": 1.0}
    assert classify_badcase({**base, "run_error": "turn_timeout"}) == "run_error"
    assert classify_badcase({**base, "answer": "  "}) == "no_answer"
    assert classify_badcase({**base, "tool_errors": 2, "success": False}) == "tool_fail"
    assert classify_badcase({**base, "success": False}) == "task_fail"
    assert classify_badcase({**base, "recall@5": 0.0}) == "retrieval_fail"
    assert classify_badcase({**base, "mrr": 0.0, "recall@5": 0.5}) == "low_mrr"
    assert classify_badcase({**base, "success": True}) == "ok"


def test_timing_stats_are_wall_clock_samples():
    """耗时统计来自逐条 QA 的墙钟样本，不是链路各步相加。"""
    stats = timing_stats([12.0, 8.0, 30.0, 10.0])
    assert stats["p50_s"] == 12.0, stats
    assert stats["max_s"] == 30.0, stats
    assert stats["total_s"] == 60.0, stats
    assert timing_stats([])["p50_s"] is None


def test_cost_and_cache_dashboard_dimensions():
    llm_events = [
        {"event_type": "llm_call", "node": "agent", "model": "main",
         "payload": {"tokens": {"prompt_tokens": 100, "completion_tokens": 50}}},
        {"event_type": "llm_call", "node": "router", "model": "small",
         "payload": {"tokens": {"prompt_tokens": 20, "completion_tokens": 10}}},
    ]
    cost = estimate_cost(llm_events, {
        "main": (1.0, 2.0),
        "small": (0.1, 0.2),
        "default": (1.0, 2.0),
    })
    assert cost["model_calls"] == {"main": 1, "small": 1}
    assert set(cost["per_node"]) == {"agent", "router"}
    assert cost["tokens_total"] == 180

    cache = cache_metrics([
        {"event_type": "tool_call", "cache_hit": True},
        {"event_type": "tool_call", "cache_hit": False},
    ])
    assert cache == {"calls": 2, "hits": 1, "hit_rate": 0.5}


def test_judge_samples_badcases_first(monkeypatch):
    from evaluation import runner

    captured = {}

    def fake_judge(samples, **_kwargs):
        captured["samples"] = samples
        return {"sample_size": len(samples)}

    class Store:
        async def get_trace(self, _trace_id):
            return []

    class Cfg:
        judge_budget_usd = 0.1
        judge_sample_size = 2
        judge_model = "judge-test"

    monkeypatch.setattr(runner.judge_mod, "judge_answer_vs_context", fake_judge)
    rows = [
        {"query_id": "ok-1", "query": "ok-1", "category": "ok"},
        {"query_id": "bad-1", "query": "bad-1", "category": "tool_fail"},
        {"query_id": "ok-2", "query": "ok-2", "category": "ok"},
        {"query_id": "bad-2", "query": "bad-2", "category": "task_fail"},
    ]
    result = __import__("asyncio").run(runner._run_judge(Cfg(), Store(), rows))

    assert result["sample_size"] == 2
    assert [s["query"] for s in captured["samples"]] == ["bad-1", "bad-2"]


def test_legacy_baseline_is_not_used(monkeypatch, tmp_path):
    from evaluation import report

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"dataset": [{"run_id": "legacy", "overall": {"recall@5": 1.2}}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "BASELINE_PATH", baseline)

    delta = report.compare_baseline(
        "dataset", {"recall@5": 0.5, "mrr": 0.5}, "new-run")
    assert delta["because_best"] is None, delta
    assert delta["deltas"] == {}, delta
    assert delta["gate"] == "block", delta


def run_all() -> None:
    test_envelope_failure_counts_as_tool_failure()
    test_text_failure_counts_as_tool_failure()
    test_duplicate_hits_do_not_inflate_retrieval_metrics()
    test_tool_failure_dominates_conflicting_success_outcome()
    test_transport_failure_and_no_calls()
    test_task_success_requires_clean_run()
    test_task_success_plan_mode_needs_verification()
    test_task_success_uses_latest_answer_and_verification()
    test_badcase_classification_covers_task_level_failures()
    test_timing_stats_are_wall_clock_samples()
    test_cost_and_cache_dashboard_dimensions()
    print("指标口径 self-check OK")


if __name__ == "__main__":
    run_all()
