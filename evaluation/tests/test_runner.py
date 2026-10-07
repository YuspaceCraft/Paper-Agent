"""M3 self-check — runner 管线：mock agent_run 后冒烟评估（零 LLM 成本）。

Run:  C:/Users/30811/miniconda3/envs/demo/python.exe evaluation/tests/test_runner.py
覆盖：manifest 加载 → 逐条 trace 合成 → 检索/工具/任务指标 → 报告组装。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation import runner, datasets
from evaluation.config import EvalConfig
from evaluation.trace_store import TraceStore, get_trace_store
from evaluation.sink import set_eval_ctx, clear_eval_ctx
from evaluation.events import record_event
from agent.observability import set_trace_id_override, clear_trace_id_override


def _fake_agent_run(store, hits_for: dict):
    async def _run(query: str, thread_id: str, *, timeout: float = 900.0) -> dict:
        # 每模拟一步真实链路：intent → 检索 → 工具 → final_answer
        set_trace_id_override(f"run-test:{thread_id.split(':')[-1]}")
        set_eval_ctx(run_id="run-test", thread_id=thread_id, source="eval")
        # Same thread, different trace/run: must not leak into the current QA.
        record_event(
            "tool_call", node="dispatcher", tool="stale_tool",
            outcome="succeeded", duration_ms=1.0,
            trace_id=f"stale:{thread_id}", thread_id=thread_id,
            run_id="stale-run", payload={"parsed": {
                "is_envelope": True, "outcome": "succeeded",
            }},
        )
        record_event("turn_start", node="graph")
        record_event("intent", node="understand", intent="literature_search")
        hits = hits_for.get(query, [])
        record_event("retrieved_context", node="dispatcher", tool="search_papers",
                     query=query, chunk_ids=hits, snapshot="ctx (fake)")
        record_event("tool_call", node="dispatcher", tool="search_papers",
                     outcome="succeeded", duration_ms=8.0, args={"query": query})
        record_event("tool_call", node="dispatcher", tool="fetch_content",
                     outcome="succeeded", duration_ms=12.0, args={"paper_name": "x"})
        record_event("llm_call", node="synthesize", model="qwen-plus",
                     duration_ms=200.0, payload={"tokens": {"prompt_tokens": 300,
                                                            "completion_tokens": 120,
                                                            "total_tokens": 420}})
        record_event("final_answer", node="synthesize", answer="answer OK")
        record_event("turn_end", node="graph", status="ok")
        clear_trace_id_override()
        clear_eval_ctx()
        return {"messages": []}
    return _run


async def _go(tmpdir: str) -> None:
    store = TraceStore(Path(tmpdir) / "runner_test.db")
    store._enabled = True
    prev = get_trace_store()
    import evaluation.trace_store as ts
    ts._INSTANCES["$default"] = store

    manifest = Path(tmpdir) / "manifest.jsonl"
    manifest.write_text("\n".join([
        '{"id": "q1", "query": "dual stream feature", '
        '"ground_truth_ids": ["p1__chunk_0001", "p1__chunk_0002"], '
        '"difficulty_level": "medium", "generation_mode": "semantic"}',
        '{"id": "q2", "query": "loss function", '
        '"ground_truth_ids": ["p1__chunk_0003"], '
        '"difficulty_level": "easy", "generation_mode": "keyword"}',
        '{"id": "q3", "query": "Dual Stream Feature", '
        '"ground_truth_ids": ["p1__chunk_0004"], '
        '"difficulty_level": "hard", "generation_mode": "keyword"}',
    ]) + "\n", encoding="utf-8")

    qas = datasets.load_manifest(manifest)
    full = datasets.deduplicate(qas)          # 去重后 2 条
    assert len(full) == 2, len(full)

    orig = runner.agent_run
    runner.agent_run = _fake_agent_run(store, {
        "dual stream feature": ["p1__chunk_0001", "p1__chunk_0009"],
        "loss function": ["p1__chunk_0003"],
    })
    try:
        cfg = EvalConfig(max_queries=0, judge_sample_size=0)
        report = await runner.run_eval(cfg, full, run_id="run-test",
                                       with_judge=False)
    finally:
        runner.agent_run = orig

    # 检索指标：q1 第 1 条命中 → recall@2=0.5, mrr=1.0；q2 recall@1=1.0
    overall = report["overall"]
    assert overall["recall@5"] > 0, overall
    assert overall["mrr"] > 0, overall
    assert "confidence_intervals" in overall, overall
    assert report["query_count"] == 2
    # 工具指标 + 任务指标
    assert report["tool_metrics"]["overall"]["calls"] == 4
    assert report["tool_metrics"]["overall"]["success_rate"] == 1.0
    assert "stale_tool" not in report["tool_metrics"]["per_tool"]
    assert report["task_metrics"]["task_success_rate"] == 1.0
    assert report["task_metrics"]["contract_success_rate"] == 1.0
    assert isinstance(report["task_dimension"], list)
    assert all(
        row["value"] != "unknown"
        for row in report["dimension"]
    ), report["dimension"]
    # 成本估算（2 次 llm_call × 420 tokens）
    assert report["cost"]["tokens_total"] == 840
    # badcase 导出结构
    assert isinstance(report["badcases"], list)
    # eval_runs 落库
    import evaluation.trace_store as ts2
    assert store._conn is not None
    # metadata 指纹
    assert "model" in report["metadata"]
    # 报告 JSON 可落盘
    import json
    json.dumps(report, ensure_ascii=False)

    await store.close()
    await asyncio.sleep(0.2)  # Windows 文件锁：等 aiosqlite worker 线程退出
    ts._INSTANCES.pop("$default", None)
    if prev is not None:
        ts._INSTANCES["$default"] = prev


def run_all() -> None:
    import shutil
    tmp = tempfile.mkdtemp(prefix="eval_runner_test_")
    try:
        asyncio.run(_go(tmp))
    finally:
        for _ in range(5):
            try:
                shutil.rmtree(tmp)
                break
            except OSError:
                import time as _t
                _t.sleep(0.2)
    print("runner 管线 self-check OK")


if __name__ == "__main__":
    import os
    os.environ.setdefault("AGENT_TRACE_ENABLED", "1")
    run_all()
