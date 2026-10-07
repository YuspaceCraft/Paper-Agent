"""Self-check for the balanced v3 agent task benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

from evaluation.datasets import (
    load_manifest,
    retrieval_enabled,
    validate_ground_truth_coverage,
)
from evaluation.metrics.contracts import evaluate_task_contract
from evaluation.metrics.retrieval import aggregate
from evaluation.metrics.task import aggregate_task_metrics
from evaluation.task_dataset import (
    DIFFICULTIES,
    TASK_LENGTHS,
    TASK_TYPES,
    build_cases,
    validate_cases,
    write_manifest,
)


def test_v3_matrix_is_balanced_and_references_current_corpus(tmp_path):
    cases = build_cases()
    summary = validate_cases(cases)
    assert summary["count"] == 36
    assert summary["task_type_mix"] == {kind: 9 for kind in TASK_TYPES}
    assert summary["difficulty_mix"] == {level: 12 for level in DIFFICULTIES}
    assert summary["task_length_mix"] == {length: 12 for length in TASK_LENGTHS}
    assert summary["retrieval_cases"] == 18
    assert summary["non_retrieval_cases"] == 18

    for row in cases:
        required = set((row.get("expected") or {}).get("required_tools") or [])
        forbidden = set((row.get("expected") or {}).get("forbidden_tools") or [])
        assert not (required & forbidden), row["id"]

    output = tmp_path / "v3.jsonl"
    result = write_manifest(output)
    assert result["count"] == 36
    rows = load_manifest(output)
    assert len(rows) == 36
    assert sum(1 for row in rows if retrieval_enabled(row)) == 18


def test_coverage_ignores_non_retrieval_tasks(tmp_path):
    rag = tmp_path / "chunks.json"
    rag.write_text(json.dumps({
        "chunks": [{"chunk_id": "paper__chunk_0001"}],
    }), encoding="utf-8")
    rows = [
        {
            "id": "rag",
            "ground_truth_ids": ["paper__chunk_0001"],
            "evaluation": {"retrieval": True},
        },
        {
            "id": "download",
            "ground_truth_ids": [],
            "evaluation": {"retrieval": False},
        },
    ]
    report = validate_ground_truth_coverage(rows, rag_path=rag)
    assert report["retrieval_query_count"] == 1
    assert report["non_retrieval_query_count"] == 1
    assert report["coverage"] == 1.0
    assert report["stale_query_ids"] == []


def test_retrieval_aggregate_skips_non_retrieval_tasks():
    rows = [
        {
            "query_id": "rag",
            "retrieval_enabled": True,
            "recall@5": 1.0,
            "precision@5": 0.2,
            "ndcg@5": 1.0,
            "hit@5": 1.0,
            "mrr": 1.0,
            "difficulty_level": "easy",
            "content_type": "body",
            "generation_mode": "keyword",
        },
        {
            "query_id": "download",
            "retrieval_enabled": False,
        },
    ]
    result = aggregate(rows, k_values=[5])
    assert result["query_count"] == 1
    assert result["skipped_query_count"] == 1
    assert result["overall"]["recall@5"] == 1.0


def test_task_contract_checks_tools_args_artifacts_and_budget(tmp_path):
    started = time.time() - 1
    artifact = tmp_path / "paper.pdf"
    artifact.write_bytes(b"%PDF-1.4 fake")
    qa = {
        "expected": {
            "required_tools": ["download_paper"],
            "forbidden_tools": ["ingest_paper"],
            "tool_args": [{
                "tool": "download_paper",
                "arg": "arxiv_id",
                "equals": "1706.03762",
            }],
            "answer_contains": ["paper.pdf"],
            "artifacts": [{
                "path": str(artifact),
                "min_size_bytes": 10,
                "modified_after_query_start": True,
            }],
        },
        "budgets": {
            "timeout_s": 30,
            "max_tool_calls": 2,
            "max_tokens": 1000,
        },
    }
    events = [{
        "event_type": "tool_call",
        "tool": "download_paper",
        "payload": {"args": {"arxiv_id": "1706.03762"}},
    }]
    result = evaluate_task_contract(
        qa,
        events,
        observed={
            "answer": "saved paper.pdf",
            "duration_s": 4.0,
            "tokens_total": 100,
        },
        query_started_at=started,
        project_root=tmp_path,
    )
    assert result["contract_success"] is True, result

    failed = evaluate_task_contract(
        qa,
        [
            *events,
            {"event_type": "tool_call", "tool": "ingest_paper",
             "payload": {"args": {"paper_name": "p"}}},
        ],
        observed={
            "answer": "wrong answer",
            "duration_s": 40.0,
            "tokens_total": 2000,
        },
        query_started_at=started,
        project_root=tmp_path,
    )
    assert failed["contract_success"] is False
    assert {item["name"] for item in failed["contract_failures"]} >= {
        "forbidden_tool:ingest_paper",
        "answer_contains:paper.pdf",
        "budget:duration_s",
        "budget:tokens_total",
    }


def test_task_metrics_break_down_by_type_difficulty_and_length():
    rows = [
        {
            "query_id": "a",
            "task_type": "rag_retrieval",
            "difficulty_level": "easy",
            "task_length": "short",
            "generation_mode": "keyword",
            "success": True,
            "status": "ok",
            "duration_s": 5.0,
            "tokens_total": 100,
            "tool_calls": 1,
            "tool_failures": 0,
            "contract_enabled": True,
            "contract_success": True,
        },
        {
            "query_id": "b",
            "task_type": "paper_download",
            "difficulty_level": "hard",
            "task_length": "long",
            "generation_mode": "handcrafted_task",
            "success": False,
            "status": "degraded",
            "duration_s": 20.0,
            "tokens_total": 400,
            "tool_calls": 2,
            "tool_failures": 1,
            "contract_enabled": True,
            "contract_success": False,
        },
    ]
    result = aggregate_task_metrics(rows)
    assert result["task_success_rate"] == 0.5
    assert result["contract_success_rate"] == 0.5
    assert result["tokens_total"] == 500
    assert {row["dimension"] for row in result["dimension"]} == {
        "task_type", "difficulty_level", "task_length", "generation_mode",
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_v3_matrix_is_balanced_and_references_current_corpus(Path(tmp))
    test_coverage_ignores_non_retrieval_tasks(Path(tempfile.mkdtemp()))
    test_retrieval_aggregate_skips_non_retrieval_tasks()
    test_task_contract_checks_tools_args_artifacts_and_budget(
        Path(tempfile.mkdtemp()))
    test_task_metrics_break_down_by_type_difficulty_and_length()
    print("task dataset self-check OK")
