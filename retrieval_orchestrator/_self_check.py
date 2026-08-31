"""
Self-check: verify evaluator and config_selector logic with synthetic data.
Run: python -m retrieval_orchestrator._self_check
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _test_evaluator():
    """Verify metrics computation."""
    from retrieval_orchestrator.evaluator import (
        _recall, _precision, _mrr, _ndcg, _hit_rate,
    )

    gt = ["c1", "c2", "c3"]

    # Perfect retrieval
    hits = ["c1", "c2", "c3", "c4", "c5"]
    assert _recall(hits, gt, 5) == 1.0, f"recall@5 should be 1.0, got {_recall(hits, gt, 5)}"
    assert _recall(hits, gt, 3) == 1.0
    assert _precision(hits, gt, 3) == 1.0
    assert _mrr(hits, gt) == 1.0
    assert _ndcg(hits, gt, 3) == 1.0
    assert _hit_rate(hits, gt, 3) == 1.0

    # Partial retrieval
    hits2 = ["c1", "c4", "c5", "c6", "c7"]
    assert _recall(hits2, gt, 3) == 1/3
    assert _precision(hits2, gt, 3) == 1/3
    assert _mrr(hits2, gt) == 1.0  # c1 is first

    # Zero recall
    hits3 = ["c9", "c10"]
    assert _recall(hits3, gt, 3) == 0.0
    assert _mrr(hits3, gt) == 0.0
    assert _hit_rate(hits3, gt, 3) == 0.0

    print("  [OK] evaluator metrics correct")


def _test_config_selector():
    """Verify filtering and ranking logic."""
    from retrieval_orchestrator.config_selector import select_optimal, QualityGateError

    experiments = [
        {
            "experiment_id": "dense_k20",
            "config": {"method": "dense", "top_k": 20},
            "evaluation_report": {"overall_metrics": {
                "recall@5": 0.92, "recall@10": 0.96, "mrr": 0.85, "ndcg@10": 0.88,
            }},
        },
        {
            "experiment_id": "dense_k10",
            "config": {"method": "dense", "top_k": 10},
            "evaluation_report": {"overall_metrics": {
                "recall@5": 0.88, "recall@10": 0.91, "mrr": 0.82, "ndcg@10": 0.84,
            }},
        },
        {
            "experiment_id": "dense_k5",
            "config": {"method": "dense", "top_k": 5},
            "evaluation_report": {"overall_metrics": {
                "recall@5": 0.72, "recall@10": 0.72, "mrr": 0.65, "ndcg@10": 0.68,
            }},
        },
        {
            "experiment_id": "sparse_k10",
            "config": {"method": "sparse", "top_k": 10},
            "evaluation_report": {"overall_metrics": {
                "recall@5": 0.55, "recall@10": 0.65, "mrr": 0.55, "ndcg@10": 0.58,
            }},
        },
    ]

    # Quality gate: checked on best AFTER filtering. sparse_k10 filtered out
    # (r10=0.65 < 0.70), dense_k5 wins with r5=0.72 >= 0.6 — no gate error.
    best = select_optimal(experiments, thresholds={"recall@10": 0.70, "mrr": 0.60})
    assert best is not None, "Should find a best config"
    assert best["experiment_id"] == "dense_k5", f"Expected dense_k5, got {best['experiment_id']}"

    # Quality gate: when best config has r5 < 0.6 → block
    all_low = [
        {"experiment_id": "bad_k5", "config": {"method": "dense", "top_k": 5},
         "evaluation_report": {"overall_metrics": {"recall@5": 0.45, "recall@10": 0.60, "mrr": 0.50}}},
    ]
    try:
        select_optimal(all_low, thresholds={"recall@10": 0.55, "mrr": 0.45})
        assert False, "Should have raised"
    except QualityGateError as e:
        assert "bad_k5" in str(e)
        print("  [OK] quality gate blocks when best r5 < 0.6")

    print("  [OK] config_selector ranking correct")


def _test_eval_dataset():
    """Verify keyword extraction and difficulty assignment."""
    from retrieval_orchestrator.eval_dataset import (
        _extract_keywords, _chunk_difficulty, generate_keyword_queries,
    )

    # Keyword extraction
    content = "[KEYWORDS: rmnet, parameter count, efficiency]\n## Section\nSome text..."
    kws = _extract_keywords(content)
    assert kws == ["rmnet", "parameter count", "efficiency"], f"Got {kws}"

    # No keywords
    assert _extract_keywords("No keywords here") == []

    # Difficulty
    assert _chunk_difficulty({"content_type": "body", "token_count": 250}) == "easy"
    assert _chunk_difficulty({"content_type": "body", "token_count": 500}) == "medium"
    assert _chunk_difficulty({"content_type": "body", "token_count": 900}) == "hard"
    assert _chunk_difficulty({"content_type": "formula", "token_count": 100}) == "hard"

    # Generate from synthetic chunks
    chunks = [
        {"chunk_id": "c0", "content_type": "body", "token_count": 500,
         "content": "[KEYWORDS: a, b, c]\n## Title\nText...",
         "section_path": "Intro", "metadata": {}},
        {"chunk_id": "c1", "content_type": "reference", "token_count": 100,
         "content": "[1] Author...",
         "section_path": "Refs", "metadata": {}},
        {"chunk_id": "c2", "content_type": "formula", "token_count": 50,
         "content": "[KEYWORDS: x, y]\nFormula...",
         "section_path": "Methods", "metadata": {}},
    ]
    qa = generate_keyword_queries(chunks, sample_size=10)
    assert len(qa) == 2, f"c0 and c2 (body+formula with keywords) should generate, got {len(qa)}"
    assert qa[0]["query"] == "a b c"
    assert qa[0]["ground_truth_ids"] == ["c0"]
    assert qa[0]["generation_mode"] == "keyword"

    print("  [OK] eval_dataset correct")


def _test_rrf_and_hybrid():
    """Verify RRF fusion logic."""
    from retrieval.fusion import rrf_fuse, weighted_fuse

    dense = [
        {"chunk_id": "c1", "score": 0.95},
        {"chunk_id": "c2", "score": 0.80},
        {"chunk_id": "c3", "score": 0.60},
    ]
    sparse = [
        {"chunk_id": "c2", "score": 0.90},
        {"chunk_id": "c3", "score": 0.85},
        {"chunk_id": "c4", "score": 0.70},
    ]

    rrf = rrf_fuse(dense, sparse, top_k=3)
    # c2 appears in both lists → should rank high
    assert rrf[0]["chunk_id"] == "c2", f"RRF: expected c2 first, got {rrf[0]['chunk_id']}"

    weighted = weighted_fuse(dense, sparse, dense_weight=0.5, top_k=3)
    assert len(weighted) == 3

    print("  [OK] RRF and hybrid fusion correct")


if __name__ == "__main__":
    print("retrieval_orchestrator self-check\n")
    try:
        _test_evaluator()
        _test_config_selector()
        _test_eval_dataset()
        _test_rrf_and_hybrid()
        print("\n[PASS] All checks passed")
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        raise
