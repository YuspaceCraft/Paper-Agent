import json

from evaluation.datasets import validate_ground_truth_coverage


def test_validate_ground_truth_coverage_detects_stale_manifest(tmp_path):
    rag = tmp_path / "chunks.json"
    rag.write_text(
        json.dumps({"chunks": [{"chunk_id": "current__chunk_0001"}]}),
        encoding="utf-8",
    )
    qas = [
        {"id": "ok", "ground_truth_ids": ["current__chunk_0001"]},
        {"id": "stale", "ground_truth_ids": ["old__chunk_0002"]},
    ]

    report = validate_ground_truth_coverage(qas, rag_path=rag)

    assert report["coverage"] == 0.5
    assert report["missing_count"] == 1
    assert report["stale_query_ids"] == ["stale"]
