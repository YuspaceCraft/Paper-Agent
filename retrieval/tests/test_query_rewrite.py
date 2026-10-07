from retrieval import RetrievalService
from retrieval.query_rewrite import fuse_query_variants, rewrite_query


def test_chinese_remote_sensing_query_is_translated():
    rewrite = rewrite_query("只给我遥感相关的内容")

    assert rewrite.primary == "remote sensing"
    assert rewrite.variants[0] == "只给我遥感相关的内容"
    assert "remote sensing" in rewrite.variants


def test_acronym_adds_expansion_without_losing_original():
    rewrite = rewrite_query("RSICC")

    assert rewrite.variants[0] == "RSICC"
    assert "remote sensing image change captioning" in rewrite.variants


def test_pronoun_uses_context_and_resolved_anchor():
    rewrite = rewrite_query(
        "它们的方法有什么区别",
        context="对比 Diffusion-RSCC 和 Cross-Temporal 两篇论文的方法",
        anchors=["Diffusion-RSCC", "Cross-Temporal"],
    )

    assert "Diffusion-RSCC" in rewrite.primary
    assert "Cross-Temporal" in rewrite.primary
    assert "method" in rewrite.primary


def test_variant_fusion_preserves_first_rank_and_best_raw_score():
    fused = fuse_query_variants(
        [
            [{"chunk_id": "a", "score": 0.2}, {"chunk_id": "b", "score": 0.1}],
            [{"chunk_id": "b", "score": 0.9}, {"chunk_id": "c", "score": 0.8}],
        ],
        top_k=3,
    )

    assert [item["chunk_id"] for item in fused] == ["b", "a", "c"]
    assert next(item for item in fused if item["chunk_id"] == "b")["raw_score"] == 0.9


def test_retrieval_service_runs_rewritten_variant():
    calls: list[str] = []

    class FakeSparse:
        def search(self, query: str, top_k: int):
            calls.append(query)
            if query == "remote sensing":
                return [{"chunk_id": "paper__chunk_0001", "score": 0.7}]
            return []

    service = RetrievalService(method="sparse", top_k=5, sparse=FakeSparse())
    hits = service.search("遥感", top_k=5)

    assert "remote sensing" in calls
    assert [hit["chunk_id"] for hit in hits] == ["paper__chunk_0001"]
