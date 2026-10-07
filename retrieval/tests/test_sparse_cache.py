from __future__ import annotations

from retrieval.sparse import SparseRetriever


def _chunks():
    return [
        {
            "chunk_id": "attention__chunk_0001",
            "content": "Transformer attention mechanisms for language models.",
        },
        {
            "chunk_id": "vision__chunk_0001",
            "content": "Remote sensing image change detection and captioning.",
        },
    ]


def test_sparse_index_cache_round_trip(tmp_path):
    cache_path = tmp_path / "sparse_index.pkl"
    first = SparseRetriever()
    first.index(_chunks(), cache_path=cache_path, cache_key="same-content")
    first_hits = first.search("attention", top_k=2)

    cached = SparseRetriever()
    cached.index(_chunks(), cache_path=cache_path, cache_key="same-content")

    assert cache_path.exists()
    assert cached.search("attention", top_k=2) == first_hits


def test_sparse_index_cache_rebuilds_when_key_changes(tmp_path):
    cache_path = tmp_path / "sparse_index.pkl"
    original = SparseRetriever()
    original.index(_chunks(), cache_path=cache_path, cache_key="old-content")

    changed_chunks = [
        {
            "chunk_id": "diffusion__chunk_0001",
            "content": "Diffusion probabilistic models for image generation.",
        }
    ]
    changed = SparseRetriever()
    changed.index(changed_chunks, cache_path=cache_path, cache_key="new-content")

    hits = changed.search("diffusion", top_k=2)
    assert [hit["chunk_id"] for hit in hits] == ["diffusion__chunk_0001"]
