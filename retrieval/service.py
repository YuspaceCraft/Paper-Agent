"""
service.py — 检索服务层
========================

RetrievalService: 一站式检索，从 optimal_retrieval_config.yaml 加载最优策略，
组合 VectorStore + Embedding + SparseRetriever + Fusion 完成检索。

DenseRetriever: 向量检索的轻量封装（供 fusion 和独立使用）。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .query_rewrite import fuse_query_variants, rewrite_query


class DenseRetriever:
    """Wraps indexer's VectorStoreAdapter + EmbeddingAdapter."""

    def __init__(self, store, embedder):
        self._store = store
        self._embedder = embedder

    def search(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[dict]:
        vec = self._embedder.embed_single(query)
        if vec is None:
            return []
        return self._store.search(vec, filters=filters, top_k=top_k)


class RetrievalService:
    """Production retrieval service backed by evaluation-optimized config.

    Usage:
        svc = RetrievalService.from_config(
            optimal_config="eval_output/optimal_retrieval_config.yaml",
            indexer_config="indexer/config.yaml",
            rag_chunks="eval_output/all_rag_chunks.json",
        )
        results = svc.search("dual stream feature extraction", top_k=5)
    """

    def __init__(
        self,
        method: str,
        top_k: int,
        dense: DenseRetriever | None = None,
        sparse=None,  # SparseRetriever, lazy import to avoid sklearn dep in API
        hybrid_mode: str = "rrf",
        dense_weight: float = 0.7,
        metadata_filter: dict | None = None,
        sparse_top_k_override: int | None = None,
        chunks_map: dict[str, dict] | None = None,
    ):
        self._method = method
        self._top_k = top_k
        self._dense = dense
        self._sparse = sparse
        self._hybrid_mode = hybrid_mode
        self._dense_weight = dense_weight
        self._metadata_filter = metadata_filter
        self._sparse_top_k_override = sparse_top_k_override or 50  # for fusion recall
        self._chunks_map = chunks_map or {}

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Execute retrieval with the configured strategy.

        Returns:
            List of {chunk_id, score, document, metadata, ...} dicts.
        """
        k = top_k or self._top_k
        rewrite = rewrite_query(query)
        variants = list(rewrite.variants) or [query]
        result_lists = [
            self._search_once(variant, top_k=max(k, self._sparse_top_k_override)
                              if len(variants) > 1 else k)
            for variant in variants
        ]
        hits = fuse_query_variants(result_lists, top_k=k)

        # ponytail: augment sparse results with document/metadata from chunk map
        for h in hits:
            cid = h.get("chunk_id", "")
            if not h.get("document") and cid in self._chunks_map:
                cm = self._chunks_map[cid]
                h.setdefault("document", cm.get("content", ""))
                metadata = dict(cm.get("metadata") or {})
                for key in ("content_type", "section_path", "token_count"):
                    metadata.setdefault(key, cm.get(key, ""))
                h.setdefault("metadata", metadata)

        return hits

    def _search_once(self, query: str, *, top_k: int) -> list[dict]:
        """Run the configured strategy for one query variant."""

        if self._method == "dense":
            hits = self._dense.search(query, top_k=top_k, filters=self._metadata_filter)

        elif self._method == "sparse":
            if self._sparse is None:
                return []
            hits = self._sparse.search(query, top_k=top_k)

        elif self._method == "hybrid":
            from .fusion import rrf_fuse, weighted_fuse

            d_hits = self._dense.search(
                query,
                top_k=max(top_k, self._sparse_top_k_override),
                filters=self._metadata_filter,
            )
            s_hits = (
                self._sparse.search(query, top_k=max(top_k, self._sparse_top_k_override))
                if self._sparse else []
            )

            if self._hybrid_mode == "weighted":
                hits = weighted_fuse(
                    d_hits, s_hits,
                    dense_weight=self._dense_weight,
                    top_k=top_k,
                )
            else:
                hits = rrf_fuse(d_hits, s_hits, top_k=top_k)
        else:
            return []

        return hits

    @classmethod
    def from_config(
        cls,
        optimal_config: str | Path,
        indexer_config: str | Path = "./indexer/config.yaml",
        rag_chunks: str | Path | None = None,
    ) -> "RetrievalService":
        """Factory: create RetrievalService from optimal_retrieval_config.yaml.

        Args:
            optimal_config: Path to optimal_retrieval_config.yaml (from eval).
            indexer_config: Path to indexer config (for vector store + embedding).
            rag_chunks: Path to rag_chunks.json (for sparse index). If None, tries
                        eval_output/all_rag_chunks.json.

        Returns:
            Configured RetrievalService ready for search().
        """
        import yaml
        from indexer.config import load_config as load_index_cfg
        from indexer.vector_store import ChromaVectorStore, QdrantVectorStore
        from indexer.embedding_adapters import create_embedding_adapter
        from .sparse import SparseRetriever

        # Load optimal strategy
        opt = yaml.safe_load(Path(optimal_config).read_text(encoding="utf-8")) or {}
        strat = opt.get("config", {})
        method = strat.get("method", "sparse")
        top_k = strat.get("top_k", 5)
        hybrid_mode = strat.get("hybrid_mode", "rrf")
        dense_weight = strat.get("dense_weight", 0.7)
        metadata_filter = strat.get("metadata_filter") or None

        # Wire indexer (ponytail: skip dense init for sparse-only, no Qdrant needed)
        dense = None
        if method in ("dense", "hybrid"):
            idx_cfg = load_index_cfg(str(indexer_config))
            vs = idx_cfg.vector_store
            if vs.backend == "qdrant":
                store = QdrantVectorStore(url=vs.qdrant.url, collection_name=vs.qdrant.collection_name)
            else:
                store = ChromaVectorStore(persist_dir=vs.chroma.persist_dir, collection_name=vs.chroma.collection_name)
            embedder = create_embedding_adapter(idx_cfg.embedding)
            dense = DenseRetriever(store, embedder)

        # Wire sparse (if chunks provided)
        sparse = None
        chunks_map: dict[str, dict] = {}
        if rag_chunks:
            raw = Path(rag_chunks).read_bytes()
            chunks = json.loads(raw).get("chunks", [])
            chunks_map = {c["chunk_id"]: c for c in chunks}
            sparse = SparseRetriever()
            cache_path = None
            cache_key = ""
            if os.getenv("RETRIEVAL_SPARSE_CACHE", "1") != "0":
                cache_root = Path(
                    os.getenv(
                        "RETRIEVAL_SPARSE_CACHE_DIR",
                        str(Path(__file__).resolve().parents[1] / ".cache" / "retrieval"),
                    )
                )
                cache_path = cache_root / "sparse_index.pkl"
                cache_key = hashlib.sha256(raw).hexdigest()
            sparse.index(
                chunks,
                cache_path=cache_path,
                cache_key=cache_key,
            )

        return cls(
            method=method,
            top_k=top_k,
            dense=dense,
            sparse=sparse,
            hybrid_mode=hybrid_mode,
            dense_weight=dense_weight,
            metadata_filter=metadata_filter,
            chunks_map=chunks_map,
        )

    @property
    def method(self) -> str:
        return self._method

    @property
    def top_k(self) -> int:
        return self._top_k
