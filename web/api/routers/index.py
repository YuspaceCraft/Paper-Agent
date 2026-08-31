"""
index.py — Indexer API endpoints.

POST   /api/index/run             — run indexing pipeline (background)
GET    /api/index/status/{task_id} — check task status
POST   /api/index/search          — vector search
GET    /api/index/stats           — collection stats
"""
from __future__ import annotations

import json
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from indexer import catalog

from ..schemas import (
    TaskStatus, IndexRunRequest, IndexRunResult,
    SearchRequest, SearchResponse, SearchResult, IndexStats,
)
from . import _task_create, _task_update, _task_get

router = APIRouter(prefix="/api/index", tags=["Indexer"])

# ponytail: lazy-init store, created on first use
_store = None
_config_path: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_store(config_path: str = ""):
    """Lazy-init the vector store from config."""
    global _store, _config_path
    if _store is not None and config_path == _config_path:
        return _store

    from indexer.config import load_config
    from indexer.vector_store import QdrantVectorStore, ChromaVectorStore

    cfg = load_config(config_path)
    vs = cfg.vector_store

    if vs.backend == "qdrant":
        _store = QdrantVectorStore(
            url=vs.qdrant.url, collection_name=vs.qdrant.collection_name,
        )
    else:
        _store = ChromaVectorStore(
            persist_dir=vs.chroma.persist_dir,
            collection_name=vs.chroma.collection_name,
        )
    _config_path = config_path
    return _store


def _run_indexing(task_id: str, rag_chunks_path: str, config_path: str,
                  meta: dict | None = None):
    """Background: run full indexing pipeline, then merge chunks + refresh caches.

    meta：原子入库路径传入的注册 payload（{metadata, page_count, chunk_count}，
    来自 _run_pipeline）。有 meta 时收尾用 register_indexed() 一次写注册+置真，
    无 meta（/api/index/run 直接跑）回退 mark_indexed() 最小注册兼容。
    """
    try:
        _task_update(task_id, status="running", progress="Loading config...")

        from indexer.pipeline import IndexerPipeline, merge_all_chunks

        pipeline = IndexerPipeline(config_path)
        stats = pipeline.run(rag_chunks_path)

        # ponytail: regenerate all_rag_chunks.json + invalidate retrieval cache
        _task_update(task_id, progress="Merging all chunks...")
        merged = merge_all_chunks()
        stats["merged_chunks"] = merged

        from .retrieval import invalidate_retrieval_service
        invalidate_retrieval_service()

        # ponytail: keep Redis catalog in sync with the vector KB — a paper is
        # "indexed" only once its chunks are in Qdrant. 原子入库路径（带 meta）
        # 注册+置 indexed 一次写入，无中间态；直接 /run 走最小注册兼容。
        paper_name = Path(rag_chunks_path).parent.name
        if meta:
            catalog.register_indexed(
                paper_name,
                metadata=meta.get("metadata") or {},
                page_count=int(meta.get("page_count") or 0),
                chunk_count=int(meta.get("chunk_count") or stats.get("total_chunks") or 0),
                indexed_chunk_count=int(stats.get("total_chunks") or 0),
            )
        else:
            catalog.mark_indexed(paper_name, stats["total_chunks"])

        _task_update(task_id, status="done", progress="Complete",
                     result=json.dumps(stats, ensure_ascii=False))

    except Exception as exc:
        _task_update(task_id, status="failed", error=str(exc))


def _run_reconcile(task_id: str, config_path: str):
    """Background: 目录 indexed 标志 ↔ 向量库实际 chunk 数对账回填。"""
    try:
        _task_update(task_id, status="running", progress="Reconciling catalog ↔ vector store...")
        from indexer.reconcile import reconcile
        report = reconcile(config_path)
        _task_update(task_id, status="done", progress="Complete",
                     result=json.dumps(report, ensure_ascii=False))
    except Exception as exc:
        _task_update(task_id, status="failed", error=str(exc))


def submit_indexing(rag_chunks_path: str, config_path: str = "", parent: str = "") -> IndexRunResult:
    """Run the indexing pipeline on a rag_chunks.json file (background).

    Shared by the /run endpoint and the composite ingest orchestrator
    (routers/background.py). `parent` marks the created task as an internal
    sub-task of a composite ingest task — hidden from /api/agent/tasks.
    Raises HTTPException(404) when the rag_chunks path is missing.
    """
    from pathlib import Path

    if not Path(rag_chunks_path).exists():
        raise HTTPException(404, f"File not found: {rag_chunks_path}")

    task_id = uuid.uuid4().hex[:12]
    # kind/paper_name 仅供任务中心展示（rag_chunks 所在目录即论文名）
    _task_create(task_id, kind="index", paper_name=Path(rag_chunks_path).parent.name,
                 parent=parent)

    thread = threading.Thread(
        target=_run_indexing,
        args=(task_id, rag_chunks_path, config_path),
        daemon=True,
    )
    thread.start()

    return IndexRunResult(task_id=task_id, status="pending")


@router.post("/run", response_model=IndexRunResult)
async def run_index(req: IndexRunRequest):
    """Run the indexing pipeline on a rag_chunks.json file (background)."""
    return submit_indexing(req.rag_chunks_path, req.config_path)


@router.get("/status/{task_id}", response_model=TaskStatus)
async def task_status(task_id: str):
    """Get the status of a background indexing task."""
    task = _task_get(task_id)
    if task is None:
        raise HTTPException(404, f"Task {task_id} not found")
    return TaskStatus(**task)


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Vector search over indexed chunks."""
    store = _get_store()

    from indexer.config import load_config
    from indexer.embedding_adapters import create_embedding_adapter

    cfg = load_config()
    adapter = create_embedding_adapter(cfg.embedding)
    query_vector = adapter.embed_single(req.query)

    docs = store.search(query_vector, filters=req.filters, top_k=req.top_k)

    results = [
        SearchResult(
            chunk_id=d["chunk_id"],
            content_type=d.get("metadata", {}).get("content_type", ""),
            section_path=d.get("metadata", {}).get("section_path", ""),
            generation_text=d.get("document", "")[:500],
            score=d.get("score", 0.0),
            metadata=d.get("metadata", {}),
        )
        for d in docs
    ]
    return SearchResponse(results=results, total=len(results))


@router.get("/stats", response_model=IndexStats)
async def stats():
    """Get vector store statistics."""
    from indexer.config import load_config

    cfg = load_config()
    vs = cfg.vector_store
    store = _get_store()

    return IndexStats(
        backend=vs.backend,
        collection_name=vs.qdrant.collection_name if vs.backend == "qdrant" else vs.chroma.collection_name,
        count=store.count(),
    )


@router.post("/reconcile", response_model=IndexRunResult)
async def reconcile(config_path: str = ""):
    """Background: 对账目录 `indexed` 标志与向量库实际 chunk 数（回填修复）。

    滚动向量库按 chunk_id 前缀统计各论文实际点数，与目录里 indexed 标志对齐：
    - 向量库有点但标志缺失/未置 → 回填 indexed=true（旧代码 desync 修复，如 Mask）；
    - 标志置真但向量库为空 → 回退 indexed=false。
    结果（含 fixed 明细）在 task result 上。
    """
    task_id = uuid.uuid4().hex[:12]
    _task_create(task_id, kind="reconcile")
    thread = threading.Thread(
        target=_run_reconcile, args=(task_id, config_path), daemon=True,
    )
    thread.start()
    return IndexRunResult(task_id=task_id, status="pending")
