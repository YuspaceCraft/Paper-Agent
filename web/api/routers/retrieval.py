"""
retrieval.py — Retrieval Service API endpoints
================================================

POST   /api/retrieval/search    — 使用评估最优策略执行检索
GET    /api/retrieval/config     — 当前检索配置

ponytail: thin HTTP wrapper around retrieval.RetrievalService.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import SearchRequest, SearchResult, SearchResponse

router = APIRouter(prefix="/api/retrieval", tags=["Retrieval"])

# ponytail: lazy-init on first search
_service = None
_service_init_error: str | None = None


def _get_service():
    """Lazy-init RetrievalService from optimal config."""
    global _service, _service_init_error
    if _service is not None:
        return _service

    from pathlib import Path
    from retrieval import RetrievalService

    # Config paths relative to project root
    optimal = Path("./eval_output/optimal_retrieval_config.yaml")
    indexer = Path("./indexer/config.yaml")
    rag = Path("./eval_output/all_rag_chunks.json")

    if not optimal.exists():
        _service_init_error = f"optimal config not found: {optimal}. Run evaluation first."
        return None

    try:
        _service = RetrievalService.from_config(
            optimal_config=optimal,
            indexer_config=indexer,
            rag_chunks=rag if rag.exists() else None,
        )
        return _service
    except Exception as e:
        _service_init_error = str(e)
        return None


def invalidate_retrieval_service():
    """Invalidate cached RetrievalService so next request rebuilds from current data.

    Call after indexing completes — new vectors are in Qdrant and
    all_rag_chunks.json has been regenerated.
    """
    global _service, _service_init_error
    _service = None
    _service_init_error = None


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """检索接口 — 使用 offline eval 选出的最优策略。

    支持覆盖 top_k；filters 暂未启用（策略自带 metadata_filter）。
    """
    svc = _get_service()
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail={
                "ok": False,
                "error": "Retrieval service unavailable",
                "error_type": "service_unavailable",
                "detail": _service_init_error or "Service not initialized",
                "next": "Run evaluation first, then re-index. See: python -m retrieval_orchestrator evaluate",
            },
        )

    hits = svc.search(req.query, top_k=req.top_k or None)
    results = [
        SearchResult(
            chunk_id=h.get("chunk_id", ""),
            content_type=(h.get("metadata") or {}).get("content_type", ""),
            section_path=(h.get("metadata") or {}).get("section_path", ""),
            generation_text=h.get("document", ""),
            score=h.get("score", 0.0),
            metadata=h.get("metadata", {}),
        )
        for h in hits
    ]
    return SearchResponse(results=results, total=len(results))


@router.get("/config")
async def get_config():
    """返回当前检索服务使用的策略配置。"""
    svc = _get_service()
    if svc is None:
        return {
            "status": "unavailable",
            "error": _service_init_error or "Service not initialized",
        }
    return {
        "status": "ready",
        "method": svc.method,
        "top_k": svc.top_k,
    }
