"""
main.py — FastAPI application entry point.

Run:
  uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload

Architecture:
  All business logic lives in pdf_pipeline/ and indexer/ modules.
  The API layer is a thin HTTP wrapper — it handles uploads, serialization,
  and task dispatch. No business logic lives here.
"""
from __future__ import annotations

# ponytail: load .env before any langchain imports — LangSmith reads
# LANGSMITH_TRACING_V2 at import time.
# main.py is at web/api/main.py → root is .parent.parent.parent (project root).
# ⚠️ 曾有 x4 的 off-by-one（指向 pre/.env 不存在）→ DASHSCOPE_API_KEY 在
# agent.graph 懒加载前拿不到，除 agent 外的 LLM 调用（notifier）报缺 key。
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import pdf, index, retrieval, reader, agent, workspace, background, creation, experiments, study, tasks, settings


# ---- startup warm-up ----

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "pdf_pipeline" / "output"
REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent.parent / "eval_output" / "paper_registry.json"


def _warmup_redis():
    """Restore Redis from JSON cold backups if Redis is empty.

    ponytail: runs synchronously at startup — cheap enough for < 100 papers.
    If Redis already has data, skips (assumes it's current).
    """
    from .routers import _get_redis

    r = _get_redis()
    if not r:
        return

    try:
        if r.dbsize() > 0:
            return  # Redis already populated
    except Exception:
        return

    restored = 0

    # Restore paper registry → dedup:* keys (only if output dir exists on disk)
    if REGISTRY_PATH.exists():
        try:
            reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            for name, meta in reg.get("papers", {}).items():
                if not (OUTPUT_DIR / name).is_dir():
                    continue  # ponytail: skip papers whose output was deleted
                r.set(f"dedup:paper:{name}", json.dumps(meta, ensure_ascii=False))
                r.sadd("dedup:papers", name)
                if h := meta.get("content_hash"):
                    r.set(f"dedup:hash:{h}", name)
                if d := meta.get("doi"):
                    r.set(f"dedup:doi:{d}", name)
                restored += 1
        except Exception:
            pass

    if restored > 0:
        print(f"[warmup] Restored {restored} papers to Redis from cold backup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup_redis()
    yield


app = FastAPI(
    title="Demo API",
    description="Research paper PDF processing, vector indexing & retrieval API",
    version="0.1.0",
    lifespan=lifespan,
)

# ponytail: permissive CORS for local dev; restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf.router)
app.include_router(index.router)
app.include_router(retrieval.router)
app.include_router(reader.router)
app.include_router(agent.router)
app.include_router(background.router)
app.include_router(workspace.router)
app.include_router(creation.router)
app.include_router(experiments.router)
app.include_router(study.router)
app.include_router(tasks.router)
app.include_router(settings.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
