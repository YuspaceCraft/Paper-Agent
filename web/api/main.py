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

import asyncio
import ipaddress
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import (
    pdf,
    index,
    retrieval,
    reader,
    agent,
    workspace,
    background,
    creation,
    experiments,
    study,
    tasks,
    settings,
    config,
    eval,
    memory,
)


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
    if os.getenv("RETRIEVAL_WARMUP", "1") != "0":
        try:
            await asyncio.to_thread(retrieval.warmup_service)
        except Exception as exc:  # noqa: BLE001 - warmup must not block startup
            print(f"[retrieval] warmup skipped: {type(exc).__name__}: {exc}")
    try:
        yield
    finally:
        from agent.supervisor import shutdown_supervisor
        from agent.tools import close_tools

        await shutdown_supervisor()
        await close_tools()


app = FastAPI(
    title="Demo API",
    description="Research paper PDF processing, vector indexing & retrieval API",
    version="0.1.0",
    lifespan=lifespan,
)

def _cors_origins() -> list[str]:
    raw = (os.getenv("DEMO_CORS_ORIGINS", "") or "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    # Packaged Electron renderer uses a file:// origin, represented as "null".
    return ["http://localhost:5173", "http://127.0.0.1:5173", "null"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def local_api_only(request, call_next):
    """Keep command/file-management APIs local unless explicitly exposed."""
    path = request.url.path
    api_token = (os.getenv("DEMO_API_TOKEN", "") or "").strip()
    if (
        api_token
        and request.method != "OPTIONS"
        and path.startswith("/api/")
        and path != "/api/health"
    ):
        presented = (
            request.headers.get("x-demo-token", "")
            or request.query_params.get("token", "")
        )
        if not presented or not secrets.compare_digest(presented, api_token):
            return JSONResponse(
                {"detail": "invalid API token"},
                status_code=401,
            )

    allow_remote = (
        os.getenv("DEMO_ALLOW_REMOTE_API", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not allow_remote:
        host = request.client.host if request.client else ""
        try:
            is_loopback = ipaddress.ip_address(
                host.split("%", 1)[0]
            ).is_loopback
        except ValueError:
            is_loopback = host in {"localhost", "testclient"}
        if not is_loopback:
            return JSONResponse(
                {"detail": "remote API access is disabled"},
                status_code=403,
            )
    return await call_next(request)


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
app.include_router(config.router)
app.include_router(eval.router)
app.include_router(memory.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
