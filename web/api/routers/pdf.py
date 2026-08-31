"""
pdf.py — PDF Pipeline API endpoints.

POST   /api/pdf/process          — upload PDF, run full pipeline (background)
GET    /api/pdf/status/{task_id} — check task status
GET    /api/pdf/outputs          — list processed papers
GET    /api/pdf/outputs/{name}   — get paper output details
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
import shutil
import threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from pydantic import BaseModel

from ..schemas import TaskStatus, PDFProcessResult, PaperOutput
from . import _task_create, _task_update, _task_get
from indexer import catalog


class ProcessLocalRequest(BaseModel):
    paper_name: str
    pdf_path: str = ""  # 可选：工作区路径（download_paper 下载到自定义目录时使用）


router = APIRouter(prefix="/api/pdf", tags=["PDF Pipeline"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"
DOWNLOAD_DIR = _PROJECT_ROOT / "data" / "downloads"
OUTPUT_DIR = _PROJECT_ROOT / "pdf_pipeline" / "output"


# ponytail: Windows MAX_PATH=260, long filenames → truncated duplicates.
# Cap at 80 chars, strip chars that break paths.
_MAX_NAME_LEN = 80
_NAME_BLACKLIST = re.compile(r'[<>:"/\\|?*()（）一-鿿㐀-䶿]')


def _sanitize_paper_name(filename: str) -> str:
    """Derive a filesystem-safe paper name from an uploaded filename.

    Rules: replace spaces → underscore, strip CJK + special chars,
    collapse multi-underscore, truncate to _MAX_NAME_LEN.
    Falls back to a random name if sanitization yields empty.
    """
    name = Path(filename).stem
    name = name.replace(' ', '_')
    name = _NAME_BLACKLIST.sub('', name)
    name = re.sub(r'_+', '_', name).strip('_.')
    if len(name) > _MAX_NAME_LEN:
        # Try to cut at last underscore within limit
        cut = name[:_MAX_NAME_LEN]
        last_us = cut.rfind('_')
        name = cut[:last_us] if last_us > _MAX_NAME_LEN // 2 else cut
    if not name:
        name = f"paper_{uuid.uuid4().hex[:8]}"
    return name


def _list_output_files(paper_name: str) -> list[str]:
    """List files in a paper's output directory (filesystem-scan only)."""
    d = OUTPUT_DIR / paper_name
    if not d.is_dir():
        return []
    return [p.name for p in d.iterdir() if p.is_file()]


def _count_chunks(paper_name: str) -> int:
    """Count chunks from rag_chunks.json, 0 if missing/corrupt."""
    rag = OUTPUT_DIR / paper_name / "rag_chunks.json"
    if not rag.exists():
        return 0
    try:
        return len(json.loads(rag.read_text(encoding="utf-8")).get("chunks", []))
    except Exception:
        return 0


def _derive_status(indexed: bool, paper_name: str, pdf_exists: bool = False) -> tuple[str, str]:
    """返回 (status, detail)。status 只两类：indexed / not_indexed。

    detail 是文件系统现场派生的诊断（非持久状态）：
      "indexed"（已入库）/ "parsed"（有解析产物，可一键入库）/ "raw"（仅有本地 PDF）/ ""。
    """
    if indexed:
        return "indexed", "indexed"
    if bool(_list_output_files(paper_name)):
        return "not_indexed", "parsed"
    if pdf_exists or (UPLOAD_DIR / f"{paper_name}.pdf").exists() or (DOWNLOAD_DIR / f"{paper_name}.pdf").exists():
        return "not_indexed", "raw"
    return "not_indexed", ""


# ponytail: paper catalog (Redis + JSON cold backup) lives in indexer.catalog.
# This router calls catalog.* — no registry logic in the API layer.


# ================================================================
# Pipeline runner
# ================================================================

def _run_pipeline(task_id: str, pdf_path: Path, paper_name: str, register: bool = True):
    """Background: parse → enhance → enrich → rag-chunk.

    register=True（默认，/api/pdf/process 兼容流）→ 结束后写目录，indexed=false
    （仅解析产物）。register=False（原子入库路径，background._run_ingest）→ 不写
    目录，由入库收尾 register_indexed() 一步置真——避免中间态/失败残留误导性
    「已解析未入库」终态。返回 result_data（含 metadata / page_count / chunk_count，
    供入库收尾复用注册 payload）；失败返回 None。
    """
    try:
        _task_update(task_id, status="running", progress="Parsing PDF...")

        from pdf_pipeline.parser import parse_pdf_docling
        from pdf_pipeline.bindings import load_bindings_json, export_bindings_json
        from pdf_pipeline.enhancer import enhance_all
        from pdf_pipeline.enricher import enrich_markdown
        from pdf_pipeline.rag_chunker import (
            rag_chunk_markdown, RAGChunkConfig,
            export_rag_report, render_rag_html,
        )

        output_dir = OUTPUT_DIR / paper_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1-2: Parse + Bindings
        result = parse_pdf_docling(str(pdf_path), export_bindings=True)
        bindings = load_bindings_json(result.bindings_path)

        page_map = None
        if result.page_map_path and Path(result.page_map_path).exists():
            page_map = json.loads(Path(result.page_map_path).read_text(encoding="utf-8"))

        # Stage 3: Enhance
        _task_update(task_id, progress="Enhancing formulas & images...")
        enhance_all(bindings, result.markdown, str(output_dir))
        export_bindings_json(bindings, result.bindings_path)

        # Stage 4: Enrich — 显式落到 output_dir。
        # 注意：enricher 默认写到 bindings["paper"]["assets_dir"]（build_bindings 按 PDF
        # stem 推导，含点号文件名如 arXiv ID "2003.12462v2" → "2003_12462v2"，与 paper_name
        # 目录名不一致），此前直接从 output_dir 读 final_enriched.md 必然 FileNotFoundError。
        _task_update(task_id, progress="Enriching Markdown...")
        raw_md = output_dir / "raw.md"
        if not raw_md.exists():
            raw_md.write_text(result.markdown, encoding="utf-8")
        enriched_path = output_dir / "final_enriched.md"
        enriched_md = enrich_markdown(
            raw_md.read_text(encoding="utf-8"), bindings,
            output_path=str(enriched_path),
        )

        # Stage 5: RAG Chunk（enriched_md 已由 enrich 返回，与文件内容一致）
        _task_update(task_id, progress="RAG chunking...")
        config = RAGChunkConfig()
        report = rag_chunk_markdown(
            enriched_md, bindings=bindings, page_map=page_map, config=config,
        )
        export_rag_report(report, str(output_dir / "rag_chunks.json"))
        html = render_rag_html(report, title=paper_name)
        (output_dir / "rag_chunks.html").write_text(html, encoding="utf-8")

        _meta = dict(result.metadata) if isinstance(result.metadata, dict) else {}
        result_data = {
            "paper_name": paper_name, "output_dir": str(output_dir),
            "chunk_count": report.total_chunks,
            "content_hash": result.content_hash,
            "doi": result.metadata.get("doi", ""),
            "files": [p.name for p in output_dir.iterdir() if p.is_file()],
            "metadata": _meta,
            "page_count": result.page_count,
        }
        # register=False（复合入库的解析阶段）不能置 "done"——否则任务在向量化
        # 之前闪现「完成」，check_task_status 会被误读为已入库，造成前后矛盾。
        _task_update(task_id, status=("running" if not register else "done"),
                     progress=("Parse complete" if not register else "Complete"),
                     result=json.dumps(result_data, ensure_ascii=False))

        if register:
            try:
                catalog.register_paper(
                    paper_name=paper_name, metadata=_meta,
                    page_count=result.page_count, chunk_count=report.total_chunks,
                )

                # Keep the sparse retrieval index in sync. merge_all_chunks aggregates
                # every paper's rag_chunks.json into all_rag_chunks.json, which
                # RetrievalService (search_papers) reads. Without this, papers parsed
                # here are invisible to search until a separate /api/index/run fires.
                from indexer.pipeline import merge_all_chunks
                merge_all_chunks()
                from .retrieval import invalidate_retrieval_service
                invalidate_retrieval_service()
            except Exception as exc:
                print(f"  [REGISTER] parse-flow registration skipped (non-fatal): {exc}")
        return result_data

    except Exception as exc:
        _task_update(task_id, status="failed", error=str(exc))
        return None


# ================================================================
# Endpoints
# ================================================================

@router.post("/process", response_model=PDFProcessResult)
async def process_pdf(file: UploadFile = File(...)):
    """Upload a PDF and run the full processing pipeline in the background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    paper_name = _sanitize_paper_name(file.filename)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = UPLOAD_DIR / f"{paper_name}.pdf"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    sha = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    content_hash = sha.hexdigest()

    existing = catalog.is_duplicate(content_hash)
    if existing:
        pdf_path.unlink(missing_ok=True)
        return PDFProcessResult(
            task_id="dup_" + content_hash[:12],
            paper_name=existing.get("title", paper_name) or paper_name,
            status="duplicate",
        )

    task_id = uuid.uuid4().hex[:12]
    _task_create(task_id, paper_name=paper_name, kind="pdf")

    thread = threading.Thread(
        target=_run_pipeline, args=(task_id, pdf_path, paper_name), daemon=True
    )
    thread.start()

    return PDFProcessResult(task_id=task_id, paper_name=paper_name, status="pending")


def _locate_and_dedup(paper_name: str, pdf_path: str = ""):
    """Locate a PDF on disk + compute its content hash + check duplicates.

    Locates in this order: 1) pdf_path (workspace-relative/absolute, resolved
    against the workspace boundary), 2) data/uploads/{paper_name}.pdf,
    3) data/downloads/{paper_name}.pdf.

    Returns (pdf: Path, paper_name: str, content_hash: str, existing: dict | None).
    Raises HTTPException(403/404) for invalid input — shared by submit_pipeline
    and the unified ingest orchestrator (routers/background.py).
    """
    from agent.providers.generic_provider import resolve_workspace_path

    paper_name = _sanitize_paper_name(paper_name)

    if pdf_path:
        try:
            pdf = resolve_workspace_path(pdf_path)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise HTTPException(404,
                f"PDF not found at workspace path '{pdf_path}'. "
                f"Use the path returned by download_paper.")
    else:
        pdf = UPLOAD_DIR / f"{paper_name}.pdf"
        if not pdf.exists():
            pdf = DOWNLOAD_DIR / f"{paper_name}.pdf"
        if not pdf.exists():
            raise HTTPException(404, f"PDF not found for '{paper_name}'. Use download_paper first, "
                                    f"or upload via /api/pdf/process.")

    sha = hashlib.sha256()
    with open(pdf, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    content_hash = sha.hexdigest()

    existing = catalog.is_duplicate(content_hash)
    return pdf, paper_name, content_hash, existing


def submit_pipeline(paper_name: str, pdf_path: str = "", parent: str = "") -> PDFProcessResult:
    """Start the full parse pipeline on a PDF on disk, in a background thread.

    Standalone endpoint path (/process-local), keeping the parse task separate
    from the agent's unified ingest (which drives _run_pipeline directly on its
    own task). `parent` (optional) marks the task as a sub-task of a composite —
    hidden from /api/agent/tasks top-level listing.
    """
    pdf, paper_name, content_hash, existing = _locate_and_dedup(paper_name, pdf_path)
    if existing:
        return PDFProcessResult(
            task_id="dup_" + content_hash[:12],
            paper_name=existing.get("title", paper_name) or paper_name,
            status="duplicate",
        )

    task_id = uuid.uuid4().hex[:12]
    _task_create(task_id, paper_name=paper_name, kind="pdf", parent=parent)

    thread = threading.Thread(
        target=_run_pipeline, args=(task_id, pdf, paper_name), daemon=True
    )
    thread.start()

    return PDFProcessResult(task_id=task_id, paper_name=paper_name, status="pending")


@router.post("/process-local", response_model=PDFProcessResult)
async def process_local(req: ProcessLocalRequest):
    """Process a PDF already on disk (downloaded by the agent or uploaded via Web UI).

    Locates the PDF in this order:
      1. req.pdf_path (workspace-relative or absolute) when provided — this is the
         path download_paper returns when the file was saved to a custom folder.
      2. data/uploads/{paper_name}.pdf
      3. data/downloads/{paper_name}.pdf
    Then runs the full pipeline. Use this instead of /process when the file is on disk.
    """
    return submit_pipeline(req.paper_name, req.pdf_path)


@router.get("/status/{task_id}", response_model=TaskStatus)
async def task_status(task_id: str):
    """Get the status of a background pipeline task."""
    task = _task_get(task_id)
    if task is None:
        raise HTTPException(404, f"Task {task_id} not found")
    return TaskStatus(**task)


@router.get("/outputs", response_model=list[PaperOutput])
async def list_outputs():
    """List all processed paper outputs + raw downloaded PDFs.

    Primary source: Redis catalog (indexer.catalog) — metadata-rich.
    Fallback: filesystem scan of pdf_pipeline/output/ + data/uploads/.
    """
    results: list[PaperOutput] = []
    seen: set[str] = set()

    for meta in catalog.list_papers():
        name = meta.get("paper_name", "")
        if not name:
            continue
        seen.add(name)
        indexed = bool(meta.get("indexed"))
        status, detail = _derive_status(indexed, name)
        results.append(PaperOutput(
            paper_name=name,
            files=_list_output_files(name),
            chunk_count=meta.get("chunk_count", _count_chunks(name)),
            status=status,
            indexed=indexed,
            detail=detail,
        ))

    # Filesystem scan for papers not in catalog (unregistered dir / Redis down)
    if OUTPUT_DIR.exists():
        for paper_dir in sorted(OUTPUT_DIR.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name in seen:
                continue
            files = _list_output_files(paper_dir.name)
            results.append(PaperOutput(
                paper_name=paper_dir.name,
                files=files,
                chunk_count=_count_chunks(paper_dir.name),
                status="not_indexed",
                indexed=False,
                detail="parsed",
            ))

    # Raw PDFs: downloaded but not yet processed (uploads + downloads both count)
    results.extend(_scan_raw_pdfs(seen))

    return results


def _scan_raw_pdfs(seen: set[str]) -> list[PaperOutput]:
    """Scan data/uploads/ and data/downloads/ for unprocessed (raw) PDFs.

    Papers already in an output dir or already listed are skipped.
    """
    results: list[PaperOutput] = []
    for d in (UPLOAD_DIR, DOWNLOAD_DIR):
        if not d.exists():
            continue
        for pdf_file in sorted(d.iterdir()):
            if not pdf_file.suffix.lower() == ".pdf":
                continue
            paper_name = pdf_file.stem
            if paper_name in seen:
                continue
            if (OUTPUT_DIR / paper_name).is_dir():
                continue  # already indexed, covered above
            seen.add(paper_name)
            results.append(PaperOutput(
                paper_name=paper_name,
                files=[pdf_file.name],
                chunk_count=0,
                status="not_indexed",
                indexed=False,
                detail="raw",
            ))
    return results


@router.get("/outputs/{paper_name}", response_model=PaperOutput)
async def get_output(paper_name: str):
    """Get output details for a specific paper (raw or indexed)."""
    paper_dir = OUTPUT_DIR / paper_name

    if paper_dir.is_dir():
        files = [p.name for p in paper_dir.iterdir() if p.is_file()]
        chunk_count = 0
        rag_json = paper_dir / "rag_chunks.json"
        if rag_json.exists():
            try:
                data = json.loads(rag_json.read_text(encoding="utf-8"))
                chunk_count = len(data.get("chunks", []))
            except Exception:
                pass
        meta = catalog.get_paper(paper_name) or {}
        indexed = bool(meta.get("indexed"))
        status, detail = _derive_status(indexed, paper_name)
        return PaperOutput(paper_name=paper_name, files=files, chunk_count=chunk_count,
                           status=status, indexed=indexed, detail=detail)

    # Check raw PDFs (uploads or downloads)
    for d in (UPLOAD_DIR, DOWNLOAD_DIR):
        pdf_path = d / f"{paper_name}.pdf"
        if pdf_path.exists():
            return PaperOutput(paper_name=paper_name, files=[pdf_path.name], chunk_count=0,
                               status="not_indexed", indexed=False, detail="raw")

    raise HTTPException(404, f"Paper {paper_name} not found")
