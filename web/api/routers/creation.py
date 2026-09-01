"""
creation.py — 创作领域 HTTP 封装（v10）。

薄封装：序列化 + 文件响应 + 参数校验，领域逻辑在 agent/domains/creation.py
（纯 Python 业务模块）。前端（写作工作区）走这里；agent 的工具直接调模块。

Endpoint:
  GET    /api/creation/docs                          list docs
  POST   /api/creation/docs                          create doc {title}
  PUT    /api/creation/docs/{doc_id}/outline         set outline {outline: [...]}
  PUT    /api/creation/docs/{doc_id}/sections/{sid}  write section {content}
  GET    /api/creation/docs/{doc_id}                 doc full state
  GET    /api/creation/docs/{doc_id}/export-docx     download .docx (FileResponse)
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.domains.creation import (
    PROJECT_ROOT,
    _doc_dir,
    _load_doc,
    _main_md_path,
    doc_create,
    doc_set_outline,
    doc_write_section,
    doc_export_docx,
)

router = APIRouter(prefix="/api/creation", tags=["creation"])


class CreateDocBody(BaseModel):
    title: str = Field(default="Untitled", min_length=1, max_length=200)


class OutlineBody(BaseModel):
    outline: list = Field(default_factory=list, description="JSON array of chapters")


class SectionBody(BaseModel):
    content: str = Field(default="", description="Section Markdown body")


@router.get("/docs")
async def list_docs(status: str = ""):
    from agent.domains.creation import doc_list

    result = json.loads(await doc_list.ainvoke({"status": status}))
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "list failed"))
    return result["data"]


@router.post("/docs", status_code=201)
async def create_doc(body: CreateDocBody):
    result = json.loads(await doc_create.ainvoke({"title": body.title}))
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "create failed"))
    return result["data"]


@router.put("/docs/{doc_id}/outline")
async def set_outline(doc_id: str, body: OutlineBody):
    result = json.loads(await doc_set_outline.ainvoke({
        "doc_id": doc_id,
        "outline": json.dumps(body.outline, ensure_ascii=False),
    }))
    if not result.get("ok"):
        raise HTTPException(
            400 if result.get("error_type") == "param_error" else 404,
            result.get("error", "set outline failed"),
        )
    return result["data"]


@router.put("/docs/{doc_id}/sections/{section_id}")
async def write_section(doc_id: str, section_id: str, body: SectionBody):
    result = json.loads(await doc_write_section.ainvoke({
        "doc_id": doc_id, "section_id": section_id, "content": body.content}))
    if not result.get("ok"):
        raise HTTPException(
            400 if result.get("error_type") == "param_error" else 404,
            result.get("error", "write section failed"),
        )
    return result["data"]


@router.get("/docs/{doc_id}")
async def get_doc(doc_id: str):
    doc = _load_doc(doc_id)
    if doc is None:
        raise HTTPException(404, f"doc '{doc_id}' not found")
    md = ""
    mp = _main_md_path(doc_id)
    if mp.exists():
        md = mp.read_text(encoding="utf-8")
    # 每章内容（编辑器加载用）；空章节 → 空串
    content_map = {}
    for item in doc.get("outline", []):
        sid = item.get("section_id")
        if not sid:
            continue
        p = _doc_dir(doc_id) / "sections" / f"{sid}.md"
        content_map[sid] = p.read_text(encoding="utf-8") if p.exists() else ""
    return {
        "doc_id": doc.get("doc_id"),
        "title": doc.get("title", ""),
        "status": doc.get("status", ""),
        "outline": doc.get("outline", []),
        "sections": doc.get("sections", {}),
        "sections_content": content_map,
        "assembled_md": md,
        "updated_at": doc.get("updated_at", ""),
    }


@router.get("/docs/{doc_id}/export-docx")
async def export_docx(doc_id: str):
    result = json.loads(await doc_export_docx.ainvoke({"doc_id": doc_id}))
    if not result.get("ok"):
        raise HTTPException(
            404 if result.get("error_type") == "param_error" else 500,
            result.get("error", "export failed"),
        )
    path = PROJECT_ROOT / result["data"]["export_path"]
    if not path.exists():
        raise HTTPException(404, "exported file not found")
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        filename=f"{doc_id}.docx")