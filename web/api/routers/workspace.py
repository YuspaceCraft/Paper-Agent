"""
workspace.py — workspace file browsing endpoints (client file explorer).

Thin wrapper over agent/providers/generic_provider.py:resolve_workspace_path
(唯一的路径安全边界 — 越界/symlink/.. 穿越即拒). No business logic —
resolve → iterdir/read → JSON.

GET  /api/workspace/list?path=<相对路径>   — directory entries
GET  /api/workspace/read?path=<相对路径>   — file content (truncated ~32KB)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from agent.providers.generic_provider import resolve_workspace_path

router = APIRouter(prefix="/api/workspace", tags=["Workspace"])

_MAX_READ_BYTES = 32_000


def _resolve(path: str) -> Path:
    try:
        return resolve_workspace_path(path)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@router.get("/list")
async def list_workspace(
    path: str = Query(default=".", description="Directory to list (relative to project root)"),
):
    p = _resolve(path)
    if not p.is_dir():
        raise HTTPException(404, f"not a directory: {path}")
    entries = []
    for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        try:
            size = e.stat().st_size if e.is_file() else None
        except OSError:
            size = None
        entries.append({"name": e.name, "is_dir": e.is_dir(), "size": size})
    return {"ok": True, "data": {"path": path, "entries": entries}}


@router.get("/read")
async def read_workspace(
    path: str = Query(..., description="File to read (relative to project root)"),
):
    p = _resolve(path)
    if not p.is_file():
        raise HTTPException(404, f"not a file: {path}")
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise HTTPException(400, f"read failed: {e}")
    # binary detection: null byte in first 8KB
    is_binary = b"\x00" in raw[:8192]
    text = raw.decode("utf-8", errors="replace")
    if len(raw) > _MAX_READ_BYTES:
        cut = raw[:_MAX_READ_BYTES].decode("utf-8", errors="ignore")
        text = f"{cut}\n...[truncated — {len(raw)} bytes total, {_MAX_READ_BYTES} shown]"
    return {"ok": True, "data": {"path": path, "is_binary": is_binary, "content": text}}
