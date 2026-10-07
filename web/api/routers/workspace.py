"""
workspace.py — workspace file browsing endpoints (client file explorer).

Thin wrapper over agent/providers/generic_provider.py:resolve_workspace_path
(唯一的路径安全边界 — 越界/symlink/.. 穿越即拒). No business logic —
resolve → iterdir/read → JSON.

GET  /api/workspace/list?path=<相对路径>&root=<project|experiments>
                                     — directory entries (root 决定基准根)
GET  /api/workspace/read?path=<相对路径>&root=<project|experiments>
                                     — file content (truncated ~32KB)
GET  /api/workspace/browse?path=<绝对路径或空>
                                     — 只读目录浏览（路径选择器用，刻意越界）

root 语义（agent/workspace_config.py）：
  project     文献问答 + 写作根 = get_project_root()（未设置时=代码根）
  experiments 实验根 = get_experiments_path()（默认 web/workspace/experiments）
"""
from __future__ import annotations

import os
import string
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from agent.providers.generic_provider import resolve_workspace_path
from agent.workspace_config import get_experiments_path, get_project_root

router = APIRouter(prefix="/api/workspace", tags=["Workspace"])

_MAX_READ_BYTES = 32_000
_MAX_LIST_ENTRIES = 2_000

_ALLOWED_ROOTS = {"project", "experiments"}


def _root_path(root: str) -> Path:
    """root → 基准目录。未知值 400（防静默回退到错误路径）。"""
    if root == "experiments":
        return get_experiments_path()
    if root == "project":
        return get_project_root()
    raise HTTPException(400, f"root must be one of {sorted(_ALLOWED_ROOTS)}")


def _resolve(path: str, root: str = "project") -> Path:
    try:
        return resolve_workspace_path(path, root=_root_path(root))
    except PermissionError as e:
        raise HTTPException(403, str(e))


@router.get("/list")
async def list_workspace(
    path: str = Query(default=".", description="Directory to list (relative to the chosen root)"),
    root: str = Query(default="project", description="Root view: project | experiments"),
):
    p = _resolve(path, root)
    if not p.is_dir():
        raise HTTPException(404, f"not a directory: {path}")
    all_entries = sorted(
        p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()),
    )
    entries = []
    for e in all_entries[:_MAX_LIST_ENTRIES]:
        try:
            size = e.stat().st_size if e.is_file() else None
        except OSError:
            size = None
        entries.append({"name": e.name, "is_dir": e.is_dir(), "size": size})
    return {
        "outcome": "succeeded",
        "data": {
            "path": path,
            "entries": entries,
            "truncated": len(all_entries) > _MAX_LIST_ENTRIES,
            "total": len(all_entries),
        },
    }


@router.get("/read")
async def read_workspace(
    path: str = Query(..., description="File to read (relative to the chosen root)"),
    root: str = Query(default="project", description="Root view: project | experiments"),
):
    p = _resolve(path, root)
    if not p.is_file():
        raise HTTPException(404, f"not a file: {path}")
    try:
        with p.open("rb") as handle:
            raw = handle.read(_MAX_READ_BYTES + 1)
        total = p.stat().st_size
    except OSError as e:
        raise HTTPException(400, f"read failed: {e}")
    truncated = len(raw) > _MAX_READ_BYTES or total > _MAX_READ_BYTES
    raw = raw[:_MAX_READ_BYTES]
    # binary detection: null byte in the bounded sample
    is_binary = b"\x00" in raw[:8192]
    text = raw.decode("utf-8", errors="replace")
    if truncated:
        text = f"{text}\n...[truncated — {total} bytes total, {_MAX_READ_BYTES} shown]"
    return {
        "outcome": "succeeded",
        "data": {
            "path": path,
            "is_binary": is_binary,
            "content": text,
            "truncated": truncated,
            "size": total,
        },
    }


@router.get("/browse")
async def browse_local(
    path: str = Query(default="", description="Absolute dir to list its subdirectories; empty → drive letters (Windows)"),
):
    """只读目录浏览 — 路径选择器（项目路径/实验根）导航用。

    刻意越界、仅列目录、绝不读写文件（独立于 resolve_workspace_path 的越界
    守卫；选择器必须能看到根外的目录用户才能选择）。桌面本地应用场景。
    """
    if not path:
        if os.name != "nt":
            return {"outcome": "succeeded", "data": {"path": "", "entries": [{"name": "/", "is_dir": True}]}}
        drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
        return {"outcome": "succeeded", "data": {"path": "", "entries": [{"name": d, "is_dir": True} for d in drives]}}
    p = Path(os.path.expanduser(path))
    if not p.is_dir():
        raise HTTPException(404, f"not a directory: {path}")
    all_entries = [
        {"name": e.name, "is_dir": True}
        for e in sorted(p.iterdir())
        if e.is_dir() and not e.name.startswith(".")
    ]
    return {
        "outcome": "succeeded",
        "data": {
            "path": str(p),
            "entries": all_entries[:_MAX_LIST_ENTRIES],
            "truncated": len(all_entries) > _MAX_LIST_ENTRIES,
            "total": len(all_entries),
        },
    }
