"""
builtin_provider.py — 科研论文核心工具，agent 存在的基础能力。

search_papers: 语义检索论文库 + 列出全部论文
fetch_content: 获取论文摘要/章节全文
download_paper: 从 arXiv 下载 PDF 到指定目录（纯下载，不解析/不切片/不入库）
ingest_paper: 解析 PDF → 向量入库（独立的显式任务，仅当用户要求入库时调用）

下载与入库解耦：download_paper 只落盘文件（支持 destination/filename 参数，
文件名默认自 arXiv 标题推导简称），绝不触发 parse/chunk/index；ingest_paper
才是显式的「入库」任务。

实现从 agent/tools.py 迁移至此。保持 httpx → API 的薄封装，
遵循 CLAUDE.md 的 FastAPI 封装原则。

## 工具返回格式约定

信封生成/解析统一收敛在 `agent/tool_contract.py`（P6）。本模块遵循：

- **结构化数据** → JSON `{"ok": true/false, "data": {...}, ...}`
  - search_papers (论文列表/搜索结果)、download_paper、ingest_paper
  - 错误响应：`{"ok": false, "error": "...", "next": "...", "error_type": "..."}`

- **内容型数据** → 纯文本 Markdown
  - fetch_content overview (标题/作者/摘要/章节列表)
  - fetch_content deep read (章节正文)

`parse_tool_result()` (tool_contract.py) 是唯一解析入口——先试 JSON envelope，
非 envelope 一律按纯文本处理（旧版双格式猜测已删除）。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import httpx
from langchain_core.tools import tool

from . import ToolDef, ToolProvider
from agent.safety import tool_allowed
from agent.providers.generic_provider import resolve_workspace_path
from agent.resolution import match_local_state
from agent.library_api import (
    api_is_down as _api_down,
    api_mark_down as _api_mark_down,
    api_timeout as _api_timeout,
)

API = os.environ.get("AGENT_API_BASE", "http://127.0.0.1:8000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


# ---- content cleaning ----
# ponytail: strip [KEYWORDS: ...] / [FORMULA_DESC: ...] / [FIGURE_DESC: ...] prefixes
# from chunk content. These are BM25 retrieval markers embedded in chunk content
# by the rag_chunker — useful for retrieval but noise once shown to the LLM.
_CONTENT_PREFIX_RE = re.compile(
    r'^\s*(?:\[(?:KEYWORDS|FORMULA_DESC|FIGURE_DESC):[^\]]+\]\s*)+',
    re.MULTILINE,
)


def _clean_content(content: str) -> str:
    """Strip retrieval-noise prefixes from chunk content."""
    return _CONTENT_PREFIX_RE.sub("", content).strip()


# ---- response helpers ----
# 信封形状统一由 agent/tool_contract.py 定义（P6 收敛），此处仅转发。
# 注意 `status` 为历史参数，从不进入信封，保留签名以兼容旧调用点。

def _ok(data: dict | list) -> str:
    from agent.tool_contract import ok as _ok_contract
    return _ok_contract(data)


def _err(status: int, detail: str, next_action: str,
         error_type: str = "unknown", **ctx) -> str:
    from agent.tool_contract import err as _err_contract
    return _err_contract(error_type, detail, next_action, **ctx)


def _parse_detail(e: httpx.HTTPStatusError) -> str:
    try:
        body = e.response.json()
        detail = body.get("detail", str(e))
        if isinstance(detail, dict):
            return json.dumps(detail, ensure_ascii=False)
        return str(detail)
    except Exception:
        return str(e)


def _backend_down_err() -> str:
    """库后端不可达 → 快速失败错误信封。

    error_type="backend_down" 让 nodes._format_error_feedback 把反馈写成
    「停止重试 + 向用户报告故障」，而不是误导性的 "Server busy. Retry once"，
    避免 agent 把整轮 TURN_TIMEOUT 烧光最后只说「回答超时」。
    """
    return _err(
        503,
        f"本地知识库后端不可达或响应超时（{API}）。"
        "请确认后端已启动（uvicorn web.api.main:app --host 0.0.0.0 --port 8000）"
        "或 AGENT_API_BASE 指向实际端口。",
        "停止重试库工具。向用户说明本地知识库后端不可用，并给出启动命令/端口检查。",
        error_type="backend_down",
    )


# ---- internal helpers ----

async def _fetch_papers(c: httpx.AsyncClient) -> list[str]:
    if _api_down(API):
        return []
    try:
        r = await c.get(f"{API}/api/reader/papers")
        r.raise_for_status()
        return [
            p.get("name", p.get("paper_name", ""))
            for p in r.json().get("papers", [])
        ]
    except Exception:
        _api_mark_down(API)
        return []


async def _fetch_sections(c: httpx.AsyncClient, paper: str) -> list[str]:
    if _api_down(API):
        return []
    try:
        r = await c.get(f"{API}/api/reader/{paper}/sections")
        r.raise_for_status()
        return [s["name"] for s in r.json().get("sections", [])]
    except Exception:
        _api_mark_down(API)
        return []


async def _paper_error(c: httpx.AsyncClient, e: httpx.HTTPStatusError,
                       paper_name: str) -> str:
    detail = _parse_detail(e)
    if e.response.status_code == 404:
        papers = await _fetch_papers(c)
        return _err(404, detail,
            f"Paper '{paper_name}' not in library. Pick from available_papers.",
            error_type="param_error", available_papers=papers)
    return _err(e.response.status_code, detail,
        "Backend error. Retry once or use search_papers().",
        error_type="unknown")


# ---- tools ----

@tool
async def search_papers(query: str = "", top_k: int = 5) -> str:
    """Search papers by topic/keyword. Use with empty query to list ALL papers.

    PRIMARY entry point. Always start here to discover what's available.

    Examples:
    - search_papers()              → list all papers in library
    - search_papers("transformer") → semantic search for "transformer"
    - search_papers("loss", 10)    → search with more results

    Returns paper metadata (list mode) or matching chunks with scores (search mode).
    """
    if _api_down(API):
        return _backend_down_err()
    async with httpx.AsyncClient(timeout=_api_timeout(15.0)) as c:
        if not query.strip():
            try:
                r = await c.get(f"{API}/api/reader/papers")
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                return _err(e.response.status_code, _parse_detail(e),
                    "Library unavailable. Retry once.",
                    error_type="transient")
            except httpx.TransportError:
                _api_mark_down(API)
                return _backend_down_err()

            papers = data.get("papers", [])
            if not papers:
                return json.dumps({
                    "ok": True, "data": {"papers": [], "count": 0},
                    "hint": "No papers indexed. Upload PDFs via the web UI first.",
                }, ensure_ascii=False, indent=2)

            trimmed = [{
                "name": p.get("paper_name", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", ""),
                "year": p.get("year", ""),
                "arxiv_id": p.get("arxiv_id", ""),
            } for p in papers]
            return _ok({"papers": trimmed, "count": len(trimmed)})

        try:
            r = await c.post(
                f"{API}/api/retrieval/search",
                json={"query": query, "top_k": top_k},
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            return _err(e.response.status_code, _parse_detail(e),
                "Search unavailable. Use search_papers() to browse all papers.",
                error_type="transient")
        except httpx.TransportError:
            _api_mark_down(API)
            return _backend_down_err()

        results = data.get("results", [])
        if not results:
            return json.dumps({
                "ok": True, "data": {"results": [], "total": 0},
                "hint": "No matches. Broaden query or use search_papers() to list all papers.",
            }, ensure_ascii=False, indent=2)

        trimmed = []
        for h in results:
            text = _clean_content((h.get("generation_text", "") or "")[:800])
            chunk_id = h.get("chunk_id", "")
            paper = (h.get("section_path", "") or "").split(" > ")[0]
            # Fallback: extract paper name from chunk_id
            # (format: {paper_name}__chunk_{number})
            if not paper and "__chunk_" in chunk_id:
                paper = chunk_id.rsplit("__chunk_", 1)[0]
            trimmed.append({
                "chunk_id": chunk_id,
                "paper": paper,
                "section": h.get("section_path", ""),
                "text": text,
                "score": h.get("score", 0),
            })
        return _ok({"results": trimmed, "total": len(trimmed)})


@tool
async def fetch_content(paper_name: str, section: str = "") -> str:
    """Read paper content. Use search_papers() FIRST to discover the paper name.

    Two modes:
    - section="" (default): returns paper metadata — title, authors, abstract,
      AND full section list with chunk counts (OVERVIEW)
    - section="Methodology": returns full body text of that section (DEEP READ)

    PRECONDITION: paper_name MUST come from a search_papers() result, not from
    user input directly. Verify the paper exists before calling.

    Examples:
    - fetch_content("RMNet")                 → abstract + section list
    - fetch_content("RMNet", "3. Methodology") → full section body text
    """
    if _api_down(API):
        return _backend_down_err()
    async with httpx.AsyncClient(timeout=_api_timeout()) as c:
        if not section:
            try:
                r = await c.get(f"{API}/api/reader/{paper_name}/abstract")
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                return await _paper_error(c, e, paper_name)
            except httpx.TransportError:
                _api_mark_down(API)
                return _backend_down_err()

            title = data.get("title", "")
            authors = data.get("authors", "")
            abstract = data.get("abstract", "")
            sections = data.get("sections", [])
            pn = data.get("paper_name", paper_name)

            lines: list[str] = []
            if title:
                lines.append(f"# {title}\n")
            if authors:
                lines.append(f"**Authors:** {authors}\n")
            if sections:
                lines.append("**Sections:**")
                for s in sections:
                    name = s.get("name", "")
                    count = s.get("chunk_count", 0)
                    lines.append(f"- {name}  ({count} chunks)")
                lines.append("")
            if abstract:
                lines.append("## Abstract\n")
                lines.append(abstract)
            if not lines:
                lines.append(f"(no metadata available for {pn})")
            return "\n".join(lines)

        try:
            r = await c.get(f"{API}/api/reader/{paper_name}/sections/{section}")
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            available = await _fetch_sections(c, paper_name)
            detail = _parse_detail(e)
            return _err(e.response.status_code, detail,
                f"Section '{section}' not found. Pick from available_sections.",
                error_type="param_error",
                available_sections=available,
                available_papers=[paper_name])
        except httpx.TransportError:
            _api_mark_down(API)
            return _backend_down_err()

        chunks = data.get("chunks", [])
        section_q = data.get("section_query", section)
        pn = data.get("paper_name", paper_name)
        total = data.get("chunk_count", len(chunks))

        lines = [f"## {section_q}  ({pn})", f"{total} chunk(s)\n"]
        for i, c in enumerate(chunks):
            content = _clean_content(c.get("content", "") or "")
            if not content:
                continue
            lines.append(content)
            if i < len(chunks) - 1:
                lines.append("")
        return "\n".join(lines)


# ---- paper management tools ----
# 下载与入库解耦：download_paper 只落盘文件（destination/filename 参数），
# 绝不触发 parse/chunk/index；ingest_paper 是独立的显式「入库」任务。


# ---- download naming helpers ----

def _sanitize_dl_name(name: str, max_len: int = 80) -> str:
    """Filesystem-safe file stem: strip path-unsafe chars, collapse underscores,
    truncate to max_len. Falls back to a random name if nothing survives."""
    n = name.replace(" ", "_")
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", n)
    n = re.sub(r'_+', '_', n).strip('_.')
    if len(n) > max_len:
        cut = n[:max_len]
        last_us = cut.rfind('_')
        n = cut[:last_us] if last_us > max_len // 2 else cut
    return n or f"paper_{uuid.uuid4().hex[:8]}"


def _short_name_from_title(title: str) -> str | None:
    """Derive a paper short name from its title, or None if not derivable.

    Academic convention is "{ShortName}: full title" — take the leading token
    before a colon / em-dash / en-dash / middle dot. Only accept a compact
    identifier-like token (2-40 ASCII chars, no spaces); otherwise None so the
    caller falls back to the arxiv_id rather than inventing a misleading name.
    """
    if not title:
        return None
    t = title.strip().strip('"\'“”‘’《》 ')
    for sep in (":", "：", "—", "–", "·", " - "):
        idx = t.find(sep)
        if idx > 0:
            t = t[:idx].strip(' \t-–—:：.·')
            break
    t = t.strip()
    if 2 <= len(t) <= 40 and re.match(r'^[A-Za-z][A-Za-z0-9_\-.]*$', t):
        return t
    return None


async def _fetch_arxiv_title(arxiv_id: str) -> str | None:
    """Best-effort fetch of a paper's arXiv title (single metadata request).

    Non-fatal: any failure returns None and the caller falls back to arxiv_id.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": 1},
            )
            r.raise_for_status()
            import feedparser  # lazy: only needed when deriving short names
            feed = feedparser.parse(r.text)
            if feed and feed.get("entries") and feed.entries[0].get("title"):
                return " ".join(feed.entries[0]["title"].split())
    except Exception:
        pass
    return None


def _resolve_stem(arxiv_id: str, title: str | None, filename: str) -> str:
    """Pick the actual file stem: explicit filename > title-derived short name > arxiv_id."""
    stem = _sanitize_dl_name(filename) if filename.strip() else ""
    if not stem:
        short = _short_name_from_title(title or "")
        stem = _sanitize_dl_name(short) if short else _sanitize_dl_name(arxiv_id)
    return stem


@tool
async def download_paper(
    arxiv_id: str,
    destination: str = "./data/downloads",
    filename: str = "",
) -> str:
    """Download a paper PDF from arXiv into a workspace folder. PURE FILE DOWNLOAD.

    Saves the PDF to `destination`/`{filename}.pdf` — destination is
    workspace-relative, so the file appears in the client's file explorer at
    exactly the folder the user asked for. When filename is empty, a short name
    is derived from the arXiv title (e.g. "RMNet"); falls back to the arxiv_id
    when no clean short name exists.

    This tool NEVER parses, chunks, or indexes the paper. To make it searchable
    in the library, call ingest_paper() afterwards as a separate, explicit step.

    Args:
        arxiv_id: arXiv paper ID (e.g. "2301.07093" or "2301.07093v2").
        destination: Workspace folder to save into (created if missing).
        filename: Optional file stem WITHOUT extension. Defaults to a short
            name derived from the arXiv title.
    """
    # Strip version suffix (e.g. "2301.07093v2" → "2301.07093"). arXiv's PDF
    # endpoint serves the latest version only at the canonical ID — the vN form
    # 404s.
    canonical_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    pdf_url = f"https://arxiv.org/pdf/{canonical_id}.pdf"

    # Resolve the destination against the workspace root — the same boundary the
    # client file explorer uses, so the file lands where the user asked to see it.
    try:
        out_dir = resolve_workspace_path(destination)
    except PermissionError as e:
        return _err(403, str(e), "Use a destination folder inside the workspace.",
                    error_type="permission_denied")

    # arXiv 身份核验（必做）：拿到 ID 对应的真实标题后才能下载。
    # 防「编造/张冠李戴的 arXiv ID → 下载到与所述论文不符的 PDF」。核验失败绝不盲下载。
    title = await _fetch_arxiv_title(canonical_id)
    if not title:
        return _err(
            422,
            f"无法验证 arXiv ID {canonical_id}（export.arxiv.org 查询失败或该 ID 无效），"
            "为避免下载到与所述论文不符的 PDF，已中止下载。",
            "先用 arxiv 子代理确认正确的 arXiv ID；若本地已有该论文的 PDF，"
            "告知我本地路径即可直接 ingest_paper。",
            error_type="unverified",
        )

    # Actual file stem: explicit filename > title-derived short name > arxiv_id.
    stem = _resolve_stem(canonical_id, title, filename)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.pdf"
    try:
        rel_path = out_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_path = str(out_path)

    if out_path.exists():
        return json.dumps({
            "ok": True, "data": {
                "paper_name": stem, "arxiv_id": canonical_id,
                "title": title,
                "filename": f"{stem}.pdf",
                "path": str(out_path), "relative_path": rel_path,
                "size_bytes": out_path.stat().st_size, "status": "raw",
                "message": (
                    f"PDF already exists at {rel_path} ({out_path.stat().st_size} bytes). "
                    "Downloaded file only — NOT parsed or indexed. "
                    f"To index it, call ingest_paper('{stem}') as a separate step."
                ),
            },
        }, ensure_ascii=False, indent=2)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
            resp = await client.get(pdf_url)
            if resp.status_code == 404:
                return _err(404,
                    f"arXiv paper '{canonical_id}' not found. Verify the ID with arxiv__get_paper_data.",
                    "Check the arxiv_id and retry.", error_type="param_error")
            resp.raise_for_status()

        out_path.write_bytes(resp.content)
        return json.dumps({
            "ok": True, "data": {
                "paper_name": stem, "arxiv_id": canonical_id,
                "title": title,
                "filename": f"{stem}.pdf",
                "path": str(out_path), "relative_path": rel_path,
                "size_bytes": len(resp.content), "status": "raw",
                "message": (
                    f"PDF downloaded to {rel_path} ({len(resp.content)} bytes). "
                    "Downloaded file only — NOT parsed or indexed. "
                    f"To index it, call ingest_paper('{stem}') as a separate step."
                ),
            },
        }, ensure_ascii=False, indent=2)

    except httpx.TimeoutException:
        return _err(408, "Download timed out (arXiv may be slow).",
            "Retry once. If persistent, the paper may be too large.", error_type="transient")


@tool
async def ingest_paper(paper_name: str, pdf_path: str = "") -> str:
    """Enqueue 入库 (ingest): ONE COMPLETE operation — parse the PDF AND build
    the vector index, making the paper searchable. Returns immediately.

    入库 is a single atomic job (parse + vectorization), never split into
    separate "parse" and "index" steps. SEPARATE from download_paper (which only
    fetches the PDF). Only call this when the user wants the paper imported into
    the library / searchable ("入库" / "导入" / "加入知识库").

    The paper must already be on disk (download_paper or Web UI upload). pdf_path
    locates the PDF when it was downloaded to a custom folder (use the
    relative_path/path returned by download_paper); when empty,
    data/uploads/{paper_name}.pdf then data/downloads/{paper_name}.pdf are searched.

    ASYNC: the job is enqueued (task_id) and this tool returns immediately.
    Progress (parse → vectorization) shows above the chat and is queryable via
    check_task_status(task_id); completion is announced to the user.

    Args:
        paper_name: Paper name as shown in search_papers() list or arxiv_id of a downloaded paper.
        pdf_path: Workspace path (relative or absolute) to the PDF to process (optional).
    """
    if _api_down(API):
        return _backend_down_err()
    async with httpx.AsyncClient(timeout=_api_timeout(15.0)) as client:
        try:
            r = await client.post(
                f"{API}/api/agent/ingest",
                json={"paper_name": paper_name, "pdf_path": pdf_path, "notify": True},
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            return _err(e.response.status_code, _parse_detail(e),
                "Failed to start the background ingest.", error_type="unknown")
        except httpx.TransportError:
            _api_mark_down(API)
            return _backend_down_err()

    data = r.json()
    return json.dumps({
        "ok": True, "data": {
            "task_id": data["task_id"],
            "paper_name": data["paper_name"],
            "status": "running",
            "message": (
                f"已后台启动入库（task_id={data['task_id']}）：解析 + 向量索引将在后台异步完成，"
                f"完成后会通知用户。可在客户端顶部任务区实时查看进度，"
                f"或用 check_task_status('{data['task_id']}') 查询。"
            ),
        },
    }, ensure_ascii=False, indent=2)


@tool
async def check_task_status(task_id: str) -> str:
    """Query a background task's status by task_id: pending | running | done | failed.

    Background tasks (e.g. the ingest enqueued by ingest_paper) run asynchronously.
    Use this whenever the user asks about task progress ("入库了吗 / 任务进展如何 /
    done yet?"). Returns current status + progress, and the result/error once done.

    Args:
        task_id: Task id returned by ingest_paper.
    """
    if _api_down(API):
        return _backend_down_err()
    async with httpx.AsyncClient(timeout=_api_timeout()) as client:
        try:
            r = await client.get(f"{API}/api/agent/tasks/{task_id}")
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return _err(404, _parse_detail(e),
                    "Task not found. It may have expired, or ask the user which paper they meant to index.",
                    error_type="param_error")
            return _err(e.response.status_code, _parse_detail(e),
                "Task status unavailable. Retry once.", error_type="transient")
        except httpx.TransportError:
            _api_mark_down(API)
            return _backend_down_err()

    return json.dumps({
        "ok": True,
        "data": {
            "task_id": data.get("task_id", task_id),
            "paper_name": data.get("paper_name", ""),
            "status": data.get("status", ""),
            "progress": data.get("progress", ""),
            "error": data.get("error") or None,
            "result": data.get("result") or None,
        },
    }, ensure_ascii=False, indent=2)


@tool
async def check_paper(term: str = "") -> str:
    """Check the LOCAL state of a paper: indexed / downloaded_not_indexed / absent.

    Fast read-only check — Redis catalog (O(1) "已入库?") + filesystem scan for
    parse output & PDFs in data/uploads & data/downloads. NEVER touches the network.

    Call this FIRST whenever the user wants to save/import ("入库") a paper:
      - state="indexed"                → already searchable in the library; don't redo
      - state="downloaded_not_indexed" → local artifact exists (output dir or PDF);
                                          matches[].pdf_path feeds ingest (action: ingest)
      - state="absent"                 → not local; first get arxiv_id from the arxiv
                                          subagent, then ingest (action: download_and_ingest)

    Per-paper entry: state 是两类语义（indexed / not_indexed），detail 是派生
    诊断（parsed=有解析产物 / raw=仅有本地 PDF / indexed / ""）。

    Args:
        term: Paper name/keyword as the user calls it. Empty term returns the
            full local snapshot grouped by state.
    """
    if _api_down(API):
        return _backend_down_err()
    async with httpx.AsyncClient(timeout=_api_timeout()) as c:
        try:
            r = await c.get(f"{API}/api/reader/local-papers")
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            return _err(e.response.status_code, _parse_detail(e),
                "Local snapshot unavailable. Retry once.",
                error_type="transient")
        except httpx.TransportError:
            _api_mark_down(API)
            return _backend_down_err()

        papers = data.get("papers", [])

    if not term.strip():
        counts: dict[str, int] = {"indexed": 0, "not_indexed": 0}
        for p in papers:
            st = p.get("state", "not_indexed")
            counts[st] = counts.get(st, 0) + 1
        return _ok({
            "term": "", "counts": counts,
            "papers": [{"paper_name": p["paper_name"], "state": p["state"],
                        "detail": p.get("detail", "")} for p in papers],
        })

    result = match_local_state(term.strip(), papers)
    return _ok({"term": term.strip(), **result})


# ---- ToolDef 描述 ----

BUILTIN_TOOLS = [search_papers, fetch_content, download_paper, ingest_paper, check_paper, check_task_status]

BUILTIN_TOOLDEFS = [
    ToolDef(
        name="search_papers",
        description=(
            "Search papers in the LOCAL library by topic/keyword; empty query lists all papers. "
            "PRIMARY entry point — start here to discover available papers. "
            "Args: query ('' = list all), top_k (max results in search mode). "
            "Returns JSON {\"ok\":true,\"data\":{papers|results}}; failure returns an error envelope."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Empty string lists all papers.",
                    "default": "",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results for search mode (ignored in list mode).",
                    "default": 5,
                },
            },
        },
        source="builtin",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
    ToolDef(
        name="fetch_content",
        description=(
            "Read content of a paper in the LOCAL library. Empty section = overview "
            "(title/authors/abstract + section list); section name = full body text. "
            "PRECONDITION: paper_name must come from a prior search_papers() result. "
            "Returns Markdown; errors as JSON envelope."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paper_name": {
                    "type": "string",
                    "description": "Paper name as returned by search_papers().",
                },
                "section": {
                    "type": "string",
                    "description": "Section heading. Empty = overview mode.",
                    "default": "",
                },
            },
            "required": ["paper_name"],
        },
        source="builtin",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
    ToolDef(
        name="download_paper",
        description=(
            "Download a paper PDF from arXiv into a workspace folder. PURE FILE DOWNLOAD — "
            "does NOT parse, chunk, or index anything. The file lands in `destination` "
            "(workspace-relative; visible in the client file explorer at exactly that folder) "
            "named `{filename}.pdf` — filename defaults to a short name derived from the arXiv "
            "title (e.g. 'RMNet'); pass filename to override. Returns the ACTUAL saved path + "
            "filename — report them verbatim, never invent a folder/name. "
            "To make the paper searchable in the library, call ingest_paper() afterwards as a "
            "separate, explicit step."
        ),
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv paper ID (e.g. '2301.07093' or '2301.07093v2').",
                },
                "destination": {
                    "type": "string",
                    "description": "Workspace folder to save into (created if missing), e.g. './data/downloads'.",
                    "default": "./data/downloads",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional file stem WITHOUT extension, e.g. 'RMNet'. Defaults to a short name derived from the arXiv title.",
                    "default": "",
                },
            },
            "required": ["arxiv_id"],
        },
        source="builtin",
        annotations={"readOnlyHint": False, "idempotentHint": True},
    ),
    ToolDef(
        name="ingest_paper",
        description=(
            "入库 (ingest): ONE COMPLETE operation — parse the downloaded PDF AND build "
            "the vector index, making the paper searchable in the library. Atomic: never "
            "split into separate parse/index steps. SEPARATE from download_paper (which "
            "only fetches the PDF): call this when the user asked to 入库 / import into "
            "the library / make searchable. PRECONDITION: the PDF must already be on disk. "
            "Args: paper_name (arxiv_id from download_paper, or name from search_papers); "
            "pdf_path (optional — the relative_path/path download_paper returned when the "
            "file was saved to a custom folder). ASYNC: enqueues the job and returns a "
            "task_id immediately — parse + vectorization finish in 1-2 minutes in the "
            "background; track via check_task_status(task_id); completion is announced to "
            "the user automatically. Returns JSON {\"ok\":true,\"data\":{task_id,paper_name,status}}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paper_name": {
                    "type": "string",
                    "description": "Paper name (arxiv_id from download_paper, or name from search_papers list).",
                },
                "pdf_path": {
                    "type": "string",
                    "description": "Workspace path (relative or absolute) to the PDF to process. Empty = auto-search data/uploads and data/downloads.",
                    "default": "",
                },
            },
            "required": ["paper_name"],
        },
        source="builtin",
        annotations={"readOnlyHint": False, "idempotentHint": False},
    ),
    ToolDef(
        name="check_task_status",
        description=(
            "Query a background task's status by task_id: pending / running / done / "
            "failed, plus progress / error / result once finished. Background tasks "
            "(the ingest enqueued by ingest_paper) run asynchronously — call this "
            "when the user asks '入库了吗 / 任务进展如何 / done yet?'. "
            "Returns JSON {\"ok\":true,\"data\":{task_id,paper_name,status,progress,error,result}}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id returned by ingest_paper.",
                },
            },
            "required": ["task_id"],
        },
        source="builtin",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
    ToolDef(
        name="check_paper",
        description=(
            "Check the LOCAL state of a paper: 'indexed' (already searchable in the "
            "library), 'downloaded_not_indexed' (local artifact exists — output dir "
            "parsed or PDF on disk; matches carry pdf_path & detail), "
            "or 'absent' (not local). Fast read-only check: Redis catalog + filesystem "
            "scan of data/uploads & data/downloads, NO network. ALWAYS call this FIRST "
            "when the user wants to save/import a paper. Empty term returns the full "
            "local snapshot grouped by binary state (indexed/not_indexed) with derived "
            "detail (parsed/raw). "
            "Returns JSON {\"ok\":true,\"data\":{term, state, matches}}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "Paper name/keyword as the user calls it. Empty = full local snapshot.",
                    "default": "",
                },
            },
        },
        source="builtin",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
]


# ---- provider ----

class BuiltinProvider(ToolProvider):
    """论文核心工具 — 来自内部 API。不需要 MCP 包装。"""

    name = "builtin"

    def __init__(self):
        self._tool_map = {
            td.name: td for td in BUILTIN_TOOLDEFS
        }
        self._func_map = {
            "search_papers": search_papers,
            "fetch_content": fetch_content,
            "download_paper": download_paper,
            "ingest_paper": ingest_paper,
            "check_paper": check_paper,
            "check_task_status": check_task_status,
        }

    async def list_tools(self) -> list[ToolDef]:
        return list(self._tool_map.values())

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if name not in self._func_map:
            raise KeyError(f"Builtin tool '{name}' not found")
        # 权限门：destructive 工具（download_paper/ingest_paper）按 role 判定。
        # 未授权角色在此拦截，返回结构化错误供 LLM 恢复，而非直接抛异常。
        td = self._tool_map.get(name)
        if td is not None and not tool_allowed(td.annotations):
            return _err(403,
                f"Action '{name}' is not authorized for the current role.",
                "Tell the user this action requires higher authorization. Do not retry.",
                error_type="permission_denied")
        fn = self._func_map[name]
        return await fn.ainvoke(arguments)
