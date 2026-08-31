"""
reader.py — Paper reading & navigation endpoints.

Thin wrappers over pdf_pipeline output artifacts (rag_chunks.json, raw.md).
No business logic — pure file read + format.

GET  /api/reader/{paper}/chunks/{chunk_id}              — chunk detail
GET  /api/reader/{paper}/chunks/{chunk_id}/context      — T8: neighbor chunks
GET  /api/reader/{paper}/sections                       — list sections
GET  /api/reader/{paper}/sections/{section_name}        — T9: chunks in section
GET  /api/reader/{paper}/abstract                       — T12: abstract from raw.md
GET  /api/reader/papers                                 — T11: metadata search
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from indexer import catalog

router = APIRouter(prefix="/api/reader", tags=["Reader"])

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "pdf_pipeline" / "output"
_PROJECT_ROOT = OUTPUT_DIR.parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"
DOWNLOAD_DIR = _PROJECT_ROOT / "data" / "downloads"
DATA_ROOT = _PROJECT_ROOT / "data"  # data/ 根目录的裸 PDF 也纳入本地快照（避免「明明存在却报 absent」）


def _resolve_paper_name(paper_name: str) -> str:
    """Resolve paper name, trying hyphens↔underscores swap then alphanumeric fuzzy match.

    Handles: exact match → -/_ swap → alphanumeric-only containment (for title→directory
    mismatches like 'Title: Subtitle' vs 'Title_Subtitle_Shortened').
    """
    if (OUTPUT_DIR / paper_name).is_dir():
        return paper_name
    # Try swapping hyphens and underscores
    swapped = paper_name.replace("-", "_") if "-" in paper_name else paper_name.replace("_", "-")
    if (OUTPUT_DIR / swapped).is_dir():
        return swapped
    # Fuzzy: strip all non-alphanumeric, compare containment (≥4 chars, aligned with
    # agent/nodes.py:_paper_matches threshold)
    norm_q = re.sub(r'[^a-zA-Z0-9]', '', paper_name).lower()
    if len(norm_q) < 4:
        return paper_name  # too short, don't guess
    best: tuple[str, float] | None = None
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir():
            continue
        norm_d = re.sub(r'[^a-zA-Z0-9]', '', d.name).lower()
        # Substring containment in either direction is a strong signal —
        # accept regardless of character-overlap score (aligned with nodes.py behavior)
        if norm_q in norm_d or norm_d in norm_q:
            score = len(norm_q) / max(len(norm_d), 1)
            if best is None or score > best[1]:
                best = (d.name, score)
    if best is not None:
        return best[0]
    return paper_name  # original, will 404 downstream


# ---- helpers ----

def _load_chunks(paper_name: str) -> list[dict]:
    name = _resolve_paper_name(paper_name)
    path = OUTPUT_DIR / name / "rag_chunks.json"
    if not path.exists():
        raise HTTPException(404, f"Paper '{paper_name}' not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("chunks", [])


def _normalize_chunk_id(chunk_id: str, paper_name: str) -> str:
    """Strip {paper_name}__ prefix if present (compat with all_rag_chunks.json IDs).
    Also handles hyphens↔underscores variation in the prefix."""
    # Try exact prefix
    prefix = f"{paper_name}__"
    if chunk_id.startswith(prefix):
        return chunk_id[len(prefix):]
    # Try swapped prefix
    swapped = paper_name.replace("-", "_") if "-" in paper_name else paper_name.replace("_", "-")
    alt = f"{swapped}__"
    if chunk_id.startswith(alt):
        return chunk_id[len(alt):]
    return chunk_id


def _chunks_index(paper_name: str) -> dict[str, dict]:
    """Lazy index chunks by chunk_id. Cached per call, not worth a global cache."""
    chunks = _load_chunks(paper_name)
    return {c["chunk_id"]: c for c in chunks}


def _read_raw_md(paper_name: str) -> str:
    name = _resolve_paper_name(paper_name)
    path = OUTPUT_DIR / name / "raw.md"
    if not path.exists():
        raise HTTPException(404, f"raw.md not found for '{paper_name}'")
    return path.read_text(encoding="utf-8")


# ---- T8: chunk context (neighbor navigation) ----

@router.get("/{paper_name}/chunks/{chunk_id}/context")
async def get_chunk_context(
    paper_name: str,
    chunk_id: str,
    window: int = Query(default=2, ge=1, le=10, description="Neighbor window size"),
):
    """Return a chunk with its N preceding and N following neighbor chunks.

    Uses prev_chunk_id / next_chunk_id chains from rag_chunks.json.
    """
    chunk_id = _normalize_chunk_id(chunk_id, paper_name)
    idx = _chunks_index(paper_name)
    if chunk_id not in idx:
        raise HTTPException(404, f"Chunk '{chunk_id}' not found in '{paper_name}'")

    # Walk backward
    prev_chunks: list[dict] = []
    cur = chunk_id
    for _ in range(window):
        prev_id = idx[cur].get("prev_chunk_id", "")
        if not prev_id or prev_id not in idx:
            break
        prev_chunks.insert(0, idx[prev_id])
        cur = prev_id

    # Walk forward
    next_chunks: list[dict] = []
    cur = chunk_id
    for _ in range(window):
        next_id = idx[cur].get("next_chunk_id", "")
        if not next_id or next_id not in idx:
            break
        next_chunks.append(idx[next_id])
        cur = next_id

    return {
        "paper_name": paper_name,
        "center": _chunk_summary(idx[chunk_id]),
        "prev": [_chunk_summary(c) for c in prev_chunks],
        "next": [_chunk_summary(c) for c in next_chunks],
        "window": window,
    }


# ---- T9: section navigation ----

# Mapping: digit/ordinal → Roman numeral
_SECTION_ORDINAL_TO_ROMAN: dict[str, str] = {
    '1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V',
    '6': 'VI', '7': 'VII', '8': 'VIII', '9': 'IX', '10': 'X',
    '1st': 'I', '2nd': 'II', '3rd': 'III', '4th': 'IV', '5th': 'V',
}


def _extract_roman_headings(paper_name: str) -> list[tuple[int, str]]:
    """Parse raw.md for Roman-numeral headings, return [(number, heading_text), ...].

    E.g. [(2, 'II. RELATED WORK'), (3, 'III. METHODOLOGY'), ...]
    Number is derived from the Roman numeral value.
    """
    raw = _read_raw_md(paper_name)
    headings: list[tuple[int, str]] = []
    roman_values: dict[str, int] = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                                     'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
    for m in re.finditer(r'^#{1,3}\s+([IVX]+)\.\s+(.+)$', raw, re.MULTILINE):
        roman = m.group(1).upper()
        heading = m.group(0).lstrip('#').strip()
        if roman in roman_values:
            headings.append((roman_values[roman], heading))
    return headings


_CN_NUMERAL: dict[str, int] = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}


def _parse_section_ordinal(section_name: str) -> int | None:
    """Extract section number from a query like '3', 'third', '第3章', '第三'."""
    key = section_name.strip().lower()
    # Pure digit
    if key.isdigit():
        return int(key)
    # Chinese ordinal: 第N章, 第三章, 第三个, 第三章节...
    m = re.match(r'第\s*(\d+|[一二三四五六七八九十]+)', key)
    if m:
        cn = m.group(1)
        return _CN_NUMERAL.get(cn) or (int(cn) if cn.isdigit() else None)
    # Bare Chinese numeral
    if key in _CN_NUMERAL:
        return _CN_NUMERAL[key]
    # English ordinal word
    word_map = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10}
    if key in word_map:
        return word_map[key]
    # 1st, 2nd, 3rd...
    ordinal_map = {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5}
    if key in ordinal_map:
        return ordinal_map[key]
    return None


def _parse_roman_prefix(section_name: str) -> int | None:
    """Extract section number from a Roman-numeral prefix like 'IV.', 'III. EXPERIMENTS'."""
    m = re.match(r'^([IVX]+)\.?\s*', section_name.strip(), re.IGNORECASE)
    if not m:
        return None
    roman_values: dict[str, int] = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                                     'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
    return roman_values.get(m.group(1).upper())


@router.get("/{paper_name}/sections")
async def list_sections(paper_name: str):
    """List all sections with chunk counts."""
    chunks = _load_chunks(paper_name)
    sections: dict[str, int] = {}
    for c in chunks:
        sec = c.get("section_path", "") or "(no section)"
        sections[sec] = sections.get(sec, 0) + 1
    return {
        "paper_name": paper_name,
        "sections": [{"name": k, "chunk_count": v} for k, v in sections.items()],
    }


@router.get("/{paper_name}/sections/{section_name:path}")
async def get_paper_section(paper_name: str, section_name: str):
    """Return all chunks in a section, with multi-level fallback matching.

    Level 1: case-insensitive substring match on section_path.
    Level 2: ordinal → Roman numeral word-boundary match (avoid false positives
              like 'IV' matching 'quantitative').
    Level 3: ordinal → raw.md heading extraction → positional match against
              the Nth main heading plus its sub-sections by document order.
    """
    chunks = _load_chunks(paper_name)
    q = section_name.lower()

    # ---- Level 1: substring match ----
    matched = [
        _chunk_summary(c) for c in chunks
        if q in (c.get("section_path", "") or "").lower()
    ]

    # Try to parse the section query as an ordinal (digit, word, Chinese, or Roman)
    ordinal = _parse_section_ordinal(section_name) or _parse_roman_prefix(section_name)

    # ---- Level 2: ordinal → Roman word-boundary ----
    if not matched and ordinal is not None:
        roman = _SECTION_ORDINAL_TO_ROMAN.get(str(ordinal))
        if roman:
            # Word-boundary regex to avoid 'IV' matching 'quantitative'
            pattern = re.compile(r'\b' + re.escape(roman) + r'\b', re.IGNORECASE)
            matched = [
                _chunk_summary(c) for c in chunks
                if pattern.search(c.get("section_path", "") or "")
            ]

    # ---- Level 3: ordinal → raw.md heading → match chunks by position ----
    if not matched and ordinal is not None:
        try:
            matched = _match_section_by_position(paper_name, ordinal, chunks)
        except HTTPException:
            pass  # raw.md missing, fall through to error

    # ---- Post-processing: expand top-level sections to include sub-sections ----
    # When the matched chunks include a Roman-numeral top-level section
    # (e.g. "IV. EXPERIMENTAL"), also include sub-section chunks
    # (A. Datasets, B. Setup, etc.) that follow it in document order.
    # Level 1/2 only match by section_path string — they miss child sections
    # because the hierarchy is flattened (e.g. "A. Datasets" doesn't contain
    # the parent heading "IV. EXPERIMENTAL").
    if matched and ordinal is not None:
        roman = _SECTION_ORDINAL_TO_ROMAN.get(str(ordinal))
        if roman:
            try:
                expanded = _match_section_by_position(paper_name, ordinal, chunks)
                if expanded:
                    matched = expanded
            except HTTPException:
                pass  # raw.md missing — keep substring/ordinal match result

    if not matched:
        # Build a helpful error: list available sections + suggest alternatives
        available = sorted(set(
            c.get("section_path", "") or "(empty)"
            for c in chunks
        ))
        # If the query is an ordinal, suggest the raw.md heading if found
        hint = ""
        ordinal = _parse_section_ordinal(section_name) or _parse_roman_prefix(section_name)
        if ordinal is not None:
            try:
                headings = _extract_roman_headings(paper_name)
                target_idx = next((i for i, (n, _) in enumerate(headings) if n == ordinal), None)
                if target_idx is not None:
                    target = headings[target_idx][1]
                    hint = (
                        f" Raw.md has heading '{target}', but no chunks reference it "
                        f"directly — its content was split into sub-sections. "
                        f"Try one of these section names: {available[:8]}."
                    )
            except HTTPException:
                pass
        if not hint:
            hint = f" Available sections: {available[:10]}."
        raise HTTPException(404,
            f"No section matching '{section_name}' in '{paper_name}'.{hint}")

    return {
        "paper_name": paper_name,
        "section_query": section_name,
        "chunk_count": len(matched),
        "chunks": matched,
    }


def _match_section_by_position(
    paper_name: str, ordinal: int, chunks: list[dict]
) -> list[dict]:
    """Match chunks by the Nth Roman-numeral heading position.

    When a heading (e.g. 'IV. EXPERIMENTS') is missing from section_paths,
    searches chunk content for the heading text to find the entry point,
    then includes all subsequent chunks until the next Roman heading.
    """
    headings = _extract_roman_headings(paper_name)
    # headings is sparse: [(2, 'II. ...'), (3, 'III. ...'), ...] — search by number
    target_idx = next((i for i, (n, _) in enumerate(headings) if n == ordinal), None)
    if target_idx is None:
        return []

    target_heading = headings[target_idx][1]   # e.g. "IV. EXPERIMENTS"
    next_heading = headings[target_idx + 1][1] if target_idx + 1 < len(headings) else None

    result: list[dict] = []
    in_target = False

    for c in chunks:
        content = c.get("content", "") or ""
        sp = c.get("section_path", "") or ""

        # Check if this chunk contains the target heading text
        # (search both content and section_path — chunks may have the heading
        #  in their section_path metadata but not in the body text)
        if not in_target and (
            target_heading in content
            or target_heading.lower() in sp.lower()
        ):
            in_target = True
            result.append(_chunk_summary(c))
            continue

        if in_target:
            # Stop at next Roman heading in content or section_path
            if next_heading and (
                next_heading in content
                or next_heading.lower() in sp.lower()
            ):
                break
            result.append(_chunk_summary(c))

    return result


# ---- T12: paper abstract ----

@router.get("/{paper_name}/abstract")
async def get_paper_abstract(paper_name: str):
    """Extract abstract from raw.md."""
    md = _read_raw_md(paper_name)

    # Extract abstract. Two formats in raw.md:
    #   1. "## Abstract" heading (legacy)
    #   2. inline "Abstract -..." label at line start (docling output — the norm)
    #  = PUA space glyph docling emits for some PDF fonts (not matched by \s).
    abstract_source = "not_found"
    m = re.search(r'##\s*Abstract[^\n]*\n+(.*?)(?=\n##\s)', md, re.DOTALL | re.IGNORECASE)
    if m:
        abstract_source = "heading"
        abstract = m.group(1).strip()
    else:
        m = re.search(
            r'(?m)^[\s]*Abstract\s*[-–—:]\s*(.*?)(?=\n\s*\n|\Z)',
            md, re.DOTALL | re.IGNORECASE,
        )
        if m:
            abstract_source = "inline"
            abstract = re.sub(r'[\s]+', ' ', m.group(1)).strip()
        else:
            abstract = ""

    # Also grab the paper title (first ## line)
    title = ""
    title_m = re.match(r'^##\s*(.+)', md)
    if title_m:
        title = title_m.group(1).strip()

    # Authors line (after title, before Abstract). Format varies:
    #   "A, B and C" (commas) or "A and B" (no comma); sometimes preceded by
    #   an "<!-- image -->" placeholder. Skip md/html artifacts, stop at Abstract.
    authors = ""
    for p in md.split("\n\n")[:5]:
        s = p.strip()
        if s.startswith("##") or s.startswith("<!--"):
            continue
        if "Abstract" in s:
            break
        if s and len(s) < 300 and any(c.isalpha() for c in s):
            authors = s
            break

    # Sections list (first-occurrence order from rag_chunks.json)
    sections: list[dict] = []
    try:
        seen: set[str] = set()
        for c in _load_chunks(paper_name):
            sp = c.get("section_path", "") or "(no section)"
            if sp not in seen:
                seen.add(sp)
                sections.append({"name": sp, "chunk_count": 1})
            else:
                # ponytail: bump count for existing entry
                for s in sections:
                    if s["name"] == sp:
                        s["chunk_count"] += 1
                        break
    except HTTPException:
        pass  # rag_chunks.json missing → empty sections

    return {
        "paper_name": paper_name,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "abstract_source": abstract_source,
        "sections": sections,
    }


# ---- T-local: tri-state local snapshot (agent decision ladder) ----
# Redis(已入库) → 文件系统(已有 PDF) → 都没有才需要网络(arXiv)。只读，无日志副作用。

def _local_pdf_path(name: str) -> str:
    """Locate a paper's source PDF in uploads/downloads/data-root, workspace-relative."""
    for d in (UPLOAD_DIR, DOWNLOAD_DIR, DATA_ROOT):
        p = d / f"{name}.pdf"
        if p.exists():
            try:
                return p.relative_to(_PROJECT_ROOT).as_posix()
            except ValueError:
                return str(p)
    return ""


def _paper_diagnosis(name: str, indexed: bool) -> str:
    """派生诊断 detail（非持久状态）：indexed | parsed | raw | ""。

    有解析产物（output 目录）→ parsed；仅有本地 PDF → raw；否则 ""。
    """
    if indexed:
        return "indexed"
    if (OUTPUT_DIR / name).is_dir() and any((OUTPUT_DIR / name).iterdir()):
        return "parsed"
    if _local_pdf_path(name):
        return "raw"
    return ""


@router.get("/local-papers")
async def list_local_papers():
    """Local paper snapshot for the agent's decision ladder（方案 B，两类语义）.

    state（对外两类）:
      "indexed"     — in Redis catalog with indexed=true (vector DB searchable)
      "not_indexed" — 其余一切；parsed/raw 由 detail 字段派生给出
    detail（派生诊断，非持久终态）:
      "indexed" | "parsed"（有解析产物）| "raw"（仅有本地 PDF）| ""
    has_pdf is false when an entry has no live PDF (e.g. parsed output whose
    source PDF was deleted) — ingest must not be handed a dead path.
    """
    snapshot: list[dict] = []
    seen: set[str] = set()

    # 1) Redis catalog first (cold backup when Redis is down) — O(1) "已入库?" check
    for meta in catalog.list_papers():
        name = meta.get("paper_name", "")
        if not name:
            continue
        seen.add(name)
        indexed = bool(meta.get("indexed"))  # cold-backup entries may lack the key
        pdf_path = _local_pdf_path(name)
        snapshot.append({
            "paper_name": name,
            "state": "indexed" if indexed else "not_indexed",
            "detail": _paper_diagnosis(name, indexed),
            "location": "catalog",
            "pdf_path": pdf_path,
            "has_pdf": bool(pdf_path),
        })

    # 2) parsed output dirs not registered in the catalog → not_indexed (name is the dir)
    if OUTPUT_DIR.is_dir():
        for paper_dir in sorted(OUTPUT_DIR.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name in seen:
                continue
            seen.add(paper_dir.name)
            pdf_path = _local_pdf_path(paper_dir.name)
            snapshot.append({
                "paper_name": paper_dir.name, "state": "not_indexed", "detail": "parsed",
                "location": "output", "pdf_path": pdf_path,
                "has_pdf": bool(pdf_path),
            })

    # 3) raw PDFs in uploads/downloads not covered above (same guards as
    #    pdf.py._scan_raw_pdfs: seen dedup + parsed-dir skip + missing-dir continue)
    for d, loc in ((UPLOAD_DIR, "uploads"), (DOWNLOAD_DIR, "downloads")):
        if not d.is_dir():
            continue
        for pdf_file in sorted(d.iterdir()):
            if not pdf_file.suffix.lower() == ".pdf":
                continue
            stem = pdf_file.stem
            if stem in seen:
                continue
            seen.add(stem)
            try:
                rel = pdf_file.relative_to(_PROJECT_ROOT).as_posix()
            except ValueError:
                rel = str(pdf_file)
            snapshot.append({
                "paper_name": stem, "state": "not_indexed", "detail": "raw",
                "location": loc, "pdf_path": rel, "has_pdf": True,
            })

    # 4) raw PDFs at data/ root not covered above（uploads/downloads 为子目录，
    #    仅扫本层文件，避免与第 3 步重复；同类 seen dedup）
    if DATA_ROOT.is_dir():
        for pdf_file in sorted(DATA_ROOT.iterdir()):
            if not pdf_file.is_file() or pdf_file.suffix.lower() != ".pdf":
                continue
            stem = pdf_file.stem
            if stem in seen:
                continue
            seen.add(stem)
            try:
                rel = pdf_file.relative_to(_PROJECT_ROOT).as_posix()
            except ValueError:
                rel = str(pdf_file)
            snapshot.append({
                "paper_name": stem, "state": "not_indexed", "detail": "raw",
                "location": "data", "pdf_path": rel, "has_pdf": True,
            })

    return {"papers": snapshot, "count": len(snapshot)}


# ---- T11: metadata search ----

@router.get("/papers")
async def search_by_metadata(
    author: str = Query(default="", description="Author name substring match"),
    year: str = Query(default="", description="Publication year"),
    keyword: str = Query(default="", description="Keyword in title substring match"),
):
    """Search papers by metadata (author, year, title keyword).

    Primary source: Redis catalog (indexer.catalog) — falls back to JSON registry.
    """
    # ponytail: resolve Query defaults to strings (FastAPI params are Query objects in tests)
    author = author.default if hasattr(author, 'default') else str(author or "")
    year = year.default if hasattr(year, 'default') else str(year or "")
    keyword = keyword.default if hasattr(keyword, 'default') else str(keyword or "")

    papers = catalog.search_by_metadata(author=author, year=year, keyword=keyword)
    return {"count": len(papers), "papers": papers}


# ---- shared helpers ----

def _chunk_summary(chunk: dict, max_content: int = 3000) -> dict[str, Any]:
    """Return a trimmed chunk suitable for agent consumption.

    max_content caps the content field per chunk. 3000 is enough for a full
    academic paragraph while keeping multi-chunk sections within context budget.
    """
    raw = chunk.get("content", "")
    return {
        "chunk_id": chunk["chunk_id"],
        "content_type": chunk.get("content_type", ""),
        "section_path": chunk.get("section_path", ""),
        "token_count": chunk.get("token_count", 0),
        "prev_chunk_id": chunk.get("prev_chunk_id", ""),
        "next_chunk_id": chunk.get("next_chunk_id", ""),
        "bound_elements": chunk.get("bound_elements", []),
        "page_no": chunk.get("metadata", {}).get("page_no", 0)
                   or chunk.get("metadata", {}).get("page_start", 0),
        "content": raw[:max_content] if len(raw) > max_content else raw,
    }
