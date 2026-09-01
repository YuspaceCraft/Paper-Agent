"""
resolution.py — Deterministic reference resolution layer.

Resolves fuzzy paper/section references BEFORE they reach the agent LLM.
"RSCC" → "Diffusion-RSCC_..." with confidence level, no LLM guessing.

Placed between understand_node and agent_node in the graph:
    START → understand → resolve → agent ↔ tools → synthesize → END

Pass-through by default: no focus_papers → no resolution. Resolution failure
→ resolved stays empty, agent discovers on its own. Never blocks the graph.
"""

from __future__ import annotations

import re
import os

import httpx

from .library_api import (
    api_is_down as _api_down,
    api_mark_down as _api_mark_down,
    api_timeout as _api_timeout,
)

API = os.environ.get("AGENT_API_BASE", "http://127.0.0.1:8000")

# (P2) 已删除 parent→subagent 的 contextvar 隐藏通道（`_resolved_ctx`）：
#   1. 不可见——父代理 model 不知道自己的 task 会被注入 resolved；
#   2. 非零状态——与 system prompt 反复声明的 zero-state 矛盾；
#   3. 无生命周期——从不 reset，长驻循环里旧 turn 的 resolved 会泄漏。
# 必要上下文由父代理按 AGENT_SYSTEM「Delegation Priority」约定显式写进 task。
# state["resolved"] 仍是 checkpointed 官方状态，仅删侧通道。

# ---- confidence levels ----

EXACT = "EXACT"    # normalized strings equal
HIGH = "HIGH"      # substring, query ≥ 8 chars
MEDIUM = "MEDIUM"  # substring, query 4-7 chars
LOW = "LOW"        # substring, query < 4 chars (noisy)
NONE = "NONE"      # no match in library

# ---- normalization (shared with nodes.py) ----

def normalize_name(name: str) -> str:
    """Normalize paper name for comparison: rm-net == RM_Net == rm net."""
    return name.replace("-", "_").replace(" ", "_").lower()


def canonicalize(name: str) -> str:
    """Aggressive name canonicalization for tri-state local matching.

    Where normalize_name keeps punctuation/underscores, canonicalize drops
    everything that isn't a letter or digit (and strips CJK + brackets). This
    bridges the disk-stem cleaning pipelines (pdf.py._sanitize_paper_name strips
    brackets/CJK/truncates; builtin_provider._sanitize_dl_name keeps brackets) so
    "Diffusion-RSCC_ Diffusion Probabilistic..." == "Diffusion-RSCC" ==
    "Diffusion_RSCC_Model_for_..." for containment purposes.
    """
    if not name:
        return ""
    s = re.sub(r"[()（）\[\]【】]", "", name)
    # CJK ranges aligned with web/api/routers/pdf.py _NAME_BLACKLIST
    s = re.sub(r"[一-鿿㐀-䶿]", "", s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# 决策塔优先级（方案 B：state 只分 indexed / not_indexed，parsed/raw 由 detail 派生）：
#   已入库 > 有解析产物(parsed) > 仅本地 PDF(raw) > catalog 占位(无本地产物)
def _local_rank(p: dict) -> int:
    if p.get("state") == "indexed":
        return 3
    return {"parsed": 2, "raw": 1}.get(p.get("detail", ""), 0)


def match_local_state(term: str, papers: list[dict]) -> dict:
    """Deterministically match a user term against a local-papers snapshot
    (endpoint /api/reader/local-papers entries; `state` 二类 + `detail` 派生诊断).

    No LLM: canonicalize both sides → exact or substring containment (the
    contained side must be ≥4 chars to avoid short false positives, e.g. "cv").
    Matches ranked indexed > parsed (detail) > raw (detail); aggregate:
      "indexed" if any indexed match, else "downloaded_not_indexed" if any
      local artifact (detail parsed/raw or has_pdf) matches, else "absent".
    """
    key = canonicalize(term)
    if not key or not papers:
        return {"state": "absent", "matches": []}

    matches: list[dict] = []
    for p in papers:
        name = p.get("paper_name", "")
        pkey = canonicalize(name)
        if not pkey:
            continue
        hit = (
            key == pkey
            or (len(key) >= 4 and key in pkey)      # containment only for long keys
            or (len(pkey) >= 4 and pkey in key)     # term is the full name + more
            or (len(key) >= 2 and pkey.startswith(key))  # short acronym prefix, e.g. "SRN"
        )
        if not hit:
            continue
        matches.append({
            "paper_name": name,
            "state": p.get("state", "not_indexed"),
            "detail": p.get("detail", ""),
            "location": p.get("location", ""),
            "pdf_path": p.get("pdf_path", ""),
            "has_pdf": bool(p.get("has_pdf")),
        })

    matches.sort(key=_local_rank, reverse=True)
    matches = matches[:4]  # cap for prompt size

    if any(m["state"] == "indexed" for m in matches):
        state = "indexed"
    elif matches and any(
        m["detail"] in ("parsed", "raw") or m["has_pdf"] for m in matches
    ):
        state = "downloaded_not_indexed"
    else:
        state = "absent"
    return {"state": state, "matches": matches}


# ---- paper fetching (with TTL cache) ----

# ponytail: single TTL cache shared by resolve_node and agent_node pre-flight.
# Both hit /api/reader/papers — one cache eliminates the duplicate HTTP call.
_papers_cache: list[dict] | None = None
_papers_cache_ts: float = 0
_PAPERS_CACHE_TTL = 30  # seconds


async def fetch_papers(force: bool = False) -> list[dict]:
    """Fetch all available papers from the reader API. Cached for 30s.

    后端不可达（熔断窗口内）直接返回缓存/空，不在 resolve 阶段空等网络。
    """
    global _papers_cache, _papers_cache_ts
    import time
    now = time.time()
    if not force and _papers_cache is not None and now - _papers_cache_ts < _PAPERS_CACHE_TTL:
        return _papers_cache
    if _api_down(API):
        return _papers_cache if _papers_cache is not None else []
    try:
        async with httpx.AsyncClient(timeout=_api_timeout()) as c:
            r = await c.get(f"{API}/api/reader/papers")
            r.raise_for_status()
            _papers_cache = r.json().get("papers", [])
            _papers_cache_ts = now
            return _papers_cache
    except Exception:
        _api_mark_down(API)
        return _papers_cache if _papers_cache is not None else []


# ---- paper matching ----

def match_paper(query: str, papers: list[dict]) -> dict:
    """Match a user query against available paper names.

    Returns a dict with query, match, confidence, level, and match_type.
    Confidence levels guide how the agent should treat the match.
    """
    if not papers:
        return {
            "query": query, "match": None, "confidence": 0.0,
            "level": NONE, "match_type": "none",
        }

    q = normalize_name(query)

    # Level 1: exact match after normalization
    for p in papers:
        name = p.get("paper_name", p.get("name", ""))
        if normalize_name(name) == q:
            return {
                "query": query, "match": name, "confidence": 1.0,
                "level": EXACT, "match_type": "exact",
            }

    # Level 2: substring containment (query in name, or name in query)
    candidates: list[tuple[str, float, bool]] = []  # (name, score, query_in_name)
    for p in papers:
        name = p.get("paper_name", p.get("name", ""))
        n = normalize_name(name)
        if q in n:
            candidates.append((name, len(q) / max(len(n), 1), True))
        elif n in q:
            candidates.append((name, len(n) / max(len(q), 1), False))

    if candidates:
        # Prefer: query-is-substring-of-name > name-is-substring-of-query
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        name, score, _ = candidates[0]
        level = HIGH if len(q) >= 8 else MEDIUM if len(q) >= 4 else LOW
        return {
            "query": query, "match": name, "confidence": score,
            "level": level, "match_type": "substring",
        }

    return {
        "query": query, "match": None, "confidence": 0.0,
        "level": NONE, "match_type": "none",
    }


# ---- section reference extraction ----

# Chinese numerals → int
_CN_NUMERAL: dict[str, int] = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}

# English ordinal words → int
_EN_ORDINAL: dict[str, int] = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
}


def extract_section_ref(query: str) -> dict | None:
    """Extract section/chapter ordinal from a user query.

    Returns {"ordinal": int, "text": str, "confidence": float} or None.
    Handles Chinese (第三章, 第三章节) and English (section 3, third chapter).
    """
    # Chinese: 第N章, 第N节, 第三章, 第三个章节, etc.
    m = re.search(r'第\s*(\d+|[一二三四五六七八九十]+)\s*个?\s*(章|节|章节)', query)
    if m:
        cn = m.group(1)
        ordinal = _CN_NUMERAL.get(cn) or (int(cn) if cn.isdigit() else None)
        if ordinal is not None:
            return {"ordinal": ordinal, "text": m.group(0), "confidence": 0.9,
                    "source": "chinese_ordinal"}

    # English: "section 3", "chapter IV", "third section"
    m = re.search(
        r'(section|chapter|part)\s+(\d+|[IVX]+)\b',
        query, re.IGNORECASE,
    )
    if m:
        num_str = m.group(2).upper()
        if num_str.isdigit():
            return {"ordinal": int(num_str), "text": m.group(0), "confidence": 0.9,
                    "source": "english_ordinal"}
        # Roman numeral
        roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                     'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
        if num_str in roman_map:
            return {"ordinal": roman_map[num_str], "text": m.group(0), "confidence": 0.85,
                    "source": "roman_numeral"}

    # English ordinal word: "third chapter", "second section"
    m = re.search(
        r'(' + '|'.join(_EN_ORDINAL) + r')\s+(section|chapter|part)',
        query, re.IGNORECASE,
    )
    if m:
        ordinal = _EN_ORDINAL.get(m.group(1).lower())
        if ordinal is not None:
            return {"ordinal": ordinal, "text": m.group(0), "confidence": 0.85,
                    "source": "english_word"}

    # Bare ordinal word at end of query: "the third"
    m = re.search(r'\b(' + '|'.join(_EN_ORDINAL) + r')\s*$', query, re.IGNORECASE)
    if m:
        ordinal = _EN_ORDINAL.get(m.group(1).lower())
        if ordinal is not None:
            return {"ordinal": ordinal, "text": m.group(0), "confidence": 0.7,
                    "source": "bare_ordinal"}

    return None


# ---- resolve_node (LangGraph node function) ----

async def resolve_node(state: dict, _config=None) -> dict:
    """Resolve fuzzy references against the paper library.

    Reads: state["focus_papers"], state["entities"], last user message.
    Writes: state["resolved"] with matched papers + optional section ref.

    Entity bag is a resolution candidate too: understand reliably reports paper
    names through `entities` (qwen-plus does), so matching only focus_papers
    left `resolved` empty for exactly the queries that need it (comparison /
    multi-paper → decide_mode plan, fallback_plan). Entities that match nothing
    are dropped silently — no false positives leak into hints.

    Pass-through: returns empty resolved when nothing to resolve.
    """
    candidates: list[str] = []
    for p in state.get("focus_papers", []) or []:
        if isinstance(p, str) and p.strip() and p.strip() not in candidates:
            candidates.append(p.strip())
    for e in state.get("entities", []) or []:
        if isinstance(e, str) and e.strip() and e.strip() not in candidates:
            candidates.append(e.strip())

    if not candidates:
        return {"resolved": {"papers": [], "section": None}}

    papers = await fetch_papers()

    # De-dupe by the matched library name; candidate order preserved.
    resolved_papers: list[dict] = []
    seen: set[str] = set()
    for c in candidates:
        m = match_paper(c, papers)
        if not m.get("match") or m["match"] in seen:
            continue
        seen.add(m["match"])
        resolved_papers.append(m)

    # Extract section reference from the last human message
    section = None
    msgs = state.get("messages", [])
    if msgs:
        last = msgs[-1]
        content = last.content if hasattr(last, "content") else str(last)
        section = extract_section_ref(content)

    resolved = {
        "papers": resolved_papers,
        "section": section,
    }
    return {"resolved": resolved}
