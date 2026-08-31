"""
catalog.py — 论文目录（Redis 单一数据源）
============================================

本地知识库的「有哪些论文」答案只从这里来。每个文件的元数据以结构化 JSON
存 Redis，与向量库（Qdrant）保持一致性：

  - 原子入库（解析+向量化一次完成）
              → register_indexed()  写完整元数据 + 去重键 + indexed=true，单次写入，无中间态
  - 仅解析（非原子兼容流 /api/pdf/process）
              → register_paper()    写元数据，indexed=false
  - 向量入库回填/修复 → mark_indexed() / patch_paper()（CLI、reconcile 对账）
  - 列论文    → list_papers()     只查 Redis，不碰向量库
  - 内容检索  → 走 retrieval 层（与本模块无关）

论文对外状态语义（方案 B，收敛为两类）:
  - "indexed"      = 目录 indexed=true（chunks 已进向量库，可检索）
  - "not_indexed"  = 其余一切；是否为 parsed/raw（有解析产物/仅有 PDF）由
                    API 层从文件系统派生，不作为终态维护。

Redis 键设计（沿用 dedup: 前缀，兼容存量数据）:
  dedup:papers         → set of paper names
  dedup:paper:{name}   → JSON 完整元数据
  dedup:hash:{sha256}  → paper_name（内容去重）
  dedup:doi:{doi}      → paper_name（DOI 去重）

JSON 冷备份: eval_output/paper_registry.json（Redis down 时兜底）
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "eval_output" / "paper_registry.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Redis connection (lazy, shared) ----

_redis: Any = None
_redis_ok: bool | None = None  # tri-state: None=untested, True=alive, False=dead


def _get_redis():
    """Lazy Redis connection, RESP2 for old Redis compatibility."""
    global _redis, _redis_ok
    if _redis_ok is False:
        return None
    if _redis is not None:
        try:
            _redis.ping()
            _redis_ok = True
            return _redis
        except Exception:
            _redis_ok = False
            _redis = None
            return None
    try:
        import redis as _redis_lib
        parsed = urlparse(REDIS_URL)
        _redis = _redis_lib.Redis(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 6379,
            protocol=2,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        if parsed.password:
            _redis.execute_command("AUTH", unquote(parsed.password))
        _redis.ping()
        _redis_ok = True
        return _redis
    except Exception:
        _redis_ok = False
        _redis = None
        return None


# ---- JSON cold backup ----

def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- public API ----

def register_paper(
    paper_name: str,
    metadata: dict,
    page_count: int = 0,
    chunk_count: int = 0,
    indexed: bool = False,
    indexed_chunk_count: int = 0,
) -> dict:
    """写入论文完整元数据 + 去重键。返回写入的 entry。

    indexed=False（默认）= 仅解析产物（/api/pdf/process 兼容流）或原子入库的
    中间阶段；原子入库应在收尾时用 register_indexed() 单次写完。
    """
    entry = {
        "paper_name": paper_name,
        "title": metadata.get("title", "") or "",
        "authors": metadata.get("authors", "") or "",
        "doi": metadata.get("doi", "") or "",
        "year": metadata.get("year", "") or "",
        "arxiv_id": metadata.get("arxiv_id", "") or "",
        "filename": metadata.get("filename", "") or "",
        "content_hash": metadata.get("content_hash", "") or "",
        "page_count": page_count,
        "chunk_count": chunk_count,
        "indexed": indexed,
        "indexed_chunk_count": indexed_chunk_count,
        "processed_at": _now_iso(),
        "indexed_at": _now_iso() if indexed else "",
    }
    content_hash = entry["content_hash"]
    doi = entry["doi"]

    r = _get_redis()
    if r:
        try:
            pipe = r.pipeline()
            pipe.set(f"dedup:paper:{paper_name}", json.dumps(entry, ensure_ascii=False))
            pipe.sadd("dedup:papers", paper_name)
            if content_hash:
                pipe.set(f"dedup:hash:{content_hash}", paper_name)
            if doi:
                pipe.set(f"dedup:doi:{doi}", paper_name)
            pipe.execute()
        except Exception:
            pass  # Redis error → cold backup still written below

    reg = _load_registry()
    reg.setdefault("papers", {})[paper_name] = entry
    if content_hash:
        reg.setdefault("by_hash", {})[content_hash] = paper_name
    if doi:
        reg.setdefault("by_doi", {})[doi] = paper_name
    _save_registry(reg)
    return entry


def register_indexed(
    paper_name: str,
    metadata: dict,
    page_count: int = 0,
    chunk_count: int = 0,
    indexed_chunk_count: int = 0,
) -> dict:
    """原子入库收尾：注册（存在则更新）并把 indexed 一步置真。

    替代「register_paper() 后再 mark_indexed()」的两段式：入库路径只应在此
    函数里写目录，避免中间态「已解析未入库」被误当终态，也避免成功入库后
    indexed 标志漏置（历史 desync 的根因之一）。
    """
    return register_paper(
        paper_name, metadata,
        page_count=page_count, chunk_count=chunk_count,
        indexed=True, indexed_chunk_count=indexed_chunk_count,
    )


def patch_paper(paper_name: str, **fields) -> dict | None:
    """原地更新已注册论文的指定字段，不触碰元数据/去重键（reconcile 用）。

    返回更新后的 entry；论文未注册返回 None。
    """
    entry = get_paper(paper_name)
    if entry is None:
        return None
    entry.update(fields)
    r = _get_redis()
    if r:
        try:
            r.set(f"dedup:paper:{paper_name}", json.dumps(entry, ensure_ascii=False))
        except Exception:
            pass
    reg = _load_registry()
    if paper_name in reg.get("papers", {}):
        reg["papers"][paper_name] = entry
        _save_registry(reg)
    return entry


def is_duplicate(content_hash: str = "", doi: str = "") -> dict | None:
    """SHA256 / DOI 去重检查。命中返回已存在的 entry，否则 None。"""
    r = _get_redis()
    if r:
        for key in ([f"dedup:hash:{content_hash}"] if content_hash else []) + \
                   ([f"dedup:doi:{doi}"] if doi else []):
            name = r.get(key)
            if name:
                meta = r.get(f"dedup:paper:{name}")
                if meta:
                    return json.loads(meta)
        return None

    reg = _load_registry()
    by_hash = reg.get("by_hash", {})
    by_doi = reg.get("by_doi", {})
    if content_hash and content_hash in by_hash:
        return reg.get("papers", {}).get(by_hash[content_hash])
    if doi and doi in by_doi:
        return reg.get("papers", {}).get(by_doi[doi])
    return None


def deregister_paper(paper_name: str) -> bool:
    """从 Redis + JSON 备份删除论文。返回是否找到。"""
    found = False

    r = _get_redis()
    if r:
        try:
            meta_raw = r.get(f"dedup:paper:{paper_name}")
            if meta_raw:
                meta = json.loads(meta_raw)
                pipe = r.pipeline()
                pipe.delete(f"dedup:paper:{paper_name}")
                pipe.srem("dedup:papers", paper_name)
                if h := meta.get("content_hash"):
                    pipe.delete(f"dedup:hash:{h}")
                if d := meta.get("doi"):
                    pipe.delete(f"dedup:doi:{d}")
                pipe.execute()
                found = True
        except Exception:
            pass

    if REGISTRY_PATH.exists():
        reg = _load_registry()
        if paper_name in reg.get("papers", {}):
            meta = reg["papers"][paper_name]
            if h := meta.get("content_hash"):
                reg.get("by_hash", {}).pop(h, None)
            if d := meta.get("doi"):
                reg.get("by_doi", {}).pop(d, None)
            del reg["papers"][paper_name]
            _save_registry(reg)
            found = True

    return found


def get_paper(paper_name: str) -> dict | None:
    """取单篇论文元数据。"""
    r = _get_redis()
    if r:
        raw = r.get(f"dedup:paper:{paper_name}")
        if raw:
            return json.loads(raw)
        return None
    return _load_registry().get("papers", {}).get(paper_name)


def list_papers() -> list[dict]:
    """列出全部论文（仅 Redis）。返回完整元数据列表。"""
    r = _get_redis()
    if r:
        papers = []
        for name in r.smembers("dedup:papers"):
            raw = r.get(f"dedup:paper:{name}")
            meta = json.loads(raw) if raw else {}
            meta.setdefault("paper_name", name)  # 老条目 JSON 缺 paper_name，用 set 键补
            papers.append(meta)
        return papers
    return [
        {**meta, "paper_name": name}
        for name, meta in _load_registry().get("papers", {}).items()
    ]


def mark_indexed(paper_name: str, indexed_chunk_count: int) -> None:
    """标记论文已写入向量库（Qdrant）。indexed=true + 计数 + 时间戳。

    若 paper 尚未注册（例如 CLI 直接对未走 pdf 管道的 rag_chunks.json 建索引），
    建最小 entry 以维持目录完整。
    """
    entry = get_paper(paper_name)
    if entry is None:
        entry = register_paper(paper_name, metadata={}, page_count=0, chunk_count=indexed_chunk_count)

    entry["paper_name"] = paper_name
    entry["indexed"] = True
    entry["indexed_chunk_count"] = indexed_chunk_count
    entry["indexed_at"] = _now_iso()

    r = _get_redis()
    if r:
        try:
            r.set(f"dedup:paper:{paper_name}", json.dumps(entry, ensure_ascii=False))
        except Exception:
            pass

    reg = _load_registry()
    if paper_name in reg.get("papers", {}):
        reg["papers"][paper_name] = entry
        _save_registry(reg)


def search_by_metadata(author: str = "", year: str = "", keyword: str = "") -> list[dict]:
    """按元数据过滤论文（author 子串 / year / title 关键词）。"""
    author = (author or "").lower()
    year = year or ""
    keyword = (keyword or "").lower()
    out = []
    for meta in list_papers():
        if author and author not in (meta.get("authors", "") or "").lower():
            continue
        if year and str(meta.get("year", "")) != year:
            continue
        if keyword and keyword not in (meta.get("title", "") or "").lower():
            continue
        out.append(meta)
    return out


def sync_to_filesystem(output_dir: str | Path) -> int:
    """删除 output 目录已不存在（文件被清理）的 stale 条目。返回删除数。"""
    out = Path(output_dir)
    stale = [
        m.get("paper_name", "") for m in list_papers()
        if m.get("paper_name") and not (out / m["paper_name"]).is_dir()
    ]
    for name in stale:
        deregister_paper(name)
    return len(stale)


# ---- self-check ----

def _selfcheck() -> int:
    """注册 → 查重 → mark_indexed → list/search → deregister 往返。"""
    r = _get_redis()
    if r is None:
        print("[catalog] Redis unavailable — self-check skipped")
        return 0

    name = "__catalog_selftest__"
    meta = {"title": "Self Test Paper", "authors": "A. Test",
            "doi": "10.0000/selftest", "year": "2024", "content_hash": "f" * 64}
    register_paper(name, meta, page_count=3, chunk_count=7)

    dup = is_duplicate(content_hash="f" * 64)
    assert dup and dup["paper_name"] == name, "is_duplicate should hit by hash"
    assert dup["year"] == "2024" and dup["page_count"] == 3, "full metadata round-trip"

    mark_indexed(name, 7)
    got = get_paper(name)
    assert got["indexed"] is True and got["indexed_chunk_count"] == 7, "mark_indexed"

    assert any(m["paper_name"] == name for m in list_papers()), "list_papers"
    assert any(m["paper_name"] == name for m in search_by_metadata(year="2024")), "search_by_metadata year"

    assert deregister_paper(name) is True, "deregister"
    assert get_paper(name) is None, "removed after deregister"
    print("[catalog] self-check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
