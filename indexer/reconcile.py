"""
reconcile.py — 目录 indexed 标志 ↔ 向量库实际内容的对账
========================================================

目录（Redis/cold backup）的 `indexed` 标志是论文对外「已入库可检索」的唯一
判据，而它只在原子入库收尾（register_indexed / mark_indexed）时写入。历史
上存在两条脱节路径：

  1. 旧 schema / 旧入库代码没有写 `indexed` 字段（entry 缺键，bool(None)=False）
     → 已入库论文永久显示「未入库」；
  2. 入库失败残留 / 标志误置 → 标志与向量库实际不一致。

reconcile 以**向量库实际点数为准**滚动对账并回填：
  - 向量库有点但标志缺失/未置  → 置 indexed=true + 真实 chunk 数（修复 1）
  - 标志置真但向量库为空      → 回退 indexed=false（修复 2）
不回写向量库、不触碰元数据/去重键，纯目录侧修复。

使用：
  CLI    : C:/Users/30811/miniconda3/envs/demo/python.exe -m indexer.reconcile
  HTTP   : POST /api/index/reconcile （后台任务，结果在 task result）
  <要求>  向量库必须可达（如 /api/index/stats 正常）。
"""
from __future__ import annotations

import json
from collections import Counter

from indexer import catalog
from indexer.config import load_config


def _build_store(config_path: str = ""):
    """按 indexer/config.yaml 构建向量库适配器（与 routers/index 同构）。"""
    from indexer.vector_store import ChromaVectorStore, QdrantVectorStore

    cfg = load_config(config_path)
    vs = cfg.vector_store
    if vs.backend == "qdrant":
        return QdrantVectorStore(
            url=vs.qdrant.url, collection_name=vs.qdrant.collection_name,
        )
    return ChromaVectorStore(
        persist_dir=vs.chroma.persist_dir, collection_name=vs.chroma.collection_name,
    )


def count_chunks_per_paper(store) -> dict[str, int]:
    """滚动向量库，按 chunk_id 前缀（`{paper}__chunk_NNNN`）统计各论文点数。"""
    hashes = store.get_existing_hashes()  # {chunk_id: content_hash}
    counter: Counter[str] = Counter()
    for cid in hashes:
        counter[cid.split("__chunk_")[0]] += 1
    return dict(counter)


def reconcile(config_path: str = "") -> dict:
    """对账并回填目录 indexed 标志。返回报告（= task result 内容）。"""
    store = _build_store(config_path)
    try:
        per_paper = count_chunks_per_paper(store)
    except Exception as exc:  # 向量库不可达 → 保持现状，绝不动数据
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "checked": 0, "fixed": [], "orphans": []}

    fixed: list[dict] = []
    seen: set[str] = set()
    for meta in catalog.list_papers():
        name = (meta.get("paper_name") or "").strip()
        if not name:
            continue
        seen.add(name)
        actual = per_paper.get(name, 0)
        should_indexed = actual > 0
        cur = meta.get("indexed")
        cur_count = meta.get("indexed_chunk_count")
        # 缺键（旧 schema）或与向量库不一致 → 回填
        if ("indexed" not in meta) or (bool(cur) != should_indexed) or (cur_count != actual):
            changed = catalog.patch_paper(
                name, indexed=should_indexed, indexed_chunk_count=actual,
            )
            fixed.append({
                "paper_name": name,
                "was": cur,                       # None=旧 schema 缺键
                "was_count": cur_count,
                "indexed": should_indexed,
                "indexed_chunk_count": actual,
                "changed": changed is not None,
            })

    # 向量库有点、目录里没有的论文（孤儿点）—— 只上报，不自动注册（无元数据）
    orphans = [name for name in per_paper if name not in seen]

    return {"ok": True, "checked": len(seen), "fixed": fixed, "orphans": orphans}


if __name__ == "__main__":
    print(json.dumps(reconcile(), ensure_ascii=False, indent=2))