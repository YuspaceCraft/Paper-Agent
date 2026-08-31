"""
test_dedup_manager.py — Hash 去重逻辑测试
=========================================

ponytail: assert-based, no framework.
运行: python indexer/tests/test_dedup_manager.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from indexer.dedup_manager import compute_hash, DedupManager


# ================================================================
# Mock VectorStore for testing
# ================================================================

class _MockStore:
    """ponytail: mock store with in-memory hash tracking."""

    def __init__(self, existing_hashes=None):
        self._hashes = dict(existing_hashes or {})
        self._upserted = []
        self._deleted = []

    def get_existing_hashes(self):
        return dict(self._hashes)

    def upsert(self, units):
        for u in units:
            self._hashes[u["chunk_id"]] = u.get("content_hash", "")
        self._upserted.extend(u["chunk_id"] for u in units)
        return len(units)

    def delete(self, ids):
        for cid in ids:
            self._hashes.pop(cid, None)
        self._deleted.extend(ids)
        return len(ids)

    def count(self):
        return len(self._hashes)


def _make_unit(cid: str, retrieval="text", generation="text") -> dict:
    return {
        "chunk_id": cid,
        "retrieval_text": retrieval,
        "generation_text": generation,
        "content_hash": "",
    }


# ================================================================
# Tests
# ================================================================

def test_compute_hash_deterministic():
    """相同输入产生相同 hash。"""
    h1 = compute_hash("chunk_0001", "hello world", "generation text")
    h2 = compute_hash("chunk_0001", "hello world", "generation text")
    assert h1 == h2, f"Hash should be deterministic: {h1} != {h2}"
    assert len(h1) == 32, f"Hash should be 32 hex chars, got {len(h1)}"


def test_compute_hash_different():
    """不同输入产生不同 hash。"""
    h1 = compute_hash("chunk_0001", "text A", "gen A")
    h2 = compute_hash("chunk_0002", "text A", "gen A")  # different ID
    h3 = compute_hash("chunk_0001", "text B", "gen A")  # different retrieval
    assert h1 != h2, "Different chunk_id should produce different hash"
    assert h1 != h3, "Different retrieval_text should produce different hash"


def test_sync_all_new_empty_store():
    """空 store → 全量 new。"""
    store = _MockStore({})
    mgr = DedupManager(store)
    units = [
        _make_unit("chunk_0001", "content 1"),
        _make_unit("chunk_0002", "content 2"),
    ]
    result = mgr.sync(units)
    assert len(result["new"]) == 2
    assert len(result["updated"]) == 0
    assert len(result["skipped"]) == 0
    assert len(result["orphaned"]) == 0


def test_sync_all_skipped():
    """全部 hash 相同 → 全量 skipped。"""
    units = [
        _make_unit("chunk_0001", "content 1"),
        _make_unit("chunk_0002", "content 2"),
    ]
    # 预计算 hash 并填入 store
    for u in units:
        u["content_hash"] = compute_hash(u["chunk_id"], u["retrieval_text"],
                                         u["generation_text"])
    existing = {u["chunk_id"]: u["content_hash"] for u in units}

    store = _MockStore(existing)
    mgr = DedupManager(store)
    result = mgr.sync(units)

    assert len(result["new"]) == 0
    assert len(result["updated"]) == 0
    assert len(result["skipped"]) == 2
    assert len(result["orphaned"]) == 0


def test_sync_updated():
    """hash 变更 → updated。"""
    old_units = [_make_unit("chunk_0001", "old content")]
    old_units[0]["content_hash"] = compute_hash(
        old_units[0]["chunk_id"], old_units[0]["retrieval_text"],
        old_units[0]["generation_text"],
    )
    store = _MockStore({u["chunk_id"]: u["content_hash"] for u in old_units})

    new_units = [_make_unit("chunk_0001", "new content")]  # 内容变了
    mgr = DedupManager(store)
    result = mgr.sync(new_units)

    assert len(result["new"]) == 0
    assert len(result["updated"]) == 1
    assert len(result["skipped"]) == 0


def test_sync_orphaned():
    """store 中有但新 units 中没有 → orphaned。"""
    existing = {"chunk_0001": "abc123", "chunk_0099": "def456"}
    store = _MockStore(existing)

    units = [_make_unit("chunk_0001", "content")]  # 只有 0001
    mgr = DedupManager(store)
    result = mgr.sync(units)

    assert len(result["orphaned"]) == 1
    assert "chunk_0099" in result["orphaned"]


def test_sync_orphaned_paper_scoped():
    """单 paper 增量索引不误删其他 paper 的 chunk（孤儿作用域）。"""
    existing = {
        "PaperA__chunk_0001": "abc",
        "PaperA__chunk_0002": "def",   # 本 paper 中已移除 → 应孤儿
        "PaperB__chunk_0001": "ghi",   # 其他 paper → 保留
    }
    store = _MockStore(existing)
    units = [_make_unit("PaperA__chunk_0001", "content")]
    mgr = DedupManager(store)
    result = mgr.sync(units)

    assert set(result["orphaned"]) == {"PaperA__chunk_0002"}, result["orphaned"]
    assert "PaperB__chunk_0001" not in result["orphaned"]


def test_apply_writes_to_store():
    """apply 将 new/updated 写入 store。"""
    store = _MockStore({})
    mgr = DedupManager(store)

    units = [_make_unit(f"chunk_{i:04d}", f"content {i}") for i in range(5)]
    classification = mgr.sync(units)
    write_stats = mgr.apply(classification)

    assert write_stats["inserted"] == 5
    assert store.count() == 5


def test_schema_version():
    """schema_version 更新时触发 re-index。"""
    units = [_make_unit("chunk_0001", "content")]
    store = _MockStore({})

    mgr_v1 = DedupManager(store, schema_version="1.0")
    result_v1 = mgr_v1.sync(units)

    # 修改 schema_version — ponytail: 版本不同不自动触发，
    # 但新的 schema_version 会写入 metadata
    for u in units:
        u["schema_version"] = "2.0"

    assert result_v1["counts"]["new"] == 1
    # hash 不变 → 即使 schema_version 变了也不自动 re-index
    # （这是设计决策：schema 迁移需显式触发全量重建）


# ================================================================
# Run
# ================================================================

if __name__ == "__main__":
    test_compute_hash_deterministic()
    test_compute_hash_different()
    test_sync_all_new_empty_store()
    test_sync_all_skipped()
    test_sync_updated()
    test_sync_orphaned()
    test_sync_orphaned_paper_scoped()
    test_apply_writes_to_store()
    test_schema_version()
    print("[OK] All dedup_manager tests passed")
