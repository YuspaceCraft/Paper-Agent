"""
test_vector_store.py — VectorStore adapter 接口契约测试
=======================================================

验证适配器接口契约 + Chroma 实现的基本行为。
ponytail: 不需要真实 Embedding API，用零向量模拟。

运行: python indexer/tests/test_vector_store.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _temp_chroma_dir():
    """ponytail: tempdir that survives Chroma's Windows file locks on cleanup."""
    path = tempfile.mkdtemp(prefix="chroma_test_")
    return path


def _cleanup_chroma_dir(path: str):
    """Try to clean up, ignoring Windows file-lock errors from Chroma."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass  # Chroma holds file locks briefly on Windows


def _make_unit(cid: str, content_hash="abc123", content_type="body",
               section_path="Introduction", retrieval_text="test text",
               generation_text="test generation text") -> dict:
    return {
        "chunk_id": cid,
        "content_hash": content_hash,
        "retrieval_text": retrieval_text,
        "generation_text": generation_text,
        "dense_vector": [0.1] * 16,  # 小维度测试向量
        "sparse_keywords": ["test", "example"],
        "hyde_questions": [],
        "metadata": {
            "content_type": content_type,
            "section_path": section_path,
            "token_count": 10,
            "ref_ids": ["[1]", "[2]"],
            "bound_elements": ["formula_001"],
            "pii_flagged": False,
        },
        "pii_flagged": False,
        "pii_findings": [],
        "schema_version": "1.0",
        "source_chunk": {},
    }


# ================================================================
# Chroma Tests
# ================================================================

def _run_store(collection_name, callback):
    """ponytail: 创建临时 Chroma store，执行测试，安全清理。"""
    try:
        from indexer.vector_store import ChromaVectorStore
    except ImportError:
        print("[SKIP] Chroma not available")
        return

    tmpdir = _temp_chroma_dir()
    try:
        store = ChromaVectorStore(persist_dir=tmpdir, collection_name=collection_name)
        callback(store)
    finally:
        # 释放 Chroma client 引用，允许 Windows 清理文件锁
        store._client = None
        store._collection = None
        _cleanup_chroma_dir(tmpdir)


def test_chroma_upsert_and_count():
    """基本 upsert + count 流程。"""
    def do(store):
        assert store.count() == 0
        units = [_make_unit(f"chunk_{i:04d}") for i in range(5)]
        n = store.upsert(units)
        assert n == 5, f"Should upsert 5 units, got {n}"
        assert store.count() == 5
        n2 = store.upsert(units)
        assert n2 == 5
        assert store.count() == 5, f"Should still be 5 after re-upsert, got {store.count()}"
    _run_store("test_collection", do)


def test_chroma_delete():
    """delete 后 count 减少。"""
    def do(store):
        units = [_make_unit(f"chunk_{i:04d}") for i in range(5)]
        store.upsert(units)
        n = store.delete(["chunk_0000", "chunk_0001"])
        assert n == 2
        assert store.count() == 3
    _run_store("test_delete", do)


def test_chroma_search():
    """向量检索返回结果。"""
    def do(store):
        units = [_make_unit(f"chunk_{i:04d}") for i in range(5)]
        store.upsert(units)
        results = store.search([0.1] * 16, top_k=3)
        assert len(results) == 3, f"Should return 3 results, got {len(results)}"
        assert "chunk_id" in results[0]
        assert "score" in results[0]
    _run_store("test_search", do)


def test_chroma_search_with_filter():
    """metadata 过滤。"""
    def do(store):
        units = [
            _make_unit("chunk_0000", content_type="body"),
            _make_unit("chunk_0001", content_type="body"),
            _make_unit("chunk_0002", content_type="reference"),
        ]
        store.upsert(units)
        results = store.search([0.1] * 16, filters={"content_type": "reference"}, top_k=5)
        assert len(results) > 0
    _run_store("test_filter", do)


def test_chroma_get_existing_hashes():
    """获取已有 hash 映射。"""
    def do(store):
        units = [
            _make_unit("chunk_0000", content_hash="hash_0000"),
            _make_unit("chunk_0001", content_hash="hash_0001"),
        ]
        store.upsert(units)
        hashes = store.get_existing_hashes()
        assert "chunk_0000" in hashes
        assert hashes["chunk_0000"] == "hash_0000"
    _run_store("test_hashes", do)


# ================================================================
# Interface compliance: mock adapter must match ABC
# ================================================================

def test_adapter_interface_compliance():
    """验证适配器接口的 5 个方法签名。"""
    from indexer.vector_store import VectorStoreAdapter

    # 验证 ABC 定义了所有必需方法
    required = {"upsert", "search", "delete", "count", "get_existing_hashes"}
    abstract_methods = set()
    for name in required:
        if hasattr(VectorStoreAdapter, name):
            attr = getattr(VectorStoreAdapter, name)
            if hasattr(attr, "__isabstractmethod__") and attr.__isabstractmethod__:
                abstract_methods.add(name)

    assert abstract_methods == required, \
        f"Missing abstract methods: {required - abstract_methods}"


# ================================================================
# Run
# ================================================================

if __name__ == "__main__":
    test_adapter_interface_compliance()
    test_chroma_upsert_and_count()
    test_chroma_delete()
    test_chroma_search()
    test_chroma_search_with_filter()
    test_chroma_get_existing_hashes()
    print("[OK] All vector_store tests passed")
