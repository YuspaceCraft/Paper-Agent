"""
test_context_assembler.py — Context Assembly 边界情况测试
=========================================================

ponytail: 无框架最小测试，assert-based self-check。
运行: python -m pytest indexer/tests/test_context_assembler.py
      或直接: python indexer/tests/test_context_assembler.py
"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from indexer.context_assembler import (
    ContextAssembler,
    _truncate_by_semantic_boundary,
    _count_tokens_heuristic,
)
from indexer.config import ContextAssemblyConfig


# ================================================================
# Mock data
# ================================================================

def _make_chunk(cid: str, content: str, ctype="body",
                section_path="Introduction",
                prev_id="", next_id="", **kwargs) -> dict:
    return {
        "chunk_id": cid,
        "content": content,
        "content_type": ctype,
        "section_path": section_path,
        "token_count": _count_tokens_heuristic(content),
        "ref_ids": kwargs.get("ref_ids", []),
        "bound_elements": kwargs.get("bound_elements", []),
        "prev_chunk_id": prev_id,
        "next_chunk_id": next_id,
        "parent_chunk_id": kwargs.get("parent_chunk_id", ""),
        "metadata": kwargs.get("metadata", {}),
    }


def test_truncate_short_text():
    """短文本不被截断。"""
    text = "Hello world, this is a short text."
    result = _truncate_by_semantic_boundary(text, 100)
    assert result == text, f"Short text should not be truncated: {result}"


def test_truncate_paragraph_boundary():
    """在段落边界截断。"""
    text = "First paragraph.\n\nSecond paragraph with more text that goes on and on. " * 20
    result = _truncate_by_semantic_boundary(text, 30)
    # 应在第一个 \n\n 附近截断
    assert "\n\n" not in result[-10:], \
        f"Should cut at paragraph boundary, got tail: ...{result[-50:]}"


def test_truncate_empty():
    """空文本处理。"""
    assert _truncate_by_semantic_boundary("", 100) == ""
    assert _truncate_by_semantic_boundary("   ", 100) == "   "


def test_assemble_single_chunk():
    """单 chunk 无邻居时的基本组装。"""
    assembler = ContextAssembler()
    chunks = [_make_chunk("chunk_0000", "[KEYWORDS: test, example]\nThis is test content.")]
    result = assembler.assemble(chunks)

    assert len(result) == 1
    r = result[0]
    assert r["chunk_id"] == "chunk_0000"
    assert "test content" in r["retrieval_text"]
    assert "test content" in r["generation_text"]
    assert r["source_chunk"] == chunks[0]


def test_assemble_with_neighbors():
    """有邻居链时 generation_text 应包含前后文。"""
    assembler = ContextAssembler()
    chunks = [
        _make_chunk("chunk_0000", "First paragraph of intro.",
                    prev_id="", next_id="chunk_0001"),
        _make_chunk("chunk_0001", "Middle paragraph of intro.",
                    prev_id="chunk_0000", next_id="chunk_0002"),
        _make_chunk("chunk_0002", "Last paragraph of intro.",
                    prev_id="chunk_0001", next_id=""),
    ]
    result = assembler.assemble(chunks)

    # chunk_0001 的 generation_text 应包含前后文
    middle = [r for r in result if r["chunk_id"] == "chunk_0001"][0]
    assert len(middle["generation_text"]) > len(middle["retrieval_text"]), \
        "generation_text should be longer than retrieval_text with neighbors"


def test_assemble_non_body():
    """非 body chunk (reference/summary) 不拼接邻居。"""
    assembler = ContextAssembler()
    chunks = [
        _make_chunk("chunk_0050", "Reference entry text.",
                    ctype="reference", section_path="References",
                    prev_id="chunk_0049", next_id="chunk_0051"),
    ]
    result = assembler.assemble(chunks)
    r = result[0]
    # reference chunk 的 generation 不应包含邻居上下文
    assert r["generation_text"] == r["source_chunk"]["content"], \
        "Non-body chunks should not include neighbor context in generation"


def test_neighbor_window_limit():
    """neighbor_window > 1 时能收集多层邻居。"""
    config = ContextAssemblyConfig(neighbor_window=2)
    assembler = ContextAssembler(config)
    chunks = [
        _make_chunk(f"chunk_{i:04d}", f"Content of chunk {i}.",
                    prev_id=f"chunk_{i-1:04d}" if i > 0 else "",
                    next_id=f"chunk_{i+1:04d}" if i < 4 else "")
        for i in range(5)
    ]
    result = assembler.assemble(chunks)
    # chunk_0002 应有 chunk_0000 和 chunk_0001 的前文，chunk_0003 和 chunk_0004 的后文
    mid = [r for r in result if r["chunk_id"] == "chunk_0002"][0]
    assert len(mid["generation_text"]) > len(mid["retrieval_text"]), \
        "window=2 should capture multi-level neighbors"


# ================================================================
# Token counting
# ================================================================

def test_token_count_mixed():
    """中英混合文本的 token 估算。"""
    text = "Hello 你好 World 世界"
    count = _count_tokens_heuristic(text)
    assert count > 0
    # 中文 ~0.6 token/char, 英文 ~0.25 token/char
    # "Hello " = 6 chars, "你好" = 2 chars, " World " = 7 chars, "世界" = 2 chars
    assert 4 <= count <= 12, f"Expected 4-12 tokens, got {count}"


def test_token_count_empty():
    assert _count_tokens_heuristic("") == 0
    assert _count_tokens_heuristic("   ") == 0


# ================================================================
# Keyword stripping
# ================================================================

def test_strip_keywords():
    text = "[KEYWORDS: attention, transformer, BERT]\nThe transformer architecture..."
    clean = ContextAssembler._strip_keywords(text)
    assert "[KEYWORDS:" not in clean
    assert "transformer architecture" in clean


# ================================================================
# Run
# ================================================================

if __name__ == "__main__":
    test_truncate_short_text()
    test_truncate_paragraph_boundary()
    test_truncate_empty()
    test_assemble_single_chunk()
    test_assemble_with_neighbors()
    test_assemble_non_body()
    test_neighbor_window_limit()
    test_token_count_mixed()
    test_token_count_empty()
    test_strip_keywords()
    print("[OK] All context_assembler tests passed")
