"""
context_assembler.py — 上下文组装器 (Small-to-Big)
=================================================

读取 rag_chunks.json，利用 prev_chunk_id / next_chunk_id 邻居链
动态拼接上下文，生成 retrieval_text（短）和 generation_text（长）。

策略:
  - retrieval_text: [KEYWORDS] + chunk 正文，截断至 retrieval_max_tokens
  - generation_text: 前序 chunk 尾部 + chunk 正文 + 后序 chunk 头部，
    截断至 generation_max_tokens（语义边界截断而非硬切）
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import ContextAssemblyConfig

# 语义边界标记（优先在这些位置截断）
_SEMANTIC_BOUNDARIES = [
    r'\n\n',           # 段落边界（最优）
    r'\n(?=#)',        # 标题前
    r'(?<=[.!?])\s+',  # 句子边界
    r'(?<=[。！？])\s*',  # 中文句子边界
]


def _count_tokens_heuristic(text: str) -> int:
    """ponytail: 快速 token 估算，无 tiktoken 依赖。"""
    if not text:
        return 0
    chinese = len(re.findall(r'[一-鿿]', text))
    other = len(text) - chinese
    return int(chinese * 0.6 + other * 0.25)


def _truncate_by_semantic_boundary(text: str, max_tokens: int) -> str:
    """在语义边界处截断文本，保留完整语义单元。

    优先在段落边界截断，其次在句子边界，最后在词边界。
    返回截断后的文本（不添加省略号，保留原始完整性）。
    """
    if _count_tokens_heuristic(text) <= max_tokens:
        return text

    # 从 max_tokens 对应的字符位置开始向前查找语义边界
    char_budget = int(max_tokens * 2.8)  # 近似: 1 token ≈ 2.8 chars (中英混合)
    if char_budget >= len(text):
        return text

    # 在 char_budget 向前 200 chars 的窗口内找最佳截断点
    search_start = max(0, char_budget - 200)
    snippet = text[search_start:char_budget + 100]

    best_cut = char_budget  # fallback: 硬截断
    best_priority = 999

    for priority, pattern in enumerate(_SEMANTIC_BOUNDARIES):
        for m in re.finditer(pattern, snippet):
            cut_pos = search_start + m.start()
            if cut_pos <= char_budget and priority < best_priority:
                best_cut = cut_pos
                best_priority = priority
                break  # 当前优先级找到即可，不需要更多
        if best_priority == priority:
            break  # 已找到当前最优优先级的截断点

    return text[:best_cut].rstrip()


class ContextAssembler:
    """Small-to-Big 上下文组装器。

    从 rag_chunks.json 加载所有 chunk，根据邻居链组装长短文本。
    """

    def __init__(self, config: ContextAssemblyConfig | None = None):
        self.config = config or ContextAssemblyConfig()

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def load_chunks(self, rag_chunks_path: str) -> list[dict]:
        """加载 rag_chunks.json，返回原始 chunk 字典列表。"""
        data = json.loads(Path(rag_chunks_path).read_text(encoding="utf-8"))
        return data.get("chunks", [])

    def assemble(self, chunks: list[dict]) -> list[dict]:
        """对所有 body chunk 执行 Small-to-Big 组装。

        返回 list of dict，每项包含:
          - chunk_id, retrieval_text, generation_text
          - metadata (原始 chunk 元数据)
          - source_chunk (原始 chunk 完整字典)

        非 body chunk (reference, summary 等) 直接原文作为两条文本。
        """
        # 构建 chunk_id → chunk 的快速查找表
        chunk_map: dict[str, dict] = {c["chunk_id"]: c for c in chunks}

        results = []
        for ch in chunks:
            assembled = self._assemble_one(ch, chunk_map)
            results.append(assembled)

        return results

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _assemble_one(self, chunk: dict, chunk_map: dict[str, dict]) -> dict:
        """为单个 chunk 组装 retrieval_text 和 generation_text。"""
        cid = chunk["chunk_id"]
        ctype = chunk.get("content_type", "body")
        content = chunk.get("content", "")

        # retrieval_text: chunk 自身内容（含 [KEYWORDS]），截断
        retrieval = _truncate_by_semantic_boundary(
            content, self.config.retrieval_max_tokens,
        )

        # generation_text: 邻居上下文扩展
        if not self.config.include_keywords:
            retrieval = self._strip_keywords(retrieval)

        generation = content  # base

        if ctype == "body":
            generation = self._build_generation_text(chunk, chunk_map)

        # 截断 generation_text
        generation = _truncate_by_semantic_boundary(
            generation, self.config.generation_max_tokens,
        )

        return {
            "chunk_id": cid,
            "retrieval_text": retrieval,
            "generation_text": generation,
            "content_type": ctype,
            "section_path": chunk.get("section_path", ""),
            "token_count": chunk.get("token_count", 0),
            "ref_ids": chunk.get("ref_ids", []),
            "bound_elements": chunk.get("bound_elements", []),
            "metadata": chunk.get("metadata", {}),
            "source_chunk": chunk,
        }

    def _build_generation_text(self, chunk: dict, chunk_map: dict[str, dict]) -> str:
        """构建含邻居上下文的 generation_text。

        格式: [prev_ctx]\n\n[chunk_content]\n\n[next_ctx]
        """
        parts = []
        window = self.config.neighbor_window

        # 前序 chunk 尾部
        prev_content = self._collect_neighbor_tails(chunk, chunk_map, window)
        if prev_content:
            parts.append(prev_content)

        # 本 chunk
        parts.append(chunk.get("content", ""))

        # 后序 chunk 头部
        next_content = self._collect_neighbor_heads(chunk, chunk_map, window)
        if next_content:
            parts.append(next_content)

        return "\n\n".join(parts)

    def _collect_neighbor_tails(
        self, chunk: dict, chunk_map: dict[str, dict], window: int,
    ) -> str:
        """沿 prev_chunk_id 链收集前序 chunk 的尾部。"""
        tails = []
        current = chunk
        for _ in range(window):
            prev_id = current.get("prev_chunk_id", "")
            if not prev_id or prev_id not in chunk_map:
                break
            prev = chunk_map[prev_id]
            prev_text = prev.get("content", "")
            # 取尾部 ~150 tokens 作为上下文
            tail = _truncate_by_semantic_boundary(prev_text[::-1], 150)[::-1]
            # 更好的方式：从末尾取字符
            tail = self._tail_tokens(prev_text, 150)
            if tail.strip():
                tails.append(tail)
            current = prev

        return "\n---\n".join(reversed(tails))

    def _collect_neighbor_heads(
        self, chunk: dict, chunk_map: dict[str, dict], window: int,
    ) -> str:
        """沿 next_chunk_id 链收集后序 chunk 的头部。"""
        heads = []
        current = chunk
        for _ in range(window):
            next_id = current.get("next_chunk_id", "")
            if not next_id or next_id not in chunk_map:
                break
            nxt = chunk_map[next_id]
            nxt_text = nxt.get("content", "")
            head = _truncate_by_semantic_boundary(nxt_text, 150)
            if head.strip():
                heads.append(head)
            current = nxt

        return "\n---\n".join(heads)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _strip_keywords(text: str) -> str:
        """移除 [KEYWORDS: ...] 前缀。"""
        return re.sub(r'^\[KEYWORDS:\s*[^\]]+\]\s*\n*', '', text)

    @staticmethod
    def _tail_tokens(text: str, max_tokens: int) -> str:
        """从文本末尾提取最多 max_tokens 个 token 的内容。"""
        if _count_tokens_heuristic(text) <= max_tokens:
            return text
        char_budget = int(max_tokens * 3.5)
        return text[-char_budget:]
