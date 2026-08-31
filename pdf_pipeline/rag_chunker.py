"""
rag_chunker.py — RAG-优化的科研文献切分与索引增强
================

针对 final_enriched.md 的四模块流水线:
  1. 结构感知切分 — LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
  2. BM25 关键词增强 — [KEYWORDS: ...] 拼接到 chunk 头部
  3. 结构化内容文本化投影 — 表格/公式/图表 → 自然语言摘要
  4. 参考文献解耦索引 — 独立切分 + 引用映射

使用方式:
  from pdf_pipeline.rag_chunker import rag_chunk_markdown

  report = rag_chunk_markdown(enriched_md, paper_metadata)
  for ch in report.chunks:
      print(ch.section_path, ch.content_type, ch.content[:100])
"""

from __future__ import annotations

import re
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ================================================================
# Token 估算 — try tiktoken, fallback to heuristic
# ================================================================

try:
    import tiktoken as _tiktoken_mod
    _TIK_ENCODER = _tiktoken_mod.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_TIK_ENCODER.encode(text))
except Exception:
    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        chinese = len(re.findall(r'[一-鿿]', text))
        other = len(text) - chinese
        return int(chinese * 0.6 + other * 0.25)


# ================================================================
# 配置
# ================================================================

@dataclass
class RAGChunkConfig:
    chunk_tokens: int = 1024
    overlap_tokens: int = 128
    min_chunk_tokens: int = 100
    max_chunk_tokens: int = 1536     # 硬上限: 1.5x chunk_tokens, 超出的强制切分
    keywords_count: int = 5

    # 转换为字符数 (近似: 1 token ≈ 3.5 英文字符)
    @property
    def chunk_chars(self) -> int:
        return int(self.chunk_tokens * 3.5)

    @property
    def overlap_chars(self) -> int:
        return int(self.overlap_tokens * 3.5)

    @property
    def min_chunk_chars(self) -> int:
        return int(self.min_chunk_tokens * 3.5)

    @property
    def max_chunk_chars(self) -> int:
        return int(self.max_chunk_tokens * 3.5)


# ================================================================
# 数据类
# ================================================================

@dataclass
class RAGChunk:
    chunk_id: str
    content: str                        # 含 [KEYWORDS: ...] 前缀
    content_type: str                   # body|table_raw|figure_desc|formula_desc|reference|table_summary|figure_summary|formula_summary
    section_path: str                   # "I. INTRODUCTION > A. Background"
    token_count: int
    ref_ids: list[str] = field(default_factory=list)
    parent_chunk_id: str = ""           # summary chunk → 所属 body chunk
    prev_chunk_id: str = ""             # 同 section 内上一个 body chunk
    next_chunk_id: str = ""             # 同 section 内下一个 body chunk
    bound_elements: list[str] = field(default_factory=list)  # 关联的 bindings element_id
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGChunkingReport:
    chunks: list[RAGChunk] = field(default_factory=list)
    total_chunks: int = 0
    quality_report: dict = field(default_factory=dict)
    config: RAGChunkConfig = field(default_factory=RAGChunkConfig)


# ================================================================
# 原子块检测与保护（表格/[FIGURE_DESC]/[FORMULA_DESC] 不可截断）
# ================================================================

# 匹配 markdown 表格（连续的 | 行）
_TABLE_RE = re.compile(
    r'(?:[ \t]*\|[^\n]+\|[ \t]*\n){2,}(?:[ \t]*\|[^\n]+\|[ \t]*\n)*',
    re.MULTILINE,
)

# [FIGURE_DESC: ...] — 单行或多行
_FIGURE_DESC_RE = re.compile(
    r'\[FIGURE_DESC:\s*[^\]]*\]',
    re.DOTALL,
)

# [FORMULA_DESC: ...]
_FORMULA_DESC_RE = re.compile(
    r'\[FORMULA_DESC:\s*[^\]]*\]',
    re.DOTALL,
)


def _find_atomic_blocks(text: str) -> list[dict]:
    """找到所有原子块，返回 [{"type": ..., "content": ..., "start": ..., "end": ...}]。"""
    blocks = []

    for m in _TABLE_RE.finditer(text):
        blocks.append({"type": "table", "content": m.group(), "start": m.start(), "end": m.end()})

    for m in _FIGURE_DESC_RE.finditer(text):
        blocks.append({"type": "figure_desc", "content": m.group(), "start": m.start(), "end": m.end()})

    for m in _FORMULA_DESC_RE.finditer(text):
        blocks.append({"type": "formula_desc", "content": m.group(), "start": m.start(), "end": m.end()})

    blocks.sort(key=lambda b: b["start"])
    # 去重叠
    filtered = []
    for b in blocks:
        if filtered and b["start"] < filtered[-1]["end"]:
            continue
        filtered.append(b)
    return filtered


def _protect_atomic_blocks(text: str, blocks: list[dict]) -> tuple[str, dict[str, dict]]:
    """原子块 → 唯一占位符。返回 (cleaned_text, {placeholder: block})。"""
    pmap = {}
    clean = text
    for i, blk in enumerate(reversed(blocks)):
        idx = len(blocks) - 1 - i
        ph = f"__ATOMIC_{idx}__"
        pmap[ph] = blk
        clean = clean.replace(blk["content"], ph, 1)
    return clean, pmap


def _restore_atomic_blocks(chunks: list[str], pmap: dict[str, dict]) -> list[str]:
    """恢复原子块占位符。"""
    result = []
    for c in chunks:
        for ph, blk in pmap.items():
            c = c.replace(ph, blk["content"])
        if c.strip():
            result.append(c)
    return result


# ================================================================
# Module 1: 结构感知切分 (LangChain MarkdownHeaderTextSplitter)
# ================================================================

# 三级标题 → metadata key
_HEADERS_TO_SPLIT = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# 子切分的分隔符优先级：结构化块边界 > 段落 > 行
_SUB_SEPARATORS = [
    "\n[FIGURE_DESC",
    "\n[FORMULA_DESC",
    "\n\n",
    "\n",
]

# LC MarkdownHeaderTextSplitter 单例
_HEADER_SPLITTER = None


def _get_header_splitter() -> MarkdownHeaderTextSplitter:
    global _HEADER_SPLITTER
    if _HEADER_SPLITTER is None:
        _HEADER_SPLITTER = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT,
            strip_headers=False,
        )
    return _HEADER_SPLITTER


def _build_section_path(metadata: dict) -> str:
    """从 MarkdownHeaderTextSplitter 的 metadata 构建面包屑路径。"""
    parts = []
    for key in ("H1", "H2", "H3"):
        if key in metadata:
            parts.append(metadata[key])
    return " > ".join(parts)


def _split_markdown_body(
    markdown: str, config: RAGChunkConfig,
) -> tuple[list[RAGChunk], str, int]:
    """
    用 LangChain 切分正文部分。

    流程:
      1. 提取原子块 → 占位符
      2. MarkdownHeaderTextSplitter → 标题感知分块
      3. 每个过大块 → RecursiveCharacterTextSplitter 子切分
      4. 恢复原子块
      5. 分离参考文献章节

    返回: (body_chunks, ref_text, next_chunk_idx)
    """
    # 1. 原子块保护
    atomic_blocks = _find_atomic_blocks(markdown)
    clean_md, pmap = _protect_atomic_blocks(markdown, atomic_blocks)

    # 2. 标题感知切分
    header_splitter = _get_header_splitter()
    lc_docs = header_splitter.split_text(clean_md)

    # 3. 子切分 + 恢复原子块
    sub_splitter = RecursiveCharacterTextSplitter(
        separators=_SUB_SEPARATORS,
        chunk_size=config.chunk_chars,
        chunk_overlap=config.overlap_chars,
        keep_separator="end",
    )

    body_chunks: list[RAGChunk] = []
    ref_text = ""
    chunk_idx = 0
    prev_chunk_content = ""

    for doc in lc_docs:
        section_path = _build_section_path(doc.metadata)
        sec_type = _classify_section_type(section_path)

        # 分离参考文献
        if sec_type == "references":
            ref_text = _restore_ref_text(doc.page_content, pmap)
            continue

        # 子切分（若块太大）
        text = doc.page_content
        if _count_tokens(text) > config.chunk_tokens:
            sub_docs = sub_splitter.split_documents([Document(page_content=text)])
            sub_texts = [d.page_content for d in sub_docs]
        else:
            sub_texts = [text]

        # 硬上限兜底（在占位符保护下切分，避免截断原子块）
        hard_texts = []
        for st in sub_texts:
            if len(st) > config.max_chunk_chars:
                hard_texts.extend(_force_split_long_chunk(st, config.max_chunk_chars))
            else:
                hard_texts.append(st)

        # 恢复原子块
        sub_texts = _restore_atomic_blocks(hard_texts, pmap)

        for st in sub_texts:
            if not st.strip():
                continue

            content_type = _detect_content_type(st)
            ref_ids = _extract_citation_ids(st)
            tokens = _count_tokens(st)

            # overlap（仅 body 段落间）
            final_content = st
            if content_type == "body" and prev_chunk_content and config.overlap_chars > 0:
                overlap = _get_overlap(prev_chunk_content, config.overlap_chars)
                if overlap:
                    final_content = overlap + "\n\n" + st

            ch = RAGChunk(
                chunk_id=f"chunk_{chunk_idx:04d}",
                content=final_content,
                content_type=content_type,
                section_path=section_path,
                token_count=tokens,
                ref_ids=ref_ids,
                metadata={
                    "paper_title": "",
                    "paper_year": "",
                    "section_type": sec_type,
                },
            )
            body_chunks.append(ch)
            chunk_idx += 1

            if content_type == "body":
                prev_chunk_content = st

            # Module 3: 为结构化内容创建文本化投影
            blk_atomics = _find_atomic_blocks(st)
            if blk_atomics:
                summary_chunks, chunk_idx = _create_textualization_chunks(
                    blk_atomics, ch.chunk_id, chunk_idx,
                )
                body_chunks.extend(summary_chunks)

    return body_chunks, ref_text, chunk_idx


def _restore_ref_text(text: str, pmap: dict[str, dict]) -> str:
    """恢复参考文献文本中的原子块占位符。"""
    for ph, blk in pmap.items():
        text = text.replace(ph, blk["content"])
    return text


def _detect_content_type(text: str) -> str:
    """判断 chunk 的内容类型。"""
    stripped = text.strip()
    if _TABLE_RE.match(stripped):
        return "table_raw"
    if _FIGURE_DESC_RE.match(stripped):
        return "figure_desc"
    if _FORMULA_DESC_RE.match(stripped):
        return "formula_desc"
    return "body"


def _classify_section_type(section_path: str) -> str:
    """从 section_path 分类章节类型。"""
    lower = section_path.lower().strip()
    cleaned = re.sub(r'^[IVX\d]+[\.、\)]\s*', '', lower)
    cleaned = re.sub(r'^[A-Z][\.、\)]\s*', '', cleaned)

    patterns = [
        ("abstract",     [r'\babstract\b', r'\b摘要\b']),
        ("introduction", [r'\bintro(duction)?\b', r'\b引言\b', r'\b绪论\b']),
        ("related_work", [r'\brelated\b', r'\bbackground\b', r'\breview\b', r'\b相关工作\b']),
        ("methods",      [r'\bmethod', r'\bapproach\b', r'\bproposed\b', r'\b方法\b', r'\b模型\b']),
        ("experiments",  [r'\bexperiment', r'\bresult', r'\bevaluation\b', r'\b实验\b', r'\b结果\b']),
        ("discussion",   [r'\bdiscussion\b', r'\banalysis\b', r'\b讨论\b', r'\b分析\b']),
        ("conclusion",   [r'\bconclusion\b', r'\bsummary\b', r'\b结论\b', r'\b总结\b']),
        ("references",   [r'\breferences?\b', r'\bbibliography\b', r'\b参考文献\b']),
    ]
    for sec_type, pat_list in patterns:
        for pat in pat_list:
            if re.search(pat, cleaned):
                return sec_type
    return "other"


# ================================================================
# Module 2: BM25 关键词增强
# ================================================================

_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'that', 'this', 'these', 'those', 'which',
    'and', 'but', 'or', 'if', 'because', 'while', 'although', 'we', 'our',
    'it', 'its', 'they', 'them', 'their', 'he', 'she', 'his', 'her',
    'using', 'used', 'use', 'based', 'also', 'et', 'al', 'via',
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
})


def _extract_abbreviations(text: str) -> list[str]:
    """提取缩写-全称对。"""
    pairs = []
    # ABBR / Full Form
    abbr_full = re.findall(
        r'\b([A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]+)?)\s*/\s*([A-Z][a-zA-Z\s]{3,40}?)(?:[,;.]|\s|$)',
        text,
    )
    for abbr, full in abbr_full:
        pairs.append(f"{abbr} / {full.strip()}")

    # (ABBR) → 回推全称
    for m in re.finditer(r'\(([A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]+)?)\)', text):
        abbr = m.group(1)
        if any(abbr in p for p in pairs):
            continue
        idx = m.start()
        if idx > 30:
            prefix = text[max(0, idx - 80):idx].strip()
            abbr_chars = list(abbr.lower())
            matched = []
            for w in reversed(prefix.split()):
                if not abbr_chars:
                    break
                if w[0].lower() == abbr_chars[-1]:
                    matched.insert(0, w)
                    abbr_chars.pop()
            if len(matched) >= 3:
                full_form = ' '.join(matched)
                if full_form.lower() != abbr.lower():
                    pairs.append(f"{abbr} / {full_form}")

    return list(dict.fromkeys(pairs))[:4]


def _extract_keywords_tf(text: str, top_n: int = 5) -> list[str]:
    """TF-based 关键词提取。"""
    clean = re.sub(r'\[(?:FORMULA_DESC|FIGURE_DESC|KEYWORDS):[^\]]+\]', ' ', text)
    clean = re.sub(r'[^a-zA-Z0-9\s一-鿿-]', ' ', clean.lower())

    words = [w.strip('-') for w in clean.split() if len(w.strip('-')) > 2]
    words = [w for w in words if w not in _STOP_WORDS]

    freq = Counter(words)
    keywords = []
    for word, _ in freq.most_common(top_n * 3):
        if len(keywords) >= top_n:
            break
        if word.isdigit():
            continue
        keywords.append(word)
    return keywords


def extract_keywords(text: str, count: int = 5) -> str:
    """提取关键词 → [KEYWORDS: kw1, kw2, ...]"""
    abbrevs = _extract_abbreviations(text)
    tf_kws = _extract_keywords_tf(text, top_n=max(2, count - len(abbrevs)))
    all_kw = abbrevs + [k for k in tf_kws if k not in ' '.join(abbrevs).lower()]
    all_kw = list(dict.fromkeys(all_kw))[:count]
    if not all_kw:
        return ""
    return f"[KEYWORDS: {', '.join(all_kw)}]"


# ================================================================
# Module 3: 结构化内容文本化投影
# ================================================================

def _summarize_table(content: str) -> str:
    """Markdown 表格 → 自然语言摘要。"""
    lines = [l.strip() for l in content.strip().split('\n') if '|' in l]
    if len(lines) < 2:
        return ""

    header_cells = [c.strip() for c in lines[0].split('|') if c.strip()]
    data_lines = lines[2:] if len(lines) > 2 else []

    if not header_cells or not data_lines:
        return ""

    n_rows, n_cols = len(data_lines), len(header_cells)

    # 识别数值列
    numeric_cols = {}
    for ci, col_name in enumerate(header_cells):
        values = []
        for row in data_lines:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if ci < len(cells):
                try:
                    v = float(cells[ci].replace(',', '').replace('%', ''))
                    values.append(v)
                except ValueError:
                    pass
        if len(values) >= n_rows * 0.5:
            numeric_cols[col_name] = values

    parts = [f"Table with {n_rows} rows and {n_cols} columns."]
    if numeric_cols:
        stats_parts = []
        for col_name, values in numeric_cols.items():
            stats_parts.append(f"{col_name} ranges {min(values):.2f} - {max(values):.2f}")
        parts.append("Key metrics: " + "; ".join(stats_parts[:5]) + ".")
    parts.append("Columns: " + ", ".join(header_cells[:8]) + ".")
    return " ".join(parts)


def _summarize_figure_desc(content: str) -> str:
    """[FIGURE_DESC: ...] → 摘要。"""
    m = re.search(r'\[FIGURE_DESC:\s*(.*?)\]', content, re.DOTALL)
    if not m:
        return ""
    desc = m.group(1).strip()
    sentences = re.split(r'(?<=[.!?])\s+', desc)
    return "Figure description: " + ' '.join(sentences[:2])


def _summarize_formula_desc(content: str) -> str:
    """[FORMULA_DESC: ...] → 摘要。"""
    m = re.search(r'\[FORMULA_DESC:\s*(.*?)\]', content, re.DOTALL)
    if not m:
        return ""
    desc = m.group(1).strip()
    sentences = re.split(r'(?<=[.!?])\s+', desc)
    return "Formula description: " + ' '.join(sentences[:2])


_ATOMIC_SUMMARIZERS = {
    "table": _summarize_table,
    "figure_desc": _summarize_figure_desc,
    "formula_desc": _summarize_formula_desc,
}

_SUMMARY_TYPE_MAP = {
    "table": "table_summary",
    "figure_desc": "figure_summary",
    "formula_desc": "formula_summary",
}


def _create_textualization_chunks(
    atomic_blocks: list[dict], parent_chunk_id: str, chunk_idx: int,
) -> tuple[list[RAGChunk], int]:
    """为原子块创建文本化投影 chunk。"""
    summaries = []
    for blk in atomic_blocks:
        summarizer = _ATOMIC_SUMMARIZERS.get(blk["type"])
        if not summarizer:
            continue
        summary_text = summarizer(blk["content"])
        if not summary_text or len(summary_text) < 20:
            continue

        ch = RAGChunk(
            chunk_id=f"chunk_{chunk_idx:04d}",
            content=summary_text,
            content_type=_SUMMARY_TYPE_MAP.get(blk["type"], f"{blk['type']}_summary"),
            section_path="",
            token_count=_count_tokens(summary_text),
            parent_chunk_id=parent_chunk_id,
            metadata={"source_type": blk["type"]},
        )
        summaries.append(ch)
        chunk_idx += 1
    return summaries, chunk_idx


# ================================================================
# bindings → chunk 关联注入
# ================================================================

def _inject_bound_elements(chunks: list[RAGChunk], bindings: dict | None) -> None:
    """
    将 bindings 中的 element_id 关联到每个 chunk。

    ponytail: 简单字符串包含检查 — 标题/引用文本出现在 chunk 内容中 → 绑定。
    摘要 chunk 通过 parent_chunk_id 继承 body chunk 的绑定。
    """
    if not bindings:
        return

    elements = bindings.get("elements", [])
    references = bindings.get("references", [])

    # 构建查找: 标题文本 → element_id，引用文本 → element_id
    caption_to_eid: dict[str, str] = {}
    for elem in elements:
        cap = elem.get("caption", "")
        if cap:
            caption_to_eid[cap] = elem["element_id"]

    ref_to_eid: dict[str, str] = {}
    for ref in references:
        rt = ref.get("ref_text", "")
        tid = ref.get("target_element_id", "")
        if rt and tid:
            ref_to_eid[rt.lower()] = tid

    # 第一遍: 直接匹配 body/raw chunk
    for ch in chunks:
        if ch.content_type in ("reference",):
            continue
        bound: list[str] = []
        content_lower = ch.content.lower()

        for cap_text, eid in caption_to_eid.items():
            if cap_text.lower() in content_lower:
                if eid not in bound:
                    bound.append(eid)

        for ref_text, eid in ref_to_eid.items():
            if ref_text in content_lower:
                if eid not in bound:
                    bound.append(eid)

        if bound:
            ch.bound_elements = bound

    # 第二遍: 摘要 chunk 继承 parent 的 bound_elements
    body_bindings: dict[str, list[str]] = {
        ch.chunk_id: ch.bound_elements for ch in chunks
    }
    for ch in chunks:
        if "summary" in ch.content_type and ch.parent_chunk_id:
            parent_bound = body_bindings.get(ch.parent_chunk_id, [])
            if parent_bound:
                ch.bound_elements = list(dict.fromkeys(parent_bound))


def _inject_page_numbers(chunks: list[RAGChunk], page_map: list[dict] | None) -> None:
    """将 page_map 中的页码信息注入到每个 chunk 的 metadata。

    ponytail: 字符串包含匹配。page_map 条目文本出现在 chunk 内容中 → 该 chunk
    属于对应页码。取所有匹配页码的 min/max 作为 page_start/page_end。
    """
    if not page_map:
        return

    for ch in chunks:
        pages: set[int] = set()
        content_lower = ch.content.lower()

        for entry in page_map:
            # 用前 80 字符作为锚点（足够唯一，减少假匹配）
            anchor = entry["text"][:80].lower()
            if len(anchor) >= 20 and anchor in content_lower:
                pages.add(entry["page_no"])

        if pages:
            ch.metadata["page_start"] = min(pages)
            ch.metadata["page_end"] = max(pages)
            # 单页 chunk 只存 page_no
            if len(pages) == 1:
                ch.metadata["page_no"] = min(pages)

    n_paged = sum(1 for ch in chunks if "page_no" in ch.metadata or "page_start" in ch.metadata)
    if n_paged:
        print(f"  [RAG-CHUNK] {n_paged}/{len(chunks)} chunks annotated with page numbers")


def _link_neighbors(chunks: list[RAGChunk]) -> None:
    """为同 section 内的 body chunk 建立双向链表。"""
    # 按 section_path 分组 body chunks
    sections: dict[str, list[int]] = {}  # section_path → [chunk indices]
    for i, ch in enumerate(chunks):
        if ch.content_type == "body" and ch.section_path:
            sections.setdefault(ch.section_path, []).append(i)

    for indices in sections.values():
        for j, idx in enumerate(indices):
            if j > 0:
                chunks[idx].prev_chunk_id = chunks[indices[j - 1]].chunk_id
            if j < len(indices) - 1:
                chunks[idx].next_chunk_id = chunks[indices[j + 1]].chunk_id


# ================================================================
# Module 4: 参考文献解耦索引
# ================================================================

# 正文引用标记: [1], [52], [3,4], [5-7]
_CITATION_RE = re.compile(r'\[(\d+(?:[,，\s-]+\d+)*)\]')


def _split_reference_entries(ref_text: str) -> list[dict]:
    """按条目切分参考文献。"""
    if not ref_text.strip():
        return []

    entries = []
    for m in re.finditer(r'(?:-\s*)?\[(\d+)\]\s', ref_text):
        entries.append({"ref_id": f"[{m.group(1)}]", "start": m.start()})

    if not entries:
        # fallback: 按空行切分
        result = []
        for p in ref_text.split('\n\n'):
            p = p.strip()
            if not p:
                continue
            m = re.match(r'(?:-\s*)?\[(\d+)\]\s+(.+)', p, re.DOTALL)
            if m:
                result.append({"ref_id": f"[{m.group(1)}]", "content": p})
        return result

    result = []
    for i, entry in enumerate(entries):
        start = entry["start"]
        end = entries[i + 1]["start"] if i + 1 < len(entries) else len(ref_text)
        content = ref_text[start:end].strip()
        author_year = _extract_ref_meta(content)
        result.append({"ref_id": entry["ref_id"], "content": content, **author_year})
    return result


def _extract_ref_meta(ref_text: str) -> dict:
    """提取参考文献的作者和年份。"""
    meta = {"authors": "", "year": ""}
    year_m = re.search(r'\b((?:19|20)\d{2})\b', ref_text)
    if year_m:
        meta["year"] = year_m.group(1)
    author_m = re.match(r'(?:-\s*)?\[\d+\]\s+([A-Z][a-z]+(?:\s+[A-Z]\.)*)', ref_text)
    if author_m:
        meta["authors"] = author_m.group(1).strip()
    return meta


def _extract_citation_ids(text: str) -> list[str]:
    """提取正文中的引用标记 ID 列表。"""
    refs = []
    for m in _CITATION_RE.finditer(text):
        for part in re.split(r'[,，\s]+', m.group(1)):
            part = part.strip()
            if '-' in part:
                try:
                    a, b = part.split('-', 1)
                    refs.extend(f"[{n}]" for n in range(int(a), int(b) + 1))
                except ValueError:
                    refs.append(f"[{part}]")
            elif part.isdigit():
                refs.append(f"[{part}]")
    return list(dict.fromkeys(refs))


def _make_ref_chunk(
    content: str, ref_ids: list[str], authors: list[str], years: list[str],
    chunk_idx: int,
) -> RAGChunk:
    """创建合并后的参考文献 chunk。"""
    return RAGChunk(
        chunk_id=f"chunk_{chunk_idx:04d}",
        content=content,
        content_type="reference",
        section_path="References",
        token_count=_count_tokens(content),
        ref_ids=ref_ids,
        metadata={
            "ref_ids": ref_ids,
            "authors": ", ".join(dict.fromkeys(authors)),
            "year": ", ".join(dict.fromkeys(years)),
            "type": "reference",
        },
    )


# ================================================================
# Overlap & Merge 辅助
# ================================================================

def _force_split_long_chunk(text: str, max_chars: int) -> list[str]:
    """硬上限切分：逐句 → 逐词，确保 chunk 不超大。"""
    if len(text) <= max_chars:
        return [text]

    # 逐句拆分
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current = current + " " + s if current else s
    if current.strip():
        chunks.append(current.strip())

    # 逐词兜底 (处理超长句)
    result = []
    for c in chunks:
        if len(c) <= max_chars:
            result.append(c)
        else:
            words = c.split()
            cur = ""
            for w in words:
                if cur and len(cur) + len(w) > max_chars:
                    result.append(cur.strip())
                    cur = w
                else:
                    cur = cur + " " + w if cur else w
            if cur.strip():
                result.append(cur.strip())
    return result


def _get_overlap(text: str, overlap_chars: int) -> str:
    """从文本末尾取 overlap_chars 字符（在词边界截断）。"""
    if len(text) <= overlap_chars:
        return text
    snippet = text[-overlap_chars:]
    cut = snippet.find(' ', max(20, len(snippet) // 3))
    if cut > 0:
        snippet = snippet[cut + 1:]
    return snippet


def _merge_small_chunks(chunks: list[RAGChunk], config: RAGChunkConfig) -> list[RAGChunk]:
    """合并过小的 chunk：body→body 合并 + 短原子块折叠到前置 body。"""
    if len(chunks) <= 1:
        return chunks

    # 需要合并的原子类型
    _FOLDABLE_TYPES = frozenset({"table_raw", "figure_desc", "formula_desc"})

    merged = []
    for ch in chunks:
        if not merged:
            merged.append(ch)
            continue

        prev = merged[-1]

        # body → body: 合并短 body 到前一个 body
        if (ch.content_type == "body" and prev.content_type == "body"
                and ch.token_count < config.min_chunk_tokens):
            prev.content = prev.content + "\n\n" + ch.content
            prev.token_count = _count_tokens(prev.content)
            prev.ref_ids = list(dict.fromkeys(prev.ref_ids + ch.ref_ids))
            prev.bound_elements = list(dict.fromkeys(prev.bound_elements + ch.bound_elements))
            continue

        # 短原子块 → 折叠到前置 body（保留内容不丢失）
        if (ch.content_type in _FOLDABLE_TYPES and prev.content_type == "body"
                and ch.token_count < config.min_chunk_tokens):
            prev.content = prev.content + "\n\n" + ch.content
            prev.token_count = _count_tokens(prev.content)
            prev.ref_ids = list(dict.fromkeys(prev.ref_ids + ch.ref_ids))
            prev.bound_elements = list(dict.fromkeys(prev.bound_elements + ch.bound_elements))
            continue

        merged.append(ch)
    return merged


# ================================================================
# 主切分流程
# ================================================================

def rag_chunk_markdown(
    markdown: str,
    paper_metadata: dict | None = None,
    bindings: dict | None = None,
    page_map: list[dict] | None = None,
    config: RAGChunkConfig | None = None,
) -> RAGChunkingReport:
    """
    RAG 优化四模块切分流水线。

    参数:
        markdown: final_enriched.md 的内容
        paper_metadata: 论文元数据 {title, authors, doi, year}
        bindings: 空间绑定数据 {elements, references}，来自 bindings.json。
                  传入后自动注入 bound_elements 到每个 chunk。
        page_map: 文本→页码映射 [{text, page_no}]，来自 page_map.json。
                  传入后自动注入 page_no 到每个 chunk。
        config: 可选的配置覆盖

    返回:
        RAGChunkingReport: 包含所有 chunk 和质检报告
    """
    if paper_metadata is None:
        paper_metadata = {}
    if config is None:
        config = RAGChunkConfig()

    # ===== Phase 1: 结构感知切分 (LangChain) =====

    body_chunks, ref_text, chunk_idx = _split_markdown_body(markdown, config)

    # 注入 paper_metadata
    for ch in body_chunks:
        ch.metadata["paper_title"] = paper_metadata.get("title", "")
        ch.metadata["paper_year"] = paper_metadata.get("year", "")

    # ===== Phase 2: BM25 关键词增强 =====

    if len(body_chunks) > 30:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(extract_keywords, ch.content, config.keywords_count): ch
                for ch in body_chunks
            }
            for fut in as_completed(futures):
                ch = futures[fut]
                try:
                    kws = fut.result()
                    if kws:
                        ch.content = kws + "\n" + ch.content
                except Exception:
                    pass
    else:
        for ch in body_chunks:
            kws = extract_keywords(ch.content, config.keywords_count)
            if kws:
                ch.content = kws + "\n" + ch.content

    # ===== Phase 3: 参考文献解耦 =====

    ref_chunks: list[RAGChunk] = []
    ref_entries = _split_reference_entries(ref_text)

    # 按 token 预算合并参考文献条目（避免每条引用一个 chunk）
    buf_content = ""
    buf_ids: list[str] = []
    buf_authors: list[str] = []
    buf_years: list[str] = []
    buf_tokens = 0

    for entry in ref_entries:
        entry_tokens = _count_tokens(entry["content"])
        # 超过单条上限则单独成 chunk（极长的参考文献条目）
        if buf_content and buf_tokens + entry_tokens > config.chunk_tokens:
            ref_chunks.append(_make_ref_chunk(
                buf_content, buf_ids, buf_authors, buf_years, chunk_idx,
            ))
            chunk_idx += 1
            buf_content = ""
            buf_ids = []
            buf_authors = []
            buf_years = []
            buf_tokens = 0

        sep = "\n\n" if buf_content else ""
        buf_content += sep + entry["content"]
        buf_ids.append(entry["ref_id"])
        if entry.get("authors"):
            buf_authors.append(entry["authors"])
        if entry.get("year"):
            buf_years.append(entry["year"])
        buf_tokens += entry_tokens

    # 最后一组
    if buf_content:
        ref_chunks.append(_make_ref_chunk(
            buf_content, buf_ids, buf_authors, buf_years, chunk_idx,
        ))
        chunk_idx += 1

    all_chunks = body_chunks + ref_chunks

    # ===== Phase 4: 合并过小 chunk =====

    all_chunks = _merge_small_chunks(all_chunks, config)

    # ===== Post-processing: bindings 注入 + 页码注入 + 邻居链 =====

    _inject_bound_elements(all_chunks, bindings)
    _inject_page_numbers(all_chunks, page_map)
    _link_neighbors(all_chunks)
    n_bound = sum(1 for ch in all_chunks if ch.bound_elements)
    if n_bound:
        print(f"  [RAG-CHUNK] {n_bound} chunks bound to {len(bindings.get('elements',[])) if bindings else 0} elements")

    # ===== Quality Check =====

    qr = _quality_check(all_chunks, ref_entries)

    report = RAGChunkingReport(
        chunks=all_chunks,
        total_chunks=len(all_chunks),
        quality_report=qr,
        config=config,
    )

    type_counts = Counter(ch.content_type for ch in all_chunks)
    print(f"  [RAG-CHUNK] {len(all_chunks)} chunks, types: {dict(type_counts)}")
    if qr["issues"]:
        print(f"  [RAG-CHUNK] WARNING: {len(qr['issues'])} quality issues (see report)")

    return report


# ================================================================
# 质检函数
# ================================================================

def _quality_check(chunks: list[RAGChunk], ref_entries: list[dict]) -> dict:
    """五项自动质检。"""
    issues = []

    # 1. TABLE 完整性
    for ch in chunks:
        if ch.content_type == "table_raw":
            lines = ch.content.strip().split('\n')
            has_sep = any(re.match(r'^\|[\s\-:|]+\|$', l) for l in lines)
            if not has_sep:
                issues.append(f"[TABLE] {ch.chunk_id}: missing separator row (possibly truncated)")

    # 2. [FORMULA_DESC] 完整性 + 对应 summary
    formula_chunks = [ch for ch in chunks if ch.content_type == "formula_desc"]
    formula_summaries = [ch for ch in chunks if ch.content_type == "formula_summary"]
    for ch in formula_chunks:
        if ch.content.count('[') != ch.content.count(']'):
            issues.append(f"[FORMULA_DESC] {ch.chunk_id}: unbalanced brackets (truncated)")
    for ch in formula_chunks:
        if not any(s.parent_chunk_id == ch.chunk_id for s in formula_summaries):
            issues.append(f"[FORMULA_DESC] {ch.chunk_id}: missing summary chunk")

    # 3. [KEYWORDS: ...] 头部
    for ch in chunks:
        if ch.content_type in ("body", "table_raw", "figure_desc", "formula_desc"):
            if not ch.content.startswith("[KEYWORDS:"):
                issues.append(f"[KEYWORDS] {ch.chunk_id}: missing keyword prefix")

    # 4. 引用映射
    body_refs = set()
    for ch in chunks:
        if ch.content_type != "reference":
            body_refs.update(ch.ref_ids)
    ref_map = {r["ref_id"] for r in ref_entries}
    unmapped = body_refs - ref_map
    orphan_refs = ref_map - body_refs
    if unmapped:
        issues.append(f"[REFS] {len(unmapped)} citation IDs not in reference list: "
                      f"{sorted(unmapped, key=lambda x: int(x[1:-1]))[:10]}")
    if orphan_refs:
        issues.append(f"[REFS] {len(orphan_refs)} references not cited in body: "
                      f"{sorted(orphan_refs, key=lambda x: int(x[1:-1]))[:10]}")

    # 5. section_path
    for ch in chunks:
        if ch.content_type in ("body", "table_raw", "figure_desc", "formula_desc"):
            if not ch.section_path:
                issues.append(f"[PATH] {ch.chunk_id}: missing section_path")
    body_paths = [ch.section_path for ch in chunks
                  if ch.content_type in ("body", "table_raw", "figure_desc", "formula_desc")
                  and ch.section_path]
    if not body_paths and any(ch.content_type == "body" for ch in chunks):
        issues.append("[PATH] All body chunks missing section_path")

    # 6. Token 分布统计
    body_tokens = sorted([c.token_count for c in chunks if c.content_type == "body"])
    all_tokens = sorted([c.token_count for c in chunks])

    def _pct(sorted_vals: list[int], p: float) -> int:
        if not sorted_vals:
            return 0
        return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]

    # 直方图 (bin = 128 tokens)
    hist_bins: dict[str, int] = {}
    for t in all_tokens:
        lo = (t // 128) * 128
        key = f"{lo}-{lo + 127}"
        hist_bins[key] = hist_bins.get(key, 0) + 1

    token_stats = {
        "body_min": min(body_tokens) if body_tokens else 0,
        "body_max": max(body_tokens) if body_tokens else 0,
        "body_median": _pct(body_tokens, 0.5),
        "body_p90": _pct(body_tokens, 0.9),
        "body_p95": _pct(body_tokens, 0.95),
        "over_max": sum(1 for t in all_tokens if t > 1536),
        "histogram": hist_bins,
    }

    # 超限告警
    oversized = [(c.chunk_id, c.token_count) for c in chunks
                 if c.token_count > 1700]
    if oversized:
        issues.append(f"[SIZE] {len(oversized)} chunks exceed 1700 tokens: "
                      f"{', '.join(f'{cid}({t}t)' for cid, t in oversized[:5])}")

    return {
        "checks_passed": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues,
        "stats": {
            "total_chunks": len(chunks),
            "body_chunks": sum(1 for c in chunks if c.content_type == "body"),
            "table_chunks": sum(1 for c in chunks if "table" in c.content_type),
            "figure_chunks": sum(1 for c in chunks if "figure" in c.content_type),
            "formula_chunks": sum(1 for c in chunks if "formula" in c.content_type),
            "reference_chunks": sum(1 for c in chunks if c.content_type == "reference"),
            "keywords_coverage": sum(1 for c in chunks if "[KEYWORDS:" in (c.content or "")),
            "ref_entries": len(ref_entries),
            "token_distribution": token_stats,
        },
    }


# ================================================================
# 便捷 API & 导出
# ================================================================

def export_rag_report(report: RAGChunkingReport, output_path: str) -> None:
    """导出 RAG chunk 报告为 JSON。"""
    data = {
        "config": {
            "chunk_tokens": report.config.chunk_tokens,
            "overlap_tokens": report.config.overlap_tokens,
        },
        "quality": report.quality_report,
        "chunks": [
            {
                "chunk_id": ch.chunk_id,
                "content_type": ch.content_type,
                "section_path": ch.section_path,
                "token_count": ch.token_count,
                "ref_ids": ch.ref_ids,
                "parent_chunk_id": ch.parent_chunk_id,
                "prev_chunk_id": ch.prev_chunk_id,
                "next_chunk_id": ch.next_chunk_id,
                "bound_elements": ch.bound_elements,
                "content": ch.content,
                "metadata": ch.metadata,
            }
            for ch in report.chunks
        ],
    }
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [RAG-CHUNK] Report exported: {output_path}")


# ================================================================
# HTML 可视化 — 检查切分质量 & 原子块完整性
# ================================================================

_CONTENT_COLORS = {
    "body":            ("#58a6ff", "#1a2332", "Body"),
    "table_raw":       ("#f39c12", "#2e2a1a", "Table Raw"),
    "figure_desc":     ("#9b59b6", "#1e1a2e", "Figure Desc"),
    "formula_desc":    ("#1abc9c", "#1a2e2a", "Formula Desc"),
    "table_summary":   ("#e67e22", "#2e241a", "Table Summary"),
    "figure_summary":  ("#8e44ad", "#1e1a2a", "Figure Summary"),
    "formula_summary": ("#16a085", "#1a2a24", "Formula Summary"),
    "reference":       ("#7f8c8d", "#1e1e2e", "Reference"),
}

_RAG_HTML_STYLE = """
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#0d1117;color:#c9d1d9;padding:20px}
  .header{text-align:center;padding:24px 0;margin-bottom:24px;border-bottom:1px solid #21262d}
  .header h1{color:#58a6ff;font-size:24px;margin-bottom:8px}
  .header .stats{color:#8b949e;font-size:13px}
  .legend{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
  .legend-item{display:flex;align-items:center;gap:6px;font-size:11px;color:#8b949e}
  .legend-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}
  .filter-bar{display:flex;gap:8px;margin:16px 0;flex-wrap:wrap}
  .filter-btn{padding:6px 14px;border-radius:16px;border:1px solid #30363d;
              background:#161b22;color:#8b949e;cursor:pointer;font-size:12px}
  .filter-btn:hover{border-color:#58a6ff;color:#c9d1d9}
  .filter-btn.active{background:rgba(88,166,255,0.15);border-color:#58a6ff;color:#58a6ff}
  .card{border-radius:8px;margin:12px 0;overflow:hidden;border:1px solid var(--border)}
  .card:hover{box-shadow:0 4px 12px rgba(0,0,0,0.3)}
  .card-header{display:flex;align-items:center;justify-content:space-between;
               padding:10px 16px;cursor:pointer;user-select:none;
               background:rgba(255,255,255,0.03)}
  .card-header:hover{background:rgba(255,255,255,0.06)}
  .card-id{font-weight:700;font-size:13px;color:var(--accent)}
  .card-type{display:inline-block;padding:2px 10px;border-radius:12px;
             font-size:10px;font-weight:600;background:var(--accent);color:#fff}
  .card-meta{display:flex;gap:12px;font-size:11px;color:#8b949e;align-items:center}
  .card-badges{display:flex;gap:6px;margin-top:4px}
  .badge{padding:1px 8px;border-radius:10px;font-size:10px;font-weight:600}
  .badge-kw{background:rgba(52,152,219,0.2);color:#3498db}
  .badge-bound{background:rgba(46,204,113,0.2);color:#2ecc71}
  .badge-ref{background:rgba(155,89,182,0.2);color:#9b59b6}
  .badge-neighbor{background:rgba(241,196,15,0.2);color:#f1c40f}
  .card-content{padding:16px;font-size:13px;line-height:1.7;white-space:pre-wrap;
                word-break:break-word;border-top:1px solid rgba(255,255,255,0.05)}
  .card-content.collapsed{max-height:200px;overflow:hidden;position:relative}
  .card-content.collapsed::after{content:'';position:absolute;bottom:0;left:0;right:0;
                                  height:40px;background:linear-gradient(transparent,var(--bg))}
  .expand-btn{display:block;width:100%;padding:6px;border:none;
              background:rgba(255,255,255,0.02);color:#8b949e;
              cursor:pointer;font-size:11px}
  .expand-btn:hover{background:rgba(255,255,255,0.06);color:#c9d1d9}
  .integrity-ok{color:#2ecc71;font-size:11px}
  .integrity-warn{color:#e74c3c;font-size:11px}
  .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:16px 0}
  .stat-card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 14px;text-align:center}
  .stat-card .value{font-size:20px;font-weight:700;color:#58a6ff}
  .stat-card .label{font-size:10px;color:#8b949e;margin-top:2px}
</style>"""


def render_rag_html(report: RAGChunkingReport, title: str = "RAG Chunk 可视化") -> str:
    """
    生成 RAG Chunk 可视化 HTML，聚焦切分质量检查：

    - 每类 chunk 用不同颜色标注
    - 表格/公式/图表标注完整性状态
    - 展示 keywords / bound_elements / ref_ids / 邻居链
    - 支持按 content_type 筛选

    返回完整 HTML 字符串。
    """
    chunks = report.chunks
    qr = report.quality_report

    # 统计
    stats_html = _render_rag_stats(qr)

    # 图例
    legend_items = ""
    for ct, (accent, _, label) in _CONTENT_COLORS.items():
        if any(ch.content_type == ct for ch in chunks):
            legend_items += (f'<div class="legend-item">'
                           f'<div class="legend-dot" style="background:{accent}"></div>'
                           f'<span>{label} ({sum(1 for c in chunks if c.content_type == ct)})</span>'
                           f'</div>')

    # 筛选按钮
    all_types = sorted(set(ch.content_type for ch in chunks))
    filter_btns = '<button class="filter-btn active" onclick="filterChunks(\'all\')">All</button>'
    for ct in all_types:
        filter_btns += f'<button class="filter-btn" onclick="filterChunks(\'{ct}\')">{_CONTENT_COLORS.get(ct, ("#888","#111",ct))[2]}</button>'

    # Chunk 卡片
    cards = ""
    for i, ch in enumerate(chunks):
        cards += _render_rag_chunk_card(ch, i)

    # 质检告警
    issues_html = ""
    if qr["issues"]:
        issue_items = "".join(f"<li>{_html_escape(i)}</li>" for i in qr["issues"])
        issues_html = (f'<div style="background:#2e1a1a;border:1px solid #5c2d2d;'
                      f'border-radius:8px;padding:16px;margin:16px 0">'
                      f'<h3 style="color:#e74c3c;margin-bottom:8px">Quality Issues ({qr["issue_count"]})</h3>'
                      f'<ul style="font-size:12px;color:#c9d1d9;padding-left:20px">{issue_items}</ul>'
                      f'</div>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html_escape(title)}</title>
{_RAG_HTML_STYLE}
</head>
<body>

<div class="header">
  <h1>RAG Chunk 可视化 — {_html_escape(title)}</h1>
  <div class="stats">{report.total_chunks} chunks · {report.config.chunk_tokens} token target</div>
</div>

{stats_html}

<div class="legend">{legend_items}</div>

<div class="filter-bar">{filter_btns}</div>

{issues_html}

<div id="chunk-list">{cards}</div>

<script>
function toggleChunk(id) {{
  const content = document.getElementById('content-' + id);
  const btn = document.getElementById('btn-' + id);
  if (content.classList.contains('collapsed')) {{
    content.classList.remove('collapsed');
    btn.textContent = 'Collapse';
  }} else {{
    content.classList.add('collapsed');
    btn.textContent = 'Expand';
  }}
}}

function filterChunks(type) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.chunk-card').forEach(card => {{
    if (type === 'all' || card.dataset.type === type) {{
      card.style.display = '';
    }} else {{
      card.style.display = 'none';
    }}
  }});
}}
</script>

</body>
</html>"""


def _render_rag_stats(qr: dict) -> str:
    s = qr["stats"]
    td = s.get("token_distribution", {})
    items = [
        (str(s["total_chunks"]), "Total"),
        (str(s["body_chunks"]), "Body"),
        (str(s["table_chunks"]), "Tables"),
        (str(s["figure_chunks"]), "Figures"),
        (str(s["formula_chunks"]), "Formulas"),
        (str(s["reference_chunks"]), "Refs"),
        (f'{s["keywords_coverage"]}/{s["total_chunks"]}', "Keywords"),
        (str(td.get("body_median", "-")), "Median tok"),
        (str(td.get("body_p90", "-")), "P90 tok"),
        (str(td.get("over_max", "0")), ">1536 tok"),
    ]
    cards = ""
    for val, label in items:
        cards += f'<div class="stat-card"><div class="value">{val}</div><div class="label">{label}</div></div>'
    return f'<div class="stats-grid">{cards}</div>'


def _render_rag_chunk_card(ch: RAGChunk, index: int) -> str:
    accent, bg, label = _CONTENT_COLORS.get(ch.content_type, ("#888", "#161b22", ch.content_type))
    content = ch.content
    is_long = len(content) > 400

    # 完整性标记（扫描 body chunk 中嵌入的表格/[FIGURE_DESC]/[FORMULA_DESC]）
    integrity = ""
    ct = ch.content_type
    if ct in ("table_raw", "body"):
        # 检查是否嵌入表格
        table_m = _TABLE_RE.search(content)
        if table_m:
            lines = table_m.group().strip().split('\n')
            has_sep = any(re.match(r'^\|[\s\-:|]+\|$', l) for l in lines)
            integrity = ('<span class="integrity-ok">TABLE OK</span>' if has_sep
                    else '<span class="integrity-warn">TABLE INCOMPLETE</span>')
    if ct in ("figure_desc", "formula_desc", "body"):
        # 检查是否嵌入 [FIGURE_DESC] 或 [FORMULA_DESC]
        f_m = _FIGURE_DESC_RE.search(content) or _FORMULA_DESC_RE.search(content)
        if f_m:
            balanced = f_m.group().count('[') == f_m.group().count(']')
            label = "FIG/FORM" if _FIGURE_DESC_RE.search(content) and _FORMULA_DESC_RE.search(content) else \
                    ("FIGURE" if _FIGURE_DESC_RE.search(content) else "FORMULA")
            integrity = (f'<span class="integrity-ok">{label} OK</span>' if balanced
                    else f'<span class="integrity-warn">{label} MISMATCH</span>')

    # 徽章
    badges = ""
    if "[KEYWORDS:" in content:
        m = re.search(r'\[KEYWORDS:\s*([^\]]+)\]', content)
        if m:
            badges += f'<span class="badge badge-kw">KW: {_html_escape(m.group(1)[:50])}</span>'
    if ch.bound_elements:
        badges += f'<span class="badge badge-bound">Bound: {", ".join(ch.bound_elements[:3])}</span>'
    if ch.ref_ids:
        badges += f'<span class="badge badge-ref">Refs: {", ".join(ch.ref_ids[:4])}</span>'
    if ch.prev_chunk_id or ch.next_chunk_id:
        neighbors = []
        if ch.prev_chunk_id:
            neighbors.append(f"prev={ch.prev_chunk_id}")
        if ch.next_chunk_id:
            neighbors.append(f"next={ch.next_chunk_id}")
        badges += f'<span class="badge badge-neighbor">{"; ".join(neighbors)}</span>'

    return f"""
<div class="chunk-card" data-type="{ch.content_type}"
     style="--accent:{accent};--border:rgba(255,255,255,0.08);--bg:{bg}">
  <div class="card-header" onclick="toggleChunk({index})">
    <div>
      <span class="card-id">#{index + 1} {ch.chunk_id}</span>
      <span class="card-type" style="margin-left:8px">{label}</span>
      {integrity}
      <div class="card-badges">{badges}</div>
    </div>
    <div class="card-meta">
      <span>{ch.token_count} tok</span>
      <span title="{_html_escape(ch.section_path)}">{_html_escape(ch.section_path[:50])}</span>
    </div>
  </div>
  <div class="card-content collapsed" id="content-{index}">
{_html_escape(content)}
  </div>
  {f'<button class="expand-btn" id="btn-{index}" onclick="toggleChunk({index})">Expand</button>' if is_long else ''}
</div>"""


def _html_escape(text: str) -> str:
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def print_quality_report(report: RAGChunkingReport) -> None:
    """打印质检报告摘要。"""
    qr = report.quality_report
    s = qr["stats"]
    td = s.get("token_distribution", {})
    print(f"\n{'=' * 60}")
    print(f"  RAG Chunk 质检报告")
    print(f"{'=' * 60}")
    print(f"  Total chunks:    {s['total_chunks']}")
    print(f"  Body:            {s['body_chunks']}")
    print(f"  Tables:          {s['table_chunks']}")
    print(f"  Figures:         {s['figure_chunks']}")
    print(f"  Formulas:        {s['formula_chunks']}")
    print(f"  References:      {s['reference_chunks']}")
    print(f"  Keyword coverage: {s['keywords_coverage']}/{s['total_chunks']}")
    if td:
        print(f"  Token range:     {td.get('body_min', 0)} - {td.get('body_max', 0)} "
              f"(median={td.get('body_median', 0)}, p90={td.get('body_p90', 0)}, "
              f"p95={td.get('body_p95', 0)})")
        over = td.get("over_max", 0)
        if over:
            print(f"  Oversized (>1536): {over} chunks")
    print(f"  Checks passed:   {qr['checks_passed']}")
    if qr["issues"]:
        print(f"\n  Issues ({qr['issue_count']}):")
        for issue in qr["issues"]:
            print(f"    - {issue}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    demo_dir = Path(__file__).resolve().parent / "output" / "MV-CC"
    demo_path = demo_dir / "final_enriched.md"
    bindings_path = demo_dir / "bindings.json"

    if demo_path.exists():
        md = demo_path.read_text(encoding="utf-8")

        # 尝试加载 bindings
        bindings = None
        if bindings_path.exists():
            bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
            print(f"  [DEMO] Loaded bindings: {len(bindings.get('elements',[]))} elements, "
                  f"{len(bindings.get('references',[]))} references")

        report = rag_chunk_markdown(md, {"title": "MV-CC", "year": "2025"}, bindings=bindings)
        print_quality_report(report)

        # JSON
        out = demo_dir / "rag_chunks.json"
        export_rag_report(report, str(out))

        # HTML 可视化
        html = render_rag_html(report, title="MV-CC RAG Chunks")
        html_path = demo_dir / "rag_chunks.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"  [RAG-CHUNK] HTML viz exported: {html_path}")
        print(f"\n  [OK] Demo output: {out}")
    else:
        print(f"[SKIP] Demo file not found: {demo_path}")
        print("[TIP] Run: python -m pdf_pipeline.cli all <pdf> first")
