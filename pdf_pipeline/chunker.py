"""
academic_chunker.py — 科研文献 Markdown 切分策略
===================

对 Docling 导出的 Markdown 进行科研文献专用切分。

设计原则：
  1. 章节边界切分 —— 在 Markdown 的 ## 标题处分割
  2. 段落完整性 —— 以空行分隔的段落为最小切分单元
  3. 摘要完整 —— 摘要作为独立整块，不切分
  4. 尺寸控制 —— 目标 ~800 字符/chunk, 范围 200-1200
  5. 搜索标签 —— 每个 chunk 注入 [Paper: 章节 | 标题] 标签

与 splitter.py 的关系：
  本模块专门处理 Docling Markdown，识别 Markdown 的 ## 结构。
  通用 splitter.py 仍处理其他文件格式。

使用方式：
  from pdf_pipeline.parser import parse_pdf_docling
  from pdf_pipeline.chunker import chunk_markdown

  result = parse_pdf_docling("paper.pdf")
  chunks = chunk_markdown(result.markdown, result.metadata)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import Counter


# ================================================================
# 数据类
# ================================================================

@dataclass
class Chunk:
    """一个文本块。"""
    chunk_id: str
    content: str
    section_type: str = "body"
    section_name: str = ""
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkingReport:
    """切分报告。"""
    total_chunks: int = 0
    avg_chunk_size: int = 0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    section_distribution: dict = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)


# ================================================================
# 配置
# ================================================================

_override_chunk_size: int | None = None


def set_chunk_size(size: int):
    """运行时覆盖 chunk 大小。"""
    global _override_chunk_size
    _override_chunk_size = size


def _get_chunk_size() -> int:
    """获取目标 chunk 大小。"""
    if _override_chunk_size is not None:
        return _override_chunk_size
    from ._config import PAPER_CHUNK_SIZE
    return PAPER_CHUNK_SIZE


# ================================================================
# Markdown 章节检测
# ================================================================

# markdown 标题模式（## 开头）
_MD_HEADER_RE = re.compile(r'^##\s+(.+)$', re.MULTILINE)

# 噪声标记（Docling 无法解析的内容）
_NOISE_PATTERNS = [
    # ponytail: 保留 <!-- image --> 和 <!-- formula-not-decoded --> 作为空间锚点
    # 仅清理表格占位符（表格已由 do_table_structure=True 正常渲染）
    re.compile(r'<!--\s*table-not-decoded\s*-->', re.IGNORECASE),
]

# 元素占位符 → 用于将 chunk 与 bindings 元素关联
_ELEMENT_PLACEHOLDER_RE = re.compile(
    r'<!--\s*(image|formula-not-decoded)\s*-->', re.IGNORECASE,
)


def _classify_section(header_text: str) -> str:
    """根据标题文本分类章节类型（多语言支持）。"""
    lower = header_text.lower().strip()

    # 去掉编号前缀 (I., II., A., 1., 一、等)
    cleaned = re.sub(r'^[IVX\d]+[\.、\)]\s*', '', lower)
    cleaned = re.sub(r'^[A-Z][\.、\)]\s*', '', cleaned)

    # 进一步清理特殊字符（土耳其语等）
    cleaned_ascii = _normalize_unicode(cleaned)

    patterns = [
        ("abstract",     [r'\babstract\b', r'\b摘要\b', r'\bözet', r'\bozet',
                          r'\b概要\b', r'\böz\b', r'\boz\b']),
        ("introduction", [r'\bintro(duction)?\b', r'\bgir[ıi]', r'\bg˙ir',
                          r'\b引言\b', r'\b绪论\b', r'\b前言\b']),
        ("related_work", [r'\brelated\s*(work|literature)\b', r'\bbackground\b',
                          r'\breview\b', r'\b相关工作\b', r'\b研究现状\b']),
        ("methods",      [r'\bmethod(ology|s)?\b', r'\bapproach\b',
                          r'\by[oö]ntem\b', r'\b[oö]nerilen\b', r'\bproposed\b',
                          r'\b方法\b', r'\b模型\b', r'\b算法\b', r'\bfark\b']),
        ("results",      [r'\bresults?\b', r'\bexperiments?\b', r'\bevaluation\b',
                          r'\bdeneysel\b', r'\bsonu[çc]', r'\b结果\b', r'\b实验\b']),
        ("discussion",   [r'\bdiscussion\b', r'\banalysis\b', r'\btart[ıi][şs]',
                          r'\b讨论\b', r'\b分析\b']),
        ("conclusion",   [r'\bconclusion\b', r'\bsummary\b', r'\bsonu[çc]\b',
                          r'\b结论\b', r'\b总结\b']),
        ("references",   [r'\breferences?\b', r'\bbibliography\b', r'\bkaynak',
                          r'\bkaynak[çc]a\b', r'\b参考文献\b']),
    ]

    matched_types: list[str] = []
    for sec_type, pat_list in patterns:
        for pat in pat_list:
            if re.search(pat, cleaned_ascii):
                matched_types.append(sec_type)
                break  # one match per section type is enough

    if len(matched_types) > 1:
        # ponytail: log ambiguity, use first match as tiebreaker
        print(f"  [CLASSIFY] \"{header_text[:60]}\" matched: {matched_types} "
              f"→ selected \"{matched_types[0]}\"")
        return matched_types[0]
    elif len(matched_types) == 1:
        return matched_types[0]

    return "other"


def _normalize_unicode(text: str) -> str:
    """将土耳其语等特殊字符规范化为 ASCII 近似，便于正则匹配。"""
    # 常见土耳其语字符 → ASCII
    replacements = {
        'ı': 'i', 'i̇': 'i',  # dotless i → i
        'ş': 's', 'ş': 's',  # s-cedilla → s
        'ğ': 'g', 'ğ': 'g',  # g-breve → g
        'ç': 'c', 'ç': 'c',  # c-cedilla → c
        'ö': 'o', 'ö': 'o',  # o-umlaut → o
        'ü': 'u', 'ü': 'u',  # u-umlaut → u
        '˙': '',             # dot above → remove
        '˘': '',             # breve → remove
        '¸': '',             # cedilla → remove
        '¨': '',             # diaeresis → remove
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


# ================================================================
# 主切分逻辑
# ================================================================

def _clean_markdown(md: str) -> str:
    """清理 Markdown 中的噪声标记和多余空白。"""
    for pattern in _NOISE_PATTERNS:
        md = pattern.sub('', md)

    # 折叠连续空行
    md = re.sub(r'\n{3,}', '\n\n', md)

    # 去行尾空格
    md = re.sub(r'[ \t]+$', '', md, flags=re.MULTILINE)

    return md.strip()


def _split_into_sections(md: str) -> list[dict]:
    """
    将 Markdown 按 ## 标题拆分为章节。

    返回: [
        {"title": "I. INTRODUCTION", "type": "introduction",
         "content": "段落1\n\n段落2\n...", "level": 2},
        ...
    ]

    标题前的所有内容（作者、摘要等）作为 preamble 处理。
    """
    # 找到所有 ## 标题的位置
    header_positions = [(m.start(), m.group(1).strip())
                        for m in _MD_HEADER_RE.finditer(md)]

    if not header_positions:
        return [{"title": "", "type": "body", "content": md, "level": 0}]

    sections = []

    # 第一个标题之前的内容 → preamble（作者信息、摘要等）
    first_header_start = header_positions[0][0]
    # 找到第一个标题所在行的行首
    pre_md = md[:first_header_start].rstrip()
    # 找第一个标题行在原文中的精确起始位置
    for i, (pos, _) in enumerate(header_positions):
        # 回退到行首
        line_start = md.rfind('\n', 0, pos) + 1
        if i == 0 and line_start > 0:
            pre_md = md[:line_start].rstrip()
        break

    if pre_md and len(pre_md.strip()) > 20:
        # 判断 preamble 类型 — default to "preamble", upgrade only on keyword match
        pre_type = "preamble"
        pre_lower = _normalize_unicode(pre_md.lower())
        if re.search(r'\babstract\b|\bozet\b|\b摘要\b|\boz\b', pre_lower):
            pre_type = "abstract"
        sections.append({
            "title": "摘要/Abstract",
            "type": pre_type,
            "content": pre_md.strip(),
            "level": 0,
        })

    # 按标题切分
    for i, (pos, title) in enumerate(header_positions):
        # 找到这个标题行的行首
        line_start = md.rfind('\n', 0, pos) + 1 if i > 0 else 0

        # 内容起始 = 标题行结束
        content_start = md.index('\n', pos) + 1 if '\n' in md[pos:] else len(md)

        # 内容结束 = 下一个标题的行首（或文末）
        if i + 1 < len(header_positions):
            next_line_start = md.rfind('\n', 0, header_positions[i + 1][0]) + 1
            content_end = next_line_start
        else:
            content_end = len(md)

        content = md[content_start:content_end].strip()

        sec_type = _classify_section(title)

        sections.append({
            "title": title,
            "type": sec_type,
            "content": content,
            "level": 2,  # 所有 Docling 导出的标题都是 ##
        })

    return sections


def _chunk_paragraphs(
    text: str,
    section_name: str,
    section_type: str,
    counter: int,
    paper_meta: dict,
) -> list[Chunk]:
    """
    将一节文本按段落切分为 chunk。

    段落边界 = 空行（\n\n）。
    每组段落累加到接近目标大小 → 切出一个 chunk。
    """
    target_size = _get_chunk_size()
    min_size = target_size // 4

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_paras: list[str] = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para) + 2  # +2 for \n\n

        if current_paras and (current_size + para_size > target_size):
            if current_size >= min_size:
                # 切出一个 chunk
                chunks.append(_make_chunk(
                    current_paras, section_name, section_type,
                    counter, paper_meta,
                ))
                counter += 1
                current_paras = [para]
                current_size = para_size
                continue

        current_paras.append(para)
        current_size += para_size

    # 最后一组
    if current_paras:
        chunks.append(_make_chunk(
            current_paras, section_name, section_type,
            counter, paper_meta,
        ))
        counter += 1

    return chunks


def _make_chunk(
    paragraphs: list[str],
    section_name: str,
    section_type: str,
    counter: int,
    paper_meta: dict,
) -> Chunk:
    """组装一个 Chunk。"""
    content = '\n\n'.join(paragraphs)
    return Chunk(
        chunk_id=f"chunk_{counter:04d}",
        content=content,
        section_type=section_type,
        section_name=section_name,
        token_count=_estimate_tokens(content),
        metadata={
            "section": section_name,
            "section_type": section_type,
            "paragraph_count": len(paragraphs),
            "paper_title": paper_meta.get("title", ""),
            "paper_year": paper_meta.get("year", ""),
        },
    )


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数。"""
    if not text:
        return 0
    chinese = len(re.findall(r'[一-鿿]', text))
    other = len(text) - chinese
    return int(chinese * 0.6 + other * 0.25)


# ================================================================
# 空间绑定注入
# ================================================================

def _inject_binding_metadata(chunks: list[Chunk], bindings: dict):
    """
    将 bindings 中的元素/引用信息注入到每个 chunk 的 metadata 中。

    ponytail: 简单字符串包含检查。标题或引用文本出现在 chunk 内容中 → 绑定。
    """
    elements = bindings.get("elements", [])
    references = bindings.get("references", [])

    if not elements:
        return

    # 构建标题文本 → element_id 映射
    caption_map: dict[str, str] = {}
    for elem in elements:
        cap = elem.get("caption", "")
        if cap:
            caption_map[cap] = elem["element_id"]

    for ch in chunks:
        bound: list[str] = []
        content_lower = ch.content.lower() if ch.content else ""

        # 标题匹配
        for cap_text, eid in caption_map.items():
            if cap_text.lower() in content_lower:
                if eid not in bound:
                    bound.append(eid)

        # 引用匹配（正文中的 "Fig. 1" / "Table I" 等）
        for ref in references:
            ref_text = ref.get("ref_text", "")
            if ref_text and ref_text.lower() in content_lower:
                tid = ref.get("target_element_id", "")
                if tid and tid not in bound:
                    bound.append(tid)

        if bound:
            ch.metadata["bound_elements"] = bound


# ================================================================
# 主入口
# ================================================================

def chunk_markdown(
    markdown: str,
    paper_metadata: dict | None = None,
    bindings: dict | None = None,
) -> ChunkingReport:
    """
    对 Docling 导出的 Markdown 执行学术切分。

    流程:
      1. 清理 markdown（去噪声标记，保留空间锚点）
      2. 按 ## 标题拆分为章节
      3. 每个章节内按段落分组切分
      4. (可选) 注入空间绑定元数据
      5. 组装 ChunkingReport

    参数:
        markdown: Docling 导出的 Markdown 文本
        paper_metadata: 论文元数据 {title, authors, doi, year}
        bindings: 空间绑定数据 {elements, references}，来自 bindings.json

    返回:
        ChunkingReport: 包含所有 chunk 的切分报告
    """
    if paper_metadata is None:
        paper_metadata = {}

    # 1. 清理
    md = _clean_markdown(markdown)
    if not md:
        return ChunkingReport(total_chunks=0)

    # 2. 拆分章节
    sections = _split_into_sections(md)

    # 3. 按章节切分
    all_chunks: list[Chunk] = []
    counter = 0

    for sec in sections:
        section_title = sec["title"]
        section_type = sec["type"]
        content = sec["content"]

        if not content:
            continue

        # 摘要 → 单 chunk（不切分）
        if section_type == "abstract" and len(content) <= _get_chunk_size():
            all_chunks.append(Chunk(
                chunk_id=f"chunk_{counter:04d}",
                content=f"## {section_title}\n\n{content}",
                section_type=section_type,
                section_name=section_title,
                token_count=_estimate_tokens(content),
                metadata={
                    "section": section_title,
                    "section_type": section_type,
                    "paper_title": paper_metadata.get("title", ""),
                    "paper_year": paper_metadata.get("year", ""),
                },
            ))
            counter += 1
        else:
            # 正文章节 → 段落分组切分
            section_chunks = _chunk_paragraphs(
                content, section_title, section_type,
                counter, paper_metadata,
            )
            # 在每个 chunk 前加上章节标题作为上下文
            for ch in section_chunks:
                ch.content = f"## {section_title}\n\n{ch.content}"
            all_chunks.extend(section_chunks)
            counter += len(section_chunks)

    # 4. 注入空间绑定元数据
    if bindings and all_chunks:
        _inject_binding_metadata(all_chunks, bindings)

    # 5. 组装报告
    sizes = [len(ch.content) for ch in all_chunks] if all_chunks else [0]
    section_dist = dict(Counter(ch.section_type for ch in all_chunks))

    report = ChunkingReport(
        total_chunks=len(all_chunks),
        avg_chunk_size=sum(sizes) // len(sizes) if sizes else 0,
        min_chunk_size=min(sizes) if sizes else 0,
        max_chunk_size=max(sizes) if sizes else 0,
        section_distribution=section_dist,
        chunks=all_chunks,
    )

    print(f"  [CHUNK] [OK] {len(sections)} sections -> "
          f"{report.total_chunks} chunks, "
          f"avg {report.avg_chunk_size} chars/chunk")

    return report


# ================================================================
# 便捷 API
# ================================================================

def parse_and_chunk(file_path: str) -> ChunkingReport:
    """
    解析 PDF → Markdown → 切分，一步完成。

    参数:
        file_path: PDF 文件路径

    返回:
        ChunkingReport: 切分报告
    """
    from .parser import parse_pdf_docling

    result = parse_pdf_docling(file_path)
    return chunk_markdown(result.markdown, result.metadata)
