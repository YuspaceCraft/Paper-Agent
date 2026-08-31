"""
paper_parser.py — 科研论文 PDF 专用解析器
==============
将 PDF 论文解析为结构化数据：元数据 + 章节 + 摘要 + 参考文献。

设计原则：
  - 多后端提取：PyMuPDF → pdfplumber → pypdf，自动降级
  - 启发式元数据：不依赖外部 API，纯正则/规则提取
  - 中英双语章节检测：支持 IMRaD 结构 + 中文学术论文章节
  - 文本清洗：去页眉页脚、修复换行断字、段落合并

输出：ParsedPaper 数据类，下游 loader/splitter 直接消费。
"""

from __future__ import annotations

import re
import io
from dataclasses import dataclass, field
from pathlib import Path

from config import PDF_EXTRACTION_BACKENDS


# ================================================================
# 数据类
# ================================================================

@dataclass
class ParsedPaper:
    """解析后的论文结构化数据。"""
    file_path: str
    full_text: str                     # 清洗后的完整文本
    metadata: dict                     # {title, authors, doi, year, journal, keywords}
    abstract: str                      # 摘要文本
    body_sections: list[dict]          # [{name, content, level, type}]
    references: str                    # 参考文献文本


# ================================================================
# 多后端 PDF 文本提取
# ================================================================

def extract_pdf_text(file_path: str) -> str:
    """
    多后端 PDF 文本提取，按优先级自动降级。

    后端优先级由 PDF_EXTRACTION_BACKENDS 配置控制，
    默认: pymupdf → pdfplumber → pypdf

    每个后端返回提取文本；如果文本太短（<100 字符）或抛出异常，
    自动尝试下一个后端。全部失败则抛出 RuntimeError。
    """
    backends = [b.strip() for b in PDF_EXTRACTION_BACKENDS if b.strip()]
    if not backends:
        backends = ["pymupdf", "pdfplumber", "pypdf"]

    errors = []
    for backend in backends:
        extractor = _BACKENDS.get(backend)
        if extractor is None:
            errors.append(f"{backend}: 未知后端")
            continue
        try:
            text = extractor(file_path)
            if text and len(text.strip()) > 100:
                print(f"  [PAPER] [OK] 使用 {backend} 提取 ({len(text)} 字符)")
                return text
            else:
                errors.append(f"{backend}: 提取文本过短 ({len(text.strip())} 字符)")
        except Exception as e:
            errors.append(f"{backend}: {e}")

    raise RuntimeError(
        f"[PAPER] [ERR] 所有 PDF 后端提取失败: {Path(file_path).name}\n" +
        "\n".join(f"  - {e}" for e in errors)
    )


def _extract_pymupdf(file_path: str) -> str:
    """
    PyMuPDF (fitz) 提取 — 列感知 + 块级排序。

    策略：
      - 先用 get_text("blocks") 获取每个文本块的坐标和内容
      - 按块位置排序：先按 y 坐标（行），同 y 内按 x 坐标（列）
      - 这样两栏排版的论文不会出现左右列交叉的问题
    """
    import fitz
    doc = fitz.open(file_path)
    all_texts = []

    for page in doc:
        blocks = page.get_text("blocks")
        # 过滤掉空块和图片块
        text_blocks = [
            b for b in blocks
            if b[6] == 0 and b[4].strip()  # type 0 = text, non-empty
        ]

        if not text_blocks:
            continue

        # 按 y 坐标排序，同 y 阈值内按 x 排序（模拟阅读顺序）
        def block_sort_key(block):
            y = block[1]  # y0
            x = block[0]  # x0
            # 将 y 量化到 ~5pt 的网格（同行的块归为一组）
            y_group = round(y / 5) * 5
            return (y_group, x)

        text_blocks.sort(key=block_sort_key)

        page_text = "\n".join(b[4].strip() for b in text_blocks)
        all_texts.append(page_text)

    doc.close()
    return "\n\n".join(all_texts)


def _extract_pdfplumber(file_path: str) -> str:
    """pdfplumber 提取 — 表格友好，纯 Python。"""
    import pdfplumber
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_pypdf(file_path: str) -> str:
    """pypdf 提取 — 零额外依赖，兜底方案。"""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


_BACKENDS = {
    "pymupdf": _extract_pymupdf,
    "pdfplumber": _extract_pdfplumber,
    "pypdf": _extract_pypdf,
}


# ================================================================
# 文本清洗
# ================================================================

def clean_paper_text(text: str) -> str:
    """
    清洗论文文本：去页眉页脚 + 修复断字 + 段落合并。

    处理步骤（按顺序）：
      1. 移除重复行（页眉/页脚 — 同一行在多页出现 > 30%）
      2. 修复换行连字符: word-\nword → wordword
      3. 修复段内换行: 非段落结束的单个 \\n → 空格
      4. 规范化空白
    """
    if not text:
        return ""

    lines = text.splitlines()

    # 步骤 1: 检测并移除页眉/页脚（高频重复行）
    cleaned_lines = _remove_repeated_lines(lines)

    # 步骤 2: 修复换行连字符
    text = "\n".join(cleaned_lines)
    text = _fix_hyphenation(text)

    # 步骤 3: 修复段内换行
    text = _fix_inline_breaks(text)

    # 步骤 4: 规范化空白
    text = _normalize_whitespace(text)

    # 步骤 5: 移除 Unicode 私有区字符（PUA, U+E000-U+F8FF）
    # 和一些无法在 GBK 编码下输出的特殊字符
    # 这些字符对语义检索无意义，且会导致 Windows 控制台报错
    text = ''.join(
        c for c in text
        if ord(c) < 0xE000 or ord(c) > 0xF8FF
    )
    # 移除 C0 控制字符（保留换行和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


def _remove_repeated_lines(lines: list[str]) -> list[str]:
    """
    移除在页间重复出现的行（页眉/页脚/页码）。

    启发式：如果一行（去除数字变化后）在超过阈值比例的行中出现，
    则视为页眉/页脚并移除。
    """
    if len(lines) < 10:
        return lines

    # 归一化：把数字替换为占位符，以便匹配"Page 1" / "Page 2" 这类模式
    def normalize(line: str) -> str:
        s = line.strip()
        if not s:
            return ""
        # 纯数字行（页码）统一为占位符
        if re.match(r'^\d{1,4}$', s):
            return "__PAGE_NUM__"
        # 包含"第X页"或"Page X"的
        s = re.sub(r'第\s*\d+\s*页', '第__页', s)
        s = re.sub(r'Page\s*\d+', 'Page__', s, flags=re.IGNORECASE)
        # 替换数字
        s = re.sub(r'\d+', '0', s)
        return s.lower()

    # 统计归一化后的行频率
    from collections import Counter
    normalized = [normalize(l) for l in lines]
    non_empty = [n for n in normalized if n]
    if not non_empty:
        return lines

    counter = Counter(non_empty)
    total = len(lines)
    threshold = max(3, total * 0.25)  # 出现超过 25% 的行视为页眉/页脚

    repeated = {pat for pat, count in counter.items() if count >= threshold}

    # 过滤
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # 跳过重复行
        if normalize(line) in repeated:
            continue
        # 跳过孤立的纯数字行（1-3 位数字，大概率是页码）
        if re.match(r'^\d{1,3}$', stripped):
            # 数字行前后至少一侧是段落边界（空行、另一页码、或短标题）
            prev_is_boundary = i == 0 or not lines[i - 1].strip() or re.match(r'^\d{1,3}$', lines[i - 1].strip())
            next_is_boundary = i + 1 >= len(lines) or not lines[i + 1].strip() or re.match(r'^\d{1,3}$', lines[i + 1].strip())
            if prev_is_boundary or next_is_boundary:
                continue
        result.append(line)

    return result


def _fix_hyphenation(text: str) -> str:
    """
    修复换行断字：word-\nword → wordword

    只修复行尾连字符（单词被行边界打断），
    不修复复合词（如 state-of-the-art 中有空格/换行分开的情况）。
    """
    # 匹配: 字母 + 连字符 + 换行 + 字母 → 合并
    # 注意: 数字间的连字符（如电话号码）不合并
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text


def _fix_inline_breaks(text: str) -> str:
    """
    修复段落内部的意外换行。

    规则：如果一行不以句末标点结束，
    且下一行不以大写字母或缩进开头，则两行属于同一段落。
    """
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    result = []
    buffer = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            # 空行 = 段落边界
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append("")
            continue

        if not buffer:
            buffer = line
            continue

        # 判断是否应该合并到 buffer
        prev_end = buffer.rstrip()[-1:] if buffer.rstrip() else ""
        curr_start = stripped[0] if stripped else ""

        # 上一行结束符不是句末标点 + 当前行开头不是大写/数字编号
        is_sentence_end = prev_end in '.。！？!?…'')'']'']'
        is_new_sentence = (
            curr_start.isupper() or
            curr_start.isdigit() or
            stripped.startswith(('(', '（', '[', '【', '"', '"', '\''))
        )

        if not is_sentence_end and not is_new_sentence:
            # 合并到上一行（加空格）
            buffer = buffer.rstrip() + " " + stripped
        else:
            result.append(buffer)
            buffer = line

    if buffer:
        result.append(buffer)

    return "\n".join(result)


def _normalize_whitespace(text: str) -> str:
    """规范化空白：多个空行 → 一个空行，去除行尾空格。"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)
    return text


# ================================================================
# 章节检测（中英双语）
# ================================================================

# 章节类型映射（正则模式 -> 归一化类型）
# 注意：不再将特定罗马数字绑定到特定章节类型。
#       编号标题（I., II., ...）用通用模式匹配，再按名称文本分类。
_SECTION_PATTERNS = [
    # === 非编号标题：直接匹配 → 类型 ===
    ("abstract", [
        r'^Abstract\s*$',
        r'^ABSTRACT\s*$',
        r'^摘\s*要\s*$',
        r'^【摘\s*要】',
        r'^内容摘要[：:]\s*$',
    ]),
    ("introduction", [
        r'^Introduction\s*$',
        r'^INTRODUCTION\s*$',
        r'^(?:引言|绪论|前言|问题的提出)\s*$',
    ]),
    ("related_work", [
        r'^(?:Related\s*Work|Background|Literature\s*Review)\s*$',
        r'^RELATED\s*WORK\s*$',
        r'^(?:相关\s*工作|研究\s*现状|文献\s*综述|国内\s*外\s*研究\s*现状)\s*$',
    ]),
    ("methods", [
        r'^(?:Method|Methodology|Approach|Experimental\s*Setup)\s*$',
        r'^METHOD(?:OLOGY|S)?\s*$',
        r'^(?:方法|实验\s*方法|研究\s*方法|模型\s*设计|模型|算法\s*设计|系统\s*设计|方案\s*设计)\s*$',
    ]),
    ("results", [
        r'^(?:Results?|Experiments?|Findings)\s*$',
        r'^RESULTS?\s*$',
        r'^EXPERIMENTS?\s*$',
        r'^(?:结果|实验\s*结果|实验\s*分析|仿真\s*实验|性能\s*评估|实验\s*评估)\s*$',
    ]),
    ("discussion", [
        r'^(?:Discussion|Analysis|Discussions?)\s*$',
        r'^DISCUSSION\s*$',
        r'^ANALYSIS\s*$',
        r'^(?:讨论|分析|结果\s*分析|对比\s*分析|讨论\s*与\s*分析)\s*$',
    ]),
    ("conclusion", [
        r'^(?:Conclusion|Summary|Future\s*Work|Concluding\s*Remarks)\s*$',
        r'^CONCLUSION\s*$',
        r'^SUMMARY\s*$',
        r'^(?:结论|总结|结束语|展望|结语|小结)\s*$',
    ]),
    ("references", [
        r'^(?:References|Bibliography|Literature\s*Cited)\s*$',
        r'^REFERENCES\s*$',
        r'^BIBLIOGRAPHY\s*$',
        r'^参考\s*文献\s*$',
        r'^【参考\s*文献】',
        r'^参考文献[：:]\s*$',
    ]),
    ("acknowledgments", [
        r'^(?:Acknowledgment|Acknowledgements?)\s*$',
        r'^ACKNOWLEDGMENTS?\s*$',
        r'^(?:致谢|鸣谢)\s*$',
    ]),
    # 非结构化摘要
    ("abstract", [
        r'^Abstract[—–\-]\s',
        r'^ABSTRACT[—–\-]\s',
    ]),
    # 关键词（标记但不作为章节）
    ("keywords", [
        r'^(?:Keywords?|Key\s*Words?|Index\s*Terms?)[：:]\s*',
        r'^(?:关键词|关键\s*词)[：:]\s*',
    ]),
]

# 编号标题分类：根据标题文本中的关键词判断章节类型
# 编号标题格式: "I. TITLE", "II. TITLE", "1. TITLE", "一、标题" 等
_NUMBERED_SECTION_KEYWORDS = [
    ("abstract",        [r'\bAbstract\b', r'\b摘要\b']),
    ("introduction",    [r'\bIntroduction\b', r'\bIntro\b', r'\b引言\b', r'\b绪论\b', r'\b前言\b']),
    ("related_work",    [r'\bRelated\s*Work\b', r'\bBackground\b', r'\bLiterature\b', r'\b相关工作\b', r'\b研究现状\b', r'\b文献综述\b']),
    ("methods",         [r'\bMethod(?:ology|s)?\b', r'\bMethodol[a-z]*\b', r'\bApproach\b', r'\bFramework\b', r'\b方法\b', r'\b实验方法\b', r'\b模型\b', r'\b算法\b', r'\b方案\b', r'\b设计\b']),
    ("results",         [r'\bResults?\b', r'\bExperiments?\b', r'\bFindings\b', r'\bEvaluation\b', r'\b结果\b', r'\b实验\b', r'\b评估\b', r'\b性能\b']),
    ("discussion",      [r'\bDiscussion\b', r'\bAnalysis\b', r'\b讨论\b', r'\b分析\b']),
    ("conclusion",      [r'\bConclusion\b', r'\bSummary\b', r'\bFuture\b', r'\b结论\b', r'\b总结\b', r'\b结束语\b', r'\b展望\b']),
    ("dataset",         [r'\bDataset\b', r'\bData\b', r'\b数据集\b', r'\b数据\b']),
]


def detect_sections(text: str) -> list[dict]:
    """
    检测论文章节结构。

    策略：
      1. 先用 _SECTION_PATTERNS 匹配无编号标题（Abstract, Introduction 等）
      2. 再用通用编号模式 ^[IVX\d]+[.、]\\s+ 匹配编号标题
      3. 对编号标题，提取标题文本后用关键词分类

    返回: [
        {"name": "Introduction", "content": "...", "level": 1, "type": "introduction"},
        ...
    ]
    """
    if not text:
        return []

    lines = text.splitlines()
    headings = []  # [(line_index, section_name, section_type, level)]

    # 编译显式模式（IGNORECASE 兼容全大写/全小写）
    compiled = []
    for sec_type, patterns in _SECTION_PATTERNS:
        for pat in patterns:
            compiled.append((re.compile(pat, re.IGNORECASE), sec_type))

    # 编译编号模式
    _numbered_re = re.compile(r'^([IVX\d]+)[.、]\s+(.+)', re.IGNORECASE)

    # 预扫描：IEEE "Abstract— text" 修复
    ieee_abstract_lines = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^Abstract[—–\-]\s{1,3}', stripped, re.IGNORECASE):
            m = re.match(r'^(Abstract[—–\-])\s{1,3}(.+)', stripped, re.IGNORECASE)
            if m:
                lines[i] = "Abstract"
                if i + 1 >= len(lines) or not lines[i + 1].strip():
                    lines.insert(i + 1, m.group(2))
                ieee_abstract_lines.add(i)

    # 扫描标题行
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # 长度限制（IEEE 特殊行除外）
        if len(stripped) > 120 and i not in ieee_abstract_lines:
            if not re.match(r'^[IVX\d]+\.\s', stripped):
                continue

        matched = False

        # 1. 尝试显式模式匹配
        for regex, sec_type in compiled:
            if regex.search(stripped):
                if _is_heading_line(i, lines, stripped):
                    level = _detect_heading_level(stripped)
                    headings.append((i, stripped, sec_type, level))
                    matched = True
                    break

        if matched:
            continue

        # 2. 尝试编号模式匹配
        num_match = _numbered_re.search(stripped)
        if num_match:
            num_str = num_match.group(1)
            section_name = num_match.group(2).strip()
            # 过滤假阳性：
            #   - 4+ 位数字（年份、DOI、基金号等）
            #   - 数字 > 30（不是合理的章节编号）
            #   - 标题包含 "Corresponding author" 等
            is_false_positive = False
            if num_str.isdigit():
                num_val = int(num_str)
                if num_val > 30 or len(num_str) >= 4:
                    is_false_positive = True
            if re.search(r'(Corresponding author|e-mail:|@\w+\.\w+|IEEE|TRANSACTIONS)', section_name, re.IGNORECASE):
                is_false_positive = True
            if section_name and not is_false_positive and _is_heading_line(i, lines, stripped):
                sec_type = _classify_numbered_section(section_name)
                level = _detect_heading_level(stripped)
                headings.append((i, stripped, sec_type, level))

    # 按位置排序，去重（如果同一行匹配多个模式）
    headings.sort(key=lambda x: x[0])
    seen_positions = set()
    unique = []
    for h in headings:
        if h[0] not in seen_positions:
            unique.append(h)
            seen_positions.add(h[0])

    if not unique:
        # 无章节标题 → 全文作为一个 "body" 章节
        return [{"name": "正文/Body", "content": text, "level": 1, "type": "body"}]

    # 按标题位置切分内容
    sections = []
    for idx, (line_idx, name, sec_type, level) in enumerate(unique):
        start = line_idx + 1  # 内容从标题下一行开始
        end = unique[idx + 1][0] if idx + 1 < len(unique) else len(lines)

        # 提取内容（跳过标题行自身）
        section_lines = lines[start:end]
        content = "\n".join(section_lines).strip()

        if content:
            sections.append({
                "name": name,
                "content": content,
                "level": level,
                "type": sec_type,
            })

    # 如果有标题之前的内容（摘要区域），尝试识别
    if unique and unique[0][0] > 0:
        pre_text = "\n".join(lines[:unique[0][0]]).strip()
        if pre_text and len(pre_text) > 50:
            # 判断是否为摘要
            first_heading_type = unique[0][2]
            if first_heading_type in ("introduction", "related_work", "methods"):
                # 标题前的内容很可能是摘要
                sections.insert(0, {
                    "name": "摘要/Abstract",
                    "content": pre_text,
                    "level": 1,
                    "type": "abstract",
                })

    return sections


def _is_heading_line(idx: int, lines: list[str], text: str) -> bool:
    """判断一行是否真的是章节标题（而非正文中恰好匹配的文字）。"""
    # 条件1：该行较短
    if len(text) > 120:
        return False

    # 条件2：上一行是空行或不存在（新段落开始）
    prev_empty = idx == 0 or not lines[idx - 1].strip()
    # 条件2b：上一行以句末标点结束（也是段落边界）
    prev_end = ""
    if idx > 0 and lines[idx - 1].strip():
        prev_end = lines[idx - 1].rstrip()[-1:]
    prev_is_sentence_end = prev_end in '.。！？!?""\'\'）)】]'
    # 条件3：下一行存在且非空
    next_exists = idx + 1 < len(lines) and bool(lines[idx + 1].strip())
    # 条件4：不以标点结尾（标题不应以句号结尾）
    no_sentence_end = not text.rstrip().endswith(('.', '。', ',', '，', ';', '；'))

    # 标准条件：前面有空行或句末标点
    if (prev_empty or prev_is_sentence_end) and next_exists and no_sentence_end:
        return True

    # 强标题词列表（即使前面没有空行也接受，适配紧凑排版）
    strong_headings = {
        "abstract", "introduction", "related work", "methodology", "methods",
        "results", "experiments", "discussion", "conclusion", "references",
        "acknowledgments", "acknowledgements", "appendix",
        "摘要", "引言", "绪论", "前言", "方法", "实验方法", "研究方法",
        "结果", "实验结果", "实验分析", "讨论", "分析", "结论", "总结",
        "参考文献", "致谢", "附录",
    }
    text_lower = text.strip().lower()
    if text_lower in strong_headings and next_exists and no_sentence_end:
        return True

    # 放宽条件：编号开头 + 下一行存在
    if re.match(r'^[IVX\d]+[.、]\s', text) and next_exists:
        return True

    # 中文：以"第X章"开头
    if re.match(r'^第[一二三四五六七八九十\d]+[章节部分]', text):
        return True

    return False


def _detect_heading_level(text: str) -> int:
    """推断标题层级：1=一级章节，2=二级子节。"""
    # 模式：X.Y 或 X.Y.Z → 二级
    if re.match(r'^\d+\.\d+', text):
        return 2
    # 模式：纯英文标题 + 冒号 → 可能是二级
    if re.match(r'^[A-Z][a-z]+:', text):
        return 2
    # 中文子节：以括号编号
    if re.match(r'^[（(]\d+[）)]', text):
        return 2
    return 1


def _classify_numbered_section(name: str) -> str:
    """
    根据章节标题文本关键词，将编号章节归类到标准类型。

    参数:
        name: 编号后的章节名称，如 "METHODOLOGY", "LEVIR-MCI DATASET"

    返回:
        标准类型: introduction | related_work | methods | results |
                  discussion | conclusion | dataset | other
    """
    if not name:
        return "other"

    for sec_type, patterns in _NUMBERED_SECTION_KEYWORDS:
        for pat in patterns:
            if re.search(pat, name, re.IGNORECASE):
                return sec_type

    return "other"


# ================================================================
# 元数据提取
# ================================================================

def extract_metadata(text: str, file_path: str) -> dict:
    """
    从论文文本中提取元数据（纯启发式，无外部 API）。

    返回: {title, authors, doi, year, journal, keywords}

    策略：
      - DOI: 高度可靠的正则匹配 (10.xxxx/...)
      - 标题: 取前 500 字符中最长的非作者行
      - 作者: 标题附近的姓名模式行
      - 年份: DOI 中提取 或 标题附近日期
      - 期刊: 页眉区域检测
      - 关键词: "Keywords" / "关键词" 后的内容
    """
    metadata = {
        "title": "",
        "authors": "",
        "doi": "",
        "year": "",
        "journal": "",
        "keywords": "",
    }

    if not text:
        return metadata

    lines = text.splitlines()
    first_block = "\n".join(lines[:80])  # 前 ~80 行 = 标题/作者区域

    # DOI — 高可靠性
    doi_match = re.search(r'\b(10\.\d{4,}/[^\s]{3,})\b', text)
    if doi_match:
        metadata["doi"] = doi_match.group(1).rstrip('.')

    # 年份 — 从 DOI 或标题区域提取
    if metadata["doi"]:
        year_match = re.search(r'/(20\d{2})/', metadata["doi"] + "/")
        if year_match:
            metadata["year"] = year_match.group(1)
    if not metadata["year"]:
        # 在标题附近查找年份
        year_match = re.search(r'(?:©|Copyright|published)\s*(20\d{2})', first_block, re.IGNORECASE)
        if year_match:
            metadata["year"] = year_match.group(1)

    # 标题 — 前 500 字符中最长的有意义行
    metadata["title"] = _extract_title(first_block, lines)

    # 作者
    metadata["authors"] = _extract_authors(first_block, metadata["title"])

    # 期刊 — 头部/页脚区域
    metadata["journal"] = _extract_journal(lines)

    # 关键词
    metadata["keywords"] = _extract_keywords(text)

    return metadata


def _extract_title(first_block: str, lines: list[str]) -> str:
    """从论文头部提取标题。

    策略：
      1. 只在全文前 20 行中查找
      2. 排除作者行、机构行、页眉页脚、DOI/版权信息
      3. 排除完整句子（正文通常以 "The", "We", "This", "In" 等开头）
      4. 偏好 Title Case / ALL CAPS 的行（论文标题特征）
      5. 存在多行标题时，合并连续的非正文行
    """
    candidates = []

    # 正文句首特征词（标题不应该是完整的句子）
    _sentence_starters = {
        'the ', 'we ', 'this ', 'our ', 'in ', 'it ', 'to ',
        'a ', 'an ', 'for ', 'with ', 'as ', 'at ', 'by ',
        'these ', 'those ', 'they ', 'their ', 'its ',
        'recently ', 'current ', 'existing ', 'most ',
        'however ', 'therefore ', 'specifically ', 'particularly ',
        'despite ', 'while ', 'although ',
    }

    for i, line in enumerate(lines[:20]):
        stripped = line.strip()
        if not stripped or len(stripped) < 8 or len(stripped) > 300:
            continue

        lower = stripped.lower()

        # 排除非标题行
        if re.search(r'@\w+\.\w+', stripped):  # 邮箱
            continue
        if re.match(r'^(Copyright|©|Published|Accepted|Received|Manuscript|Digital Object|Vol\.?|No\.?|pp\.?|DOI[: ]|https?://|arXiv:|See https)', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d{1,6}$', stripped):  # 纯数字（页码/文章ID）
            continue
        if re.match(r'^(Abstract|摘要|Introduction|引言)', stripped, re.IGNORECASE):
            continue
        if re.match(r'^(IEEE|ACM|Springer|TRANSACTIONS|JOURNAL|LETTERS|PROCEEDINGS)', stripped):
            continue
        if re.match(r'^\d+[A-Z]', stripped):  # "1University of..."
            continue
        if stripped.count(',') >= 3 and len(stripped) < 200:  # 作者行
            continue
        if re.search(r'(Corresponding author|e-mail:|@\w+\.\w+)', stripped, re.IGNORECASE):
            continue
        # 排除完整句子开头的正文行
        if any(lower.startswith(s) for s in _sentence_starters):
            continue
        # 排除长正文行（包含 Figure/Table 引用 或 以小写开头）
        if len(stripped) > 100 and stripped[0].islower():
            continue
        if re.search(r'(shown in Figure|is shown|we propose|our method|this paper|in this|et al\.)', stripped, re.IGNORECASE):
            continue

        candidates.append((i, stripped))

    if not candidates:
        # 放宽条件兜底
        for i, line in enumerate(lines[:30]):
            stripped = line.strip()
            if stripped and 5 < len(stripped) < 300:
                if not re.search(r'@\w+\.\w+', stripped) and not re.match(r'^\d{1,6}$', stripped):
                    candidates.append((i, stripped))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0])

    # 得分：早期行 + 长度适中 + Title Case 加分
    def score(c):
        idx, text = c
        position_score = 1.0 if idx < 5 else 0.5 if idx < 10 else 0.1
        length_score = min(len(text), 120)  # 120 字符以上不加分
        # Title Case / ALL CAPS 加分
        words = text.split()
        if words:
            caps_count = sum(1 for w in words if w and w[0].isupper())
            caps_ratio = caps_count / len(words)
            caps_bonus = 1.5 if caps_ratio > 0.7 else 1.0
        else:
            caps_bonus = 1.0
        return position_score * length_score * caps_bonus

    best = max(candidates, key=score)
    title = best[1]

    # 多行标题合并：向前后两个方向合并 Title Case 行
    # 先向后合并
    best_idx = best[0]
    for offset in range(1, 3):
        next_idx = best_idx + offset
        if next_idx >= len(lines):
            break
        next_stripped = lines[next_idx].strip()
        if not next_stripped or len(next_stripped) < 5:
            break
        if _is_author_or_meta_line(next_stripped):
            break
        # Title Case 检查
        words = next_stripped.split()
        if words:
            caps_count = sum(1 for w in words if w and w[0].isupper())
            if caps_count / len(words) < 0.5:
                break
        combined = title + " " + next_stripped
        if len(combined) > 300:
            break
        title = combined

    # 再向前合并（处理标题第一行不是最佳候选的情况）
    for offset in range(1, 3):
        prev_idx = best_idx - offset
        if prev_idx < 0:
            break
        prev_stripped = lines[prev_idx].strip()
        if not prev_stripped or len(prev_stripped) < 5:
            break
        if _is_author_or_meta_line(prev_stripped):
            break
        words = prev_stripped.split()
        if words:
            caps_count = sum(1 for w in words if w and w[0].isupper())
            if caps_count / len(words) < 0.5:
                break
        combined = prev_stripped + " " + title
        if len(combined) > 300:
            break
        title = combined

    return title[:300]


def _is_author_or_meta_line(text: str) -> bool:
    """检查一行是否是作者/机构/元数据行（非标题）。"""
    if re.search(r'@\w+\.\w+', text):
        return True
    if re.match(r'^\d+[A-Z]', text):  # 机构行 "1University of..."
        return True
    if re.match(r'^\d{1,6}$', text):  # 纯数字
        return True
    # 作者行：包含逗号 + 首字母大写人名模式
    if text.count(',') >= 1 and re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text):
        if re.search(r'(Student Member|Senior Member|Fellow|IEEE|ACM|University|Institute|Laboratory|College|School|Department)', text, re.IGNORECASE):
            return True
    if re.match(r'^(Abstract|Introduction|I\.\s|IEEE|TRANSACTIONS|JOURNAL|Keywords|Index Terms|Manuscript|Received|Accepted|Published|Digital Object)', text, re.IGNORECASE):
        return True
    if re.search(r'(Corresponding author|e-mail:)', text, re.IGNORECASE):
        return True
    if text.lower().startswith(('the ', 'we ', 'this ', 'our ', 'in this', 'for the')):
        return True
    return False


def _extract_authors(first_block: str, title: str) -> str:
    """从论文头部提取作者列表。"""
    lines = first_block.splitlines()
    author_candidates = []

    # 定位标题行
    title_idx = -1
    for i, line in enumerate(lines):
        if title and title[:50] in line:
            title_idx = i
            break

    # 搜索标题后的作者行
    search_start = max(0, title_idx) if title_idx >= 0 else 0
    for i in range(search_start, min(search_start + 20, len(lines))):
        stripped = lines[i].strip()
        if not stripped:
            continue
        # 去除上标数字和特殊符号，方便匹配姓名
        cleaned = re.sub(r'[\d,†*∗†‡§¶‖]+', '', stripped)
        cleaned = cleaned.strip()
        # 作者行特征：包含多个首字母大写的姓名
        if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', cleaned):
            if not re.search(r'@\w+\.\w+', stripped):  # 排除纯邮箱行
                if len(stripped) < 500:  # 作者行通常不会太长
                    author_candidates.append(stripped)

    if author_candidates:
        return author_candidates[0][:300]

    return ""


def _extract_journal(lines: list[str]) -> str:
    """从页眉/第一页提取期刊名。"""
    # 检查前 20 行的常见期刊名模式
    for line in lines[:20]:
        stripped = line.strip()
        if not stripped:
            continue
        # 常见期刊关键词
        for kw in ['Transactions on', 'Journal of', 'Proceedings of',
                    'Conference on', 'Symposium on', 'Letters', 'Review',
                    'IEEE', 'ACM', 'Springer', 'Elsevier',
                    '学报', '期刊', '杂志']:
            if kw.lower() in stripped.lower():
                return stripped[:200]
    return ""


def _extract_keywords(text: str) -> str:
    """提取关键词列表。"""
    # 英文
    m = re.search(
        r'(?:Keywords?|Key\s*Words?|Index\s*Terms?)\s*[：:]\s*(.+?)(?:\n\n|\n[A-Z])',
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        kw = m.group(1).strip().replace('\n', ' ')
        return re.sub(r'\s{2,}', ' ', kw)[:500]

    # 中文
    m = re.search(
        r'(?:关键词|关键\s*词)\s*[：:]\s*(.+?)(?:\n\n|\n[^\s]{2,})',
        text, re.DOTALL
    )
    if m:
        kw = m.group(1).strip().replace('\n', ' ')
        return re.sub(r'\s{2,}', ' ', kw)[:500]

    return ""


# ================================================================
# 摘要 / 参考文献 提取
# ================================================================

def extract_abstract(text: str, sections: list[dict]) -> str:
    """提取摘要文本。"""
    # 从章节列表中找
    for sec in sections:
        if sec.get("type") == "abstract":
            return sec["content"]

    # 头部模式匹配（兜底）
    m = re.search(
        r'(?:^Abstract[—–\-:\s]+\s*|^ABSTRACT[—–\-:\s]+\s*|^摘\s*要[：:\s]+|^【摘\s*要】\s*)'
        r'(.+?)(?:\n(?:[A-Z][a-z]+\s*$|引言|Introduction|1\.|I\.))',
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    return ""


def extract_references(text: str, sections: list[dict]) -> str:
    """提取参考文献文本。"""
    for sec in sections:
        if sec.get("type") == "references":
            return sec["content"]
    return ""


# ================================================================
# 主编排函数
# ================================================================

def parse_pdf(file_path: str) -> ParsedPaper:
    """
    解析 PDF 论文的主入口。

    流程: 提取文本 → 清洗 → 提取元数据 → 检测章节 → 摘取摘要/参考文献

    参数:
        file_path: PDF 文件路径

    返回:
        ParsedPaper: 结构化论文数据
    """
    path = Path(file_path)
    print(f"  [PAPER] 正在解析: {path.name}")

    # 1. 提取原始文本
    raw_text = extract_pdf_text(file_path)

    # 2. 清洗文本
    cleaned_text = clean_paper_text(raw_text)

    # 3. 提取元数据
    metadata = extract_metadata(cleaned_text, file_path)

    # 4. 检测章节
    sections = detect_sections(cleaned_text)

    # 5. 提取摘要和参考文献
    abstract = extract_abstract(cleaned_text, sections)
    references = extract_references(cleaned_text, sections)

    # 6. 过滤正文章节（排除摘要和参考文献）
    body_sections = [
        s for s in sections
        if s["type"] not in ("abstract", "references")
    ]

    print(f"  [PAPER] [OK] {path.name}: "
          f"标题={metadata.get('title', '?')[:40]}..., "
          f"{len(body_sections)} 章节, "
          f"摘要={'Y' if abstract else 'N'}, "
          f"DOI={'Y' if metadata.get('doi') else 'N'}")

    return ParsedPaper(
        file_path=str(path.absolute()),
        full_text=cleaned_text,
        metadata=metadata,
        abstract=abstract,
        body_sections=body_sections,
        references=references,
    )
