"""
docling_parser.py — 基于 Docling 的科研论文 PDF → Markdown 解析器
===================

使用 IBM Docling 将 PDF 转换为结构化 Markdown，保留：
  - 标题层级（##, ###）
  - 段落、列表、表格
  - 正确的阅读顺序（自动处理多栏排版）

策略：
  Docling 核心能力是 PDF → Markdown（深度学习布局分析 + 阅读顺序），
  我们在这个可靠的 Markdown 输出上做切分，而不是自己拼装元素树。

与 paper_parser.py 的关系：
  - paper_parser.py：启发式规则 + 多后端文本提取
  - docling_parser.py：Docling 深度学习模型 → Markdown → 结构完整

使用方式：
  from pdf_pipeline.parser import parse_pdf_docling
  result = parse_pdf_docling("paper.pdf")
  print(result.markdown)  # 完整的 Markdown 文本
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# 确保 HF 镜像优先设置（国内网络环境）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 预导入 docling（Windows 上必须主线程早期初始化，否则 segfault）
from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: E402
from docling.datamodel.pipeline_options import (  # noqa: E402
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.datamodel.base_models import InputFormat  # noqa: E402


# ================================================================
# 数据类
# ================================================================

@dataclass
class DoclingParseResult:
    """
    Docling 解析结果。

    Attributes:
        file_path: 源文件路径
        markdown: Docling 导出的完整 Markdown 文本
        metadata: 元数据 {title, authors, doi, year}
        page_count: 总页数
    """
    file_path: str
    markdown: str
    metadata: dict = field(default_factory=dict)
    page_count: int = 0
    bindings_path: str = ""  # bindings.json 侧车文件路径
    page_map_path: str = ""  # page_map.json 侧车文件路径
    content_hash: str = ""   # SHA256 of PDF bytes


# ================================================================
# Pipeline 配置（CPU-only，轻量）
# ================================================================

_converter = None


def _make_pipeline_options():
    """创建轻量级 pipeline 配置（CPU-only, 无 OCR）。"""
    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU,
        num_threads=2,
    )
    # 关闭重度功能 → 大幅降低内存
    opts.do_ocr = False
    # 表结构识别是 OOM（std::bad_alloc）重灾区；默认开启，内存不足时
    # 可设 DOCLING_TABLE_STRUCTURE=0 关闭（牺牲表格结构化，保全文覆盖）
    opts.do_table_structure = os.getenv("DOCLING_TABLE_STRUCTURE", "1") != "0"
    # threaded pipeline 默认按 4 页一组批处理，多页页面平面累积导致
    # std::bad_alloc（26 页论文在 page 8 起整段失败）；逐页处理降低峰值内存
    opts.ocr_batch_size = 1
    opts.layout_batch_size = 1
    opts.table_batch_size = 1
    opts.do_picture_description = False
    opts.do_picture_classification = False
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.generate_table_images = False
    return opts


def _get_converter():
    """获取 DocumentConverter 单例。"""
    global _converter
    if _converter is None:
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=_make_pipeline_options(),
                ),
            },
        )
    return _converter


def _reset_converter():
    """释放 converter（OOM 恢复）。"""
    global _converter
    _converter = None
    import gc
    gc.collect()


# ================================================================
# 主入口
# ================================================================

def parse_pdf_docling(file_path: str, export_bindings: bool = False) -> DoclingParseResult:
    """
    使用 Docling 解析 PDF，返回完整的 Markdown 文本。

    流程:
      1. Docling 布局分析 → 识别标题/段落/表格/公式/图表
      2. 导出为 Markdown（保留结构）
      3. 提取元数据
      4. (可选) 导出空间绑定 → bindings.json

    参数:
        file_path: PDF 文件路径
        export_bindings: 是否导出 bindings.json 侧车文件

    返回:
        DoclingParseResult: markdown + metadata + (可选) bindings_path
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

    safe_name = path.name.encode('ascii', 'replace').decode('ascii')
    print(f"  [DOCLING] 解析中 (CPU/轻量): {safe_name}")

    converter = _get_converter()

    try:
        conv_result = converter.convert(str(path))
    except MemoryError:
        _reset_converter()
        raise RuntimeError(f"Docling OOM: {path.name}。请尝试更小的 PDF。")
    except Exception as e:
        msg = str(e)
        if "bad allocation" in msg or "bad_alloc" in msg:
            _reset_converter()
            raise RuntimeError(f"Docling 内存不足: {path.name}。原始: {e}")
        raise

    doc = conv_result.document

    # ---- Markdown 导出 ----
    markdown = doc.export_to_markdown(enable_chart_tables=False)

    # ---- 元数据提取 (Docling structured metadata first, regex fallback) ----
    doc_meta = _extract_docling_metadata(doc)
    metadata = _extract_metadata_from_markdown(markdown, file_path, doc_meta)

    # ---- arXiv ID ----
    arxiv_id = _extract_arxiv_id(markdown)
    if arxiv_id:
        metadata["arxiv_id"] = arxiv_id

    # ---- SHA256 内容指纹 ----
    content_hash = _compute_sha256(file_path)
    metadata["content_hash"] = content_hash

    # ---- 页码 ----
    page_count = _count_pages(doc)

    # ---- 文本→页码映射 ----
    page_map = _build_page_map(doc)

    safe_name = path.name.encode('ascii', 'replace').decode('ascii')
    safe_title = metadata.get('title', '?')[:40].encode('ascii', 'replace').decode('ascii')
    print(f"  [DOCLING] [OK] {safe_name}: "
          f"{len(markdown)} 字符 Markdown, "
          f"~{page_count} 页, "
          f"标题={safe_title}")

    result = DoclingParseResult(
        file_path=str(path.absolute()),
        markdown=markdown,
        metadata=metadata,
        page_count=page_count,
        content_hash=content_hash,
    )

    # 空间绑定导出 → pdf_pipeline/output/{pdf_stem}/
    if export_bindings:
        from .bindings import build_bindings, export_bindings_json
        bindings = build_bindings(doc, markdown, str(path))
        bpath = str(Path(bindings["paper"]["assets_dir"]) / "bindings.json")
        export_bindings_json(bindings, bpath)
        result.bindings_path = bpath

        # 导出 page_map 用于 chunk→页码注入
        pmap_path = str(Path(bindings["paper"]["assets_dir"]) / "page_map.json")
        _export_page_map(page_map, pmap_path)
        result.page_map_path = pmap_path

        n_elem = len(bindings.get("elements", []))
        n_refs = len(bindings.get("references", []))
        print(f"  [BINDINGS] {Path(bpath).name}: {n_elem} elements, {n_refs} references, "
              f"{len(page_map)} page anchors")

    return result


# ================================================================
# 元数据提取（从 Markdown 文本中启发式提取）
# ================================================================

def _extract_metadata_from_markdown(md: str, file_path: str,
                                     doc_meta: dict | None = None) -> dict:
    """从 Markdown 中提取元数据，优先使用 Docling 结构化元数据。

    Priority: Docling document metadata → markdown regex heuristics.
    Each field is tagged with its source in _sources dict.
    """
    metadata = {
        "title": "",
        "authors": "",
        "doi": "",
        "year": "",
        "filename": Path(file_path).name,
        "_sources": {},  # ponytail: provenance tags per field
    }

    # ---- Layer 1: Docling structured metadata ----
    if doc_meta:
        if doc_meta.get("title"):
            metadata["title"] = str(doc_meta["title"])[:300]
            metadata["_sources"]["title"] = "docling"
        if doc_meta.get("authors"):
            metadata["authors"] = str(doc_meta["authors"])[:500]
            metadata["_sources"]["authors"] = "docling"
        if doc_meta.get("doi"):
            metadata["doi"] = str(doc_meta["doi"])
            metadata["_sources"]["doi"] = "docling"
        if doc_meta.get("year"):
            metadata["year"] = str(doc_meta["year"])
            metadata["_sources"]["year"] = "docling"

    # ---- Layer 2: Markdown regex fallback for missing fields ----
    lines = md.splitlines()
    first_10 = lines[:10] if len(lines) >= 10 else lines

    if not metadata["title"]:
        for line in first_10:
            stripped = line.strip()
            if stripped.startswith("## ") and len(stripped) > 5:
                metadata["title"] = stripped[3:].strip()[:300]
                metadata["_sources"]["title"] = "regex_heuristic"
                break

    if not metadata["authors"]:
        found_title = False
        author_lines = []
        for line in lines[:20]:
            stripped = line.strip()
            if stripped.startswith("## ") and len(stripped) > 5:
                found_title = True
                continue
            if found_title and stripped:
                if re.search(r'@\w+\.\w+', stripped) or re.search(r'[A-Z][a-z]+\s+[A-Z]', stripped):
                    author_lines.append(stripped)
                elif author_lines:
                    break
        if author_lines:
            metadata["authors"] = "; ".join(author_lines)[:500]
            metadata["_sources"]["authors"] = "regex_heuristic"

    if not metadata["doi"]:
        doi_match = re.search(r'\b(10\.\d{4,}/[^\s]{3,})\b', md)
        if doi_match:
            metadata["doi"] = doi_match.group(1).rstrip('.')
            metadata["_sources"]["doi"] = "regex_heuristic"

    if not metadata["year"]:
        if metadata["doi"]:
            year_match = re.search(r'/(20\d{2})/', metadata["doi"] + "/")
            if year_match:
                metadata["year"] = year_match.group(1)
                metadata["_sources"]["year"] = "regex_heuristic"
        if not metadata["year"]:
            year_match = re.search(r'©\s*(20\d{2})', md)
            if year_match:
                metadata["year"] = year_match.group(1)
                metadata["_sources"]["year"] = "regex_heuristic"

    return metadata


def _extract_docling_metadata(doc) -> dict:
    """Extract metadata from Docling document object.

    Returns a dict with available fields (title, authors, doi, year).
    All values are strings; empty string means not found.
    """
    meta: dict[str, str] = {}
    try:
        # Docling may expose metadata via document.info or similar accessors
        if hasattr(doc, 'name') and doc.name:
            # doc.name often contains the paper title for academic PDFs
            meta["title"] = str(doc.name).strip()
        # Try PDF info dict
        if hasattr(doc, '_document') and hasattr(doc._document, 'info'):
            info = doc._document.info
            for field in ('title', 'author', 'doi', 'year'):
                val = getattr(info, field, None) or info.get(field, None)
                if val:
                    key = "authors" if field == "author" else field
                    meta[key] = str(val).strip()
    except Exception:
        pass
    return meta


def _count_pages(doc) -> int:
    """估算页数。"""
    try:
        pages = set()
        for item, _ in doc.iterate_items():
            if hasattr(item, 'prov') and item.prov:
                for p in item.prov:
                    if hasattr(p, 'pageno') and p.pageno:
                        pages.add(p.pageno)
        return len(pages) if pages else 0
    except Exception:
        return 0


def _compute_sha256(file_path: str) -> str:
    """计算 PDF 文件的 SHA256 哈希（去重指纹）。"""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _extract_arxiv_id(markdown: str) -> str:
    """从 Markdown 文本中提取 arXiv ID。"""
    # arXiv ID 格式: arXiv:XXXX.XXXXX 或 arXiv:XXXX.XXXXXvN
    m = re.search(r'ar[xX]iv\s*:\s*(\d{4}\.\d{4,5}(?:v\d+)?)', markdown)
    if m:
        return m.group(1)
    # 也匹配 abs/格式的 arXiv URL
    m = re.search(r'ar[xX]iv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)', markdown)
    if m:
        return m.group(1)
    return ""


def _build_page_map(doc) -> list[dict]:
    """从 Docling document 构建文本→页码映射。

    遍历所有 TextItem 和 SectionHeaderItem，记录文本片段与页码的对应关系。
    后续 chunker 用此映射确定每个 chunk 的页码范围。

    Returns:
        [{text: str, page_no: int}, ...]
    """
    entries: list[dict] = []
    for item, _level in doc.iterate_items():
        page_no = None
        if hasattr(item, 'prov') and item.prov:
            for p in item.prov:
                # ponytail: docling v2 uses page_no (underscore), not pageno
                pn = getattr(p, 'page_no', None) or getattr(p, 'pageno', None)
                if pn:
                    page_no = pn
                    break

        if page_no is None:
            continue

        text = ""
        if hasattr(item, 'text') and item.text:
            text = item.text.strip()
        elif hasattr(item, 'label'):
            # SectionHeaderItem 等 — 用 label 文本
            pass

        if not text or len(text) < 20:
            continue

        # 只保留前 120 字符作为匹配锚点（足够唯一定位，且不受后续 enrich 影响）
        entries.append({
            "text": text[:120],
            "page_no": page_no,
        })

    # ponytail: collision detection — duplicate 120-char prefixes -> warning
    _check_page_map_collisions(entries)

    return entries


def _check_page_map_collisions(entries: list[dict]):
    """Detect duplicate text anchors in the page map and warn."""
    seen: dict[str, list[int]] = {}
    for e in entries:
        seen.setdefault(e["text"], []).append(e["page_no"])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        pages = sorted(set(p for v in dupes.values() for p in v))
        print(f"  [PAGE-MAP] WARNING: {len(dupes)} duplicate anchors detected "
              f"(pages {pages} share same prefix). "
              f"These chunks may have incorrect page attribution.")


def _export_page_map(page_map: list[dict], output_path: str) -> str:
    """导出 page_map 为 JSON 文件。返回路径。"""
    Path(output_path).write_text(
        json.dumps(page_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
