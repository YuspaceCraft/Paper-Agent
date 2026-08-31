"""
binding_export.py — Docling PDF 空间绑定 & 引用回溯
=================

从 Docling 布局分析结果中提取图片/表格/公式的空间坐标，
回溯正文中的引用关系，输出 bindings.json 侧车文件。

三部分:
  1. extract_bindings_from_doc()  — bbox + caption 提取
  2. backtrack_references()       — 多语言引用正则匹配
  3. validate_bindings()          — 完整性校验

ponytail: 一个文件，三个函数，全在 300 行内。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF: PDF 页面渲染 + 图片裁剪

# Docling 类型（模块级导入，避免 agent/__init__.py 级联）
from docling_core.types.doc import (  # noqa: E402
    DocItem,
    DocItemLabel,
    TextItem,
    TableItem,
    PictureItem,
    FormulaItem,
)


# ================================================================
# 数据类
# ================================================================

@dataclass
class BoundElement:
    """一个空间绑定元素（图/表/公式）。"""
    element_id: str           # "fig_1", "table_2", "formula_3"
    type: str                 # "picture" | "table" | "formula"
    page_no: int
    bbox: list[float]         # [l, t, r, b]
    caption: str              # 提取的标题文本，空串表示未找到
    caption_page: int


# ================================================================
# 标题检测模式（多语言）
# ================================================================

_CAPTION_PATTERNS: list[re.Pattern] = [
    # 英文
    re.compile(r'(?:Figure|Fig\.?)\s*\d+', re.IGNORECASE),
    re.compile(r'(?:Table|TABLE)\s*[IVX\d]+', re.IGNORECASE),
    re.compile(r'(?:Eq(?:uation)?\.?)\s*\(?\d+\)?', re.IGNORECASE),
    # 土耳其语
    re.compile(r'(?:[SŞ]ekil)\s*\.?\s*\d+', re.IGNORECASE),
    re.compile(r'(?:TABLO|Tablo)\s*\.?\s*[IVX\d]+', re.IGNORECASE),
    re.compile(r'(?:Denklem)\s*\.?\s*\(?\d+\)?', re.IGNORECASE),
    # 中文
    re.compile(r'(?:图|图表)\s*\d+'),
    re.compile(r'(?:表|表格)\s*[IVX\d]+'),
    re.compile(r'(?:公式|方程)\s*\(?\d+\)?'),
]


def _is_caption(text: str) -> bool:
    """判断文本块是否是图表标题。"""
    stripped = text.strip()
    # ponytail: 去掉行首非字母字符（Docling 可能插入 cedilla 等杂音）
    cleaned = re.sub(r'^[^a-zA-Z一-鿿ŞşĞğÇçÖöÜüİı]+', '', stripped)
    for pat in _CAPTION_PATTERNS:
        if pat.match(cleaned):
            return True
    return False


def _caption_matches_type(caption: str, elem_type: str) -> bool:
    """验证标题文本与元素类型一致（防止 TABLE 标题匹配到 formula）。"""
    c = caption.strip().lower()
    if elem_type == "picture":
        # 图片标题: figure/fig/şekil/sekil/图
        return any(kw in c for kw in ("figure", "fig.", "fig ", "şekil", "sekil", "图"))
    elif elem_type == "table":
        # 表格标题: table/tablo/表
        return any(kw in c for kw in ("table", "tablo", "表"))
    elif elem_type == "formula":
        # 公式通常无标题，跳过
        return False
    return False


# ================================================================
# Stage 1: bbox + caption + 内容提取
# ================================================================

def _resolve_caption_from_cref(item, doc) -> str:
    """从 Docling cref 指针解析标题文本（更可靠）。"""
    captions = getattr(item, 'captions', []) or []
    for cap in captions:
        cref = getattr(cap, 'cref', '')
        # cref 格式: "#/texts/15" → index 15 in doc.texts
        m = re.search(r'/texts/(\d+)', cref)
        if m:
            idx = int(m.group(1))
            if idx < len(doc.texts):
                t = doc.texts[idx]
                if hasattr(t, 'text') and t.text:
                    return t.text.strip()
    return ""


def extract_bindings_from_doc(doc, markdown: str = "",
                              assets_dir: str = "") -> dict:
    """
    遍历 Docling document，提取所有图片/表格/公式及其 bbox + 标题 +
    公式文本。

    参数:
        doc: DoclingDocument
        markdown: 导出的 markdown 文本（未使用，保留接口兼容）
        assets_dir: 资源输出目录，空串表示不创建资源文件

    返回 bindings dict（可直接序列化为 bindings.json）。
    """
    # 按页收集文本项（用于标题匹配后备方案）
    page_texts: dict[int, list[dict]] = {}

    for item, _level in doc.iterate_items():
        if not isinstance(item, DocItem):
            continue
        if isinstance(item, TextItem) and item.prov:
            page_no = item.prov[0].page_no
            text = item.text.strip() if item.text else ""
            if text and len(text) > 3:
                if page_no not in page_texts:
                    page_texts[page_no] = []
                bbox = item.prov[0].bbox
                page_texts[page_no].append({
                    "text": text,
                    "t": float(bbox.t),
                    "b": float(bbox.b),
                    "l": float(bbox.l),
                })

    # 收集图片/表格/公式
    elements: list[dict] = []
    counters: dict[str, int] = {}

    for item, _level in doc.iterate_items():
        if not isinstance(item, DocItem) or not item.prov:
            continue

        elem_type = None
        if isinstance(item, PictureItem):
            elem_type = "picture"
        elif isinstance(item, TableItem):
            elem_type = "table"
        elif isinstance(item, FormulaItem):
            elem_type = "formula"

        if elem_type is None:
            continue

        counters.setdefault(elem_type, 0)
        counters[elem_type] += 1
        cnt = counters[elem_type]

        bbox = item.prov[0].bbox
        page_no = item.prov[0].page_no
        bbox_list = [round(float(bbox.l), 1), round(float(bbox.t), 1),
                     round(float(bbox.r), 1), round(float(bbox.b), 1)]

        elem_id = f"{elem_type}_{cnt}"

        # 标题：优先用 cref 解析，后备为近邻匹配
        caption = _resolve_caption_from_cref(item, doc)
        if not caption:
            caption = _find_caption_nearby(
                bbox_list, page_no, page_texts.get(page_no, []),
                elem_type=elem_type,
                max_gap=400.0,
            )

        elem_data = {
            "element_id": elem_id,
            "type": elem_type,
            "page_no": page_no,
            "bbox": bbox_list,
            "caption": caption,
            "caption_page": page_no,
        }

        # 公式文本（Stage 3 多模态增强用）
        if elem_type == "formula":
            orig = getattr(item, 'orig', '') or ''
            text = getattr(item, 'text', '') or ''
            elem_data["formula_text"] = orig or text
            elem_data["formula_path"] = (
                f"{elem_id}.txt" if assets_dir else ""
            )

        # 图片资源路径（Stage 3 裁剪用）
        if elem_type == "picture":
            elem_data["image_path"] = (
                f"{elem_id}.png" if assets_dir else ""
            )

        elements.append(elem_data)

    page_count = len(doc.pages) if hasattr(doc, 'pages') else 0

    # ponytail: caption miss logging
    _log_caption_coverage(elements)

    return {
        "paper": {"file": "", "pages": page_count},
        "elements": elements,
        "references": [],
    }


def _find_caption_nearby(
    bbox: list[float],
    page_no: int,
    page_items: list[dict],
    elem_type: str = "",
    max_gap: float = 200.0,
) -> str:
    """
    在同一页内找离元素最近的标题文本。

    ponytail: 简单垂直距离扫描。标题通常在元素正下方或上方，
    取垂直距离最小且匹配标题模式的文本。
    同时验证标题类型与元素类型一致（TABLE 标题不匹配 picture）。
    """
    elem_mid_y = (bbox[1] + bbox[3]) / 2.0
    best_text = ""
    best_dist = float("inf")

    for pi in page_items:
        if not _is_caption(pi["text"]):
            continue
        # 验证标题类型与元素类型匹配
        if not _caption_matches_type(pi["text"], elem_type):
            continue
        text_mid_y = (pi["t"] + pi["b"]) / 2.0
        gap = abs(text_mid_y - elem_mid_y)
        if gap < best_dist and gap < max_gap:
            best_dist = gap
            best_text = pi["text"]

    return best_text


def _log_caption_coverage(elements: list[dict]):
    """Log how many elements have captions, by type. Warn if coverage is low."""
    by_type: dict[str, dict] = {}
    for e in elements:
        t = e.get("type", "unknown")
        by_type.setdefault(t, {"total": 0, "captioned": 0})
        by_type[t]["total"] += 1
        if e.get("caption"):
            by_type[t]["captioned"] += 1

    for t, counts in sorted(by_type.items()):
        total = counts["total"]
        captioned = counts["captioned"]
        missing = total - captioned
        if missing > 0:
            pct = missing / total * 100
            level = "WARNING" if pct > 30 else "INFO"
            print(f"  [BINDINGS] {level}: {missing}/{total} {t} elements "
                  f"have no caption ({pct:.0f}%)")


# ================================================================
# Stage 2: 引用回溯
# ================================================================

# --- element_id 匹配辅助 ---

def _roman_to_int(s: str) -> int | None:
    """罗马数字 → 整数，失败返回 None。"""
    try:
        vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        result = 0
        prev = 0
        for ch in s.upper():
            v = vals.get(ch)
            if v is None:
                return None
            result += v
            if v > prev:
                result -= 2 * prev
            prev = v
        return result
    except (ValueError, KeyError):
        return None


def _normalize_ref_number(num_str: str, ref_type: str) -> str:
    """将引用编号统一为整数形式的字符串。"""
    num_str = num_str.strip()
    # 罗马数字 → 整数
    if re.match(r'^[IVX]+$', num_str, re.IGNORECASE):
        val = _roman_to_int(num_str)
        if val is not None:
            return str(val)
    # 阿拉伯数字
    m = re.search(r'(\d+)', num_str)
    if m:
        return m.group(1)
    return num_str


def _build_element_lookup(elements: list[dict]) -> dict[str, str]:
    """
    构建 {(type, number): element_id} 查找表。

    例如: {("picture", "1"): "picture_1", ("table", "3"): "table_3"}
    type 使用 element 自身的 type 字段（picture/table/formula）。
    """
    lookup: dict[str, str] = {}
    for elem in elements:
        eid = elem["element_id"]  # e.g. "picture_1", "table_2"
        etype = elem["type"]      # "picture" | "table" | "formula"
        # 从 element_id 末尾取数字
        m = re.search(r'(\d+)$', eid)
        if m:
            key = f"{etype}_{m.group(1)}"
            lookup[key] = eid
    return lookup


# --- 多语言引用正则 ---

def _build_ref_regex() -> re.Pattern:
    """构建匹配正文中图表引用的组合正则（多语言）。"""
    patterns = [
        # 英文 Figure / Fig.
        r'(?:(?:Figure|Fig\.?)\s*(\d+))',
        # 土耳其语 Şekil / Sekil
        r'(?:[ŞS]ekil\s*\.?\s*(\d+))',
        # 中文 图
        r'(?:图\s*(\d+))',
        # 英文 Table
        r'(?:Table\s*\.?\s*([IVX\d]+))',
        # 土耳其语 TABLO / Tablo
        r'(?:TABLO|Tablo)\s*\.?\s*([IVX\d]+)',
        # 中文 表
        r'(?:表\s*([IVX\d]+))',
        # 英文 Eq. / Equation
        r'(?:Eq(?:uation)?\.?\s*\(?(\d+)\)?)',
        # 土耳其语 Denklem
        r'(?:Denklem\s*\.?\s*\(?(\d+)\)?)',
        # 中文 公式
        r'(?:公式|方程)\s*\(?(\d+)\)?',
    ]
    return re.compile('|'.join(patterns), re.IGNORECASE)


_REF_PATTERN_MAP: list[tuple[str, int]] = [
    # (pattern_prefix, type_index)
    # 九个模式，按在 _build_ref_regex 中的顺序
]

_REF_REGEX = _build_ref_regex()


def _classify_ref_type(match: re.Match) -> str:
    """根据匹配到的分组确定引用类型（返回 element type: picture/table/formula）。"""
    groups = match.groups()
    # 模式顺序: Figure(0) | Şekil(1) | 图(2) | Table(3) | TABLO(4) | 表(5) | Eq(6) | Denklem(7) | 公式(8)
    # groups 中只有一个非 None
    for i, g in enumerate(groups):
        if g is not None:
            if i <= 2:
                return "picture"   # ponytail: "figure" ref → "picture" element type
            elif i <= 5:
                return "table"
            else:
                return "formula"
    return "unknown"


def backtrack_references(markdown: str, bindings: dict) -> list[dict]:
    """
    扫描 markdown 全文，匹配正文中对图表公式的引用，
    并与 bindings["elements"] 中的元素关联。

    返回: reference dict 列表
    """
    elements = bindings.get("elements", [])
    elem_lookup = _build_element_lookup(elements)

    refs: list[dict] = []
    seen: set[tuple[int, str]] = set()  # (char_pos, target_id) 去重

    for m in _REF_REGEX.finditer(markdown):
        ref_text = m.group(0)
        ref_type = _classify_ref_type(m)
        ref_number_raw = next(g for g in m.groups() if g is not None)
        ref_number = _normalize_ref_number(ref_number_raw, ref_type)
        char_pos = m.start()

        # 匹配 element
        lookup_key = f"{ref_type}_{ref_number}"
        target_id = elem_lookup.get(lookup_key)

        # 去重同一位置
        dedup_key = (char_pos // 100, lookup_key)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # 提取上下文
        ctx_start = max(0, char_pos - 150)
        ctx_end = min(len(markdown), char_pos + 200)
        context = markdown[ctx_start:ctx_end].replace('\n', ' ').strip()

        refs.append({
            "ref_text": ref_text,
            "ref_type": ref_type,
            "ref_number": ref_number,
            "target_element_id": target_id or "",
            "char_pos": char_pos,
            "context": context[:300],
        })

    return refs


# ================================================================
# Stage 4: 质量校验
# ================================================================

def validate_bindings(bindings: dict, chunk_report=None) -> dict:
    """
    完整性校验:
      - total_elements: 总元素数
      - elements_without_refs: 正文中从未引用的元素
      - orphan_refs: 匹配不到任何元素的引用
      - ref_coverage: 被引用的元素比例
    """
    elements = bindings.get("elements", [])
    references = bindings.get("references", [])

    # 收集被引用的 element_id
    ref_targets: set[str] = set()
    for r in references:
        tid = r.get("target_element_id", "")
        if tid:
            ref_targets.add(tid)

    # 收集所有 element_id
    all_ids = {e["element_id"] for e in elements}

    elements_without_refs = sorted(all_ids - ref_targets)
    orphan_refs = [
        r for r in references
        if not r.get("target_element_id")
    ]

    result = {
        "total_elements": len(elements),
        "total_references": len(references),
        "elements_without_refs": elements_without_refs,
        "elements_without_ref_count": len(elements_without_refs),
        "orphan_refs": orphan_refs,
        "orphan_ref_count": len(orphan_refs),
        "ref_coverage": (len(ref_targets) / len(all_ids)) if all_ids else 0,
    }

    if chunk_report:
        # 每 chunk 绑定元素数
        chunks_with_bindings = 0
        for ch in chunk_report.chunks:
            bound = ch.metadata.get("bound_elements", [])
            if bound:
                chunks_with_bindings += 1
        result["chunks_total"] = chunk_report.total_chunks
        result["chunks_with_bindings"] = chunks_with_bindings

    return result


# ================================================================
# JSON I/O
# ================================================================

def export_bindings_json(bindings: dict, output_path: str) -> str:
    """导出 bindings 为 JSON 文件。返回路径。"""
    data = dict(bindings)
    # 补充 paper.file
    if not data["paper"].get("file"):
        stem = Path(output_path).stem.replace("_bindings", "")
        data["paper"]["file"] = stem + ".pdf"
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_bindings_json(path: str) -> dict:
    """加载 bindings.json。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ================================================================
# 端到端便捷 API
# ================================================================

def _crop_images_from_pdf(pdf_path: str, elements: list[dict],
                          output_dir: Path) -> int:
    """
    使用 PyMuPDF 从 PDF 页面按 bbox 裁剪图片，保存为 PNG。

    Docling bbox 坐标原点为 BOTTOMLEFT，PyMuPDF 为 TOPLEFT，
    需转换: y_top = page_height - bbox_top。

    返回成功裁剪的图片数。
    """
    doc = fitz.open(pdf_path)
    cropped = 0

    for elem in elements:
        if elem["type"] != "picture":
            continue

        page_no = elem["page_no"] - 1  # Docling 1-indexed → fitz 0-indexed
        if page_no < 0 or page_no >= len(doc):
            continue

        page = doc[page_no]
        page_h = page.rect.height

        l, t, r, b = elem["bbox"]
        # BOTTOMLEFT → TOPLEFT
        y0 = page_h - t
        y1 = page_h - b
        clip = fitz.Rect(l, y0, r, y1)

        if clip.is_empty or clip.width < 10 or clip.height < 10:
            continue

        # 渲染 300 DPI → 清晰度足够 VLM 识别
        mat = fitz.Matrix(2.5, 2.5)  # ~300 DPI
        pix = page.get_pixmap(clip=clip, matrix=mat)
        fpath = output_dir / f"{elem['element_id']}.png"
        pix.save(str(fpath))

        elem["image_path"] = str(fpath)
        cropped += 1

    doc.close()
    return cropped


def build_bindings(doc, markdown: str, pdf_path: str) -> dict:
    """
    完整流程：提取元素 → 回溯引用 → 裁剪关键图片 → 保存公式。

    输出目录: pdf_pipeline/output/{pdf_stem}/
    只保存有关键引用（有 caption + 正文引用）的图片。
    """
    # 输出目录 — ponytail: sanitize spaces/special chars
    import re
    pdf_stem = re.sub(r'[^\w一-鿿-]', '_', Path(pdf_path).stem)
    pdf_stem = re.sub(r'__+', '_', pdf_stem).strip('_')[:80]
    output_dir = Path(__file__).resolve().parent / "output" / pdf_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    assets_dir = str(output_dir)
    bindings = extract_bindings_from_doc(doc, markdown, assets_dir)
    bindings["paper"]["file"] = Path(pdf_path).name
    bindings["paper"]["assets_dir"] = assets_dir

    # 保存原始 Markdown（Stage 4 注入用）
    (output_dir / "raw.md").write_text(markdown, encoding="utf-8")

    # 先回溯引用，确定哪些元素被正文引用
    bindings["references"] = backtrack_references(markdown, bindings)

    # 收集被引用的 element_id
    ref_targets = {r["target_element_id"] for r in bindings["references"]
                   if r.get("target_element_id")}

    # 过滤图片：只保留有 caption + 被正文引用的关键图片
    key_pictures = []
    dropped = 0
    for elem in bindings["elements"]:
        if elem["type"] == "picture":
            has_caption = bool(elem.get("caption"))
            is_referenced = elem["element_id"] in ref_targets
            if has_caption and is_referenced:
                key_pictures.append(elem)
            else:
                dropped += 1
        else:
            key_pictures.append(elem)
    bindings["elements"] = key_pictures

    # 裁剪关键图片（bbox → PNG）
    n_cropped = _crop_images_from_pdf(pdf_path, bindings["elements"], output_dir)
    if n_cropped:
        print(f"  [CROP] {n_cropped} key images → {output_dir}")
    if dropped:
        print(f"  [CROP] {dropped} non-essential images skipped")

    # 保存公式文本文件
    for elem in bindings["elements"]:
        if elem["type"] == "formula" and elem.get("formula_text"):
            fpath = output_dir / f"{elem['element_id']}.txt"
            fpath.write_text(elem["formula_text"], encoding="utf-8")
            elem["formula_path"] = str(fpath)

    return bindings
