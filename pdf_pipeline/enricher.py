"""
enricher.py — Stage 4 富化 Markdown 生成
==========

将 Stage 3 增强后的公式描述和图片描述注入回原始 Markdown，
替换 <!-- formula-not-decoded --> 和 <!-- image --> 占位符。

注入规则:
  - 公式: <!-- formula-not-decoded --> → [FORMULA_DESC: ...]
  - 关键图片: <!-- image --> → [FIGURE_DESC: ...]
  - 被跳过图片: 移除占位符（无 caption）或保留 caption
"""

from __future__ import annotations

import re
from pathlib import Path


def inject_enhancements(markdown: str, bindings: dict) -> str:
    """
    将 bindings 中的 formula_desc / picture_desc 注入 Markdown，
    替换对应占位符。

    映射方式: element_id 后缀序号 → 第 N 个同类型占位符（1-indexed）。
    例如 formula_3 → 第 3 个 <!-- formula-not-decoded -->，与元素过滤无关。
    """
    elements = bindings.get("elements", [])
    formulas = [e for e in elements if e["type"] == "formula"]
    pictures = [e for e in elements if e["type"] == "picture"]

    # 找到所有占位符
    formula_phs = list(re.finditer(
        r'<!--\s*formula-not-decoded\s*-->', markdown
    ))
    image_phs = list(re.finditer(
        r'<!--\s*image\s*-->', markdown
    ))

    # 构建替换: (start, end, text)，从后往前应用以避免坐标偏移
    replacements: list[tuple[int, int, str]] = []

    for elem in formulas:
        desc = elem.get("formula_desc", "")
        if not desc:
            continue
        n = int(elem["element_id"].split("_")[1])
        if n - 1 < len(formula_phs):
            m = formula_phs[n - 1]
            replacements.append((m.start(), m.end(), desc))

    for elem in pictures:
        # ponytail: 使用 element_id 后缀序号而非 list 索引，防止过滤导致的错位
        n = int(elem["element_id"].split("_")[1])
        if n - 1 >= len(image_phs):
            continue
        m = image_phs[n - 1]
        desc = elem.get("picture_desc", "")
        caption = elem.get("caption", "")
        if desc:
            replacements.append(
                (m.start(), m.end(), f"[FIGURE_DESC: {desc}]")
            )
        elif caption:
            replacements.append(
                (m.start(), m.end(), f"[FIGURE_CAPTION: {caption}]")
            )
        # ponytail: 无 caption 无 desc → 留空（装饰图片不注入任何内容）

    # 从后往前替换
    replacements.sort(key=lambda x: x[0], reverse=True)
    result = markdown
    for start, end, text in replacements:
        result = result[:start] + text + result[end:]

    # ponytail: post-injection integrity check
    _check_enrichment_integrity(result, formulas, pictures, formula_phs, image_phs)

    return result


def _check_enrichment_integrity(
    result: str,
    formulas: list,
    pictures: list,
    formula_phs: list,
    image_phs: list,
):
    """Post-injection integrity check: detect unmapped elements and leftover placeholders."""
    # Count remaining placeholders
    leftover_formula = len(re.findall(r'<!--\s*formula-not-decoded\s*-->', result))
    leftover_image = len(re.findall(r'<!--\s*image\s*-->', result))

    # Count elements with descriptions that couldn't be mapped
    formulas_with_desc = [e for e in formulas if e.get("formula_desc")]
    pictures_with_desc = [e for e in pictures if e.get("picture_desc")]
    unmapped_formula = max(0, len(formulas_with_desc) - len(formula_phs))
    unmapped_picture = max(0, len(pictures_with_desc) - len(image_phs))

    if leftover_formula > 0 or unmapped_formula > 0:
        print(f"  [ENRICH] WARNING: {unmapped_formula} formula descriptions unmapped "
              f"(only {len(formula_phs)} placeholders for {len(formulas_with_desc)} elements), "
              f"{leftover_formula} placeholders left uninjected")
    if leftover_image > 0 or unmapped_picture > 0:
        print(f"  [ENRICH] WARNING: {unmapped_picture} picture descriptions unmapped "
              f"(only {len(image_phs)} placeholders for {len(pictures_with_desc)} elements), "
              f"{leftover_image} placeholders left uninjected")


def enrich_markdown(markdown: str, bindings: dict,
                    output_path: str = "") -> str:
    """
    完整富化: 注入增强 → 保存 final_enriched.md。

    Returns: 富化后的 Markdown 文本。
    """
    enriched = inject_enhancements(markdown, bindings)

    if not output_path:
        # 默认保存到 bindings 同目录
        assets_dir = bindings.get("paper", {}).get("assets_dir", "")
        if assets_dir:
            output_path = str(Path(assets_dir) / "final_enriched.md")

    if output_path:
        Path(output_path).write_text(enriched, encoding="utf-8")
        n_f = len([e for e in bindings.get("elements", [])
                    if e["type"] == "formula" and e.get("formula_desc")])
        n_p = len([e for e in bindings.get("elements", [])
                    if e["type"] == "picture" and e.get("picture_desc")])
        print(f"  [ENRICH] {n_f} formulas + {n_p} figures injected")
        print(f"  [ENRICH] → {output_path}")

    return enriched
