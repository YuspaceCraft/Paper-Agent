"""
pdf_pipeline CLI — 命令行工具：PDF → Markdown → Chunk 可视化
===========================================================

用法:
  python -m pdf_pipeline.cli parse <pdf_path>       # 解析 PDF → Markdown
  python -m pdf_pipeline.cli visualize <pdf_path>   # 全部 + HTML 可视化
  python -m pdf_pipeline.cli bindings <pdf_path>    # 仅导出空间绑定 JSON
  python -m pdf_pipeline.cli enhance <pdf_path>    # Stage 3 多模态增强
  python -m pdf_pipeline.cli enrich <pdf_path>     # Stage 4 富化 Markdown
  python -m pdf_pipeline.cli all <pdf_path>            # 完整四阶段流程
  python -m pdf_pipeline.cli rag-chunk <pdf_path>     # RAG 优化切分 → JSON + HTML
  python -m pdf_pipeline.cli rag-visualize <json>     # 从 rag_chunks.json 重新生成 HTML
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import argparse

# ---- 必须在导入任何其他模块前初始化 docling ----
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _slugify(name: str, max_len: int = 80) -> str:
    """Sanitize a filename stem into a safe directory/filename token.

    Replaces runs of chars that aren't alphanumeric, CJK, dash, or underscore
    with a single underscore. Also strips leading/trailing non-alnum chars.
    """
    import unicodedata
    result: list[str] = []
    for ch in name:
        cat = unicodedata.category(ch)
        # Letters (L*), numbers (N*), dash, underscore → keep
        if cat.startswith("L") or cat.startswith("N") or ch in ("-", "_"):
            result.append(ch)
        elif ch in (" ", ".", ",", "(", ")", "[", "]", "（", "）", "、", "："):
            result.append("_")
        else:
            result.append("_")
    slug = "".join(result)
    # Collapse multiple underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug[:max_len] if len(slug) > max_len else slug

# ---- 必须在导入任何其他模块前初始化 docling ----
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from docling.document_converter import DocumentConverter  # noqa: F401
except ImportError as e:
    print(f"[WARN] docling 未安装: {e}")
    print("[TIP] pip install docling")


def cmd_parse(args):
    """解析 PDF → 导出 Markdown。"""
    from .parser import parse_pdf_docling

    result = parse_pdf_docling(args.file)

    md_path = _slugify(Path(args.file).stem) + "_docling.md"
    Path(md_path).write_text(result.markdown, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  Docling 解析完成")
    print(f"{'=' * 60}")
    print(f"  文件: {Path(args.file).name}")
    print(f"  Markdown: {len(result.markdown):,} 字符, ~{result.page_count} 页")
    if result.metadata.get("title"):
        print(f"  标题: {result.metadata['title'][:80]}")
    print(f"\n  [OK] Markdown 已导出: {md_path}")


def cmd_visualize(args):
    """解析 + 切分 + 生成 HTML 可视化（两步展示 + 空间绑定）。"""
    from .parser import parse_pdf_docling
    from .chunker import chunk_markdown

    html_path = args.html or f"{_slugify(Path(args.file).stem)}_viz.html"

    parsed = parse_pdf_docling(args.file, export_bindings=True)

    _bindings = None
    if parsed.bindings_path:
        try:
            from .bindings import load_bindings_json
            _bindings = load_bindings_json(parsed.bindings_path)
        except Exception:
            pass

    report = chunk_markdown(parsed.markdown, parsed.metadata, bindings=_bindings)

    from .viz import render_html_two_step
    html = render_html_two_step(parsed.markdown, report, title=Path(args.file).name, bindings=_bindings)
    Path(html_path).write_text(html, encoding="utf-8")

    n_elem = len(_bindings.get("elements", [])) if _bindings else 0
    n_refs = len(_bindings.get("references", [])) if _bindings else 0
    print(f"\n  [OK] HTML: {html_path}")
    print(f"  [OK] Step 1: Docling Markdown | Step 2: {report.total_chunks} chunks | Step 3: {n_elem} elements, {n_refs} refs")


def cmd_bindings(args):
    """仅导出空间绑定 JSON（不切分）。"""
    from .parser import parse_pdf_docling
    result = parse_pdf_docling(args.file, export_bindings=True)
    if result.bindings_path:
        from .bindings import load_bindings_json
        b = load_bindings_json(result.bindings_path)
        n_elem = len(b.get("elements", []))
        n_refs = len(b.get("references", []))
        print(f"\n  [OK] bindings.json: {result.bindings_path}")
        print(f"  [OK] {n_elem} elements, {n_refs} references")
    else:
        print("[WARN] 未生成 bindings")


def cmd_enhance(args):
    """Stage 3: 多模态语义增强（公式 LLM + 图片 VLM）。"""
    from .parser import parse_pdf_docling
    from .bindings import load_bindings_json, export_bindings_json
    from .enhancer import enhance_all

    # 先确保 bindings 已生成
    result = parse_pdf_docling(args.file, export_bindings=True)
    if not result.bindings_path:
        print("[ERROR] bindings 生成失败，无法进行增强")
        return

    bindings = load_bindings_json(result.bindings_path)
    output_dir = str(Path(result.bindings_path).parent)

    print(f"\n{'=' * 60}")
    print(f"  Stage 3: 多模态语义增强")
    print(f"{'=' * 60}")
    print(f"  文件: {Path(args.file).name}")

    summary = enhance_all(bindings, result.markdown, output_dir)

    # 回写增强后的 bindings
    export_bindings_json(bindings, result.bindings_path)

    print(f"\n  [OK] 公式增强: {summary['formulas_enhanced']} 个")
    print(f"  [OK] 图片增强: {summary['images_enhanced']} 个")
    print(f"  [OK] bindings.json 已更新: {result.bindings_path}")


def cmd_enrich(args):
    """Stage 4: 将增强描述注入 Markdown → final_enriched.md。"""
    from .parser import parse_pdf_docling
    from .bindings import load_bindings_json
    from .enricher import enrich_markdown

    output_dir = Path(__file__).resolve().parent / "output" / _slugify(Path(args.file).stem)
    bpath = output_dir / "bindings.json"
    raw_md_path = output_dir / "raw.md"

    if not bpath.exists():
        print(f"[ERROR] bindings 不存在: {bpath}")
        print("[TIP] 先运行: python -m pdf_pipeline.cli enhance <pdf>")
        return

    bindings = load_bindings_json(str(bpath))

    # 优先用缓存的 raw.md，避免重新解析
    if raw_md_path.exists():
        markdown = raw_md_path.read_text(encoding="utf-8")
    else:
        result = parse_pdf_docling(args.file, export_bindings=False)
        markdown = result.markdown

    print(f"\n{'=' * 60}")
    print(f"  Stage 4: 富化 Markdown 生成")
    print(f"{'=' * 60}")
    print(f"  文件: {Path(args.file).name}")

    enriched = enrich_markdown(markdown, bindings)

    if args.show:
        for pat, label in [
            (r'\[FORMULA_DESC:', 'Formula injection'),
            (r'\[FIGURE_DESC:', 'Figure injection'),
        ]:
            for m in re.finditer(pat, enriched):
                ctx = enriched[max(0, m.start()-40):m.end()+120]
                ctx = ctx.replace('\n', ' ')[:150]
                print(f"  [{label}] ...{ctx}...")


def cmd_rag_chunk(args):
    """RAG 优化切分: final_enriched.md → 四模块流水线 + HTML 可视化。"""
    from .rag_chunker import (
        rag_chunk_markdown, RAGChunkConfig,
        export_rag_report, print_quality_report, render_rag_html,
    )
    from .bindings import load_bindings_json

    pdf_stem = Path(args.file).stem
    output_dir = Path(__file__).resolve().parent / "output" / _slugify(pdf_stem)
    enriched_path = output_dir / "final_enriched.md"
    bindings_path = output_dir / "bindings.json"

    if not enriched_path.exists():
        print(f"[ERROR] final_enriched.md 不存在: {enriched_path}")
        print("[TIP] 先运行: python -m pdf_pipeline.cli all <pdf>")
        return

    markdown = enriched_path.read_text(encoding="utf-8")

    # 加载 bindings（若存在）
    bindings = None
    if bindings_path.exists():
        try:
            bindings = load_bindings_json(str(bindings_path))
        except Exception:
            pass

    # 加载 page_map（若存在）
    page_map_path = output_dir / "page_map.json"
    page_map = None
    if page_map_path.exists():
        try:
            import json as _json
            page_map = _json.loads(page_map_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    config = RAGChunkConfig()
    if args.chunk_tokens:
        config.chunk_tokens = args.chunk_tokens
    if args.overlap_tokens:
        config.overlap_tokens = args.overlap_tokens

    report = rag_chunk_markdown(markdown, bindings=bindings, page_map=page_map, config=config)
    print_quality_report(report)

    # JSON 导出
    out_path = args.output or str(output_dir / "rag_chunks.json")
    export_rag_report(report, out_path)

    # HTML 可视化
    html_path = args.html or str(output_dir / "rag_chunks.html")
    html = render_rag_html(report, title=pdf_stem)
    Path(html_path).write_text(html, encoding="utf-8")
    print(f"  [RAG-CHUNK] HTML viz: {html_path}")


def cmd_rag_visualize(args):
    """从已有 rag_chunks.json 重新生成 HTML 可视化。"""
    import json
    from .rag_chunker import render_rag_html, RAGChunk, RAGChunkingReport, RAGChunkConfig

    json_path = Path(args.file)
    if not json_path.exists():
        print(f"[ERROR] rag_chunks.json 不存在: {json_path}")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # 重建 RAGChunkingReport
    chunks = []
    for cd in data["chunks"]:
        ch = RAGChunk(
            chunk_id=cd["chunk_id"],
            content=cd["content"],
            content_type=cd["content_type"],
            section_path=cd.get("section_path", ""),
            token_count=cd.get("token_count", 0),
            ref_ids=cd.get("ref_ids", []),
            parent_chunk_id=cd.get("parent_chunk_id", ""),
            prev_chunk_id=cd.get("prev_chunk_id", ""),
            next_chunk_id=cd.get("next_chunk_id", ""),
            bound_elements=cd.get("bound_elements", []),
            metadata=cd.get("metadata", {}),
        )
        chunks.append(ch)

    cfg = data.get("config", {})
    config = RAGChunkConfig(
        chunk_tokens=cfg.get("chunk_tokens", 1024),
        overlap_tokens=cfg.get("overlap_tokens", 128),
    )

    report = RAGChunkingReport(
        chunks=chunks,
        total_chunks=len(chunks),
        quality_report=data.get("quality", {}),
        config=config,
    )

    html = render_rag_html(report, title=json_path.stem)
    html_path = args.html or str(json_path.with_suffix(".html"))
    Path(html_path).write_text(html, encoding="utf-8")
    print(f"  [OK] HTML viz: {html_path}")


def cmd_all(args):
    """完整五阶段流程: parse → bindings → enhance → enrich → rag-chunk。"""
    from .parser import parse_pdf_docling
    from .bindings import load_bindings_json, export_bindings_json
    from .enhancer import enhance_all
    from .enricher import enrich_markdown
    from .rag_chunker import (
        rag_chunk_markdown, RAGChunkConfig,
        export_rag_report, print_quality_report, render_rag_html,
    )

    pdf_stem = Path(args.file).stem
    output_dir = Path(__file__).resolve().parent / "output" / _slugify(pdf_stem)

    # ---- Stage 1: PDF → Markdown + 资源保存 ----
    print(f"\n{'=' * 60}")
    print(f"  Stage 1: PDF → Markdown + 资源提取")
    print(f"{'=' * 60}")
    result = parse_pdf_docling(args.file, export_bindings=True)
    bindings = load_bindings_json(result.bindings_path)

    n_tables = sum(1 for e in bindings["elements"] if e["type"] == "table")
    n_formulas = sum(1 for e in bindings["elements"] if e["type"] == "formula")
    n_pictures = sum(1 for e in bindings["elements"] if e["type"] == "picture")
    print(f"  文本+表格 → raw.md ({len(result.markdown):,} chars)")
    print(f"  元素提取: {n_tables} tables, {n_formulas} formulas, {n_pictures} pictures")

    # ---- Stage 2: 空间绑定校验 + 引用回溯 ----
    print(f"\n{'=' * 60}")
    print(f"  Stage 2: 空间绑定 & 引用回溯")
    print(f"{'=' * 60}")
    n_refs = len(bindings.get("references", []))
    ref_targets = {r["target_element_id"] for r in bindings["references"]
                   if r.get("target_element_id")}
    print(f"  引用关系: {n_refs} references → {len(ref_targets)} elements")

    # ---- Stage 3: 多模态语义增强 ----
    print(f"\n{'=' * 60}")
    print(f"  Stage 3: 多模态语义增强")
    print(f"{'=' * 60}")
    summary = enhance_all(bindings, result.markdown, str(output_dir))
    export_bindings_json(bindings, result.bindings_path)

    # ---- Stage 4: 增强注入 raw.md → final_enriched.md ----
    print(f"\n{'=' * 60}")
    print(f"  Stage 4: 增强注入 → final_enriched.md")
    print(f"{'=' * 60}")
    raw_md = output_dir / "raw.md"
    markdown_src = raw_md.read_text(encoding="utf-8") if raw_md.exists() else result.markdown
    enrich_markdown(markdown_src, bindings)

    # ---- Stage 5: RAG 优化切分 → rag_chunks.json + HTML ----
    print(f"\n{'=' * 60}")
    print(f"  Stage 5: RAG 优化切分")
    print(f"{'=' * 60}")
    config = RAGChunkConfig()
    if args.chunk_tokens:
        config.chunk_tokens = args.chunk_tokens
    if args.overlap_tokens:
        config.overlap_tokens = args.overlap_tokens

    enriched_path = output_dir / "final_enriched.md"
    enriched_md = enriched_path.read_text(encoding="utf-8")

    # 加载 page_map（若存在）
    page_map_path = output_dir / "page_map.json"
    page_map = None
    if page_map_path.exists():
        try:
            import json as _json
            page_map = _json.loads(page_map_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    report = rag_chunk_markdown(enriched_md, bindings=bindings, page_map=page_map, config=config)
    print_quality_report(report)

    export_rag_report(report, str(output_dir / "rag_chunks.json"))
    html = render_rag_html(report, title=pdf_stem)
    (output_dir / "rag_chunks.html").write_text(html, encoding="utf-8")

    print(f"\n  [OK] 全流程完成: {output_dir}")
    print(f"  [OK] rag_chunks.json ({report.total_chunks} chunks)")
    print(f"  [OK] rag_chunks.html")


def main():
    parser = argparse.ArgumentParser(description="Docling PDF -> Markdown -> Chunk 可视化")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("parse", help="PDF -> Markdown")
    p.add_argument("file")

    p = subparsers.add_parser("visualize", help="PDF -> HTML 可视化")
    p.add_argument("file")
    p.add_argument("--html", default=None, help="HTML 输出路径")

    p = subparsers.add_parser("bindings", help="仅导出空间绑定 JSON")
    p.add_argument("file")

    p = subparsers.add_parser("enhance", help="Stage 3: 多模态语义增强")
    p.add_argument("file")

    p = subparsers.add_parser("enrich", help="Stage 4: 注入增强 → final_enriched.md")
    p.add_argument("file")
    p.add_argument("--show", action="store_true",
                   help="展示注入点上下文")

    p = subparsers.add_parser("all", help="完整流程: parse → enrich → rag-chunk")
    p.add_argument("file")
    p.add_argument("--chunk-tokens", type=int, default=None,
                   help="目标 chunk 大小 (tokens, default 1024)")
    p.add_argument("--overlap-tokens", type=int, default=None,
                   help="重叠大小 (tokens, default 128)")

    p = subparsers.add_parser("rag-chunk", help="RAG 优化切分 (需先 run all)")
    p.add_argument("file")
    p.add_argument("--chunk-tokens", type=int, default=None,
                   help="目标 chunk 大小 (tokens, default 1024)")
    p.add_argument("--overlap-tokens", type=int, default=None,
                   help="重叠大小 (tokens, default 128)")
    p.add_argument("--output", default=None, help="输出 JSON 路径")
    p.add_argument("--html", default=None, help="输出 HTML 路径")

    p = subparsers.add_parser("rag-visualize", help="从 rag_chunks.json 生成 HTML 可视化")
    p.add_argument("file", help="rag_chunks.json 路径")
    p.add_argument("--html", default=None, help="输出 HTML 路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    {"parse": cmd_parse, "visualize": cmd_visualize,
     "bindings": cmd_bindings, "enhance": cmd_enhance, "enrich": cmd_enrich,
     "all": cmd_all, "rag-chunk": cmd_rag_chunk,
     "rag-visualize": cmd_rag_visualize}[args.command](args)


if __name__ == "__main__":
    main()
