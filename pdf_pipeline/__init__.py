"""
pdf_pipeline — 科研文献 PDF 预处理管道
========================================

独立可复用的 PDF 处理模块，将 PDF 转化为结构化 Markdown +
空间绑定 + 学术切分 + 可视化。

零依赖 agent/ 内部模块，可独立导入无 segfault 风险。

快速开始:
  from pdf_pipeline import parse_pdf_docling, chunk_markdown, build_bindings

  result = parse_pdf_docling("paper.pdf", export_bindings=True)
  report = chunk_markdown(result.markdown, result.metadata)
"""

from .parser import parse_pdf_docling, DoclingParseResult
from .bindings import (
    build_bindings, export_bindings_json, load_bindings_json,
    validate_bindings, BoundElement,
)
from .chunker import chunk_markdown, set_chunk_size, Chunk, ChunkingReport, parse_and_chunk
from .viz import (
    render_html_two_step, render_chunks_text, export_chunks_json,
    visualize_pdf, SECTION_COLORS, DEFAULT_COLOR,
)

from .enhancer import enhance_formulas, enhance_images, enhance_all
from .enricher import inject_enhancements, enrich_markdown
from .rag_chunker import (
    rag_chunk_markdown, export_rag_report, print_quality_report,
    render_rag_html, RAGChunk, RAGChunkingReport, RAGChunkConfig,
    extract_keywords,
)

__all__ = [
    # parser
    "parse_pdf_docling",
    "DoclingParseResult",
    # bindings
    "build_bindings",
    "export_bindings_json",
    "load_bindings_json",
    "validate_bindings",
    "BoundElement",
    # enhancer
    "enhance_formulas",
    "enhance_images",
    "enhance_all",
    # enricher
    "inject_enhancements",
    "enrich_markdown",
    # chunker
    "chunk_markdown",
    "set_chunk_size",
    "Chunk",
    "ChunkingReport",
    "parse_and_chunk",
    # viz
    "render_html_two_step",
    "render_chunks_text",
    "export_chunks_json",
    "visualize_pdf",
    "SECTION_COLORS",
    "DEFAULT_COLOR",
    # rag_chunker
    "rag_chunk_markdown",
    "export_rag_report",
    "print_quality_report",
    "render_rag_html",
    "RAGChunk",
    "RAGChunkingReport",
    "RAGChunkConfig",
    "extract_keywords",
]
