"""
chunk_viz.py — Chunk 可视化（两步展示：Docling Markdown → Chunks）
============

提供多种可视化方式：

  1. HTML 卡片视图 —— 两步展示:
     Step 1: Docling 导出的完整 Markdown（可读的结构化文本）
     Step 2: 切分后的 Chunk 列表（带颜色编码）

  2. 纯文本摘要 —— 终端友好

  3. Streamlit 组件 —— 集成到 Web 界面

使用方式：
  from pdf_pipeline.viz import render_html_two_step

  html = render_html_two_step(parse_result.markdown, report)
"""

from __future__ import annotations

import html as _html
from collections import Counter
from pathlib import Path

from .chunker import Chunk, ChunkingReport


# ================================================================
# 颜色方案
# ================================================================

SECTION_COLORS = {
    "abstract":     {"bg": "#1b2a1b", "border": "#2d4a2d", "accent": "#4caf50", "label": "摘要"},
    "introduction": {"bg": "#1a1a3e", "border": "#2d2d6b", "accent": "#667eea", "label": "引言"},
    "related_work": {"bg": "#1e1a2e", "border": "#3d2d5c", "accent": "#9b59b6", "label": "相关工作"},
    "methods":      {"bg": "#1e2a1a", "border": "#3d5c2d", "accent": "#3498db", "label": "方法"},
    "results":      {"bg": "#2e1a1a", "border": "#5c2d2d", "accent": "#e74c3c", "label": "结果"},
    "discussion":   {"bg": "#2e2a1a", "border": "#5c4d2d", "accent": "#f39c12", "label": "讨论"},
    "conclusion":   {"bg": "#1a2e2a", "border": "#2d5c4d", "accent": "#1abc9c", "label": "结论"},
    "references":   {"bg": "#2a1a2e", "border": "#4d2d5c", "accent": "#95a5a6", "label": "参考文献"},
    "other":        {"bg": "#1e1e2e", "border": "#3d3d5c", "accent": "#7f8c8d", "label": "其他"},
}
DEFAULT_COLOR = {"bg": "#1e1e2e", "border": "#3d3d5c", "accent": "#95a5a6", "label": ""}


# ================================================================
# HTML: 两步展示（Markdown → Chunks）
# ================================================================

def render_html_two_step(
    markdown: str,
    report: ChunkingReport | None = None,
    title: str = "PDF 解析 & Chunk 可视化",
    bindings: dict | None = None,
) -> str:
    """
    生成两步 HTML：
      Step 1: Docling 导出的完整 Markdown
      Step 2: 切分后的 Chunk 列表
      Step 3: (可选) 空间绑定 — 元素表 + 引用匹配

    参数:
        markdown: Docling 导出的 Markdown 文本
        report: 可选的切分报告
        title: 页面标题
        bindings: 可选的空间绑定数据 {elements, references}

    返回:
        str: 完整 HTML
    """
    md_html = _render_markdown_section(markdown, title)
    chunks_html = _render_chunks_section(report) if report else ""
    bindings_html = _render_bindings_section(bindings) if bindings else ""
    has_bindings = bool(bindings and bindings.get("elements"))

    stats = _compute_chunk_stats(report.chunks) if report else {}
    legend = _render_legend(stats.get("section_counts", {})) if report else ""

    # 构建第三个 tab（仅在 bindings 存在时）
    bindings_tab = ""
    if has_bindings:
        bindings_tab = '<li class="tab" onclick="switchTab(event, \'tab-bindings\')">Spatial Bindings</li>'
    bindings_content = ""
    if has_bindings:
        bindings_content = f'<div id="tab-bindings" class="tab-content">{bindings_html}</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
{_STYLES}
</head>
<body>

<div class="header">
  <h1>PDF 解析 & Chunk 可视化</h1>
  <div class="stats">
    完整流程：Docling PDF → Markdown → {f'{report.total_chunks} chunks' if report else '切分'}
  </div>
</div>

{_render_stats_grid(stats) if stats else ""}
{legend}

<ul class="tabs">
  <li class="tab active" onclick="switchTab(event, 'tab-markdown')">Step 1: Docling Markdown 输出</li>
  <li class="tab" onclick="switchTab(event, 'tab-chunks')">Step 2: Chunk 切分结果</li>
  {bindings_tab}
</ul>

<div id="tab-markdown" class="tab-content active">
  {md_html}
</div>

<div id="tab-chunks" class="tab-content">
  {chunks_html}
</div>
{bindings_content}

<script>
function switchTab(evt, tabId) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  evt.currentTarget.classList.add('active');
  document.getElementById(tabId).classList.add('active');
}}

function toggleChunk(id) {{
  const content = document.getElementById('content-' + id);
  const btn = document.getElementById('btn-' + id);
  if (content.classList.contains('collapsed')) {{
    content.classList.remove('collapsed');
    btn.textContent = '收起';
  }} else {{
    content.classList.add('collapsed');
    btn.textContent = '展开完整内容';
  }}
}}
</script>

</body>
</html>"""


def _render_markdown_section(md: str, title: str) -> str:
    """将 Markdown 转为 HTML 展示。"""
    md_escaped = _html.escape(md)

    # 简单的 Markdown → HTML 转换
    # ## → h2
    md_escaped = _MD_HEADER_HTML.sub(r'<h2>\1</h2>', md_escaped)
    # 空行 → 段落边界
    paragraphs = md_escaped.split('\n\n')
    html_paras = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith('<h2>'):
            html_paras.append(p)
        else:
            # 行内换行 → <br>
            p = p.replace('\n', '<br>')
            html_paras.append(f'<p>{p}</p>')
    body = '\n'.join(html_paras)

    return f"""
<div class="md-section">
  <div class="md-header">
    <span class="step-badge">Step 1</span>
    Docling PDF → Markdown 转换结果
    <span style="font-size:12px;color:#8b949e;margin-left:12px;">
      {len(md):,} 字符 | ~{md.count(chr(10))+1} 行
    </span>
  </div>
  <div class="md-content">{body}</div>
</div>"""


def _render_chunks_section(report: ChunkingReport) -> str:
    """渲染所有 chunk 卡片。"""
    cards = ""
    for i, ch in enumerate(report.chunks):
        cards += _render_chunk_card(ch, i)
    return f"""
<div class="md-header">
  <span class="step-badge">Step 2</span>
  学术切分结果 — {report.total_chunks} 个 Chunk
  <span style="font-size:12px;color:#8b949e;margin-left:12px;">
    平均 {report.avg_chunk_size} 字符 | 范围 {report.min_chunk_size}-{report.max_chunk_size}
  </span>
</div>
<div class="chunks-container">{cards}</div>"""


def _render_chunk_card(chunk: Chunk, index: int) -> str:
    """渲染单个 chunk 卡片。"""
    colors = SECTION_COLORS.get(chunk.section_type, DEFAULT_COLOR)
    label = colors["label"]
    content = chunk.content
    is_long = len(content) > 500

    return f"""
<div class="chunk-card"
     style="--card-bg: {colors['bg']}; --card-border: {colors['border']}; --card-accent: {colors['accent']};">
  <div class="chunk-header" onclick="toggleChunk({index})">
    <div>
      <span class="chunk-id">#{index + 1}</span>
      <span class="chunk-section-tag" style="margin-left:8px;">{label} {chunk.section_type}</span>
    </div>
    <div class="chunk-meta">
      <span>{len(content)} 字符</span>
      <span>~{chunk.token_count} tokens</span>
      <span>{_html.escape(chunk.section_name[:60])}</span>
    </div>
  </div>
  <div class="chunk-content collapsed" id="content-{index}">
{_html.escape(content)}
  </div>
  {f'<button class="expand-btn" id="btn-{index}" onclick="toggleChunk({index})">▼ 展开完整内容</button>' if is_long else ''}
</div>"""


def _render_bindings_section(bindings: dict) -> str:
    """渲染空间绑定 tab。"""
    elements = bindings.get("elements", [])
    references = bindings.get("references", [])

    # 元素表
    rows = ""
    for e in elements:
        bbox_str = f"[{e['bbox'][0]:.0f}, {e['bbox'][1]:.0f}, {e['bbox'][2]:.0f}, {e['bbox'][3]:.0f}]"
        cap = _html.escape(e.get("caption", "")[:80])
        cap = cap or "(无标题)"
        # 统计引用该元素的 ref 数量
        eid = e["element_id"]
        ref_count = sum(1 for r in references if r.get("target_element_id") == eid)
        rows += f"""<tr>
          <td style="font-weight:600;">{_html.escape(eid)}</td>
          <td>{e['type']}</td>
          <td>{e['page_no']}</td>
          <td style="font-size:12px;">{bbox_str}</td>
          <td style="font-size:12px;">{cap}</td>
          <td>{ref_count}</td>
        </tr>"""

    elements_table = f"""
    <h3>布局元素 ({len(elements)})</h3>
    <table class="bindings-table">
      <tr><th>ID</th><th>类型</th><th>页码</th><th>bbox</th><th>标题</th><th>引用数</th></tr>
      {rows}
    </table>""" if rows else "<p>未检测到布局元素。</p>"

    # 引用列表
    ref_rows = ""
    for r in references:
        ref_rows += f"""<tr>
          <td>{_html.escape(r.get('ref_text', ''))}</td>
          <td>{r.get('ref_type', '')}</td>
          <td>{_html.escape(r.get('target_element_id', '(未匹配)'))}</td>
          <td style="font-size:12px;">{_html.escape(r.get('context', '')[:100])}</td>
        </tr>"""

    refs_table = f"""
    <h3>引用回溯 ({len(references)})</h3>
    <table class="bindings-table">
      <tr><th>引用文本</th><th>类型</th><th>目标元素</th><th>上下文</th></tr>
      {ref_rows}
    </table>""" if ref_rows else "<p>未检测到正文引用。</p>"

    return f"""
    <div class="bindings-section">
      <div class="md-header">
        <span class="step-badge">Step 3</span>
        空间绑定 — {len(elements)} 元素, {len(references)} 引用
      </div>
      {elements_table}
      <div style="margin-top:24px;"></div>
      {refs_table}
    </div>"""


def _render_stats_grid(stats: dict) -> str:
    """渲染统计卡片。"""
    if not stats:
        return ""
    return f"""<div class="stats-grid">
  <div class="stat-card"><div class="value">{stats.get('total_chunks', 0)}</div><div class="label">总 Chunks</div></div>
  <div class="stat-card"><div class="value">{stats.get('avg_size', 0)}</div><div class="label">平均字符数</div></div>
  <div class="stat-card"><div class="value">{stats.get('total_tokens', 0)}</div><div class="label">估算 Tokens</div></div>
  <div class="stat-card"><div class="value">{stats.get('min_size', 0)}–{stats.get('max_size', 0)}</div><div class="label">大小范围</div></div>
</div>"""


def _render_legend(section_counts: dict[str, int]) -> str:
    """渲染颜色图例。"""
    if not section_counts:
        return ""
    items = ""
    for st, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        c = SECTION_COLORS.get(st, DEFAULT_COLOR)
        items += f"""<div class="legend-item">
    <div class="legend-dot" style="background:{c['accent']};"></div>
    <span>{c['label']} {st} ({count})</span></div>"""
    return f'<div class="section-legend">{items}</div>'


def _compute_chunk_stats(chunks: list[Chunk]) -> dict:
    """计算 chunk 统计。"""
    sizes = [len(ch.content) for ch in chunks] if chunks else [0]
    sc = dict(Counter(ch.section_type for ch in chunks))
    return {
        "total_chunks": len(chunks),
        "avg_size": sum(sizes) // len(sizes) if sizes else 0,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "total_tokens": sum(ch.token_count for ch in chunks),
        "section_counts": sc,
    }


# ================================================================
# HTML 样式
# ================================================================

_MD_HEADER_HTML = __import__('re').compile(r'^## (.+)$', __import__('re').MULTILINE)

_STYLES = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 20px;
  }
  .header { text-align: center; padding: 24px 0; margin-bottom: 24px; border-bottom: 1px solid #21262d; }
  .header h1 { color: #58a6ff; font-size: 28px; margin-bottom: 8px; }
  .header .stats { color: #8b949e; font-size: 14px; }

  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0; }
  .stat-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 12px 16px; text-align: center; }
  .stat-card .value { font-size: 24px; font-weight: 700; color: #58a6ff; }
  .stat-card .label { font-size: 12px; color: #8b949e; margin-top: 4px; }

  .section-legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8b949e; }
  .legend-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }

  /* Tabs */
  .tabs { display: flex; gap: 0; margin: 24px 0 0 0; border-bottom: 2px solid #21262d; }
  .tab {
    padding: 10px 20px; cursor: pointer; font-size: 14px; font-weight: 600;
    color: #8b949e; background: transparent; border: none;
    border-bottom: 2px solid transparent; margin-bottom: -2px;
    transition: color 0.15s, border-color 0.15s;
  }
  .tab:hover { color: #c9d1d9; }
  .tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }

  .tab-content { display: none; padding-top: 16px; }
  .tab-content.active { display: block; }

  /* Markdown section */
  .md-section { background: #161b22; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; margin-bottom: 16px; }
  .md-header {
    padding: 12px 16px; background: rgba(88,166,255,0.08); border-bottom: 1px solid #21262d;
    font-weight: 600; font-size: 14px; color: #58a6ff;
  }
  .step-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px;
    background: #58a6ff; color: #fff; margin-right: 8px;
  }
  .md-content {
    padding: 20px 24px; font-size: 14px; line-height: 1.8; white-space: pre-wrap;
    word-break: break-word; max-height: 70vh; overflow-y: auto;
  }
  .md-content h2 {
    color: #58a6ff; font-size: 18px; margin: 20px 0 10px 0;
    padding-bottom: 6px; border-bottom: 1px solid #21262d;
  }
  .md-content p { margin: 8px 0; }

  /* Chunk cards */
  .chunks-container { display: flex; flex-direction: column; gap: 12px; }
  .chunk-card {
    background: var(--card-bg, #161b22); border: 1px solid var(--card-border, #21262d);
    border-left: 4px solid var(--card-accent, #58a6ff); border-radius: 8px; overflow: hidden;
  }
  .chunk-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
  .chunk-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; background: rgba(255,255,255,0.03); cursor: pointer; user-select: none;
  }
  .chunk-header:hover { background: rgba(255,255,255,0.06); }
  .chunk-id { font-weight: 700; font-size: 14px; color: var(--card-accent, #58a6ff); }
  .chunk-meta { display: flex; gap: 12px; font-size: 12px; color: #8b949e; }
  .chunk-meta span { display: flex; align-items: center; gap: 4px; }
  .chunk-section-tag {
    display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px;
    font-weight: 600; background: var(--card-accent, #58a6ff); color: #fff; opacity: 0.9;
  }
  .chunk-content {
    padding: 16px; font-size: 14px; line-height: 1.7; white-space: pre-wrap;
    word-break: break-word; border-top: 1px solid rgba(255,255,255,0.05);
  }
  .chunk-content.collapsed { max-height: 150px; overflow: hidden; position: relative; }
  .chunk-content.collapsed::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 40px;
    background: linear-gradient(transparent, var(--card-bg, #161b22));
  }
  .expand-btn {
    display: block; width: 100%; padding: 8px; border: none;
    background: rgba(255,255,255,0.03); color: #8b949e;
    cursor: pointer; font-size: 12px;
  }
  .expand-btn:hover { background: rgba(255,255,255,0.08); color: #c9d1d9; }

  /* Bindings table */
  .bindings-section { margin-top: 16px; }
  .bindings-table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  .bindings-table th {
    background: rgba(88,166,255,0.1); color: #58a6ff; padding: 8px 10px;
    text-align: left; font-weight: 600; border-bottom: 2px solid #30363d;
  }
  .bindings-table td {
    padding: 6px 10px; border-bottom: 1px solid #21262d;
  }
  .bindings-table tr:hover td { background: rgba(255,255,255,0.03); }
  .bindings-section h3 {
    color: #c9d1d9; font-size: 15px; margin: 16px 0 8px 0;
    padding-bottom: 6px; border-bottom: 1px solid #21262d;
  }
</style>"""


# ================================================================
# 纯文本摘要
# ================================================================

def render_chunks_text(report: ChunkingReport) -> str:
    """终端友好的纯文本 chunk 摘要。"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  Chunk 摘要 — {report.total_chunks} chunks, "
                 f"平均 {report.avg_chunk_size} 字符")
    lines.append("=" * 80)

    for i, ch in enumerate(report.chunks):
        preview = ch.content[:200].replace('\n', ' ')
        lines.append(f"\n--- Chunk #{i + 1} ---")
        lines.append(f"  类型: {ch.section_type} | 章节: {ch.section_name[:60]}")
        lines.append(f"  大小: {len(ch.content)} 字符 | ~{ch.token_count} tokens")
        lines.append(f"  预览: {preview}...")

    sc = dict(Counter(ch.section_type for ch in report.chunks))
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  分布: {sc}")
    lines.append("=" * 80)

    return "\n".join(lines)


# ================================================================
# JSON 导出
# ================================================================

def export_chunks_json(report: ChunkingReport, output_path: str | None = None) -> str:
    """将切分结果导出为 JSON。"""
    import json

    sizes = [len(ch.content) for ch in report.chunks]
    data = {
        "stats": {
            "total_chunks": report.total_chunks,
            "avg_chunk_size": sum(sizes) // len(sizes) if sizes else 0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "section_distribution": report.section_distribution,
        },
        "chunks": [
            {
                "chunk_id": ch.chunk_id,
                "section_type": ch.section_type,
                "section_name": ch.section_name,
                "content": ch.content,
                "token_count": ch.token_count,
            }
            for ch in report.chunks
        ],
    }

    json_str = json.dumps(data, ensure_ascii=False, indent=2)

    if output_path:
        Path(output_path).write_text(json_str, encoding="utf-8")
        print(f"  [VIZ] JSON 已导出: {output_path}")

    return json_str


# ================================================================
# Streamlit 集成
# ================================================================

def render_chunks_streamlit(markdown: str, report: ChunkingReport):
    """在 Streamlit 中渲染两步可视化。"""
    import streamlit as st

    st.markdown("## PDF 解析 & Chunk 可视化")

    tab1, tab2 = st.tabs([
        "Step 1: Docling Markdown 输出",
        f"Step 2: Chunk 切分结果 ({report.total_chunks} chunks)",
    ])

    with tab1:
        st.markdown(f"**Docling PDF → Markdown 结果**")
        st.caption(f"{len(markdown):,} 字符 | ~{markdown.count(chr(10))+1} 行")
        with st.expander("查看完整 Markdown", expanded=False):
            st.text_area(
                "markdown", markdown, height=500,
                key="md_view", label_visibility="collapsed",
            )

    with tab2:
        # 统计
        cols = st.columns(4)
        cols[0].metric("Chunks", report.total_chunks)
        cols[1].metric("平均字符", f"{report.avg_chunk_size:,}")
        cols[2].metric("大小范围", f"{report.min_chunk_size}-{report.max_chunk_size}")
        total_tok = sum(ch.token_count for ch in report.chunks)
        cols[3].metric("估算 Tokens", f"{total_tok:,}")

        # 章节筛选
        all_types = sorted(set(ch.section_type for ch in report.chunks))
        selected = st.multiselect("按类型筛选", all_types, default=all_types)

        filtered = [ch for ch in report.chunks if ch.section_type in selected]
        st.caption(f"显示 {len(filtered)}/{report.total_chunks} chunks")

        for i, ch in enumerate(filtered):
            colors = SECTION_COLORS.get(ch.section_type, DEFAULT_COLOR)
            st.markdown(
                f"""<div style="background:{colors['bg']};border-left:4px solid {colors['accent']};
                border-radius:8px;padding:8px 16px;margin:12px 0 4px 0;">
                <span style="font-weight:700;color:{colors['accent']};">#{i + 1}</span>
                <span style="display:inline-block;padding:2px 8px;border-radius:12px;
                font-size:11px;font-weight:600;background:{colors['accent']};color:#fff;margin-left:8px;">
                {colors['label']} {ch.section_type}</span>
                <span style="font-size:12px;color:#8b949e;margin-left:12px;">
                {len(ch.content)} 字符 · {ch.section_name[:60]}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            with st.expander(f"查看内容", expanded=i < 2):
                st.text_area(
                    f"chunk_{i}", ch.content, height=200,
                    key=f"chunk_{i}", label_visibility="collapsed",
                )


# ================================================================
# 端到端
# ================================================================

def visualize_pdf(
    file_path: str,
    output_html: str | None = None,
) -> tuple[str, ChunkingReport]:
    """
    解析 PDF → 生成两步 HTML 可视化。

    返回:
        (markdown, ChunkingReport)
    """
    from .chunker import parse_and_chunk

    # 获取 markdown
    from .parser import parse_pdf_docling
    parsed = parse_pdf_docling(file_path)
    markdown = parsed.markdown

    # 切分
    report = parse_and_chunk(file_path)

    if output_html:
        html = render_html_two_step(markdown, report, title=Path(file_path).name)
        Path(output_html).write_text(html, encoding="utf-8")
        print(f"  [VIZ] HTML 已导出: {output_html}")

    # 文本摘要写入文件
    txt_path = Path(file_path).stem + "_chunks.txt"
    Path(txt_path).write_text(render_chunks_text(report), encoding="utf-8")
    print(f"  [VIZ] 文本摘要: {txt_path}")

    return markdown, report
