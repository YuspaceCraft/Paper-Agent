"""
chunk_viz_page.py — PDF 解析 & Chunk 可视化页面 (Streamlit)
=================

两步展示：
  Step 1: Docling PDF → Markdown 转换结果
  Step 2: 学术切分结果（每个 chunk 内容可见）

运行方式：
  streamlit run web/chunk_viz_page.py
"""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

# ---- 必须最早设置 ----
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---- 预导入 docling（避免 Streamlit 多线程 segfault） ----
try:
    import docling.document_converter  # noqa: F401
except ImportError:
    pass

# 项目根目录
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

st.set_page_config(
    page_title="PDF 解析 & Chunk 可视化",
    page_icon="📄",
    layout="wide",
)


def main():
    st.title("📄 PDF 解析 & Chunk 可视化")
    st.caption("Docling PDF → Markdown → 学术切分 → 可视化验证")

    # ---- 侧边栏 ----
    with st.sidebar:
        st.markdown("## ⚙️ 配置")
        from pdf_pipeline.chunker import _get_chunk_size
        default_size = _get_chunk_size()
        chunk_size = st.slider(
            "目标 Chunk 大小（字符）",
            300, 2000, default_size, 100,
        )

        st.divider()
        st.markdown("### ℹ️ 说明")
        st.markdown("""
        **流程:**
        1. Docling 布局分析 → Markdown
        2. 按 `##` 章节拆分
        3. 按段落分组切分

        **Docling** (IBM) 深度学习
        PDF 解析，识别标题层级、
        多栏排版、阅读顺序。
        """)

    # ---- 主区域 ----
    tab1, tab2 = st.tabs(["📤 上传 PDF", "📂 本地文件"])

    with tab1:
        uploaded = st.file_uploader("拖拽 PDF 文件", type=["pdf"], key="upload")
        if uploaded:
            _process(uploaded, chunk_size)

    with tab2:
        from config import DATA_DIR
        pdfs = list(DATA_DIR.rglob("*.pdf")) if DATA_DIR.exists() else []
        if not pdfs:
            st.warning(f"`{DATA_DIR}` 目录中暂无 PDF")
        else:
            selected = st.selectbox(
                "选择 PDF",
                options=[str(f.relative_to(DATA_DIR)) for f in sorted(pdfs)],
            )
            if selected and st.button("解析并可视化", type="primary"):
                full_path = DATA_DIR / selected
                with open(full_path, "rb") as f:
                    data = f.read()

                class FakeUpload:
                    name = full_path.name
                    def __init__(self, d): self._d = d
                    def getbuffer(self):
                        import io; return io.BytesIO(self._d)
                    def read(self): return self._d

                _process(FakeUpload(data), chunk_size)


def _process(uploaded, chunk_size: int):
    """处理 PDF 并展示两步结果。"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        # 覆盖 chunk 大小
        from pdf_pipeline.chunker import set_chunk_size, _get_chunk_size
        old = _get_chunk_size()
        set_chunk_size(chunk_size)

        # Step 1: Docling 解析
        with st.spinner(f"🔍 Docling 解析 `{uploaded.name}` ..."):
            from pdf_pipeline.parser import parse_pdf_docling
            parsed = parse_pdf_docling(tmp_path, export_bindings=True)

        # 加载 bindings
        _bindings = None
        if parsed.bindings_path:
            try:
                from pdf_pipeline.bindings import load_bindings_json
                _bindings = load_bindings_json(parsed.bindings_path)
            except Exception:
                pass

        # Step 2: 切分
        with st.spinner("✂️ 学术切分..."):
            from pdf_pipeline.chunker import chunk_markdown
            report = chunk_markdown(parsed.markdown, parsed.metadata, bindings=_bindings)

        set_chunk_size(old)

        # ---- 渲染两步结果 ----
        _render_results(parsed, report, uploaded.name, _bindings)

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _render_results(parsed, report, filename: str, bindings: dict | None = None):
    """渲染两步可视化。"""
    from pdf_pipeline.viz import SECTION_COLORS, DEFAULT_COLOR

    st.success(f"✅ `{filename}` 解析完成")

    # 统计卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Markdown 大小", f"{len(parsed.markdown):,} 字符")
    c2.metric("Chunks", report.total_chunks)
    c3.metric("平均大小", f"{report.avg_chunk_size:,}")
    c4.metric("章节数", len(report.section_distribution))

    # 两步 Tab
    tab_a, tab_b = st.tabs([
        "📝 Step 1: Docling Markdown 输出",
        f"✂️ Step 2: Chunk 切分结果 ({report.total_chunks})",
    ])

    with tab_a:
        st.markdown("### Docling PDF → Markdown 转换结果")
        st.caption(f"{len(parsed.markdown):,} 字符 — 保留了完整的文档结构和阅读顺序")
        with st.expander("查看完整 Markdown", expanded=False):
            st.text_area(
                "markdown", parsed.markdown, height=600,
                key="md_full", label_visibility="collapsed",
            )

        # 元数据
        meta = parsed.metadata
        if any(meta.values()):
            st.markdown("**提取的元数据:**")
            st.json(meta)

    with tab_b:
        st.markdown(f"### 切分结果 — {report.total_chunks} Chunks")
        st.caption(
            f"平均 {report.avg_chunk_size} 字符 | "
            f"范围 {report.min_chunk_size}-{report.max_chunk_size} | "
            f"分布: {report.section_distribution}"
        )

        # 筛选
        all_types = sorted(set(ch.section_type for ch in report.chunks))
        selected = st.multiselect("按类型筛选", all_types, default=all_types)
        filtered = [ch for ch in report.chunks if ch.section_type in selected]
        st.caption(f"显示 {len(filtered)}/{report.total_chunks} chunks")

        # 渲染每个 chunk
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
                {len(ch.content)} 字符 | {ch.section_name[:60]}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            with st.expander("查看内容", expanded=i < 3):
                st.text_area(
                    f"chunk_{i}", ch.content, height=200,
                    key=f"chunk_{i}", label_visibility="collapsed",
                )

    # ---- 空间绑定（可选） ----
    if bindings and bindings.get("elements"):
        st.divider()
        st.markdown("### 📍 空间绑定 (Spatial Bindings)")
        elements = bindings.get("elements", [])
        references = bindings.get("references", [])
        c1, c2 = st.columns(2)
        c1.metric("布局元素", len(elements))
        c2.metric("引用匹配", len(references))
        with st.expander(f"元素详情 ({len(elements)} 图片/表格/公式)", expanded=False):
            for e in elements:
                st.caption(f"**{e['element_id']}** ({e['type']}) — p.{e['page_no']} "
                           f"| {e.get('caption', '(无标题)')[:80]}")
        if references:
            with st.expander(f"引用回溯 ({len(references)} 条)", expanded=False):
                for r in references:
                    tid = r.get('target_element_id', '(未匹配)')
                    st.caption(f"`{r.get('ref_text', '')}` → **{tid}** "
                               f"| {r.get('context', '')[:100]}")

    # ---- 导出 ----
    st.divider()
    st.markdown("### 📥 导出")
    c1, c2 = st.columns(2)
    with c1:
        from pdf_pipeline.viz import render_html_two_step
        html_out = render_html_two_step(parsed.markdown, report, title=filename, bindings=bindings)
        st.download_button(
            "⬇️ HTML 可视化 (两步展示)",
            html_out,
            f"{Path(filename).stem}_viz.html",
            "text/html",
            use_container_width=True,
        )
    with c2:
        from pdf_pipeline.viz import export_chunks_json
        json_out = export_chunks_json(report)
        st.download_button(
            "⬇️ JSON 数据",
            json_out,
            f"{Path(filename).stem}_chunks.json",
            "application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
