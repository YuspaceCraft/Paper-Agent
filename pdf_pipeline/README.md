# pdf_pipeline — PDF 文献预处理管道

独立可复用的科研文献 PDF 预处理模块。五阶段管道：
结构化提取 → 空间绑定 → 多模态增强 → 富化注入 → RAG 优化切分。

## 架构

```
PDF
 ├── parser.py → Docling 解析 → Markdown → raw.md
 │   ├── 元数据提取 (title/authors/doi/year/arxiv_id)
 │   ├── SHA256 内容指纹 (去重用)
 │   └── page_map.json (文本→页码锚点)
 ├── bindings.py → bbox + caption + 引用回溯 → bindings.json
 │   ├── PyMuPDF 裁剪 → picture_N.png
 │   └── FormulaItem.orig → formula_N.txt
 ├── enhancer.py (Stage 3) → LLM/VLM 语义增强
 │   ├── qwen3.6-max-preview → formula_desc（批处理，6 条/次）
 │   └── qwen3.7-plus (vision) → picture_desc（并行 4 worker）
 ├── enricher.py (Stage 4) → 增强注入 → final_enriched.md
 │   └── [FORMULA_DESC] / [FIGURE_DESC] 替换占位符
 ├── rag_chunker.py (Stage 5) → 四模块 RAG 切分
 │   ├── Module 1: 结构感知切分 + 硬上限强制切分 (max_chunk_tokens)
 │   ├── Module 2: BM25 关键词增强
 │   ├── Module 3: 结构化内容文本化 (表格/公式/图片 → 自然语言摘要)
 │   ├── Module 4: 参考文献解耦
 │   ├── 页码注入 (page_map → chunk.metadata.page_no)
 │   └── 质控: 短 chunk 合并 + token 分布统计 (min/max/median/p90/p95)
 └── viz.py → HTML/JSON 可视化

输出: pdf_pipeline/output/{pdf_stem}/
```

## 已实现功能 (Stage 1-2)

| 功能 | 模块 | 状态 |
|------|------|------|
| PDF → Markdown 解析 | parser.py | ✅ |
| 表格结构识别 | parser.py (do_table_structure=True) | ✅ |
| 元数据提取 (title/authors/doi/year/arxiv_id) | parser.py | ✅ |
| SHA256 内容指纹 (上传去重) | parser.py | ✅ |
| 文本→页码锚点 (page_map.json) | parser.py | ✅ |
| 图片/表格/公式空间坐标提取 | bindings.py | ✅ |
| 标题自动关联 (cref + 近邻匹配) | bindings.py | ✅ |
| 公式原始文本提取 | bindings.py | ✅ |
| 图片裁剪保存 (PyMuPDF, 300 DPI) | bindings.py | ✅ |
| 关键图片过滤 (有 caption + 正文引用) | bindings.py | ✅ |
| 多语言引用回溯 (EN/TR/CN) | bindings.py | ✅ |
| bindings.json 导出 | bindings.py | ✅ |
| 引用完整性校验 | bindings.py | ✅ |
| HTML 可视化 (含 Spatial Bindings tab) | viz.py | ✅ |

## 已实现功能 (Stage 3-4)

| 功能 | 模块 | 状态 |
|------|------|------|
| 公式 LLM 增强 (批处理, 6条/次, 失败逐条重试) | enhancer.py | ✅ |
| 图片 VLM 描述 (并行请求, 4 worker) | enhancer.py | ✅ |
| `.env` 自动加载 DASHSCOPE_API_KEY | enhancer.py | ✅ |
| 增强注入 Markdown (formula_desc/picture_desc) | enricher.py | ✅ |
| final_enriched.md 生成 | enricher.py | ✅ |

## 已实现功能 (Stage 5 — RAG 切分)

| 功能 | 模块 | 状态 |
|------|------|------|
| 结构感知切分 + 硬上限强制切分 (max_chunk_tokens=1536) | rag_chunker.py | ✅ |
| 原子块保护 (占位符期间强制切分，避免截断) | rag_chunker.py | ✅ |
| 短 chunk 合并 (body→body + 原子块折叠) | rag_chunker.py | ✅ |
| Token 分布统计 (min/max/median/p90/p95 + 直方图) | rag_chunker.py | ✅ |
| BM25 关键词增强 ([KEYWORDS: ...] 前缀) | rag_chunker.py | ✅ |
| 结构化内容文本化 (table/figure/formula → summary chunk) | rag_chunker.py | ✅ |
| 参考文献解耦 (条目级切分 + 元数据提取) | rag_chunker.py | ✅ |
| chunk ↔ bindings 关联 (bound_elements) | rag_chunker.py | ✅ |
| 邻居链 (prev_chunk_id / next_chunk_id) | rag_chunker.py | ✅ |
| 质检报告 (完整性/路径/原子块检查) | rag_chunker.py | ✅ |
| rag_chunks.json 导出 (下游唯一入口) | rag_chunker.py | ✅ |
| RAG chunk HTML 可视化 | rag_chunker.py | ✅ |
| Chunk→页码注入 (page_map 匹配) | rag_chunker.py | ✅ |

## 待实现

| 功能 | 依赖 | 状态 |
|------|------|------|
| Ragas 质量评估集成 | eval/ | ❌ |
| Embedding / 混合检索 | rag_chunks.json | ❌ |
| Context assembly (邻居 chunk + parent chunk) | rag_chunks.json | ❌ |

## 快速开始

### CLI

```bash
# 完整五阶段流程（一条命令出最终产物）
python -m pdf_pipeline.cli all data/paper.pdf
# → output/{paper}/rag_chunks.json + rag_chunks.html

# 单独执行各阶段
python -m pdf_pipeline.cli parse data/paper.pdf       # Stage 1: PDF → Markdown
python -m pdf_pipeline.cli bindings data/paper.pdf     # Stage 2: 空间绑定 JSON
python -m pdf_pipeline.cli enhance data/paper.pdf      # Stage 3: 多模态语义增强
python -m pdf_pipeline.cli enrich data/paper.pdf       # Stage 4: 富化 Markdown
python -m pdf_pipeline.cli visualize data/paper.pdf    # 快速 HTML 预览（旧 chunker）
python -m pdf_pipeline.cli rag-chunk data/paper.pdf    # Stage 5: RAG 切分（单独跑）
python -m pdf_pipeline.cli rag-visualize rag_chunks.json  # 从 JSON 重新生成 HTML
```

### Python API

```python
from pdf_pipeline import (
    parse_pdf_docling, enhance_all, enrich_markdown,
    rag_chunk_markdown, RAGChunkConfig,
    export_rag_report, render_rag_html,
)
from pdf_pipeline.bindings import load_bindings_json

# Stage 1-2: 解析 + 空间绑定
result = parse_pdf_docling("paper.pdf", export_bindings=True)
bindings = load_bindings_json(result.bindings_path)

# Stage 3: 多模态增强
enhance_all(bindings, result.markdown, "output_dir")

# Stage 4: 富化注入
# 显式传 output_path 落盘到目标目录；不传时默认写到 bindings 的
# assets_dir（build_bindings 按 PDF stem 推导，含点号文件名如 arXiv ID
# "2003.12462v2" 会得到 "2003_12462v2"，与调用方约定的目录名可能不一致）
enriched = enrich_markdown(result.markdown, bindings,
                           output_path="output/paper/final_enriched.md")

# Stage 5: RAG 优化切分
report = rag_chunk_markdown(enriched, bindings=bindings)
export_rag_report(report, "rag_chunks.json")
html = render_rag_html(report, title="paper")
```

> **注意**：`enhancer` 的 LLM/VLM 客户端已在 `_get_client()` 设 `timeout=120, max_retries=1`（与 `agent/nodes.py` 对齐），避免上游 API 无响应时单次调用挂满默认 600s。
>
> **大 PDF 稳定性**：`parser` 已把 docling 各阶段 `*_batch_size` 压到 1（逐页处理，避免 std::bad_alloc 截断）；仍不稳时可用环境变量 `DOCLING_TABLE_STRUCTURE=0`（关表格模型）、后端进程加 `OMP_NUM_THREADS=1`。详见 TROUBLESHOOTING「docling 解析大 PDF 截断 / 偶发 segfault / 挂起」。

## 依赖

- **docling** / **docling-core** — IBM Docling PDF 解析
- **PyMuPDF** (fitz) — PDF 页面渲染 + 图片裁剪
- **openai** — LLM/VLM API 调用 (DashScope 兼容接口)
- 可选: streamlit (Web UI), python-dotenv (.env 加载)

## 设计约束

1. **绑定优先**: 任何增强调用前必须先完成空间绑定
2. **关键图片过滤**: 仅保存有 caption + 正文引用的图片
3. **成本分级**: 装饰性图片/无引用图片跳过；表格不需要多模态增强
4. **高密度输出**: VLM 图片描述 ≤90 tokens，格式 [Function]+[Mechanism]+[Claim]
5. **切分完整性**: 引用与被引用对象分离时必须补偿上下文 (Stage 4)
6. **输出隔离**: 所有产物在 `pdf_pipeline/output/{pdf_stem}/` 下，独立于源文件
