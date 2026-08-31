# Pipeline 任务状态

**最后更新**: 2026-07-08

## 总体进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一：结构化提取 | ✅ 完成 | Docling 解析 + 公式文本/图片裁剪保存 |
| 阶段二：空间绑定校验 | ⚠️ 部分完成 | bindings.json + 内容文件已输出，缺 bound_context |
| 阶段三：多模态语义增强 | ✅ 完成 | enhancer.py: 公式 LLM + 图片 VLM 增强 |
| 阶段四：精准注入与切分 | ⚠️ 部分完成 | 增强描述注入已完成，引用感知切分/Metadata 注入待开发 |
| 质量评估 | ❌ 未开始 | Ragas 自动评分 |

## 阶段一详情（✅ 完成）

- [x] PDF → Markdown（Docling, `do_table_structure=True`）
- [x] bbox + page_no 提取（`extract_bindings_from_doc()`）
- [x] 标题关联（cref 解析 + 近邻匹配后备）
- [x] 公式文本提取 + 独立文件保存（`formula_text` 来自 `FormulaItem.orig` → `formula_N.txt`）
- [x] 资源目录（`pdf_pipeline/output/{pdf_stem}/`，独立于源文件路径）
- [x] 图片裁剪保存（PyMuPDF 按 bbox 从 PDF 页面裁剪 → `picture_N.png`，300 DPI）
- [x] 表格解析（Docling markdown 已足够理解，不需要多模态增强）
- [ ] Layout Parser 栏位检测（未集成，Docling 已自带布局分析）
- [ ] 扫描版 PDF 兜底（`do_ocr` 当前为 False）
- [ ] 旋转页面预处理

## 阶段二详情（⚠️ 部分完成）

- [x] bindings.json 输出（element_id, type, page_no, bbox, caption, formula_text, formula_path, image_path）
- [x] 公式内容独立保存（`pdf_pipeline/output/{pdf_stem}/formula_N.txt`）
- [x] 多语言引用回溯（EN/TR/CN 正则匹配，15/15 全匹配）
- [x] 类型一致性校验（_caption_matches_type 防 TABLE 标题错配到 formula）
- [x] HTML 可视化（Spatial Bindings tab）
- [x] Streamlit 集成（bindings 展示 expander）
- [ ] **bound_context 收集**：对每个引用句，取前 2 句上下文存入元素
- [ ] Layout Parser 双栏检测（跨栏元素归属）
- [ ] 置信度字段（confidence score）

## 阶段三详情（✅ 完成）

- [x] 图片预保存（PyMuPDF bbox 裁剪 → `picture_N.png`）
- [x] 公式预保存（`FormulaItem.orig` → `formula_N.txt`）
- [x] 公式增强：`enhance_formulas()` — 读取 `formula_N.txt` + 引用上下文 → LLM 解释
  - Model: `qwen3.6-max-preview`, Prompt 适配纯文本公式（非 LaTeX）
  - 输出: `formula_N_enhanced.txt` + `formula_desc` 字段
- [x] 图片增强：`enhance_images()` — 读取 `picture_N.png` + caption + 摘要 → VLM 描述
  - Model: `qwen3.7-plus` (vision), 图片 base64 编码
  - 输出: `picture_N_description.txt` + `picture_desc` 字段
- [x] CLI 集成：`docling_cli.py enhance <pdf>`
- [x] `.env` 自动加载：`DASHSCOPE_API_KEY` 从项目根目录 `.env` 读取
- [x] 表格增强：**不需要** — Docling markdown 表格已可理解

### 用法

```bash
# 完整三阶段流程
python docling_cli.py enhance data/paper.pdf

# 或编程使用
from pdf_pipeline.enhancer import enhance_all
from pdf_pipeline.bindings import load_bindings_json
bindings = load_bindings_json("pdf_pipeline/output/paper/bindings.json")
enhance_all(bindings, markdown, output_dir)
```

## 阶段四详情（⚠️ 部分完成）

- [x] 富化 Markdown 生成（`final_enriched.md`）
  - `inject_enhancements()`: formula_desc 替换 `<!-- formula-not-decoded -->` 占位符
  - `inject_enhancements()`: picture_desc 替换 `<!-- image -->` 占位符
  - 映射方式: element_id 后缀序号 → 第 N 个同类型占位符
  - CLI: `python docling_cli.py enrich <pdf>` / `python docling_cli.py all <pdf>`
- [ ] 引用感知切分
  - 一级：按标题层级切分
  - 二级：保护公式/表格/图片块不被截断
  - 三级：引用句与被引用对象分离时，bound_context 注入目标 Chunk 头部
- [ ] Chunk Metadata 注入
  - paper_id, section_title, page_num, element_type, source_bbox, confidence_score

## 核心约束检查清单

- [x] 绑定优先原则：元素必须先有 bbox+type 才能后续处理
- [ ] 成本分级策略：行内公式/装饰性图片/已有解释的公式跳过增强
- [ ] 置信度门控：VLM 描述 < 0.7 → 降级为原始 Caption
- [ ] 切分完整性红线：引用与被引用对象分离时必须补偿上下文
- [ ] 验证前置：Ragas ContextRelevancy / Faithfulness 自动评分
- [ ] Metadata 不可省略：无 Metadata 的 Chunk 禁止入库
- [ ] 版本隔离：工具/Prompt/阈值变更需创建新版本

## 技术栈对照

| 组件 | 计划工具 | 实际实现 | 原因 |
|------|---------|---------|------|
| PDF 解析 | Marker | **Docling** | 用户决策沿用 |
| 版面分析 | Layout Parser | Docling 内置 | Docling 自带 layout analysis |
| 表格解析 | GMFT | Docling `do_table_structure` | 轻量模型覆盖常规表格 |
| 公式描述 | Qwen2.5-VL-7B | **qwen3.6-max-preview** | DashScope LLM, 纯文本公式 |
| 图片描述 | Qwen2.5-VL-72B | **qwen3.7-plus** (vision) | DashScope VLM, ≤90 tokens 高密度 |
| 引用回溯 | 自研正则 | 自研正则 ✅ | 多语言 EN/TR/CN |
| 语义切分 | LangChain | 自研 academic_chunker | 已有实现 |
| 质量评估 | Ragas/TruLens | 待定 | Stage 4 |

## 关键文件

| 文件 | 用途 |
|------|------|
| `pdf_pipeline/` | **独立模块** — PDF 预处理管道（零 agent 依赖） |
| `pdf_pipeline/README.md` | 模块文档：功能 + API + 开发状态 |
| `pdf_pipeline/parser.py` | PDF → Markdown，可选 bindings 导出 |
| `pdf_pipeline/bindings.py` | Stage 1+2 核心：提取/bindings/回溯/校验 + 图片裁剪/公式保存 |
| `pdf_pipeline/enhancer.py` | Stage 3：公式 LLM + 图片 VLM 多模态语义增强 |
| `pdf_pipeline/output/` | **输出目录** — 公式/图片 + bindings + 增强结果（按 PDF stem 分目录） |
| `pdf_pipeline/chunker.py` | 学术切分 + bindings 注入 |
| `pdf_pipeline/viz.py` | HTML 可视化（含 Spatial Bindings tab） |
| `pdf_pipeline/cli.py` | CLI 实现 |
| `pdf_pipeline/_config.py` | 管道专属配置（独立于项目 config.py） |
| `agent/binding_export.py` | → 向后兼容重导出 `pdf_pipeline.bindings` |
| `agent/docling_parser.py` | → 向后兼容重导出 `pdf_pipeline.parser` |
| `agent/academic_chunker.py` | → 向后兼容重导出 `pdf_pipeline.chunker` |
| `agent/chunk_viz.py` | → 向后兼容重导出 `pdf_pipeline.viz` |
| `web/chunk_viz_page.py` | Streamlit UI（消费者） |
| `docling_cli.py` | CLI 入口（薄包装） |
