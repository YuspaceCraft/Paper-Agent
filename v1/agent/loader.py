"""
loader.py — 文档加载器（多文件 + 多格式）
=========
扫描目录、识别文件格式、批量加载为 LangChain Document 对象。

升级能力：
  1. 支持单个文件或整个目录
  2. 支持 .txt / .md / .pdf / .csv / .docx
  3. 失败文件自动跳过（不中断整体流程）
  4. 保留 metadata 方便溯源

LangChain 的 Document 对象包含两个核心属性：
  - page_content: 文档的文本内容
  - metadata:    文档的元数据（来源路径、文件名、文件类型）
"""

from pathlib import Path
from langchain_core.documents import Document


# ============================================================
# 支持的文件格式
# ============================================================
# 每个条目: (后缀列表, 加载器说明, 加载函数)
# 添加新格式只需在这里加一行，不用改其他代码
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".txt":  "纯文本",
    ".md":   "Markdown",
    ".csv":  "CSV 表格",
    ".json": "JSON",
    ".py":   "Python 源码",
    ".js":   "JavaScript 源码",
    ".html": "HTML",
    ".xml":  "XML",
    ".yaml": "YAML",
    ".yml":  "YAML",
}


def _load_text_file(path: Path) -> list[Document]:
    """
    加载纯文本类文件（txt, md, csv, py, js, html, xml, yaml, json 等）。

    所有文本格式统一用 UTF-8 读取，
    内容封装为单个 Document（后续由 splitter 切分）。
    """
    try:
        content = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # UTF-8 失败时尝试 GBK（Windows 常见编码）
        content = path.read_text(encoding="gbk", errors="ignore").strip()
    except Exception:
        # 其他错误也尝试 GBK
        content = path.read_text(encoding="utf-8", errors="ignore").strip()

    if not content:
        print(f"  [LOAD] [WARN] 文件为空: {path.name}")
        return []

    print(f"  [LOAD] [OK] {path.name} ({len(content)} 字符)")
    return [Document(
        page_content=content,
        metadata={
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": path.suffix.lower(),
        }
    )]


def _load_csv_file(path: Path) -> list[Document]:
    """
    加载 CSV 文件。

    CSV 会把每一行转为可读文本（"列名: 值" 格式），
    然后将全部行合并为一个 Document。

    这样做的原因是：RAG 检索是按"块"匹配的，
    将整个 CSV 作为一个文本块，LLM 可以一次看到所有数据行。
    """
    import csv

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                line = " | ".join(f"{k}: {v}" for k, v in row.items())
                rows.append(line)
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                line = " | ".join(f"{k}: {v}" for k, v in row.items())
                rows.append(line)

    if not rows:
        print(f"  [LOAD] [WARN] CSV 无数据行: {path.name}")
        return []

    content = "\n".join(rows)
    print(f"  [LOAD] [OK] {path.name} ({len(rows)} 行)")
    return [Document(
        page_content=content,
        metadata={
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": ".csv",
        }
    )]


def _load_pdf_file(path: Path) -> list[Document]:
    """
    加载 PDF 论文 — 使用 paper_parser 进行结构化提取。

    提取内容：
      - 元数据：标题、作者、DOI、年份、期刊、关键词
      - 摘要（完整保留）
      - 正文章节（IMRaD + 中文等价章节）
      - 参考文献（单独保留）

    返回单个 Document（含完整元数据和序列化的章节数据），
    下游 splitter 负责章节感知切分。

    如果 paper_parser 不可用（缺少依赖）或解析失败，
    自动降级到 PyPDFLoader（每页一个 Document）。
    """
    # --- 主路径：paper_parser 结构化提取 ---
    try:
        from agent.paper_parser import parse_pdf
    except ImportError as e:
        print(f"  [LOAD] [WARN] paper_parser 不可用 ({e})，降级为 PyPDFLoader")
        return _load_pdf_file_fallback(path)

    try:
        paper = parse_pdf(str(path))
    except Exception as e:
        print(f"  [LOAD] [WARN] paper_parser 解析失败: {e}")
        print(f"  [LOAD] [INFO] 降级为 PyPDFLoader 加载 {path.name}")
        return _load_pdf_file_fallback(path)

    # 组装丰富的元数据
    meta = paper.metadata
    doc = Document(
        page_content=paper.full_text,
        metadata={
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": ".pdf",
            # 论文专属元数据（检索时可用于筛选和来源展示）
            "paper_title": meta.get("title", ""),
            "paper_authors": meta.get("authors", ""),
            "paper_doi": meta.get("doi", ""),
            "paper_year": meta.get("year", ""),
            "paper_journal": meta.get("journal", ""),
            "paper_keywords": meta.get("keywords", ""),
            # 内部字段（_ 前缀，供 splitter 消费，不进入向量检索）
            "_has_abstract": bool(paper.abstract),
            "_section_count": len(paper.body_sections),
            "_abstract_text": paper.abstract,
            "_body_sections": [
                {
                    "name": s["name"],
                    "content": s["content"],
                    "level": s.get("level", 1),
                    "type": s.get("type", "other"),
                }
                for s in paper.body_sections
            ],
            "_references_text": paper.references,
        }
    )
    print(f"  [LOAD] [OK] {path.name} "
          f"(标题: {meta.get('title', '?')[:50]}, "
          f"{len(paper.body_sections)} 个章节, "
          f"摘要: {'Y' if paper.abstract else 'N'})")
    return [doc]


def _load_pdf_file_fallback(path: Path) -> list[Document]:
    """
    降级方案：使用 PyPDFLoader 按页加载 PDF。

    这是原 _load_pdf_file 的实现，保留用于以下场景：
      - paper_parser 导入失败（缺少 pymupdf/pdfplumber）
      - 单个 PDF 解析失败（损坏/扫描件/非标准格式）
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        print(f"  [LOAD] [ERR] 缺少 pypdf 库，跳过 PDF: {path.name}")
        print(f"  [LOAD] [TIP] 请运行: pip install pypdf")
        return []

    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = path.name
        doc.metadata["file_type"] = ".pdf"
    print(f"  [LOAD] [OK] {path.name} ({len(docs)} 页) [降级模式]")
    return docs


def _load_docx_file(path: Path) -> list[Document]:
    """
    加载 Word 文档（.docx）。

    需要安装: pip install docx2txt
    """
    try:
        import docx2txt
    except ImportError:
        print(f"  [LOAD] [ERR] 缺少 docx2txt 库，跳过 DOCX: {path.name}")
        print(f"  [LOAD] [TIP] 请运行: pip install docx2txt")
        return []

    text = docx2txt.process(str(path))
    if not text or not text.strip():
        print(f"  [LOAD] [WARN] 文档为空: {path.name}")
        return []

    print(f"  [LOAD] [OK] {path.name} ({len(text)} 字符)")
    return [Document(
        page_content=text.strip(),
        metadata={
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": ".docx",
        }
    )]


# ============================================================
# 格式 → 加载器 的分发表
# ============================================================
_FORMAT_LOADERS = {
    ".txt":  _load_text_file,
    ".md":   _load_text_file,
    ".json": _load_text_file,
    ".py":   _load_text_file,
    ".js":   _load_text_file,
    ".html": _load_text_file,
    ".xml":  _load_text_file,
    ".yaml": _load_text_file,
    ".yml":  _load_text_file,
    ".csv":  _load_csv_file,
    ".pdf":  _load_pdf_file,
    ".docx": _load_docx_file,
}


# ============================================================
# 公开 API
# ============================================================

def discover_files(
    path: str | Path,
    extensions: set[str] | None = None,
) -> list[Path]:
    """
    扫描目录，返回所有支持格式的文件路径。

    参数:
        path:       目录路径或单个文件路径
        extensions: 限定扫描的文件后缀（默认: 所有支持格式）

    返回:
        list[Path]: 文件路径列表
    """
    root = Path(path)

    # 单个文件 → 直接返回
    if root.is_file():
        ext = root.suffix.lower()
        supported = extensions or set(_FORMAT_LOADERS.keys())
        if ext in supported:
            return [root]
        else:
            return []

    # 目录 → 递归扫描
    if not root.is_dir():
        raise FileNotFoundError(
            f"[ERR] 路径不存在: {root.absolute()}"
        )

    supported = extensions or set(_FORMAT_LOADERS.keys())
    files = []
    for ext in supported:
        files.extend(root.rglob(f"*{ext}"))

    # 排序保证可复现
    files.sort()
    return files


def load_documents(path: str | Path) -> list[Document]:
    """
    加载文档：自动识别路径类型（文件/目录）和文件格式。

    参数:
        path: 文件路径 或 目录路径
              - 文件 → 加载该文件
              - 目录 → 递归加载所有支持格式的文件

    返回:
        list[Document]: LangChain Document 列表
    """
    root = Path(path)
    print(f"[LOAD] 扫描路径: {root.absolute()}")

    # 1. 发现所有需要处理的文件
    files = discover_files(root)

    if not files:
        # 目录存在但没有支持格式的文件
        if root.is_dir():
            print(f"[LOAD] [WARN] 目录下未找到支持的文件格式")
            print(f"[LOAD] [TIP] 支持: {', '.join(_FORMAT_LOADERS.keys())}")
            print(f"[LOAD] [TIP] PDF 需要: pip install pypdf")
            print(f"[LOAD] [TIP] DOCX 需要: pip install docx2txt")
        return []

    print(f"[LOAD] 发现 {len(files)} 个文件待加载")
    print(f"[LOAD] " + "-" * 40)

    # 2. 逐个加载，失败跳过
    all_docs = []
    failed = []

    for file_path in files:
        ext = file_path.suffix.lower()
        loader = _FORMAT_LOADERS.get(ext)

        if loader is None:
            print(f"  [LOAD] [SKIP] 不支持的格式: {file_path.name}")
            failed.append((file_path.name, "不支持的文件格式"))
            continue

        try:
            docs = loader(file_path)
            if docs:
                all_docs.extend(docs)
        except Exception as e:
            print(f"  [LOAD] [ERR] 加载失败 {file_path.name}: {e}")
            failed.append((file_path.name, str(e)))

    # 3. 汇总
    print(f"[LOAD] " + "-" * 40)
    print(f"[LOAD] [OK] 加载完成: {len(all_docs)} 个文档段")

    if failed:
        print(f"[LOAD] [WARN] {len(failed)} 个文件加载失败（已跳过）:")
        for fname, reason in failed:
            print(f"  - {fname}: {reason}")

    return all_docs
