"""
splitter.py — 智能文档切分器（多策略路由）
==========
根据文件类型自动选择最优切分策略，保护每种格式的语义结构。

设计原则：
  不同文件类型的"语义单元"完全不同，一刀切的字符切分是破坏性的：
    - 代码文件切在函数中间 → 语法结构被破坏 → 检索到的代码片段不可用
    - CSV 按行切分 → 列对齐丢失 → 无法理解表格含义
    - PDF 论文切在图表跨页处 → 上下文断裂 → 引用失效

路由策略：
  file_type metadata (由 loader.py 设置)
    │
    ├── .txt  .md  .html  .xml  .yaml  .docx → 文本策略（RecursiveCharacterTextSplitter）
    ├── .py   .js   .ts   .java .go   .rs    → 代码策略（AST / 语义边界）
    ├── .pdf                                   → 论文策略（页边界 + 语义合并）
    ├── .csv                                   → 表格策略（结构提取，不向量化原始行）
    └── 未知格式                               → 通用文本策略（降级兜底）
"""

import re
import ast as _ast
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, PAPER_CHUNK_SIZE, PAPER_CHUNK_OVERLAP, INCLUDE_REFERENCES


# ================================================================
#  策略 1: 文本策略 — RecursiveCharacterTextSplitter
# ================================================================
# 适用于：纯文本、Markdown、HTML、XML、YAML、Word 文档等
# 这些格式的语义边界 = 段落/章节/句子，递归切分天然合适。
# ================================================================

def _split_text(docs: list[Document]) -> list[Document]:
    """通用文本切分，使用递归字符切分器。"""
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 保留 file_type metadata
    for chunk in chunks:
        if "file_type" not in chunk.metadata:
            chunk.metadata["file_type"] = docs[0].metadata.get("file_type", ".txt")

    return chunks


# ================================================================
#  策略 2: 代码策略 — AST / 语义边界切分
# ================================================================
# 问题：RecursiveCharacterTextSplitter 会在代码中间任意切分，
#       导致函数被腰斩、类定义不完整、import 与使用分离。
# 方案：Python 用 AST 提取顶层节点（函数/类），
#       JS/TS/Java/Go/Rust 用正则匹配函数签名边界。
# 结果：每个代码块是一个完整的函数/类/模块声明。
# ================================================================

def _split_python(doc: Document) -> list[Document]:
    """
    用 AST 将 Python 源码按函数和类边界切分。

    每个顶层定义（函数/类/async 函数）成为一个独立 chunk。
    模块级代码（import、全局变量、装饰器）前置在每个 chunk 前面，
    确保 LLM 能看到导入和依赖。
    """
    source = doc.page_content
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        # 代码有语法错误时降级为文本切分
        print(f"  [SPLIT] [WARN] Python 语法错误, 降级为文本切分: {doc.metadata.get('filename', '?')}")
        return _split_text([doc])

    lines = source.splitlines(keepends=True)
    chunks = []

    # 收集模块级前置代码（import、全局变量、装饰器定义等）
    preamble_end = 0
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            preamble_end = max(preamble_end, node.end_lineno)
        elif isinstance(node, (_ast.Assign, _ast.AnnAssign)):
            # 全局变量赋值
            preamble_end = max(preamble_end, node.end_lineno)
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            break  # 遇到第一个函数/类就停止

    preamble = "".join(lines[:preamble_end]) if preamble_end > 0 else ""

    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            # 取装饰器到函数/类结束之间的所有行
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            end = node.end_lineno
            body = "".join(lines[start - 1:end])
            chunk_text = (preamble + "\n" + body).strip() if preamble else body.strip()

            node_type = "class" if isinstance(node, _ast.ClassDef) else "function"
            name = node.name
            docstring = _ast.get_docstring(node)
            summary = docstring.split("\n")[0] if docstring else ""

            # 在 page_content 中注入搜索标签
            # 为什么：嵌入模型只看 page_content，metadata 不会被嵌入。
            # 加了 "Code: function hello" 这样的标签后，
            # 用户搜索 "hello 函数" 时向量相似度更高。
            filename = doc.metadata.get("filename", "")
            search_tags = f"[Code: {node_type} {name}"
            if summary:
                search_tags += f" — {summary}"
            search_tags += f" in {filename}]"

            chunks.append(Document(
                page_content=search_tags + "\n" + chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_type": "code",
                    "node_type": node_type,
                    "node_name": name,
                    "summary": summary,
                    "line_range": f"{start}-{end}",
                }
            ))

    if not chunks:
        # 没有函数/类的脚本（如纯配置），整体作为一个块
        return [Document(
            page_content=source,
            metadata={**doc.metadata, "chunk_type": "code", "node_type": "module"}
        )]

    return chunks


# JS/TS/Java/Go/Rust 的函数/类/接口模式
_CODE_PATTERNS = {
    ".js": [
        # function 声明 / 箭头函数赋值 / 类声明 / 方法定义 / 模块导出
        r'(?:(?:export\s+)?(?:async\s+)?function\s+\w+[^{]*\{)',
        r'(?:(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{)',
        r'(?:(?:export\s+)?class\s+\w+)',
    ],
    ".ts": [
        r'(?:(?:export\s+)?(?:async\s+)?function\s+\w+[^{]*\{)',
        r'(?:(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{)',
        r'(?:(?:export\s+)?(?:abstract\s+)?class\s+\w+)',
        r'(?:(?:export\s+)?interface\s+\w+)',
        r'(?:(?:export\s+)?type\s+\w+\s*=)',
    ],
    ".java": [
        r'(?:(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?\w+(?:<[^>]+>)?\s+\w+\s*\([^)]*\)\s*(?:throws\s+\w+)?\s*\{)',
    ],
    ".go": [
        r'(?:func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\([^)]*\))',
    ],
    ".rs": [
        r'(?:(?:pub\s+)?(?:async\s+)?fn\s+\w+[^{]*\{)',
        r'(?:(?:pub\s+)?(?:struct|enum|trait|impl)\s+\w+)',
    ],
}


def _split_code_by_pattern(doc: Document) -> list[Document]:
    """
    用正则模式匹配函数/类边界，适用于 JS/TS/Java/Go/Rust。

    工作原理：
      1. 找到所有函数/类声明的起始行号
      2. 通过大括号计数确定每个声明的结束行号
      3. 在声明边界处切分，每个声明 + 头部（import 等）组成一个 chunk
    """
    source = doc.page_content
    lines = source.splitlines(keepends=True)
    ext = doc.metadata.get("file_type", ".js")
    patterns = _CODE_PATTERNS.get(ext, _CODE_PATTERNS[".js"])

    # 找所有匹配的起始行
    starts = []
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.search(pat, line):
                starts.append(i)
                break
    starts.sort()

    if not starts:
        # 没有可识别的函数/类，整个文件作为一个块
        return [Document(
            page_content=source,
            metadata={**doc.metadata, "chunk_type": "code", "node_type": "module"}
        )]

    # 计算大括号平衡，确定每个声明块的结束行
    def find_block_end(start_line: int) -> int:
        depth = 0
        for i in range(start_line, len(lines)):
            depth += lines[i].count("{") - lines[i].count("}")
            if depth <= 0 and i > start_line:
                return i + 1
        return len(lines)

    # 提取头部（import/require 等）
    preamble_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "require(", "//", "/*", "*", "package ", "use ")) or not stripped:
            preamble_end = i + 1
        else:
            break
    preamble = "".join(lines[:preamble_end]).strip()

    chunks = []
    for start in starts:
        end = find_block_end(start)
        body = "".join(lines[start:end])
        chunk_text = (preamble + "\n\n" + body).strip() if preamble else body.strip()

        # 提取声明名
        first_line = lines[start].strip()
        name = "unknown"
        name_match = re.search(
            r'(?:function|class|interface|type|struct|enum|trait|impl|fn|func)\s+(\w+)',
            first_line
        )
        if name_match:
            name = name_match.group(1)

        # 在 page_content 中注入搜索标签
        # 同样原因：嵌入模型只看 page_content，metadata 不会被嵌入
        filename = doc.metadata.get("filename", "")
        search_tags = f"[Code: declaration {name} in {filename}]"

        chunks.append(Document(
            page_content=search_tags + "\n" + chunk_text,
            metadata={
                **doc.metadata,
                "chunk_type": "code",
                "node_type": "declaration",
                "node_name": name,
                "line_range": f"{start + 1}-{end}",
            }
        ))

    return chunks


def _split_code(docs: list[Document]) -> list[Document]:
    """代码策略路由器：根据具体语言选择最优拆分方式。"""
    all_chunks = []
    for doc in docs:
        file_type = doc.metadata.get("file_type", "")
        if file_type == ".py":
            chunks = _split_python(doc)
        elif file_type in (".js", ".ts", ".java", ".go", ".rs"):
            chunks = _split_code_by_pattern(doc)
        else:
            chunks = _split_text([doc])

        for chunk in chunks:
            chunk.metadata.setdefault("file_type", file_type)
        all_chunks.extend(chunks)

    return all_chunks


# ================================================================
#  策略 3: PDF 论文策略 — 章节感知切分
# ================================================================
# 问题：原方案按页加载 → 跨页合并 → RecursiveCharacterTextSplitter 盲切，
#       论文的 IMRaD 章节结构被破坏，摘要被切碎。
# 方案：利用 paper_parser 提供的章节数据，在章节边界处切分。
#       - 摘要永不切分（完整保留）
#       - 短章节（≤800 字符）保持完整
#       - 长章节在章节内按段落边界进一步切分
#       - 每个 chunk 注入搜索标签 [Paper: 章节名 | 论文标题 | 年份]
# 深层逻辑：论文的"语义单元"是章节，不是字符数。
# ================================================================

# 跨页标志（保留，用于降级模式）
_CROSS_PAGE_MARKERS = [
    r"接上页[：:]",
    r"续表\s*\d*",
    r"\(续\)",
    r"continued\b",
    r"Table\s+\d+\s*[-–—]\s*continued",
    r"图\s*\d+\s*[-–—]\s*续",
]

# 不完整表格特征
_INCOMPLETE_TABLE = re.compile(
    r"^(\s*\||\+[-]+\+|│).*[^|│]$",
    re.MULTILINE,
)


def _split_pdf(docs: list[Document]) -> list[Document]:
    """
    PDF 论文切分策略 — 章节感知。

    路由逻辑：
      - 如果 doc 包含 _body_sections 元数据 → 增强切分（paper_parser 产物）
      - 如果 doc 包含 page 元数据 → 降级模式（PyPDFLoader 产物）
      - 其他 → 通用文本切分

    增强切分下：
      1. 摘要保持为 1 个完整 chunk
      2. 正文章节在章节边界处切分
      3. 短章节（≤ PAPER_CHUNK_SIZE）保持完整
      4. 长章节内部按段落进一步切分
      5. 参考文献可选纳入
    """
    if not docs:
        return []

    all_chunks = []

    for doc in docs:
        # 检测格式：增强（有 _body_sections）vs 降级（有 page）
        if "_body_sections" in doc.metadata:
            all_chunks.extend(_split_pdf_enhanced(doc))
        elif "page" in doc.metadata:
            # 降级模式：使用原有跨页合并逻辑
            all_chunks.extend(_split_pdf_legacy_merge([doc]))
        else:
            # 未知格式：降级为文本切分
            all_chunks.extend(_split_text([doc]))

    return all_chunks


def _split_pdf_enhanced(doc: Document) -> list[Document]:
    """
    增强论文切分：基于 paper_parser 输出的章节结构。

    每个章节是一个语义单元，在章节边界处切分；
    长章节内部按 PAPER_CHUNK_SIZE 进一步切分。

    搜索标签：
      每个 chunk 的开头注入 [Paper: 章节名 | 论文标题 | 年份]，
      帮助嵌入模型将检索查询与论文结构对齐。
    """
    chunks = []
    paper_title = doc.metadata.get("paper_title", "")
    paper_authors = doc.metadata.get("paper_authors", "")
    paper_year = doc.metadata.get("paper_year", "")
    filename = doc.metadata.get("filename", "")

    # 基础元数据（去除内部字段 _xxx）
    base_metadata = {
        k: v for k, v in doc.metadata.items()
        if not k.startswith("_")
    }
    base_metadata["chunk_type"] = "paper"

    # --- 1. 摘要 chunk（完整保留，永不切分）---
    abstract = doc.metadata.get("_abstract_text", "")
    if abstract:
        search_tags = _build_paper_tags(
            "摘要/Abstract", paper_title, paper_authors, paper_year
        )
        chunks.append(Document(
            page_content=search_tags + "\n" + abstract.strip(),
            metadata={
                **base_metadata,
                "section_name": "摘要/Abstract",
                "section_type": "abstract",
                "is_abstract": True,
            }
        ))

    # --- 2. 正文章节 chunk（章节边界 + 段落内部切分）---
    body_sections = doc.metadata.get("_body_sections", [])
    paper_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PAPER_CHUNK_SIZE,
        chunk_overlap=PAPER_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )

    for sec in body_sections:
        section_name = sec.get("name", "")
        section_content = sec.get("content", "").strip()
        section_type = sec.get("type", "other")

        if not section_content:
            continue

        search_tags = _build_paper_tags(
            section_name, paper_title, paper_authors, paper_year
        )

        # 短章节（如结论、小结）：保持完整，不切分
        if len(section_content) <= PAPER_CHUNK_SIZE:
            chunks.append(Document(
                page_content=search_tags + "\n" + section_content,
                metadata={
                    **base_metadata,
                    "section_name": section_name,
                    "section_type": section_type,
                }
            ))
        else:
            # 长章节：在章节内按段落边界切分
            temp_doc = Document(page_content=section_content, metadata={})
            sub_chunks = paper_splitter.split_documents([temp_doc])
            for i, sub in enumerate(sub_chunks):
                chunks.append(Document(
                    page_content=search_tags + "\n" + sub.page_content,
                    metadata={
                        **base_metadata,
                        "section_name": section_name,
                        "section_type": section_type,
                        "section_part": f"{i + 1}/{len(sub_chunks)}",
                    }
                ))

    # --- 3. 参考文献（可选）---
    if INCLUDE_REFERENCES:
        references = doc.metadata.get("_references_text", "")
        if references:
            search_tags = _build_paper_tags(
                "参考文献/References", paper_title, paper_authors, paper_year
            )
            # 参考文献截断前 2000 字符（避免噪声过多）
            ref_text = references[:2000]
            chunks.append(Document(
                page_content=search_tags + "\n" + ref_text,
                metadata={
                    **base_metadata,
                    "section_name": "参考文献/References",
                    "section_type": "references",
                    "is_references": True,
                }
            ))

    return chunks


def _build_paper_tags(section_name: str, title: str = "",
                      authors: str = "", year: str = "") -> str:
    """
    构建论文块的搜索标签。

    标签放在 page_content 开头，帮助嵌入模型在查询时
    将 NL 问题与论文结构对齐。
    格式: [Paper: 方法 | Change-Agent: A Novel... | 2024]
    """
    tags = f"[Paper: {section_name}"
    if title:
        short_title = title[:80] + ("..." if len(title) > 80 else "")
        tags += f" | {short_title}"
    if year:
        tags += f" | {year}"
    tags += "]"
    return tags


def _split_pdf_legacy_merge(docs: list[Document]) -> list[Document]:
    """
    降级模式：原有的跨页合并 + 文本切分逻辑。

    保留给以下场景：
      - paper_parser 降级到 PyPDFLoader 产生的 per-page Document
      - 旧向量数据库中的 pdf chunk_type 数据

    逻辑与原来的 _split_pdf 完全一致。
    """
    if not docs:
        return []

    # 跨页检测与合并
    merged = []
    pending = None

    for doc in docs:
        page = doc.metadata.get("page", 0)

        is_continuation = any(
            re.search(pattern, doc.page_content, re.IGNORECASE)
            for pattern in _CROSS_PAGE_MARKERS
        )

        prev_incomplete = False
        if pending:
            prev_incomplete = bool(
                _INCOMPLETE_TABLE.search(pending.page_content)
            )

        if pending and (is_continuation or prev_incomplete):
            pending.page_content += "\n\n" + doc.page_content
            pending.metadata["pages"] = f"{pending.metadata.get('page', 0)}-{page}"
        else:
            if pending:
                merged.append(pending)
            pending = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "pages": str(page)}
            )

    if pending:
        merged.append(pending)

    # 文本切分
    all_chunks = []
    for doc in merged:
        chunks = _split_text([doc])
        for chunk in chunks:
            chunk.metadata["chunk_type"] = "pdf"
            chunk.metadata["pages"] = doc.metadata.get("pages", "?")
        all_chunks.extend(chunks)

    return all_chunks


# ================================================================
#  策略 4: 表格策略 — 结构化抽取（不向量化原始行）
# ================================================================
# 问题：CSV/Excel 按行切块嵌入向量检索 → 列对齐丢失 → 语义混乱。
#       "name: Alice, age: 30, city: Beijing" 这种文本嵌入后
#       问 "谁在 Beijing" 可能召回，但问 "age 列的平均值" 完全失灵。
# 方案：不向量化原始数据行。改为：
#       1. 抽取表的 Schema（列名、类型、行数、列含义）
#       2. 抽样前几条作为数据示例
#       3. 生成一段"表格描述"文本 → 这才是被嵌入和检索的内容
#       4. 原始 CSV 文件路径保存在 metadata 中
# 深层逻辑：表格数据的查询应该是 SQL/代码执行，不是向量相似度。
#           向量检索只负责"找到相关的表格"，不负责"查询数据"。
# ================================================================

def _split_csv(docs: list[Document]) -> list[Document]:
    """
    表格策略：将 CSV 转为"表格描述文档"。
    只嵌入描述（列名、类型、行数、样本行），不嵌入原始数据行。

    输出：每个 CSV 一个 Document，page_content 是表格的描述，
         metadata 中保存原始文件路径和数据预览。
    """
    import csv
    import io

    results = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown.csv")
        content = doc.page_content

        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
        except Exception:
            # CSV 解析失败，降级为文本切分
            print(f"  [SPLIT] [WARN] CSV 解析失败, 降级为文本: {filename}")
            results.extend(_split_text([doc]))
            continue

        if not rows:
            print(f"  [SPLIT] [WARN] CSV 无数据: {filename}")
            continue

        headers = rows[0]
        data_rows = rows[1:]
        total_rows = len(data_rows)

        # 推断列类型（基于前 20 行采样）
        sample_size = min(20, total_rows)
        col_types = {}
        for col_idx, header in enumerate(headers):
            values = [
                row[col_idx] for row in data_rows[:sample_size]
                if col_idx < len(row) and row[col_idx].strip()
            ]
            col_types[header] = _infer_column_type(values)

        # 抽样本行（前 5 行）
        sample_rows = data_rows[:5]
        sample_text = ""
        for row in sample_rows:
            # CSV 行可能列数不一致，做安全处理
            row_text = " | ".join(
                f"{headers[i] if i < len(headers) else 'col'+str(i)}: "
                f"{row[i] if i < len(row) else ''}"
                for i in range(len(headers))
            )
            sample_text += f"  {row_text}\n"

        # 生成表格描述文本（这就是被嵌入检索的内容）
        description = f"""表格文件: {filename}
            列数: {len(headers)} | 数据行数: {total_rows}

            列定义:
            {_format_schema(headers, col_types)}

            数据样本（前 5 行）:
            {sample_text}"""

        # 注入搜索标签，确保 "员工表" "employees" 等关键词可被检索
        search_tags = (
            f"[Table: {filename} — {len(headers)} columns, "
            f"{total_rows} rows: {', '.join(headers[:5])}"
            f"{'...' if len(headers) > 5 else ''}]"
        )

        results.append(Document(
            page_content=search_tags + "\n" + description,
            metadata={
                **doc.metadata,
                "chunk_type": "table_description",
                "total_rows": total_rows,
                "columns": headers,
                "column_types": col_types,
                "sample_rows": sample_rows,
            }
        ))

    return results


def _infer_column_type(values: list[str]) -> str:
    """推断列的语义类型（数值/日期/分类/文本）。"""
    if not values:
        return "unknown"

    # 数值检测
    numeric = 0
    for v in values:
        try:
            float(v.replace(",", "").replace("%", "").strip())
            numeric += 1
        except ValueError:
            pass
    if numeric > len(values) * 0.8:
        return "numeric"

    # 日期检测
    date_patterns = [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",
        r"\d{4}年\d{1,2}月\d{1,2}日",
    ]
    date_count = 0
    for v in values:
        if any(re.search(p, v) for p in date_patterns):
            date_count += 1
    if date_count > len(values) * 0.8:
        return "date"

    # 分类检测（唯一值少）
    unique = len(set(v.strip().lower() for v in values))
    if unique <= min(20, len(values) * 0.5):
        return f"category ({unique} unique values)"

    return "text"


def _format_schema(headers: list[str], col_types: dict) -> str:
    """格式化列定义，便于 LLM 理解表格结构。"""
    lines = []
    for h in headers:
        lines.append(f"  - {h} ({col_types.get(h, 'unknown')})")
    return "\n".join(lines)


# ================================================================
#  路由器 — 统一的 split_documents 入口
# ================================================================

# file_type → 策略的映射表
_ROUTE_MAP = {
    # 文本类 → 通用文本策略
    ".txt":  _split_text,
    ".md":   _split_text,
    ".html": _split_text,
    ".xml":  _split_text,
    ".yaml": _split_text,
    ".yml":  _split_text,
    ".json": _split_text,
    ".docx": _split_text,

    # 代码类 → 代码策略
    ".py":  _split_code,
    ".js":  _split_code,
    ".ts":  _split_code,
    ".java": _split_code,
    ".go":  _split_code,
    ".rs":  _split_code,

    # PDF → 论文策略
    ".pdf": _split_pdf,

    # 表格 → 表格策略
    ".csv": _split_csv,
}


def split_documents(docs: list[Document]) -> list[Document]:
    """
    智能文档切分入口 — 根据 file_type 自动路由到对应策略。

    参数:
        docs: 原始文档列表（必须带 file_type metadata，由 loader.py 设置）

    返回:
        list[Document]: 切分后的文档块列表

    路由逻辑:
        查看每个 doc 的 metadata["file_type"]，分发给对应策略。
        同一批 docs 可能包含多种文件类型，按 type 分组后分别处理。
    """
    if not docs:
        print("[SPLIT] [WARN] 没有文档需要切分")
        return []

    # 按 file_type 分组
    groups: dict[str, list[Document]] = {}
    untyped = []
    for doc in docs:
        ft = doc.metadata.get("file_type", "")
        if ft:
            groups.setdefault(ft, []).append(doc)
        else:
            untyped.append(doc)

    print(f"[SPLIT] 共 {len(docs)} 个文档, "
          f"涵盖 {len(groups)} 种文件类型: {list(groups.keys())}")

    all_chunks = []

    for file_type, group in groups.items():
        strategy = _ROUTE_MAP.get(file_type)
        if strategy is None:
            print(f"  [SPLIT] [FALLBACK] 未知类型 {file_type}, 降级为文本切分")
            strategy = _split_text

        print(f"  [SPLIT] {file_type}: {len(group)} 个文档 → {strategy.__name__}")
        chunks = strategy(group)
        all_chunks.extend(chunks)

    # 无 file_type 的文档用文本策略兜底
    if untyped:
        print(f"  [SPLIT] [FALLBACK] {len(untyped)} 个无类型标记的文档 → 文本切分")
        all_chunks.extend(_split_text(untyped))

    print(f"[SPLIT] [OK] 切分完成: {len(docs)} 个文档 → {len(all_chunks)} 个块")
    return all_chunks
