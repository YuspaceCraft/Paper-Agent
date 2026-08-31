"""
tool_summarizer.py — 工具结果摘要器
===================
在工具执行后立即对原始输出做摘要，确保 LLM 只看到关键信息，
不接触原始噪音数据。

核心原则：
  - 短输出（≤2000 字符）：规则提取关键字段
  - 长输出（>2000 字符）：LLM 摘要（温度 0，max_tokens 256）
  - 错误输出：提取错误码和关键描述

设计意图：
  工具原始输出可能包含大量冗余文本（PDF 全文、长检索结果等），
  直接喂给 LLM 会导致上下文爆炸和注意力分散。
  摘要器作为"中间件"在工具返回后、LLM 看到前运行。

使用方式：
  from agent.tool_summarizer import summarize_tool_result

  raw = search_literature.invoke({"query": "transformer", "top_k": 5})
  summary = summarize_tool_result("search_literature", raw)
  # summary.key_findings → ["找到 5 篇相关文献", "DenseNet (2017): 密集连接卷积网络..."]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ToolResultSummary:
    """工具执行结果的摘要。"""
    tool_name: str
    success: bool = True
    key_findings: list[str] = field(default_factory=list)  # 1-3 条关键发现
    error_message: str = ""
    skipped: bool = False            # 因超过重试次数被跳过
    skip_reason: str = ""
    raw_length: int = 0              # 原始输出字符数


# ================================================================
#  规则提取 — 按工具类型匹配
# ================================================================

def _extract_file_list(text: str, max_items: int = 8) -> list[str]:
    """从文件列表类输出中提取文件名。"""
    findings = []

    # 模式 1: 带编号列表 "1. file1.pdf  2. file2.pdf"
    numbered = re.findall(r'\d+[.)]\s*(\S+\.(?:pdf|PDF|txt|md|csv|json))', text)
    if numbered:
        count = len(numbered)
        findings.append(f"目录包含 {count} 个文件: {', '.join(numbered[:max_items])}")
        if count > max_items:
            findings.append(f"...及其他 {count - max_items} 个文件")
        return findings

    # 模式 2: 纯文件名换行
    files = re.findall(r'(\S+\.(?:pdf|PDF|txt|md|csv|json))', text)
    if files:
        count = len(files)
        findings.append(f"目录包含 {count} 个文件: {', '.join(files[:max_items])}")
        if count > max_items:
            findings.append(f"...及其他 {count - max_items} 个文件")
        return findings

    # 模式 3: 路径列表
    paths = re.findall(r'(?:/[\w./-]+|[\w./-]+\.(?:pdf|PDF|txt|md))', text)
    if paths:
        count = len(paths)
        basenames = [p.rsplit('/', 1)[-1] for p in paths[:max_items]]
        findings.append(f"目录包含 {count} 个项目: {', '.join(basenames)}")
        return findings

    return []


def _extract_search_results(text: str, max_items: int = 3) -> list[str]:
    """从检索结果中提取关键信息。"""
    findings = []

    # ---- 质量检测：空结果 ----
    if not text or not text.strip():
        return ["⚠️ 检索返回空结果"]

    # ---- 质量检测：明显的错误/失败消息 ----
    # 只有在整个结果仅为"未找到"消息（无实质文档块）时才视为失败。
    # 语义检索常返回 "关键词X未精确匹配，但以下是相关文档：..." 这种混合输出，
    # 简单匹配 "未找到" 会导致误判。
    if "向量数据库为空" in text:
        findings.append("❌ 检索失败: " + text.strip()[:150])
        return findings
    if "未找到" in text:
        # 检查后续是否有实质文档块内容（50+ 字符的实际文本）
        idx = text.find("未找到")
        after_text = text[idx:] if idx >= 0 else text
        # 取"未找到"之后的部分，跳过第一行（即"未找到"所在行）
        after_lines = after_text.split("\n", 1)
        body_after = after_lines[1] if len(after_lines) > 1 else ""
        # 去掉分隔线和空白
        body_cleaned = re.sub(r'[-=]{3,}.*?[-=]{3,}', '', body_after).strip()
        if len(body_cleaned) < 80:
            # 确实没有实质内容 → 视为检索失败
            findings.append("❌ 检索失败: " + text.strip()[:150])
            return findings
        # 有实质内容 → 不视为失败，继续正常提取（但标注部分未匹配）
        findings.append("ℹ️ 部分关键词未精确匹配，以下为语义相关文档")

    # ---- 质量检测：检测文档块内容是否实质 ----
    # 按 "文档块 N" 或 "--- 文档块 N" 分割提取每个块的实际内容
    chunk_bodies = []
    # 方法：按文档块分隔符切分，取每个分段中分隔符之后的内容
    blocks = re.split(r'(?:^|\n)---?\s*文档块\s*\d+.*?---?\s*\n', text)
    if len(blocks) > 1:
        # 第一个分段是分隔符之前的前言文本，跳过
        chunk_bodies = [b.strip() for b in blocks[1:] if b.strip()]
    if not chunk_bodies:
        # 回退：按 "文档块 N"（无 ---）切分
        blocks = re.split(r'(?:^|\n)文档块\s*\d+[^\n]*\n', text)
        if len(blocks) > 1:
            chunk_bodies = [b.strip() for b in blocks[1:] if b.strip()]
    if not chunk_bodies:
        # 最终回退：没有明显的文档块分隔，将整个文本作为一个块检查
        chunk_bodies = [text.strip()] if text.strip() else []

    # 统计文档块数量
    chunk_count = len(re.findall(r'块\s*\d+|Chunk\s*\d+|来源[：:]', text))
    if chunk_count == 0 and chunk_bodies:
        chunk_count = len(chunk_bodies)

    # ---- 质量检测：检查每个块的内容长度 ----
    short_chunks = 0
    total_content_len = 0
    filename_only_chunks = 0
    for body in chunk_bodies:
        body_stripped = body.strip()
        body_len = len(body_stripped)
        total_content_len += body_len
        if body_len < 80:
            short_chunks += 1
        if body_len > 0:
            lines = [l for l in body_stripped.split('\n') if l.strip()]
            if len(lines) <= 2 and body_len < 150:
                filename_only_chunks += 1

    # ---- 质量检测：截断检测（仅检查正文内容，排除标题元数据行的正常截断） ----
    truncation_count = 0
    for body in chunk_bodies:
        # 在正文中检测 "..." 截断（标题摘要行的 "...(N字)" 截断是正常的，不计入）
        truncation_count += len(re.findall(r'\.\.\.(?!\.)', body))

    # ---- 构建发现列表 ----
    if chunk_count > 0:
        if short_chunks >= max(chunk_count * 0.6, 1) and chunk_count >= 2:
            avg_len = total_content_len // max(chunk_count, 1)
            findings.append(
                f"⚠️ 检索到 {chunk_count} 个文档块，但其中 {short_chunks} 个内容不完整"
                f"（平均 {avg_len} 字符/块），建议检查文献索引是否正常"
            )
        elif filename_only_chunks >= max(chunk_count * 0.5, 1) and chunk_count >= 2:
            findings.append(
                f"⚠️ 检索到 {chunk_count} 个文档块，但内容似乎仅为元数据/文件名，"
                f"缺少实质文本，可能索引异常"
            )
        else:
            avg_len = total_content_len // max(chunk_count, 1)
            findings.append(f"检索到 {chunk_count} 个相关文档块（平均 {avg_len} 字符/块）")

    if truncation_count >= 3:
        findings.append(f"⚠️ 检测到 {truncation_count} 处内容截断，检索结果可能不完整")

    # 提取来源文件名
    sources = set()
    for m in re.finditer(r'(?:文件|来源|filename)[：:\s]*(\S+\.pdf)', text, re.IGNORECASE):
        sources.add(m.group(1))
    if sources:
        findings.append(f"来源文件: {', '.join(sorted(sources))}")

    # 提取论文标题
    titles = re.findall(r'(?:论文|标题|title)[：:\s]*[「"]([^「"]+)[」"]', text)
    if not titles:
        titles = re.findall(r'\[(?:论文|Paper)\]\s*(.+?)(?:\n|$)', text)
    if titles:
        findings.append(f"相关论文: {'; '.join(titles[:max_items])}")

    # 什么都没有时取前 200 字符
    if not findings:
        cleaned = text.strip()[:200]
        if cleaned:
            findings.append(cleaned)

    return findings[:max_items]


def _extract_error(text: str) -> list[str]:
    """从错误输出中提取关键信息。"""
    # 匹配 [ERR] 前缀
    err_match = re.search(r'\[ERR\]\s*(.+)', text)
    if err_match:
        return [f"错误: {err_match.group(1)[:150]}"]

    # 匹配常见错误模式
    for pattern in [r'(Error|Exception|错误)[：:]\s*(.+?)(?:\n|$)',
                    r'(FileNotFound|Permission denied|Not found)[：:]?\s*(.+?)(?:\n|$)',
                    r'(向量数据库为空|未找到|无相关)',]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return [f"错误: {m.group(0)[:150]}"]

    return [f"错误: {text[:150]}"]


def _extract_status(text: str) -> list[str]:
    """从系统状态输出中提取关键信息。"""
    findings = []
    for line in text.split('\n'):
        line = line.strip()
        if any(kw in line for kw in ['文档块', '向量库', '已索引', 'count', '文件数', '工具总数']):
            findings.append(line)
    return findings[:3] if findings else [text[:200]]


def _extract_memory(text: str) -> list[str]:
    """从长期记忆/对话上下文输出中提取关键信息。"""
    findings = []
    count_match = re.search(r'(\d+)\s*(?:条|项|个)', text)
    if count_match:
        findings.append(f"找到 {count_match.group(1)} 条记忆/记录")

    # 提取每条记忆的摘要
    items = re.findall(r'\d+[.)]\s*(.+?)(?=\n\d+[.)]|\n\n|$)', text)
    if items:
        findings.append(f"关键条目: {'; '.join(i[:80] for i in items[:3])}")

    return findings[:3] if findings else [text[:200]]


def _extract_file_operation(text: str) -> list[str]:
    """从文件操作结果中提取关键信息（create_directory, move_file, organize_paper 等）。

    返回格式化的操作摘要，明确区分成功/失败/提示信息。

    采用多层匹配策略，兼容 MCP 工具的多种输出格式：
      1. 标记前缀: [OK] / [ERR] / [INFO] / [WARN]
      2. 结构化标签: [DIR] / [FILE] / [STATS] / [SEARCH] / [SUBDIR]
      3. 自然语言模式: "Created", "Moved", "Found", "already exists" 等
      4. 纯文本回退: 提取最有意义的前几行
    """
    findings = []

    # ---- Layer 1: 标记前缀 ----
    ok_match = re.search(r'\[OK\]\s*(.+)', text)
    if ok_match:
        findings.append(f"✅ 已完成: {ok_match.group(1).strip()[:120]}")

    info_match = re.search(r'\[INFO\]\s*(.+)', text)
    if info_match:
        findings.append(f"ℹ️ {info_match.group(1).strip()[:120]}")

    warn_match = re.search(r'\[WARN\]\s*(.+)', text)
    if warn_match:
        findings.append(f"⚠️ {warn_match.group(1).strip()[:120]}")

    err_match = re.search(r'\[ERR\]\s*(.+)', text)
    if err_match:
        findings.append(f"❌ 失败: {err_match.group(1).strip()[:120]}")

    # ---- Layer 2: 结构化标签 ----
    if not findings:
        dirs = re.findall(r'\[DIR\]\s*(.+?)(?:\n|$)', text)
        files = re.findall(r'\[FILE\]\s*(.+?)(?:\n|$)', text)
        stats = re.findall(r'\[STATS\]\s*(.+?)(?:\n|$)', text)
        search_matches = re.findall(r'\[SEARCH\]\s*(.+?)(?:\n|$)', text)
        subdirs = re.findall(r'\[SUBDIR\]\s*(.+?)(?:\n|$)', text)
        if dirs or files or stats or search_matches or subdirs:
            found_counts = []
            if subdirs:
                found_counts.append(f"{len(subdirs)} 个分类目录")
                for sd in subdirs[:3]:
                    findings.append(f"📁 {sd.strip()}")
            if dirs:
                for d in dirs[:2]:
                    findings.append(f"📂 {d.strip()}")
            if files:
                found_counts.append(f"{len(files)} 个文件")
                for f_item in files[:3]:
                    findings.append(f"📄 {f_item.strip()}")
            if stats:
                for st in stats:
                    findings.append(st.strip())
            if search_matches:
                found_counts.append(f"搜索结果: {len(search_matches)} 条")
                for sm in search_matches[:1]:
                    findings.append(sm.strip())
            if found_counts:
                findings.insert(0, f"共 {', '.join(found_counts)}")

    # ---- Layer 3: 自然语言模式（MCP 工具常见输出格式）----
    if not findings:
        text_lower = text.lower()

        # 成功模式
        success_patterns = [
            (r'(?:created|已创建|创建了?|新建了?)\s*(.+?)(?:\n|$)', '📁 已创建'),
            (r'(?:moved?|已移动|移动了?|移到了?)\s*(.+?)(?:\n|$)', '📄 已移动'),
            (r'(?:copied|已复制)\s*(.+?)(?:\n|$)', '📄 已复制'),
            (r'(?:deleted|已删除|removed)\s*(.+?)(?:\n|$)', '🗑️ 已删除'),
            (r'(?:renamed|已重命名)\s*(.+?)(?:\n|$)', '✏️ 已重命名'),
            (r'(?:organized?|已归类|分类了?)\s*(.+?)(?:\n|$)', '📁 已归类'),
            (r'(?:success|成功|完成|done|ok)\s*[:：]?\s*(.+?)(?:\n|$)', '✅'),
            (r'(?:found|找到|发现)\s*(\d+)\s*(?:files?|director\w+|papers?|items?|篇|个)', '🔍'),
            (r'(?:listed?|列出)\s*(\d+)\s*(?:files?|director\w+|papers?|items?|篇|个)', '📋'),
        ]
        for pattern, prefix in success_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                # 提取匹配的完整行
                line_start = max(text.rfind('\n', 0, m.start()), 0)
                line_end = text.find('\n', m.end())
                if line_end == -1:
                    line_end = len(text)
                full_line = text[line_start:line_end].strip()
                findings.append(f"{prefix} {full_line[:120]}")
                break  # 只取第一个成功模式

        # 失败/冲突模式（如果没有匹配到成功模式）
        if not findings:
            fail_patterns = [
                (r'(?:already\s+exists?|已存在|已经存在)', '⚠️ 资源已存在'),
                (r'(?:not\s+found|未找到|找不到|不存在)', '❌ 未找到'),
                (r'(?:permission\s+denied|权限不足|拒绝访问)', '❌ 权限不足'),
                (r'(?:failed?|失败|出错|error)', '❌ 操作失败'),
                (r'(?:empty|为空|空目录|no\s+files?)', 'ℹ️ 目录为空'),
            ]
            for pattern, prefix in fail_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    line_start = max(text.rfind('\n', 0, m.start()), 0)
                    line_end = text.find('\n', m.end())
                    if line_end == -1:
                        line_end = len(text)
                    full_line = text[line_start:line_end].strip()
                    findings.append(f"{prefix}: {full_line[:120]}")
                    break

        # 统计数字回退
        if not findings:
            count_match = re.search(
                r'(\d+)\s*(?:files?|director\w+|papers?|documents?|items?|entries?|个|篇|条|项)',
                text, re.IGNORECASE,
            )
            if count_match:
                findings.append(f"共 {count_match.group(1)} 项")

    # ---- Layer 4: 纯文本回退（取第一段有意义的内容）----
    if not findings:
        # 取第一行非空行作为摘要
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # 跳过无意义的头尾行
            meaningful = [l for l in lines if len(l) > 10 and not l.startswith('==')]
            if meaningful:
                findings.append(meaningful[0][:200])
            else:
                findings.append(lines[0][:200])

    return findings[:3]


# ================================================================
#  工具名 → 提取策略映射
# ================================================================

_EXTRACTION_RULES: dict[str, callable] = {
    "list_directory": _extract_file_list,
    "list_allowed_directories": _extract_file_list,
    "search_literature": _extract_search_results,
    "search_papers": _extract_search_results,
    "get_paper_detail": _extract_search_results,
    "get_paper_data": _extract_search_results,
    "compare_papers": _extract_search_results,
    "get_system_status": _extract_status,
    "get_conversation_context": _extract_memory,
    "search_long_term_memory": _extract_memory,
    # === 文件管理工具 ===
    "create_directory":       _extract_file_operation,
    "move_file":              _extract_file_operation,
    "organize_paper":         _extract_file_operation,
    "search_files":           _extract_file_list,
    "get_file_info":          _extract_file_operation,
    "list_paper_categories":  _extract_file_operation,
}


def _extract_by_rules(tool_name: str, text: str) -> list[str] | None:
    """规则提取，无法识别则返回 None 触发 LLM 摘要。"""
    # 错误输出优先处理
    if text.startswith("[ERR]") or "Error" in text[:200]:
        return _extract_error(text)

    # 按工具名查找策略
    extractor = _EXTRACTION_RULES.get(tool_name)
    if extractor:
        return extractor(text)

    # 模糊匹配工具名
    for key, fn in _EXTRACTION_RULES.items():
        if key in tool_name.lower() or tool_name.lower() in key:
            return fn(text)

    # 文件管理工具名提示集（处理 MCP 适配后的名称变体）
    _FILE_MGMT_NAME_HINTS = {
        "create_directory", "move_file", "organize_paper",
        "search_files", "get_file_info", "list_paper_categories",
    }
    if tool_name.lower() in _FILE_MGMT_NAME_HINTS or any(
        hint in tool_name.lower() for hint in _FILE_MGMT_NAME_HINTS
    ):
        return _extract_file_operation(text)

    # 通用回退 — 取第一段有意义的内容
    cleaned = text.strip()
    if len(cleaned) <= 300:
        return [cleaned]

    # 取第一段（到第一个空行为止）
    first_para = cleaned.split('\n\n')[0] if '\n\n' in cleaned else cleaned[:300]
    return [first_para + ("..." if len(cleaned) > 300 else "")]


# ================================================================
#  LLM 摘要 — 长输出压缩
# ================================================================

_LLM_SUMMARIZE_PROMPT = """\
提取以下工具输出中的 1-3 条关键发现。每条不超过 60 字。
只返回中文摘要，不返回解释。
如果输出是错误信息，摘要以"错误:"开头。

工具名: {tool_name}
输出内容:
{text}

关键发现:"""


def _summarize_with_llm(tool_name: str, text: str) -> list[str]:
    """使用 LLM 对超长工具输出做摘要（温度 0，max_tokens 256）。"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from config import LLM_MODEL, OPENAI_API_KEY, DASHSCOPE_BASE_URL

        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            model=LLM_MODEL,
            temperature=0,
            max_tokens=256,
        )
        prompt = _LLM_SUMMARIZE_PROMPT.format(
            tool_name=tool_name,
            text=text[:4000],  # 最多喂 4000 字符
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        # 按行拆分，过滤空行
        lines = [l.strip().lstrip('-').strip() for l in content.split('\n') if l.strip()]
        return lines[:3] if lines else [content[:200]]
    except Exception as e:
        # LLM 不可用 → 回退到规则提取
        return _extract_by_rules(tool_name, text) or [text[:300]]


# ================================================================
#  结构化提取 — 从 Pydantic model 直接提取摘要（零 regex 损耗）
# ================================================================

def _summarize_search_result(result) -> ToolResultSummary:
    """从 SearchResult 结构化字段直接构建摘要。"""
    findings = []
    if result.hit_count > 0:
        findings.append(
            f"检索到 {result.hit_count} 个相关文档块"
            f"（平均 {result.avg_chunk_length} 字符/块）"
        )
    if result.sources:
        findings.append(f"来源文件: {', '.join(result.sources[:3])}")
    if result.paper_titles:
        findings.append(f"相关论文: {'; '.join(result.paper_titles[:3])}")
    if result.quality_warning:
        findings.append(f"⚠️ {result.quality_warning}")
    if result.truncation_count >= 3:
        findings.append(f"⚠️ 检测到 {result.truncation_count} 处内容截断")
    if result.short_chunk_count > 0:
        findings.append(f"⚠️ {result.short_chunk_count} 个文档块内容不完整")

    return ToolResultSummary(
        tool_name="search_literature",
        success=result.hit_count > 0 and not result.quality_warning.startswith("❌"),
        key_findings=findings[:3] if findings else ["检索完成"],
        raw_length=len(result.raw_formatted),
    )


def _summarize_file_list_result(result) -> ToolResultSummary:
    """从 FileListResult 结构化字段直接构建摘要。"""
    findings = []
    if result.item_count > 0:
        findings.append(f"目录包含 {result.item_count} 个项目")
    if result.directories:
        findings.append(f"目录: {', '.join(result.directories[:5])}")
    if result.files:
        findings.append(f"文件: {', '.join(result.files[:8])}")
    if not findings and result.items:
        findings.append(f"共 {len(result.items)} 项")
    return ToolResultSummary(
        tool_name="list_directory",
        success=True,
        key_findings=findings[:3] if findings else ["目录为空"],
        raw_length=len(result.raw_formatted),
    )


def _summarize_file_op_result(result) -> ToolResultSummary:
    """从 FileOperationResult 结构化字段直接构建摘要。"""
    status_icon = {"ok": "✅", "error": "❌", "exists": "⚠️", "empty": "ℹ️", "skipped": "⏭️"}

    findings = []
    icon = status_icon.get(result.status, "")
    if result.details:
        findings.append(f"{icon} {result.details[:120]}" if icon else result.details[:120])
    elif result.target:
        findings.append(f"{icon} {result.operation}: {result.target}" if icon else f"{result.operation}: {result.target}")

    if result.stats:
        stats_str = ", ".join(f"{k}={v}" for k, v in result.stats.items())
        findings.append(f"统计: {stats_str}")

    return ToolResultSummary(
        tool_name=result.operation,
        success=result.status == "ok",
        key_findings=findings[:3] if findings else [f"{result.operation}完成"],
        error_message=result.details if result.status == "error" else "",
        raw_length=len(result.raw_formatted),
    )


def _summarize_memory_result(result) -> ToolResultSummary:
    """从 MemoryResult 结构化字段直接构建摘要。"""
    findings = []
    if result.count > 0:
        label = "条记忆" if result.source == "ltm" else "条记录"
        findings.append(f"找到 {result.count} {label}")
    if result.keywords:
        findings.append(f"关键词: {', '.join(result.keywords[:5])}")
    if result.items:
        findings.append(f"关键条目: {'; '.join(item[:80] for item in result.items[:3])}")
    return ToolResultSummary(
        tool_name="memory_search",
        success=result.count > 0,
        key_findings=findings[:3] if findings else ["未找到相关记忆"],
        raw_length=len(result.raw_formatted),
    )


def _summarize_status_result(result) -> ToolResultSummary:
    """从 SystemStatusResult 结构化字段直接构建摘要。"""
    findings = []
    if result.vector_count > 0:
        findings.append(f"向量库文档块: {result.vector_count}")
    else:
        findings.append("向量库: 为空")
    if result.ltm_count > 0:
        findings.append(f"长期记忆: {result.ltm_count} 条")
    if result.conversation_turns > 0:
        findings.append(f"对话轮数: {result.conversation_turns}")
    if result.embedding_model:
        findings.append(f"模型: {result.embedding_model[:40]}")
    return ToolResultSummary(
        tool_name="get_system_status",
        success=True,
        key_findings=findings[:3],
        raw_length=len(result.raw_formatted),
    )


def _summarize_query_rewrite_result(result) -> ToolResultSummary:
    """从 QueryRewriteResult 结构化字段直接构建摘要。"""
    findings = []
    if result.needs_rewrite:
        findings.append(f"查询已重写: {result.original[:60]} → {result.rewritten[:80]}")
    else:
        findings.append(f"查询已清晰，无需重写 (类型: {result.query_type})")
    if result.explanation:
        findings.append(f"理由: {result.explanation[:100]}")
    return ToolResultSummary(
        tool_name="rewrite_query",
        success=True,
        key_findings=findings[:3],
        raw_length=len(result.raw_formatted),
    )


# ================================================================
#  _summarize_string — 原 regex 路径（重命名，作为 string fallback）
# ================================================================

def _summarize_string(
    tool_name: str,
    text: str,
    max_findings: int = 3,
    llm_threshold: int = 2000,
) -> ToolResultSummary:
    """对字符串格式的工具输出做摘要（regex 规则 + LLM 回退）。

    这是原有 summarize_tool_result() 逻辑的保留路径，
    用于 MCP 外部工具（其返回不受我们控制的字符串）。
    """
    # 错误输出
    if text.startswith("[ERR]") or text.startswith("[WARN]"):
        findings = _extract_error(text)
        return ToolResultSummary(
            tool_name=tool_name,
            success=False,
            key_findings=findings,
            error_message=findings[0] if findings else text[:150],
            raw_length=len(text),
        )

    # 短输出 → 规则提取
    if len(text) <= llm_threshold:
        findings = _extract_by_rules(tool_name, text)
        if findings is None:
            findings = [text[:300]]
        return ToolResultSummary(
            tool_name=tool_name,
            success=True,
            key_findings=findings[:max_findings],
            raw_length=len(text),
        )

    # 长输出 → LLM 摘要
    findings = _summarize_with_llm(tool_name, text)
    return ToolResultSummary(
        tool_name=tool_name,
        success=True,
        key_findings=findings[:max_findings],
        raw_length=len(text),
    )


# ================================================================
#  主入口 — isinstance 分发 + string fallback
# ================================================================

def summarize_tool_result(
    tool_name: str,
    raw_output,
    max_findings: int = 3,
    llm_threshold: int = 2000,
) -> ToolResultSummary:
    """
    对工具输出做摘要。

    分发策略:
      1. Pydantic model → 结构化提取（零 token 消耗，直接从字段组装）
      2. str → 原 regex/LLM 路径（MCP 外部工具 fallback）

    参数:
        tool_name:     工具名称（用于 string fallback 时的提取策略选择）
        raw_output:    工具输出（Pydantic model 或 str）
        max_findings:  最多提取几条关键发现
        llm_threshold: string 路径中超过此字符数触发 LLM 摘要

    返回:
        ToolResultSummary
    """
    # ── 结构化分发 ──
    # 延迟导入避免循环依赖（tool_models 可能 import 此模块的 ToolResultSummary）
    from agent.tool_models import (
        SearchResult,
        FileListResult,
        FileOperationResult,
        MemoryResult,
        SystemStatusResult,
        QueryRewriteResult,
    )

    if isinstance(raw_output, SearchResult):
        summary = _summarize_search_result(raw_output)
        summary.tool_name = tool_name
        return summary

    if isinstance(raw_output, FileListResult):
        summary = _summarize_file_list_result(raw_output)
        summary.tool_name = tool_name
        return summary

    if isinstance(raw_output, FileOperationResult):
        summary = _summarize_file_op_result(raw_output)
        summary.tool_name = tool_name
        return summary

    if isinstance(raw_output, MemoryResult):
        summary = _summarize_memory_result(raw_output)
        summary.tool_name = tool_name
        return summary

    if isinstance(raw_output, SystemStatusResult):
        summary = _summarize_status_result(raw_output)
        summary.tool_name = tool_name
        return summary

    if isinstance(raw_output, QueryRewriteResult):
        summary = _summarize_query_rewrite_result(raw_output)
        summary.tool_name = tool_name
        return summary

    # ── String fallback（MCP 外部工具 / 向后兼容）──
    if raw_output is None:
        return ToolResultSummary(
            tool_name=tool_name,
            success=False,
            error_message="工具返回 None",
            raw_length=0,
        )

    text = str(raw_output).strip() if raw_output else ""

    if not text:
        return ToolResultSummary(
            tool_name=tool_name,
            success=False,
            error_message="工具返回空结果",
            raw_length=0,
        )

    return _summarize_string(tool_name, text, max_findings=max_findings,
                             llm_threshold=llm_threshold)
