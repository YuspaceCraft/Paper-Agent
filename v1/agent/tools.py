"""
tools.py — Agent 工具集
========
将现有 RAG 组件封装为 LangChain Tool，供 Agent 调用。

工具列表：
  search_literature      — 两阶段文献检索（粗筛 + 精排）
  get_paper_detail       — 按标题/作者精确检索论文详情
  compare_papers         — 多类型混合检索，对比分析
  search_long_term_memory— 检索长期记忆中的相关事实
  get_conversation_context— 获取当前对话历史和状态
  rewrite_query          — 指代消解 + 查询优化
  add_to_memory          — 手动添加长期记忆
  get_system_status      — 查询向量库状态

设计要点：
  - 所有工具通过 _ToolContext 共享全局资源（惰性加载）
  - 返回结构化字符串，便于 Agent 解析
  - 错误不抛异常，返回错误描述字符串
  - MCP 工具自动做 sync 兼容包装（StructuredTool → 支持 .invoke()）
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool, StructuredTool

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI
    from agent.memory import BaseMemory
    from agent.ltm import LongTermMemory

logger = logging.getLogger(__name__)


# ================================================================
#  _ToolContext — 全局资源共享
# ================================================================

class _ToolContext:
    """存储 Agent 工具所需的全局资源（惰性加载）。"""

    def __init__(self):
        self._vector_store: Chroma | None = None
        self._reranker = None
        self._llm: ChatOpenAI | None = None
        self._embeddings = None
        self._memory: BaseMemory | None = None
        self._retrieval_scope: str = "local"

    # ---- 延迟加载 ----

    @property
    def embeddings(self):
        if self._embeddings is None:
            from agent.embedder import get_embeddings
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def vector_store(self) -> Chroma | None:
        if self._vector_store is None:
            from agent.store import load_vector_store, store_exists
            if not store_exists():
                return None
            self._vector_store = load_vector_store(self.embeddings)
        return self._vector_store

    @property
    def reranker(self):
        if self._reranker is None:
            from agent.retriever import Reranker
            from config import RERANK_MODEL
            self._reranker = Reranker(model_name=RERANK_MODEL)
        return self._reranker

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            from agent.generator import create_llm
            self._llm = create_llm()
        return self._llm

    @property
    def memory(self) -> BaseMemory | None:
        return self._memory

    @memory.setter
    def memory(self, value: BaseMemory):
        self._memory = value

    @property
    def ltm(self) -> LongTermMemory | None:
        from agent.memory import HybridMemory
        if isinstance(self._memory, HybridMemory):
            return self._memory.long_term
        return None

    @property
    def retrieval_scope(self) -> str:
        return self._retrieval_scope

    @retrieval_scope.setter
    def retrieval_scope(self, value: str):
        self._retrieval_scope = value

    # ---- 工具函数可调用 ----

    def do_search(self, query: str, top_k: int = 5):
        """执行二阶段检索并返回结构化 SearchResult。"""
        if self.vector_store is None:
            from agent.tool_models import SearchResult
            return SearchResult(
                quality_warning="❌ 检索失败: 向量数据库为空，请先导入文献。",
                raw_formatted="[ERR] 向量数据库为空，请先导入文献。",
            )
        from agent.retriever import create_retriever_with_rerank, build_search_result
        retriever = create_retriever_with_rerank(
            self.vector_store, reranker=self.reranker, final_k=top_k,
        )
        docs = retriever.invoke(query)
        if not docs:
            from agent.tool_models import SearchResult
            return SearchResult(
                raw_formatted=f"未找到与 \"{query}\" 相关的文献。",
            )

        result = build_search_result(docs)

        # 质量检测：如果大部分文档块内容极短，追加警告
        if result.short_chunk_count >= max(result.hit_count * 0.5, 1) and result.hit_count > 0:
            result.quality_warning = (
                f"⚠️ 检索质量警告: {result.short_chunk_count}/{result.hit_count} 个文档块"
                f"内容不完整（平均 {result.avg_chunk_length} 字符/块）。"
                f"文献索引可能异常，建议重新导入文献。"
                f"请勿基于不完整的信息编造论文内容，应诚实告知用户当前无法检索到有效信息。"
            )
            # 更新 raw_formatted 追加警告
            result.raw_formatted += (
                f"\n\n[WARN] 检索质量警告: {result.short_chunk_count}/{result.hit_count}"
                f" 个文档块内容不完整（平均 {result.avg_chunk_length} 字符/块）。"
                f"文献索引可能异常，建议重新导入文献。"
                f"请勿基于不完整的信息编造论文内容，应诚实告知用户当前无法检索到有效信息。"
            )

        return result

    def do_memory_search(self, query: str, top_k: int = 3):
        """检索长期记忆，返回结构化 MemoryResult。"""
        from agent.tool_models import MemoryResult

        if self.ltm is None:
            return MemoryResult(
                source="ltm",
                raw_formatted="[INFO] 长期记忆不可用（需要 HybridMemory 模式）。",
            )
        facts = self.ltm.retrieve(query, top_k=top_k)
        if not facts:
            return MemoryResult(
                source="ltm",
                raw_formatted="长期记忆中没有相关信息。",
            )

        items = [f.content for f in facts]
        keywords: list[str] = []
        seen_kw: set[str] = set()
        for f in facts:
            for kw in (f.keywords or [])[:5]:
                if kw not in seen_kw:
                    keywords.append(kw)
                    seen_kw.add(kw)

        # 构建 raw_formatted（旧版文本格式）
        lines = [f"[长期记忆] 找到 {len(facts)} 条相关记忆:"]
        for i, f in enumerate(facts, 1):
            kw = ", ".join(f.keywords[:5]) if f.keywords else "无"
            lines.append(f"  {i}. {f.content} (关键词: {kw})")
        raw = "\n".join(lines)

        return MemoryResult(
            items=items,
            count=len(facts),
            keywords=keywords[:10],
            source="ltm",
            raw_formatted=raw,
        )

    def do_rewrite(self, query: str):
        """执行查询重写，返回结构化 QueryRewriteResult。"""
        from agent.tool_models import QueryRewriteResult

        if self._memory is None:
            return QueryRewriteResult(
                original=query, rewritten=query,
                raw_formatted=query,
            )
        from agent.query_rewriter import get_query_rewriter
        rewriter = get_query_rewriter()
        result = rewriter.rewrite(query, self._memory, self.llm)

        # raw_formatted 保持旧版文本格式
        lines = [
            f"原始查询: {result.original}",
            f"重写查询: {result.rewritten}",
            f"查询类型: {result.query_type}",
            f"是否重写: {'是' if result.needs_rewrite else '否'}",
        ]
        if result.explanation:
            lines.append(f"理由: {result.explanation}")
        raw = "\n".join(lines)

        return QueryRewriteResult(
            original=result.original,
            rewritten=result.rewritten,
            query_type=result.query_type,
            needs_rewrite=result.needs_rewrite,
            explanation=result.explanation or "",
            raw_formatted=raw,
        )


# 全局单例（模块级别）
_ctx = _ToolContext()


def set_agent_memory(memory: BaseMemory) -> None:
    """设置 Agent 工具使用的对话记忆对象。"""
    _ctx.memory = memory


def get_agent_memory() -> BaseMemory | None:
    """获取当前对话记忆对象。"""
    return _ctx.memory


def set_agent_scope(scope: str) -> None:
    """设置 Agent 工具使用的检索范围（"local" | "online" | "hybrid"）。"""
    _ctx.retrieval_scope = scope


def get_agent_scope() -> str:
    """获取当前检索范围。"""
    return _ctx.retrieval_scope


def set_agent_resources(
    embeddings=None,
    reranker=None,
    vector_store=None,
) -> None:
    """
    注入外部模型实例（如 Streamlit @st.cache_resource 缓存的），
    避免 _ToolContext 重复加载模型 → 避免 CUDA OOM。
    """
    if embeddings is not None:
        _ctx._embeddings = embeddings
    if reranker is not None:
        _ctx._reranker = reranker
    if vector_store is not None:
        _ctx._vector_store = vector_store


def invalidate_vector_store() -> None:
    """强制下次调用时重新加载向量库（上传文件后使用）。"""
    _ctx._vector_store = None


# ================================================================
#  Tool 1: search_literature — 文献检索
# ================================================================

@tool
def search_literature(query: str, top_k: int = 5):
    """
    【本地知识库】在已索引的科研文献中检索与查询相关的内容。

    使用二阶段检索（向量粗筛 + 交叉编码器精排）返回最相关的文献片段。
    适用于：查找具体方法、实验结果、论文结论等。
    注意：此工具仅搜索本地已上传的 PDF 论文，不会在线搜索。

    :param query: 检索查询文本（建议使用完整的问题或关键词）
    :param top_k: 返回的文档块数量，默认 5。事实查询建议 3，综述建议 8
    :return: 格式化后的检索结果，包含文献来源和内容
    """
    if top_k < 1:
        top_k = 3
    if top_k > 20:
        top_k = 20
    return _ctx.do_search(query, top_k=top_k)


# ================================================================
#  Tool 2: get_paper_detail — 精确论文检索
# ================================================================

@tool
def get_paper_detail(paper_title: str = "", author: str = "", year: str = ""):
    """
    【本地知识库】按论文标题、作者或年份精确检索某篇论文的详细信息。

    至少需要提供一个检索条件。会返回论文的摘要、方法、结论等章节内容。
    适用于：用户询问某篇特定论文的详细内容时。
    注意：此工具仅搜索本地已上传的 PDF 论文。

    :param paper_title: 论文标题（支持部分匹配）
    :param author: 作者姓名
    :param year: 发表年份
    :return: 论文的详细信息
    """
    if not paper_title and not author and not year:
        from agent.tool_models import SearchResult
        return SearchResult(
            quality_warning="❌ 请至少提供论文标题、作者或年份中的一个条件。",
            raw_formatted="[ERR] 请至少提供论文标题、作者或年份中的一个条件。",
        )

    # 构建精确查询
    parts = []
    if paper_title:
        parts.append(paper_title)
    if author:
        parts.append(f"作者: {author}")
    if year:
        parts.append(f"({year})")
    query = " ".join(parts)

    return _ctx.do_search(query, top_k=5)


# ================================================================
#  Tool 3: compare_papers — 多论文对比
# ================================================================

@tool
def compare_papers(topic: str, aspects: str = "方法、实验、性能"):
    """
    【本地知识库】针对某个主题进行多维度对比检索，适合比较不同论文的方法或结果。

    使用多类型混合检索（各类 chunk_type 各取 Top-3），再从所有候选中
    Reranker 精排合并。适用于：对比分析、综述等复杂任务。
    注意：此工具仅对比本地已上传的 PDF 论文。

    :param topic: 对比主题（如 "变化检测方法"）
    :param aspects: 关注的维度，逗号分隔（默认 "方法、实验、性能"）
    :return: 多维度检索后的格式化结果
    """
    from agent.tool_models import SearchResult

    if _ctx.vector_store is None:
        return SearchResult(
            quality_warning="❌ 检索失败: 向量数据库为空，请先导入文献。",
            raw_formatted="[ERR] 向量数据库为空，请先导入文献。",
        )

    from agent.retriever import multi_type_retrieve, build_search_result

    # 对每个维度分别检索后合并
    aspect_list = [a.strip() for a in aspects.replace("，", ",").split(",") if a.strip()]
    all_docs = []
    seen = set()

    for aspect in aspect_list[:4]:  # 最多4个维度
        query = f"{topic} {aspect}"
        docs = multi_type_retrieve(
            _ctx.vector_store, query,
            reranker=_ctx.reranker, k_per_type=3,
        )
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    if not all_docs:
        return SearchResult(
            raw_formatted=f"未找到与 \"{topic}\" 相关的对比文献。",
        )

    # 去重后限制数量
    all_docs = all_docs[:10]

    # 使用 build_search_result 构建结构化结果
    result = build_search_result(all_docs)

    # 覆盖 raw_formatted 为多维度对比的特定格式
    lines = [f"[多维度对比检索] 主题: {topic} | 维度: {aspects} | 找到 {len(all_docs)} 个相关块\n"]
    for i, doc in enumerate(all_docs, 1):
        fn = doc.metadata.get("filename", "?")
        sec = doc.metadata.get("section_name", "")
        src = f"{fn}" + (f" [{sec}]" if sec else "")
        content = doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content
        lines.append(f"--- 块 {i} ({src}) ---\n{content}\n")
    result.raw_formatted = "\n".join(lines)

    return result


# ================================================================
#  Tool 4: search_long_term_memory — 长期记忆检索
# ================================================================

@tool
def search_long_term_memory(query: str, top_k: int = 3):
    """
    检索长期记忆中与查询相关的事实和用户偏好。

    长期记忆存储了之前对话中提取的关键信息（用户偏好、重要结论等）。
    适用于：需要参考之前讨论内容、用户偏好时。

    :param query: 检索查询
    :param top_k: 返回的记忆条目数
    :return: 相关的长期记忆内容
    """
    return _ctx.do_memory_search(query, top_k=top_k)


# ================================================================
#  Tool 5: get_conversation_context — 对话上下文
# ================================================================

@tool
def get_conversation_context(include_summary: bool = True):
    """
    获取当前对话的历史上下文，包括近期对话、摘要和状态。

    适用于：Agent 需要了解之前讨论了什么、当前分析进度时。

    :param include_summary: 是否包含对话摘要
    :return: 对话上下文的文本表示
    """
    from agent.tool_models import MemoryResult

    if _ctx.memory is None:
        return MemoryResult(
            source="conversation",
            raw_formatted="[INFO] 当前无对话记忆。",
        )

    parts = []
    items: list[str] = []

    # 摘要
    if include_summary:
        summary = _ctx.memory.get_summary()
        if summary:
            parts.append(f"[对话摘要]\n{summary}")
            items.append(summary[:200])

    # 近期对话历史
    history = _ctx.memory.get_history_text()
    if history:
        parts.append(history)
        # 提取 Q&A 作为 items
        for line in history.split("\n"):
            if line.startswith("Q: ") or line.startswith("A: "):
                items.append(line[:200])

    # 状态
    from agent.memory import HybridMemory
    if isinstance(_ctx.memory, HybridMemory):
        state_text = _ctx.memory.get_state_text()
        if state_text:
            parts.append(f"[当前状态]\n{state_text}")

    raw = "\n\n".join(parts) if parts else "[INFO] 对话上下文为空（可能是首轮对话）。"

    return MemoryResult(
        items=items[:10],
        count=len(items),
        source="conversation",
        raw_formatted=raw,
    )


# ================================================================
#  Tool 6: rewrite_query — 查询优化
# ================================================================

@tool
def rewrite_query(query: str):
    """
    对用户查询进行指代消解和优化。将模糊指代（如"这篇论文"、"该方法"）
    替换为具体实体名称，并判断查询类型（fact/review/compare/general）。

    建议在检索前先调用此工具优化查询，以获得更好的检索结果。
    如果查询本身已经很清晰，也会返回类型分类信息。

    :param query: 原始用户查询
    :return: 重写后的查询及类型信息
    """
    return _ctx.do_rewrite(query)


# ================================================================
#  Tool 7: add_to_memory — 添加长期记忆
# ================================================================

@tool
def add_to_memory(content: str, keywords: str = ""):
    """
    将一条重要信息添加到长期记忆。使用场景：
    - 用户明确表达了偏好（如"我喜欢简短回答"）
    - 对话中得出的重要结论需要跨会话保留
    - 用户要求记住某个信息

    :param content: 要记住的内容（一句话）
    :param keywords: 关键词，逗号分隔（可选，不填则自动提取）
    :return: 结构化 MemoryResult（含 status 和 memory_id）
    """
    from agent.tool_models import MemoryResult

    if _ctx.ltm is None:
        return MemoryResult(
            source="ltm",
            raw_formatted="[ERR] 长期记忆不可用（需要 HybridMemory 模式）。",
        )

    kw_list = [k.strip() for k in keywords.replace("，", ",").split(",") if k.strip()] if keywords else None
    try:
        sid = _ctx.ltm.add(content, keywords=kw_list)
        keywords_used = kw_list or []
        return MemoryResult(
            items=[content],
            count=1,
            keywords=keywords_used,
            source="ltm",
            memory_id=str(sid),
            raw_formatted=f"[OK] 已添加到长期记忆 (id={sid})",
        )
    except Exception as e:
        return MemoryResult(
            source="ltm",
            raw_formatted=f"[ERR] 添加失败: {e}",
        )


# ================================================================
#  Tool 8: get_system_status — 系统状态
# ================================================================

@tool
def get_system_status():
    """
    查询向量数据库和模型状态：已索引文档块数量、嵌入/精排/生成模型信息、长期记忆条数。

    注意：此工具仅报告向量检索系统的内部状态，不涉及文件系统中的 PDF 文件。
    如需查看文件系统中的论文文件，请使用 list_directory 或 search_files。
    :return: 系统状态摘要
    """
    from config import LLM_MODEL, RERANK_MODEL, LOCAL_EMBEDDING_MODEL
    from agent.tool_models import SystemStatusResult

    lines = ["[系统状态]"]
    vector_count = 0

    # 向量库状态
    if _ctx.vector_store is not None:
        try:
            vector_count = _ctx.vector_store._collection.count()
            lines.append(f"  向量库文档块: {vector_count}")
        except Exception:
            lines.append("  向量库: 已加载（无法获取计数）")
    else:
        lines.append("  向量库: 为空（请先导入文献）")

    # 模型信息
    lines.append(f"  嵌入模型: {LOCAL_EMBEDDING_MODEL}")
    lines.append(f"  精排模型: {RERANK_MODEL}")
    lines.append(f"  生成模型: {LLM_MODEL}")

    # 记忆状态
    ltm_count = _ctx.ltm.count() if _ctx.ltm is not None else 0
    turn_count = _ctx.memory.turn_count() if _ctx.memory is not None else 0
    if _ctx.ltm is not None:
        lines.append(f"  长期记忆: {ltm_count} 条")
    if _ctx.memory is not None:
        lines.append(f"  对话轮数: {turn_count}")

    return SystemStatusResult(
        vector_count=vector_count,
        embedding_model=LOCAL_EMBEDDING_MODEL,
        rerank_model=RERANK_MODEL,
        llm_model=LLM_MODEL,
        ltm_count=ltm_count,
        conversation_turns=turn_count,
        raw_formatted="\n".join(lines),
    )


# ================================================================
#  _ensure_sync_compatible — MCP 工具同步包装
# ================================================================

def _ensure_sync_compatible(tool_obj):
    """
    确保工具支持同步 .invoke() 调用。

    背景：
      langchain_mcp_adapters 返回的 StructuredTool 可能只有 coroutine
      没有 func，导致 StructuredTool._run() 检查 if self.func 失败
      并抛出 "StructuredTool does not support sync invocation"。

    修复策略：
      直接在原实例上设置 func 属性（asyncio 桥接），
      不创建新 StructuredTool 实例，避免 Pydantic 验证问题。

    参数:
        tool_obj: 待检查的 LangChain Tool / StructuredTool

    返回:
        支持同步 invoke() 的工具（原实例，已 patch func）
    """
    # 情况 1: 已有同步 func → 直接返回
    func_val = getattr(tool_obj, "func", None)
    if func_val is not None:
        return tool_obj

    # 情况 2: 查找可用的异步函数
    async_func = None
    if hasattr(tool_obj, "coroutine"):
        async_func = getattr(tool_obj, "coroutine", None)
    if async_func is None:
        async_func = getattr(tool_obj, "_arun", None)
    if async_func is None:
        async_func = getattr(tool_obj, "arun", None)
        if async_func is not None:
            # 旧版 LangChain arun 方法 → 包装为协程调用
            _legacy_arun_ref = async_func

            async def _legacy_arun_wrapper(*args, **kwargs):
                return await _legacy_arun_ref(*args, **kwargs)
            async_func = _legacy_arun_wrapper

    if async_func is None:
        logger.warning(
            f"工具 '{getattr(tool_obj, 'name', '?')}' 既无 func 也无 coroutine，跳过"
        )
        return tool_obj

    # ---- 构建同步桥接函数 ----
    tool_name = getattr(tool_obj, "name", "unknown_tool")

    def _make_sync_bridge(_async_fn):
        """
        用 asyncio 桥接将 async 函数包装为同步函数。

        兼容三种运行时环境：
          a) 无 running loop → asyncio.run()（最常见）
          b) 有 running loop（Streamlit/asyncio 服务）→ 新建子 loop
          c) RuntimeError → 回退到新建 loop
        """
        def _sync_func(**kwargs):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 已有运行中的事件循环 — 不能直接用 asyncio.run()
                    # 新建独立 loop 执行
                    new_loop = asyncio.new_event_loop()
                    try:
                        return new_loop.run_until_complete(
                            _async_fn(**kwargs)
                        )
                    finally:
                        new_loop.close()
                else:
                    return loop.run_until_complete(_async_fn(**kwargs))
            except RuntimeError:
                return asyncio.run(_async_fn(**kwargs))
        return _sync_func

    # ---- 直接 patch 原实例的 func 属性 ----
    # 使用 object.__setattr__ 绕过 Pydantic BaseModel 的字段验证
    # （Pydantic v2 在某些模式下不允许直接属性赋值）
    try:
        sync_func = _make_sync_bridge(async_func)
        object.__setattr__(tool_obj, "func", sync_func)
        logger.debug(f"[TOOLS] 已 patch func on '{tool_name}' → 支持同步调用")
    except Exception as e:
        logger.warning(f"[TOOLS] patch func on '{tool_name}' 失败: {e}")

    return tool_obj


def _ensure_all_sync_compatible(tools: list) -> list:
    """对工具列表中的每个工具做 sync 兼容检查（就地 patch）。"""
    for i, t in enumerate(tools):
        try:
            tools[i] = _ensure_sync_compatible(t)
        except Exception as e:
            logger.warning(
                f"[TOOLS] 工具兼容检查失败: {getattr(t, 'name', '?')} — {e}"
            )
    return tools


# ================================================================
#  Tool 列表（供 Agent 初始化使用）
# ================================================================

# arXiv MCP 工具名称 — 用于区分在线检索 vs 文件管理工具
_ARXIV_TOOL_NAMES = {
    "search_papers", "get_paper_data", "get_full_paper_text",
    "list_categories", "update_categories",
}


def get_tools_for_scope(retrieval_scope: str = "local") -> list:
    """返回按检索范围过滤后的工具列表。

    参数:
        retrieval_scope: "local" | "online" | "hybrid"

    返回:
        过滤后的 LangChain Tool 列表

    - "local":  本地检索工具 + 上下文工具 + 文件系统 MCP（无 arXiv）
    - "online": 上下文工具（无 search_literature）+ arXiv MCP + 文件系统 MCP
    - "hybrid": 全部工具
    """
    # 上下文工具（所有 scope 都需要）
    always_tools = [
        search_long_term_memory,
        get_conversation_context,
        rewrite_query,
        add_to_memory,
        get_system_status,
    ]

    # 本地检索工具（仅 local / hybrid）
    if retrieval_scope in ("local", "hybrid"):
        search_tools = [search_literature, get_paper_detail, compare_papers]
    else:
        search_tools = []

    core = search_tools + always_tools

    # MCP 工具（按 scope 过滤）
    try:
        from mcphub import get_mcp_tools
        all_mcp = get_mcp_tools()
        all_mcp = _ensure_all_sync_compatible(all_mcp)

        # 分离 arXiv 工具和文件系统工具
        arxiv_tools = [t for t in all_mcp if t.name in _ARXIV_TOOL_NAMES]
        fs_tools = [t for t in all_mcp if t.name not in _ARXIV_TOOL_NAMES]

        if retrieval_scope == "local":
            core.extend(fs_tools)  # 仅文件系统
        elif retrieval_scope == "online":
            core.extend(arxiv_tools)  # arXiv 在线检索
            core.extend(fs_tools)     # 文件管理仍然有用
        elif retrieval_scope == "hybrid":
            core.extend(all_mcp)  # 全部
    except Exception:
        pass

    return core


# ── 意图 → 允许的工具名白名单（硬约束，非 soft prompt）──

# 文件管理意图：只允许文件系统操作工具
_FILE_MANAGEMENT_ALLOWED_TOOLS = {
    # 文件系统 MCP 工具
    "list_directory", "create_directory", "move_file",
    "search_files", "get_file_info", "organize_paper",
    "batch_classify", "list_paper_categories", "list_allowed_directories",
}

# 知识检索意图（本地）：排除文件系统工具，只保留检索 + 上下文
_KNOWLEDGE_RETRIEVAL_LOCAL_ALLOWED = {
    "search_literature", "get_paper_detail", "compare_papers",
    "search_long_term_memory", "get_conversation_context",
    "rewrite_query", "add_to_memory", "get_system_status",
}

# 知识检索意图（在线）：排除本地检索工具，只保留 arXiv + 上下文
_KNOWLEDGE_RETRIEVAL_ONLINE_ALLOWED = {
    "search_papers", "get_paper_data", "get_full_paper_text",
    "list_categories", "update_categories",
    "search_long_term_memory", "get_conversation_context",
    "rewrite_query", "add_to_memory", "get_system_status",
}

# 知识检索意图（hybrid）：本地检索 + arXiv + 上下文
_KNOWLEDGE_RETRIEVAL_HYBRID_ALLOWED = (
    _KNOWLEDGE_RETRIEVAL_LOCAL_ALLOWED | _KNOWLEDGE_RETRIEVAL_ONLINE_ALLOWED
)


def get_tools_for_intent_and_scope(
    intent_type: str = "knowledge_retrieval",
    retrieval_scope: str = "local",
    active_skill: str = "",
) -> list:
    """按意图类型 + 检索范围硬过滤工具列表（核心改进）。

    与 get_tools_for_scope() 的关键区别：
      - 这是一个**硬约束**，不只是 system prompt 中的建议
      - 不在白名单中的工具对 LLM 完全不可见，彻底杜绝"乱调用工具"
      - 文件管理意图下排除所有检索工具，检索意图下排除所有文件工具
      - active_skill 非空时，合并技能声明的 tool_names 到白名单

    参数:
        intent_type:     "knowledge_retrieval" | "file_management" | 其他
        retrieval_scope: "local" | "online" | "hybrid"
        active_skill:    Skills 系统激活的技能名称（空=无激活技能）

    返回:
        硬过滤后的 LangChain Tool 列表（仅包含该意图下允许的工具）
    """
    if intent_type == "file_management":
        allowed = _FILE_MANAGEMENT_ALLOWED_TOOLS
    elif intent_type == "knowledge_retrieval":
        if retrieval_scope == "online":
            allowed = _KNOWLEDGE_RETRIEVAL_ONLINE_ALLOWED
        elif retrieval_scope == "hybrid":
            allowed = _KNOWLEDGE_RETRIEVAL_HYBRID_ALLOWED
        else:
            allowed = _KNOWLEDGE_RETRIEVAL_LOCAL_ALLOWED
    else:
        # general_chat / out_of_domain / 未知 → 只给最小工具集
        allowed = {"get_system_status", "get_conversation_context"}

    # Skills: 合并激活技能的工具白名单
    if active_skill:
        try:
            from skills import skill_registry
            skill_tool_names = skill_registry.get_skill_tool_names(active_skill)
            if skill_tool_names:
                allowed = allowed | set(skill_tool_names)
            # 加载技能专用工具
            skill_tools = skill_registry.get_skill_tools(active_skill)
        except Exception:
            skill_tools = []
    else:
        skill_tools = []

    # 获取全部工具，然后按白名单过滤
    all_tools = get_tools_for_scope(retrieval_scope)
    filtered = [t for t in all_tools if t.name in allowed]

    # 合并技能专用工具
    if active_skill and skill_tools:
        # 防止技能工具名与核心工具重名
        existing_names = {t.name for t in filtered}
        for st in skill_tools:
            if hasattr(st, "name") and st.name not in existing_names:
                filtered.append(st)
            elif hasattr(st, "name"):
                import logging
                logging.getLogger(__name__).warning(
                    f"技能工具 '{st.name}' 与核心工具重名，跳过"
                )

    # 兜底：如果过滤后为空（MCP 未加载），至少返回核心工具
    if not filtered:
        core_by_name = {
            "search_literature": search_literature,
            "get_paper_detail": get_paper_detail,
            "compare_papers": compare_papers,
            "search_long_term_memory": search_long_term_memory,
            "get_conversation_context": get_conversation_context,
            "rewrite_query": rewrite_query,
            "add_to_memory": add_to_memory,
            "get_system_status": get_system_status,
        }
        filtered = [t for name, t in core_by_name.items() if name in allowed]

    return filtered


def get_all_tools() -> list:
    """返回所有可用工具的列表（含 MCP 工具），供 Agent 使用。

    现在委托给 get_tools_for_scope()，基于当前 _ctx.retrieval_scope 过滤。
    """
    return get_tools_for_scope(_ctx.retrieval_scope)


def get_core_tools() -> list:
    """返回核心工具（检索 + 上下文 + MCP），用于简单 Agent。

    现在委托给 get_tools_for_scope()，基于当前 _ctx.retrieval_scope 过滤。
    """
    return get_tools_for_scope(_ctx.retrieval_scope)
