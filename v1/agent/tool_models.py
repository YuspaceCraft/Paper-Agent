"""
tool_models.py — 工具返回类型的 Pydantic 模型
========================
定义每个工具的返回结构，替代原有的 raw string 返回。
summarizer 通过 isinstance 分发直接从结构化字段提取摘要，
消除 str → regex 反向解析的 round-trip 损耗。

**结构化信息交换（Agent v2.1）：**
  - __str__() 返回紧凑 JSON，包含 type + 所有结构化字段 + summary
  - LLM 通过 create_agent 接收到的 ToolMessage 内容是结构化 JSON
  - 下游模块可通过 model_dump() 获取完整结构化字段
  - raw_formatted 保留用于人类可读场景和向后兼容

使用方式:
  from agent.tool_models import SearchResult, FileListResult, ...

  result = SearchResult(
      hit_count=5,
      sources=["ResNet.pdf", "DenseNet.pdf"],
      chunks=[...],
      raw_formatted="--- 文档块 1 ...",  # 人类可读摘要
  )
  # LLM 收到 str(result) → JSON（含 type/hit_count/sources/summary 等）
  # 代码中可用 result.model_dump() → 完整 dict
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


# ================================================================
#  SearchResult — 检索类工具输出
# ================================================================

class SearchResult(BaseModel):
    """search_literature / get_paper_detail / compare_papers 的输出。

    chunks 保留原始检索到的文档块 (content + metadata)，
    供反思阶段做忠实度验证时直接使用。
    raw_formatted 保留人类可读摘要。
    注意：__str__() 返回结构化 JSON（LLM 直接消费结构化数据）。
    """
    chunks: list[dict] = Field(default_factory=list)
    hit_count: int = 0
    sources: list[str] = Field(default_factory=list)   # 去重后的来源文件名
    paper_titles: list[str] = Field(default_factory=list)  # 检测到的论文标题
    avg_chunk_length: int = 0
    # 质量标记
    quality_warning: str = ""      # 空=正常
    truncation_count: int = 0      # 截断处数
    short_chunk_count: int = 0     # 内容过短的文档块数
    # 人类可读摘要（供日志和回退使用）
    raw_formatted: str = ""

    def __str__(self) -> str:
        """结构化 JSON 输出（Agent v2.1: LLM 直接消费结构化数据）。"""
        return _to_json({
            "type": "SearchResult",
            "hit_count": self.hit_count,
            "sources": self.sources[:5],
            "paper_titles": self.paper_titles[:5],
            "avg_chunk_length": self.avg_chunk_length,
            "quality_warning": self.quality_warning or None,
            "truncation_count": self.truncation_count,
            "short_chunk_count": self.short_chunk_count,
            "summary": self.raw_formatted[:2000],
        })


# ================================================================
#  FileListResult — 文件列表类工具输出
# ================================================================

class FileListResult(BaseModel):
    """list_directory / search_files 的输出。"""
    items: list[str] = Field(default_factory=list)       # 文件/目录名
    item_count: int = 0
    directories: list[str] = Field(default_factory=list)  # 仅目录
    files: list[str] = Field(default_factory=list)        # 仅文件
    raw_formatted: str = ""

    def __str__(self) -> str:
        """结构化 JSON 输出（Agent v2.1: LLM 直接消费结构化数据）。"""
        return _to_json({
            "type": "FileListResult",
            "item_count": self.item_count,
            "directories": self.directories[:20],
            "files": self.files[:20],
            "summary": self.raw_formatted[:2000],
        })


# ================================================================
#  FileOperationResult — 文件操作类工具输出
# ================================================================

class FileOperationResult(BaseModel):
    """create_directory / move_file / organize_paper 等 MCP 工具的输出。"""
    operation: str = ""      # create / move / delete / organize / classify
    target: str = ""         # 操作对象（路径/文件名/目录名）
    status: str = ""         # ok / error / exists / empty / skipped
    details: str = ""        # 人类可读的操作描述
    stats: dict = Field(default_factory=dict)  # 统计信息 {files_moved: 3, ...}
    raw_formatted: str = ""

    def __str__(self) -> str:
        """结构化 JSON 输出（Agent v2.1: LLM 直接消费结构化数据）。"""
        return _to_json({
            "type": "FileOperationResult",
            "operation": self.operation,
            "target": self.target,
            "status": self.status,
            "details": self.details[:300],
            "stats": self.stats,
            "summary": self.raw_formatted[:2000],
        })


# ================================================================
#  MemoryResult — 记忆检索类工具输出
# ================================================================

class MemoryResult(BaseModel):
    """search_long_term_memory / get_conversation_context / add_to_memory 的输出。"""
    items: list[str] = Field(default_factory=list)   # 记忆/对话条目的文本
    count: int = 0
    keywords: list[str] = Field(default_factory=list) # 关联关键词
    source: str = ""  # "ltm" | "conversation" | "hybrid"
    # add_to_memory 专用字段
    memory_id: str = ""   # 新增记忆的 ID
    raw_formatted: str = ""

    def __str__(self) -> str:
        """结构化 JSON 输出（Agent v2.1: LLM 直接消费结构化数据）。"""
        return _to_json({
            "type": "MemoryResult",
            "count": self.count,
            "source": self.source,
            "keywords": self.keywords[:10],
            "items": [item[:200] for item in self.items[:5]],
            "memory_id": self.memory_id or None,
            "summary": self.raw_formatted[:2000],
        })


# ================================================================
#  SystemStatusResult — 系统状态工具输出
# ================================================================

class SystemStatusResult(BaseModel):
    """get_system_status 的输出。"""
    vector_count: int = 0
    indexed_files: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    rerank_model: str = ""
    llm_model: str = ""
    ltm_count: int = 0
    conversation_turns: int = 0
    raw_formatted: str = ""

    def __str__(self) -> str:
        """结构化 JSON 输出（Agent v2.1: LLM 直接消费结构化数据）。"""
        return _to_json({
            "type": "SystemStatusResult",
            "vector_count": self.vector_count,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "ltm_count": self.ltm_count,
            "conversation_turns": self.conversation_turns,
            "summary": self.raw_formatted[:2000],
        })


# ================================================================
#  QueryRewriteResult — 查询重写工具输出
# ================================================================

class QueryRewriteResult(BaseModel):
    """rewrite_query 的输出。"""
    original: str = ""
    rewritten: str = ""
    query_type: str = "general"    # fact / review / compare / general
    needs_rewrite: bool = False
    explanation: str = ""
    raw_formatted: str = ""

    def __str__(self) -> str:
        """结构化 JSON 输出（Agent v2.1: LLM 直接消费结构化数据）。"""
        return _to_json({
            "type": "QueryRewriteResult",
            "original": self.original[:200],
            "rewritten": self.rewritten[:500],
            "query_type": self.query_type,
            "needs_rewrite": self.needs_rewrite,
            "explanation": self.explanation[:200],
            "summary": self.raw_formatted[:2000],
        })


# ================================================================
#  辅助函数
# ================================================================

def _to_json(data: dict) -> str:
    """将 dict 序列化为紧凑 JSON 字符串（无换行，节省 token）。"""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
