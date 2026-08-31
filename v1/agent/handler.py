"""
handler.py — Web 层与 Agent 层之间的薄接口
=========
职责：
  1. 注入 Streamlit 缓存的资源 (embeddings, reranker, vector_store)
  2. 调用 agent_query()，返回结构化结果
  3. PDF 上传处理

Web 层只需调用 handle_query() / handle_upload()，不感知内部逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from config import DATA_DIR
from agent.agent import agent_query, LoopResult
from agent.loader import load_documents
from agent.splitter import split_documents


# ================================================================
#  handle_query — Web 查询入口
# ================================================================

def handle_query(
    question: str,
    memory,
    *,
    agent_type: str = "react",
    retrieval_scope: str | None = None,
    embeddings=None,
    reranker=None,
    vector_store=None,
    progress_callback: Callable[[str, str], None] | None = None,
    resume_plan: dict | None = None,
) -> LoopResult:
    """
    Web 层查询入口：注入资源 → 调用 Agent → 返回结构化结果。

    参数:
        question:          用户问题
        memory:            对话记忆对象
        agent_type:        Agent 类型
        retrieval_scope:   检索范围（None = 由 Agent 根据意图自动判定）
        embeddings:        Streamlit 缓存的嵌入模型（可选）
        reranker:          Streamlit 缓存的重排模型（可选）
        vector_store:      Streamlit 缓存的向量库（可选）
        progress_callback: UI 进度回调
        resume_plan:       澄清后恢复时传入的序列化 ConfirmedPlan，跳过 PLANNING

    返回:
        LoopResult，web 层根据 .status 做不同渲染
    """
    from agent.tools import set_agent_resources

    # 注入外部缓存的资源
    if embeddings or reranker or vector_store:
        set_agent_resources(
            embeddings=embeddings,
            reranker=reranker,
            vector_store=vector_store,
        )

    return agent_query(
        question=question,
        memory=memory,
        agent_type=agent_type,
        retrieval_scope=retrieval_scope,
        progress_callback=progress_callback,
        resume_plan=resume_plan,
    )


# ================================================================
#  handle_upload — PDF 上传处理
# ================================================================

@dataclass
class UploadResult:
    """上传处理的结构化结果。"""
    success: bool = False
    filename: str = ""
    message: str = ""
    chunk_count: int = 0
    doc_count: int = 0


def handle_upload(
    file_bytes: bytes,
    filename: str,
    *,
    embeddings,
    vector_store=None,
    store_version: int = 0,
) -> UploadResult:
    """
    处理 PDF 文件上传：保存 → 解析 → 切分 → 嵌入 → 追加到向量库。

    参数:
        file_bytes:   文件字节内容
        filename:     文件名
        embeddings:   嵌入模型实例
        vector_store: 当前向量库（None 则新建）
        store_version: 向量库版本号（用于缓存失效）

    返回:
        UploadResult
    """
    from agent.store import create_vector_store

    # 1. 保存文件
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / filename
    dest.write_bytes(file_bytes)

    # 2. 加载文档
    docs = load_documents(str(dest))
    if not docs:
        return UploadResult(
            success=False,
            filename=filename,
            message=f"PDF 解析失败或内容为空: {filename}",
        )

    # 3. 切分文档
    chunks = split_documents(docs)
    if not chunks:
        return UploadResult(
            success=False,
            filename=filename,
            message="文档切分后无内容",
        )

    # 4. 嵌入 + 写入向量库
    if vector_store is None:
        vector_store = create_vector_store(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    return UploadResult(
        success=True,
        filename=filename,
        message=f"已成功将 {len(chunks)} 个文档块写入向量库",
        chunk_count=len(chunks),
        doc_count=len(docs),
    )


__all__ = [
    "handle_query",
    "handle_upload",
    "UploadResult",
    "LoopResult",
]
