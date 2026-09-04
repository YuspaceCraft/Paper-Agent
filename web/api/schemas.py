"""
schemas.py — Pydantic models for FastAPI request/response validation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from datetime import datetime


# ---- Task tracking ----

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending | running | done | failed
    progress: str = ""
    error: str | None = None
    result: dict | None = None
    created_at: str = ""
    updated_at: str = ""
    # ---- agent background-task extras (optional; empty for pdf/index tasks) ----
    kind: str = ""        # ingest | ""（上传解析/索引等非 agent 任务）
    paper_name: str = ""
    notify: bool = False  # 完成时是否需要前端通知 agent 再告知用户
    stage: str = ""       # ingest 任务当前阶段："parse" 解析中 | "index" 向量化入库中 | "" 其他


# ---- PDF Pipeline ----

class PDFProcessResult(BaseModel):
    task_id: str
    paper_name: str
    status: str


class PaperOutput(BaseModel):
    paper_name: str
    files: list[str]
    chunk_count: int = 0
    status: str = "not_indexed"  # "indexed"=in vector DB | "not_indexed"=一切未入库
    indexed: bool = False  # true ⟺ chunks are in Qdrant (dense KB)
    detail: str = ""       # 派生诊断（非持久状态）: "indexed"|"parsed"|"raw"|""


# ---- Indexer ----

class IndexRunRequest(BaseModel):
    rag_chunks_path: str
    config_path: str = ""


class IndexRunResult(BaseModel):
    task_id: str
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict | None = None


class SearchResult(BaseModel):
    chunk_id: str
    content_type: str = ""
    section_path: str = ""
    generation_text: str = ""
    score: float = 0.0
    metadata: dict = {}


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int


class IndexStats(BaseModel):
    backend: str
    collection_name: str
    count: int


# ---- Agent ----

class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    thread_id: str = Field(default="default", description="Conversation session ID")
    mode: Literal["auto", "react", "plan"] = Field(
        default="auto",
        description="执行模式覆盖：auto = decide_mode 启发式；react/plan = 客户端显式指定",
    )


class AgentChatResponse(BaseModel):
    answer: str
    intent: str = ""
    thread_id: str = "default"
    mode: str = ""
    error: str | None = None


class AgentHealthResponse(BaseModel):
    status: str
    model: str
    tools: int


# ---- Background tasks (agent-driven) ----

class AgentIngestRequest(BaseModel):
    paper_name: str = Field(..., min_length=1, description="Paper name / arxiv_id to index")
    pdf_path: str = Field("", description="Workspace path to the PDF (optional)")
    notify: bool = Field(True, description="是否在完成时通知 agent 再告知用户")


class AgentIngestResult(BaseModel):
    task_id: str
    paper_name: str
    status: str  # running


class AgentNotifyRequest(BaseModel):
    thread_id: str = Field(default="default", description="Conversation session ID")
    task: dict = Field(..., description="Background task status dict (TaskStatus shape)")
