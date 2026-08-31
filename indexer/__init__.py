"""
indexer — Knowledge Indexing & Storage Orchestration
=====================================================

基于 pdf_pipeline 输出的 rag_chunks.json，构建多粒度向量索引。
完全解耦上游 Pipeline，以 rag_chunks.json 为唯一输入契约。

四模块架构:
  1. ContextAssembler — Small-to-Big 上下文组装
  2. MultiGranularityEmbedder — Dense + BM25 双路向量化
  3. DedupManager — Content Hash 去重 + 增量同步
  4. VectorStoreAdapter — 向量库适配器 (Chroma / Qdrant)

Quick start:
    from indexer import IndexerPipeline

    pipeline = IndexerPipeline("indexer/config.yaml")
    stats = pipeline.run("pdf_pipeline/output/MV-CC/rag_chunks.json")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

# ================================================================
# Data Models
# ================================================================


@dataclass
class IndexUnit:
    """贯穿索引管道各阶段的标准数据单元。"""
    chunk_id: str
    content_hash: str = ""
    retrieval_text: str = ""            # 短文本，用于 Dense Embedding
    generation_text: str = ""           # 长文本，含邻居上下文，用于 LLM 生成
    dense_vector: list[float] | None = None
    sparse_keywords: list[str] = field(default_factory=list)
    hyde_questions: list[str] = field(default_factory=list)
    source_chunk: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    schema_version: str = "1.0"
    pii_flagged: bool = False
    pii_findings: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        """基于 retrieval_text + generation_text + chunk_id 计算内容指纹。"""
        payload = f"{self.chunk_id}|{self.retrieval_text}|{self.generation_text}"
        self.content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
        return self.content_hash


# ================================================================
# Re-exports
# ================================================================

from .config import (
    IndexerConfig, load_config,
    EmbeddingConfig, APIEmbeddingConfig, LocalEmbeddingConfig,
)
from .context_assembler import ContextAssembler
from .embedding_adapters import (
    EmbeddingAdapter, APIEmbeddingAdapter,
    SentenceTransformerAdapter, create_embedding_adapter,
)
from .embedder import MultiGranularityEmbedder
from .vector_store import VectorStoreAdapter, ChromaVectorStore, QdrantVectorStore
from .dedup_manager import DedupManager, compute_hash
from .pipeline import IndexerPipeline, export_eval_manifest, structured_log

__all__ = [
    # Models
    "IndexUnit",
    # Config
    "IndexerConfig",
    "load_config",
    "EmbeddingConfig",
    "APIEmbeddingConfig",
    "LocalEmbeddingConfig",
    # Embedding adapters
    "EmbeddingAdapter",
    "APIEmbeddingAdapter",
    "SentenceTransformerAdapter",
    "create_embedding_adapter",
    # Sub-modules
    "ContextAssembler",
    "MultiGranularityEmbedder",
    "VectorStoreAdapter",
    "ChromaVectorStore",
    "QdrantVectorStore",
    "DedupManager",
    "compute_hash",
    # Pipeline
    "IndexerPipeline",
    "export_eval_manifest",
    "structured_log",
]
