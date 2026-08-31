"""
retriever.py — 检索器（向量检索 + Reranker 二阶段精排）
===========
检索管道：粗筛（向量相似度） → 精排（Reranker 重打分） → 格式化

二阶段检索流程：
  用户问题
    │
    ▼
  Stage 1: 向量检索 (fast)
    Chroma ANN 搜索 → 取 candidate_k 个候选（默认 20）
    │
    ▼
  Stage 2: Reranker 精排 (accurate)
    自动适配两种架构：
      - Cross-encoder (bge-reranker-base): (query, doc) pair → 相关性分数
      - Bi-encoder  (Qwen3-Embedding):   instruction 引导 + 余弦相似度
    → 重排序 → 取 final_k 个（默认 4）
    │
    ▼
  格式化输出 → 喂给 LLM

为什么需要 Reranker？
  向量检索用独立编码做余弦相似度，会把"词像但意不同"的文档排前面。
  Cross-encoder 让 query 和 doc 在模型内部交互（attention across tokens），
  精度远高于 bi-encoder 余弦相似度。
"""

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from config import (
    TOP_K,
    RERANK_CANDIDATE_K,
    RERANK_FINAL_K,
    RERANK_MODEL,
    HF_ENDPOINT,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
)


# ================================================================
#  Reranker — 二阶段精排
# ================================================================

def _detect_device():
    """自动检测最佳设备。"""
    import torch
    if EMBEDDING_DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    if EMBEDDING_DEVICE in ("auto", ""):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _is_cross_encoder_model(model_name: str) -> bool:
    """判断模型是否为 cross-encoder 架构。

    Cross-encoder 模型（如 bge-reranker-base）需要拼接 (query, doc)
    对输入，直接输出相关性分数。它们不能用 bi-encoder 方式（分别编码
    再算余弦相似度）来使用。

    识别规则：
      - 模型名含 "reranker" 的通常是 cross-encoder
      - Qwen3-Embedding 系列是 bi-encoder（即使被当作 reranker 用）
    """
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        # Qwen3-Embedding 是 bi-encoder，通过 instruction 模板切换角色
        return False
    if "reranker" in name_lower:
        return True
    # sentence-transformers 的 cross-encoder 通常以 "cross-encoder" 开头
    if "cross-encoder" in name_lower or "cross_encoder" in name_lower:
        return True
    return False


class Reranker:
    """
    二阶段精排器，自动适配 bi-encoder 和 cross-encoder 两种架构。

    Bi-encoder 模式 (如 Qwen3-Embedding-0.6B):
      1. query 添加 instruction 前缀后单独编码
      2. doc 用原文单独编码
      3. 计算余弦相似度 → 作为相关性分数
      优势：可以和 embedding 阶段共享模型，省显存

    Cross-encoder 模式 (如 BAAI/bge-reranker-base):
      1. 将 (query, doc) 拼接后输入模型
      2. 模型直接输出相关性分数（logit）
      优势：query 和 doc 在模型内部交互，精度远高于 bi-encoder 余弦相似度
    """

    # Bi-encoder 模式的 instruction 模板（仅 Qwen3-Embedding 等使用）
    RERANK_QUERY_TEMPLATE = (
        "Instruct: Given a query, identify the most relevant documents.\n"
        "Query: {query}"
    )

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None
        self._device = None
        self._is_cross_encoder = _is_cross_encoder_model(model_name)

    def _ensure_model(self):
        """延迟加载模型，自动选择正确的架构。"""
        if self._model is not None:
            return

        self._device = _detect_device()

        if self._is_cross_encoder:
            self._load_cross_encoder()
        else:
            self._load_bi_encoder()

    def _load_cross_encoder(self):
        """加载 cross-encoder 模型。"""
        from sentence_transformers import CrossEncoder
        print(f"  [RERANK] [LOAD] 正在加载 Cross-Encoder 模型: {self.model_name} ({self._device})...")
        self._model = CrossEncoder(
            self.model_name,
            device=self._device,
            max_length=512,
        )
        # 验证模型加载成功
        test_score = self._model.predict([("test query", "test document")], show_progress_bar=False)
        print(f"  [RERANK] [OK] Cross-Encoder 就绪 ({self._device}), "
              f"max_length=512, 预热分数={test_score[0]:.4f}")

    def _load_bi_encoder(self):
        """加载 bi-encoder 模型（如 Qwen3-Embedding 复用为 reranker）。"""
        from sentence_transformers import SentenceTransformer
        print(f"  [RERANK] [LOAD] 正在加载 Bi-Encoder 模型: {self.model_name} ({self._device})...")
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
            device=self._device,
        )
        if self._device == "cuda":
            self._model.half()
            print("  [RERANK] [INFO] 已启用 FP16 半精度加速")
        self._model.encode("预热", show_progress_bar=False)
        print(f"  [RERANK] [OK] Bi-Encoder 就绪 ({self._device})")

    @property
    def is_cross_encoder(self) -> bool:
        return self._is_cross_encoder

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = RERANK_FINAL_K,
    ) -> list[Document]:
        """
        对候选文档重新打分排序。

        参数:
            query:     用户查询
            documents: Stage 1 返回的候选文档列表
            top_k:     最终返回的数量

        返回:
            list[Document]: 按相关性重排后的 top_k 文档
        """
        if not documents:
            return []
        if len(documents) <= top_k:
            return documents

        self._ensure_model()

        if self._is_cross_encoder:
            return self._rerank_cross_encoder(query, documents, top_k)
        else:
            return self._rerank_bi_encoder(query, documents, top_k)

    def _rerank_cross_encoder(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """Cross-encoder 重排：将 (query, doc) 对输入模型，直接输出分数。"""
        doc_texts = [doc.page_content for doc in documents]

        doc_lens = [len(t) for t in doc_texts]
        print(f"  [RERANK] 正在 Cross-Encoder 重排 {len(documents)} 个候选...")
        print(f"  [RERANK] [INFO] 文档长度: min={min(doc_lens)}, max={max(doc_lens)}, "
              f"avg={sum(doc_lens)//len(doc_lens)} 字符")

        # CrossEncoder.predict() 接受 [(query, doc), ...] 格式
        pairs = [(query, text) for text in doc_texts]
        scores = self._model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        # scores 形状应为 (N,) — 每对 (query, doc) 一个分数
        scores = np.asarray(scores)
        if scores.ndim > 1:
            print(f"  [RERANK] [WARN] predict() 返回形状 {scores.shape}，期望 (N,)，"
                  f"取最后一列作为相关性分数")
            scores = scores[:, -1]  # 取 relevant 列（二分类时 label=1）
        scores = scores.flatten()

        # 诊断：分数分布
        s_min, s_max = float(scores.min()), float(scores.max())
        s_mean, s_std = float(scores.mean()), float(scores.std())
        s_spread = s_max - s_min
        print(f"  [RERANK] [DIAG] 分数范围: {s_min:.4f} ~ {s_max:.4f} "
              f"(spread={s_spread:.4f}, mean={s_mean:.4f}, std={s_std:.4f})")
        if s_spread < 0.05:
            print(f"  [RERANK] [WARN] 分数过于集中 (spread={s_spread:.4f})，"
                  f"cross-encoder 难以区分候选文档！")

        # 按分数降序排列
        scored = list(zip(scores.tolist(), documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 诊断：reranker 改变了多少排名
        old_top_k_filenames = {d.metadata.get("filename", "?") for d in documents[:top_k]}
        new_top_k_filenames = {d.metadata.get("filename", "?") for _, d in scored[:top_k]}
        overlap = old_top_k_filenames & new_top_k_filenames
        print(f"  [RERANK] [DIAG] 重排前后 Top-{top_k} 重叠: {len(overlap)}/{top_k} "
              f"(新增: {new_top_k_filenames - old_top_k_filenames}, "
              f"移除: {old_top_k_filenames - new_top_k_filenames})")

        top_docs = [doc for _, doc in scored[:top_k]]
        return top_docs

    def _rerank_bi_encoder(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """Bi-encoder 重排：分别编码后计算余弦相似度。"""
        doc_texts = [doc.page_content for doc in documents]

        # query 加 instruction 前缀（引导模型进入"检索"模式）
        augmented_query = self.RERANK_QUERY_TEMPLATE.format(query=query)

        print(f"  [RERANK] 正在 Bi-Encoder 重排 {len(documents)} 个候选...")

        # 编码 query（带 instruction）和 docs（不带）
        query_embedding = self._model.encode(
            [augmented_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        doc_embeddings = self._model.encode(
            doc_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 计算余弦相似度
        scores = self._model.similarity(query_embedding, doc_embeddings)[0]

        # 按分数降序排列
        scored = list(zip(scores.tolist(), documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_docs = [doc for _, doc in scored[:top_k]]

        top_score = scored[0][0] if scored else 0
        bottom_score = scored[-1][0] if scored else 0
        print(f"  [RERANK] [OK] Bi-Encoder 分数范围: {bottom_score:.4f} ~ {top_score:.4f}, 取 Top-{top_k}")

        return top_docs


# ================================================================
#  检索器创建
# ================================================================

def create_retriever(
    vector_store: Chroma,
    k: int = TOP_K,
    filter_dict: dict | None = None,
) -> BaseRetriever:
    """
    创建基础检索器（单阶段：向量检索 → 直接返回）。

    用于不需要精排的场景，或作为 Stage 1 的粗筛器。
    """
    search_kwargs: dict = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return retriever


def create_retriever_with_rerank(
    vector_store: Chroma,
    reranker: Reranker | None = None,
    candidate_k: int = RERANK_CANDIDATE_K,
    final_k: int = RERANK_FINAL_K,
    filter_dict: dict | None = None,
) -> BaseRetriever:
    """
    创建带二阶段精排的检索器（推荐用于生产环境）。

    Stage 1: Chroma 向量检索 → candidate_k 个候选
    Stage 2: Reranker 精排 → 取 final_k 个

    参数:
        vector_store: Chroma 向量数据库
        reranker:     Reranker 实例（None 则自动创建）
        candidate_k:  Stage 1 粗筛数量（建议 15~30）
        final_k:      Stage 2 精排后最终数量
        filter_dict:  可选的 metadata 过滤
    """
    if reranker is None:
        reranker = Reranker(model_name=RERANK_MODEL)

    coarse = create_retriever(vector_store, k=candidate_k, filter_dict=filter_dict)

    print(f"[RETRIEVE] [SETUP] 二阶段检索器: "
          f"粗筛 Top-{candidate_k} → 精排 Top-{final_k}")

    return _TwoStageRetriever(coarse, reranker, final_k)


class _TwoStageRetriever(BaseRetriever):
    """二阶段检索器：粗筛 → 精排。"""

    def __init__(self, coarse_retriever, reranker, final_k):
        super().__init__()
        self._coarse = coarse_retriever
        self._reranker = reranker
        self._final_k = final_k

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Stage 1
        candidates = self._coarse.invoke(query)
        print(f"[RETRIEVE] Stage 1: 向量粗筛 → {len(candidates)} 个候选")

        # Stage 2
        top_docs = self._reranker.rerank(query, candidates, top_k=self._final_k)
        return top_docs


# ================================================================
#  内容格式化
# ================================================================

def format_retrieved_docs(docs: list[Document]) -> str:
    """将检索结果格式化为 LLM 上下文字符串（保留向后兼容）。"""
    if not docs:
        return "（未找到相关文档）"

    parts = []
    for i, doc in enumerate(docs, 1):
        chunk_type = doc.metadata.get("chunk_type", "")
        filename = doc.metadata.get("filename", "未知文件")
        source_desc = _format_source(doc, chunk_type, filename)
        parts.append(
            f"--- 文档块 {i} {source_desc} ---\n"
            f"{doc.page_content}"
        )

    result = "\n\n".join(parts)
    print(f"[RETRIEVE] [INFO] 已格式化 {len(docs)} 个文档块")
    return result


def build_search_result(docs: list[Document]) -> "SearchResult":
    """从检索到的 Document 列表构建结构化 SearchResult。

    同时填充 raw_formatted 为旧版字符串格式，确保向后兼容。
    """
    from agent.tool_models import SearchResult

    if not docs:
        return SearchResult(raw_formatted="（未找到相关文档）")

    # 提取结构化字段
    sources: list[str] = []
    titles: list[str] = []
    chunks: list[dict] = []
    total_content_len = 0
    short_chunk_count = 0
    seen_sources: set[str] = set()

    for doc in docs:
        content = doc.page_content
        fn = doc.metadata.get("filename", "未知文件")
        section = doc.metadata.get("section_name", "")
        paper_title = doc.metadata.get("paper_title", "")
        paper_year = doc.metadata.get("paper_year", "")
        chunk_type = doc.metadata.get("chunk_type", "")

        # 来源去重
        if fn not in seen_sources:
            sources.append(fn)
            seen_sources.add(fn)

        # 论文标题去重
        if paper_title:
            title_with_year = paper_title[:80] + (f" ({paper_year})" if paper_year else "")
            if title_with_year not in titles:
                titles.append(title_with_year)

        # 文档块
        content_len = len(content.strip())
        total_content_len += content_len
        if content_len < 80:
            short_chunk_count += 1

        chunks.append({
            "content": content,
            "filename": fn,
            "section": section,
            "paper_title": paper_title,
            "paper_year": paper_year,
            "chunk_type": chunk_type,
        })

    avg_len = total_content_len // len(docs) if docs else 0

    # 质量检测
    quality_warning = ""
    if short_chunk_count >= max(len(docs) * 0.5, 1) and len(docs) >= 2:
        quality_warning = (
            f"⚠️ 检索质量警告: {short_chunk_count}/{len(docs)} 个文档块内容不完整"
            f"（平均 {avg_len} 字符/块），文献索引可能异常"
        )

    # raw_formatted: 等同于旧版 format_retrieved_docs() 输出
    raw_formatted = format_retrieved_docs(docs)

    return SearchResult(
        chunks=chunks,
        hit_count=len(docs),
        sources=sources,
        paper_titles=titles,
        avg_chunk_length=avg_len,
        quality_warning=quality_warning,
        short_chunk_count=short_chunk_count,
        raw_formatted=raw_formatted,
    )


def _format_source(doc: Document, chunk_type: str, filename: str) -> str:
    """根据 chunk_type 生成来源描述。"""
    if chunk_type == "code":
        node_type = doc.metadata.get("node_type", "")
        node_name = doc.metadata.get("node_name", "")
        lines = doc.metadata.get("line_range", "")
        return f"[{node_type}: {node_name} | {filename}:{lines}]"
    elif chunk_type == "table_description":
        columns = doc.metadata.get("columns", [])
        total_rows = doc.metadata.get("total_rows", 0)
        return f"[表格 {filename} | {len(columns)} 列, {total_rows} 行]"
    elif chunk_type == "pdf":
        pages = doc.metadata.get("pages", "?")
        return f"[PDF {filename} | 页码: {pages}]"
    elif chunk_type == "paper":
        section = doc.metadata.get("section_name", "")
        paper_title = doc.metadata.get("paper_title", "")
        paper_year = doc.metadata.get("paper_year", "")
        pages = doc.metadata.get("pages", "")
        desc = f"[{section}"
        if paper_title:
            desc += f" | {paper_title[:60]}"
        if paper_year:
            desc += f" ({paper_year})"
        if pages:
            desc += f" | p.{pages}"
        desc += f" | {filename}]"
        return desc
    else:
        return f"[{filename}]"


# ================================================================
#  多类型混合检索（带 Reranker）
# ================================================================

def multi_type_retrieve(
    vector_store: Chroma,
    query: str,
    reranker: Reranker | None = None,
    k_per_type: int = 3,
    types: list[str] | None = None,
) -> list[Document]:
    """多类型混合检索：每种类型各取 top-k，Reranker 精排后合并。"""
    if types is None:
        retriever = create_retriever_with_rerank(
            vector_store, reranker=reranker,
            candidate_k=RERANK_CANDIDATE_K, final_k=k_per_type * 3,
        )
        return retriever.invoke(query)

    all_docs = []
    seen = set()
    for chunk_type in types:
        retriever = create_retriever_with_rerank(
            vector_store, reranker=reranker,
            candidate_k=RERANK_CANDIDATE_K, final_k=k_per_type,
            filter_dict={"chunk_type": chunk_type},
        )
        docs = retriever.invoke(query)
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
        print(f"  [RETRIEVE] {chunk_type}: {len(docs)} 个结果")

    print(f"[RETRIEVE] [OK] 混合检索完成: {len(types)} 种类型, 共 {len(all_docs)} 块")
    return all_docs
