"""
embedder.py — 嵌入模型(Qwen3-Embedding-0.6B + DashScope API 双模式)
===========
将文本转换为向量。默认使用本地 Qwen3-Embedding-0.6B 模型。

双模式设计：
  EMBEDDING_PROVIDER = "local"     → 本地 Qwen3-Embedding-0.6B(免费、离线)
  EMBEDDING_PROVIDER = "dashscope" → 阿里云 DashScope API(需要联网 + Key)

Qwen3-Embedding-0.6B 特性：
  - 0.6B 参数（~1.2GB 显存/内存）
  - 支持 instruction-aware 嵌入（不同任务用不同指令模板）
  - 同时可作为 reranker 使用（计算 query-doc 相似度）
  - sentence-transformers 兼容

GPU 加速：
  设置 EMBEDDING_DEVICE=cuda 使用 NVIDIA GPU(需 CUDA + PyTorch GPU 版)
  设置 EMBEDDING_DEVICE=auto 自动检测最佳设备
  FP16 半精度在 CUDA 上自动开启，速度翻倍，精度损失可忽略

切换模型后需要重建向量数据库：
  python -m src.cli ingest --rebuild
"""

import os
import torch
from langchain_core.embeddings import Embeddings
from config import (
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    EMBEDDING_MODEL,
    DASHSCOPE_API_KEY,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
)

# ================================================================
#  GPU 设备检测
# ================================================================

def _detect_device(preferred: str) -> str:
    """根据用户偏好和实际环境选择最佳设备。"""
    preferred = preferred.strip().lower()

    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  [DEVICE] CUDA 可用: {name}")
            return "cuda"
        print("  [DEVICE] [WARN] 设置了 CUDA 但未检测到 GPU，回退到 CPU")
        return "cpu"

    # auto: CUDA > MPS > CPU
    if preferred in ("auto", ""):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  [DEVICE] 自动选择 CUDA: {name}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  [DEVICE] 自动选择 MPS (Apple Silicon)")
            return "mps"
        print("  [DEVICE] 自动选择 CPU")
        return "cpu"

    print(f"  [DEVICE] [WARN] 未知设备 '{preferred}'，回退到 CPU")
    return "cpu"


# ================================================================
#  Qwen3-Embedding 的 instruction 模板
# ================================================================
# Qwen3-Embedding 系列模型在训练时使用了任务指令。
# 查询时添加对应指令可以显著提升检索质量。
# 文档索引时不需要指令（模型直接用原文）。
# ================================================================

QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve relevant documents that answer the question.\n"
    "Query: "
)

QWEN3_RERANK_INSTRUCTION = (
    "Instruct: Given a query, judge whether the document is relevant.\n"
    "Query: "
)


# ================================================================
#  本地嵌入模型（Qwen3-Embedding-0.6B Instruct）
# ================================================================

class LocalEmbeddings_Instruct(Embeddings):
    """
    本地嵌入模型，基于 sentence-transformers。

    Qwen3-Embedding 会为查询自动添加 instruction 前缀，
    文档索引保持原文。这是 instruction-tuned 模型的标准用法。

    GPU 加速：
      - CUDA: 自动使用 FP16 半精度，速度翻倍
      - MPS:  Apple Silicon GPU 加速
      - batch_size 越大吞吐越高，但更吃显存
    """

    def __init__(
        self,
        model_name: str,
        query_instruction: str = QWEN3_QUERY_INSTRUCTION,
        device: str | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.device = device or _detect_device(EMBEDDING_DEVICE)
        self.batch_size = batch_size
        self._model = None  # 延迟加载

    def _ensure_model(self):
        """延迟加载模型，根据设备启用对应优化。"""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"  [EMBED] [LOAD] 正在加载本地模型: {self.model_name}")
        print(f"  [EMBED] [LOAD] 目标设备: {self.device}, batch_size: {self.batch_size}")

        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
            device=self.device,
        )

        # CUDA 下半精度加速（FP16），显存减半 + 速度翻倍
        if self.device == "cuda":
            self._model.half()
            print("  [EMBED] [INFO] 已启用 FP16 半精度加速")

        # 预热
        self._model.encode("预热", show_progress_bar=False)
        # 获取向量维度
        try:
            dims = self._model.get_embedding_dimension()
        except AttributeError:
            dims = self._model.get_sentence_embedding_dimension()
        print(f"  [EMBED] [OK] 本地模型就绪 ({dims} 维, {self.device})")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入文档块（索引阶段）。

        文档侧不加 instruction 前缀——Qwen3-Embedding
        训练时文档侧就是不添加前缀的原文。
        """
        if not texts:
            return []
        self._ensure_model()

        texts = [self._preprocess(t) for t in texts]
        print(f"  [EMBED] 本地嵌入 {len(texts)} 条文档...")
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入查询文本（查询阶段）。

        查询侧添加 instruction 前缀——这会让模型进入"检索模式"，
        产生的向量与文档向量在同一个空间但更聚焦于"寻找答案"。
        """
        self._ensure_model()
        text = self._preprocess(text)
        augmented = self.query_instruction + text
        embedding = self._model.encode(
            [augmented],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return embedding[0].tolist()

    def _preprocess(self, text: str) -> str:
        """轻量清洗。"""
        import re
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

# ================================================================
#  兼容所有 sentence-transformers 模型
# ================================================================
class LocalEmbeddings(Embeddings):
    """
    本地嵌入模型，基于 sentence-transformers。

    Qwen3-Embedding 会为查询自动添加 instruction 前缀，
    文档索引保持原文。这是 instruction-tuned 模型的标准用法。

    GPU 加速：
      - CUDA: 自动使用 FP16 半精度，速度翻倍
      - MPS:  Apple Silicon GPU 加速
      - batch_size 越大吞吐越高，但更吃显存
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device or _detect_device(EMBEDDING_DEVICE)
        self.batch_size = batch_size
        self._model = None  # 延迟加载

    def _ensure_model(self):
        """延迟加载模型，根据设备启用对应优化。"""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"  [EMBED] [LOAD] 正在加载本地模型: {self.model_name}")
        print(f"  [EMBED] [LOAD] 目标设备: {self.device}, batch_size: {self.batch_size}")

        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
            device=self.device,
        )

        # CUDA 下半精度加速（FP16），显存减半 + 速度翻倍
        if self.device == "cuda":
            self._model.half()
            print("  [EMBED] [INFO] 已启用 FP16 半精度加速")

        # 预热
        self._model.encode("预热", show_progress_bar=False)
        # 获取向量维度
        try:
            dims = self._model.get_embedding_dimension()
        except AttributeError:
            dims = self._model.get_sentence_embedding_dimension()
        print(f"  [EMBED] [OK] 本地模型就绪 ({dims} 维, {self.device})")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入文档块（索引阶段）。

        文档侧不加 instruction 前缀——Qwen3-Embedding
        训练时文档侧就是不添加前缀的原文。
        """
        if not texts:
            return []
        self._ensure_model()

        texts = [self._preprocess(t) for t in texts]
        print(f"  [EMBED] 本地嵌入 {len(texts)} 条文档...")
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入查询文本（查询阶段）。

        查询侧添加 instruction 前缀——这会让模型进入"检索模式"，
        产生的向量与文档向量在同一个空间但更聚焦于"寻找答案"。
        """
        self._ensure_model()
        text = self._preprocess(text)
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return embedding[0].tolist()

    def _preprocess(self, text: str) -> str:
        """轻量清洗。"""
        import re
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


# ================================================================
#  DashScope API 嵌入模型（保留）
# ================================================================

class DashScopeEmbeddings(Embeddings):
    """DashScope API 嵌入模型（需要联网 + API Key）。"""

    MAX_BATCH_SIZE = 25

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.url = (
            "https://dashscope.aliyuncs.com/api/v1/services/"
            "embeddings/text-embedding/text-embedding"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import requests as _requests

        all_embeddings = []
        total = len(texts)
        for batch_start in range(0, total, self.MAX_BATCH_SIZE):
            batch = texts[batch_start:batch_start + self.MAX_BATCH_SIZE]
            batch = [self._preprocess(t) for t in batch]

            batch_num = batch_start // self.MAX_BATCH_SIZE + 1
            total_batches = (total + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE
            print(f"  [EMBED] API 批次 {batch_num}/{total_batches}: "
                  f"{len(batch)} 条...")

            resp = _requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": {"texts": batch},
                    "parameters": {"text_type": "document"},
                },
                timeout=120,
            )
            result = resp.json()
            if "output" not in result:
                raise RuntimeError(
                    f"[EMBED] API 错误: {result.get('message', result)}"
                )
            items = result["output"]["embeddings"]
            items.sort(key=lambda x: x.get("text_index", 0))
            all_embeddings.extend([item["embedding"] for item in items])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _preprocess(self, text: str) -> str:
        import re
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


# ================================================================
#  工厂函数
# ================================================================

def get_embeddings() -> Embeddings:
    """创建嵌入模型，根据 EMBEDDING_PROVIDER 选择提供方。"""
    if EMBEDDING_PROVIDER == "local":
        device = _detect_device(EMBEDDING_DEVICE)
        print(f"[EMBED] [SETUP] 使用本地嵌入模型: {LOCAL_EMBEDDING_MODEL} ({device})")
        return LocalEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            device=device,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

    elif EMBEDDING_PROVIDER == "dashscope":
        print(f"[EMBED] [SETUP] 使用 DashScope 嵌入模型: {EMBEDDING_MODEL}")
        return DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=DASHSCOPE_API_KEY,
        )

    else:
        raise ValueError(
            f"[EMBED] [ERR] 未知的嵌入提供方: {EMBEDDING_PROVIDER}\n"
            f"支持的值: local, dashscope"
        )
