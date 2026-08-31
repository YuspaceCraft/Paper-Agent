"""
embedding_adapters.py — Embedding 后端适配器
=============================================

抽象接口 + API (DashScope/OpenAI) + Local (sentence-transformers) 实现。
切换后端仅需修改 config.yaml 的 embedding.backend 字段。

用法:
    from indexer.embedding_adapters import create_embedding_adapter

    adapter = create_embedding_adapter(config.embedding)
    vectors = adapter.embed_batch(["text1", "text2"])
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .config import EmbeddingConfig, APIEmbeddingConfig, LocalEmbeddingConfig


# ================================================================
# Abstract Interface
# ================================================================

class EmbeddingAdapter(ABC):
    """Embedding 后端抽象接口。"""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """批量向量化。返回与输入等长的向量列表，失败项为 None。"""
        ...

    def embed_single(self, text: str) -> list[float] | None:
        """单条向量化（默认回退到 batch(size=1)）。"""
        result = self.embed_batch([text])
        return result[0] if result else None


# ================================================================
# API Embedding (DashScope / OpenAI-compatible)
# ================================================================

class APIEmbeddingAdapter(EmbeddingAdapter):
    """兼容 OpenAI Embedding API 的云端后端。

    支持 DashScope、OpenAI、vLLM 等兼容服务。
    """

    def __init__(self, config: APIEmbeddingConfig):
        self.config = config
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            try:
                from dotenv import load_dotenv
                _env_path = Path(__file__).resolve().parent.parent / ".env"
                if _env_path.exists():
                    load_dotenv(_env_path)
                    api_key = os.getenv(self.config.api_key_env)
            except ImportError:
                pass
        if not api_key:
            raise RuntimeError(
                f"{self.config.api_key_env} 未设置。"
                f"请在环境变量或项目根目录 .env 中配置。"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.api_base,
        )

    # ---- embed ----

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []

        self._ensure_client()
        vectors: list[list[float] | None] = [None] * len(texts)

        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start:start + self.config.batch_size]
            batch_indices = list(range(start, min(start + len(batch), len(texts))))

            for attempt in range(self.config.max_retries):
                try:
                    resp = self._client.embeddings.create(
                        model=self.config.model,
                        input=batch,
                        timeout=self.config.timeout,
                    )
                    for i, item in enumerate(resp.data):
                        vectors[batch_indices[i]] = item.embedding
                    break
                except Exception as exc:
                    if attempt == self.config.max_retries - 1:
                        print(f"  [EMBED-API] Batch [{start}:{start+len(batch)}] FAILED: {exc}")
                        for j, text in enumerate(batch):
                            vectors[batch_indices[j]] = self._embed_single_fallback(text)
                    else:
                        wait = 2 ** attempt
                        print(f"  [EMBED-API] Retry {attempt+1}/{self.config.max_retries} in {wait}s: {exc}")
                        time.sleep(wait)

        return vectors

    def _embed_single_fallback(self, text: str) -> list[float] | None:
        """单条 embedding（batch 失败时的降级）。"""
        try:
            resp = self._client.embeddings.create(
                model=self.config.model,
                input=[text],
                timeout=self.config.timeout,
            )
            return resp.data[0].embedding
        except Exception as exc:
            print(f"  [EMBED-API] Single embedding FAILED: {exc}")
            return None


# ================================================================
# Local Embedding (sentence-transformers / BGE / Qwen3-Embedding)
# ================================================================

class SentenceTransformerAdapter(EmbeddingAdapter):
    """本地 embedding 模型后端。

    使用 transformers.AutoModel 直接加载（绕过 sentence_transformers 的 DLL 冲突）。
    支持 BGE、Qwen3-Embedding、GTE 等所有 HuggingFace embedding 模型。

    自动检测模型架构 → 选择正确的 pooling 策略:
      - 仅 Encoder (BERT/BGE):  mean pooling
      - Decoder (Qwen3):        last-token pooling
    """

    def __init__(self, config: LocalEmbeddingConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._dim: int | None = None
        self._pooling_mode: str = ""   # "mean" | "lasttoken"
        self._device = None

    # ----------------------------------------------------------------
    # Path resolution
    # ----------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        import os as _os
        from pathlib import Path as _Path

        if _os.path.isabs(model_name) or model_name.startswith(("./", "../", ".\\")):
            return model_name

        project_root = _Path(__file__).resolve().parent.parent
        candidate = project_root / model_name
        if candidate.exists():
            return str(candidate)

        return model_name

    # ----------------------------------------------------------------
    # Model loading (raw transformers → no DLL conflict)
    # ----------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return

        import sys as _sys

        model_path = self._resolve_model_path(self.config.model_name)
        print(f"  [EMBED-LOCAL] Model path: {model_path}")
        _sys.stdout.flush()

        # 检测 pooling 模式
        pooling_config_path = Path(model_path) / "1_Pooling" / "config.json"
        if pooling_config_path.exists():
            import json as _json
            pool_cfg = _json.loads(pooling_config_path.read_text(encoding="utf-8"))
            if pool_cfg.get("pooling_mode_lasttoken"):
                self._pooling_mode = "lasttoken"
            elif pool_cfg.get("pooling_mode_mean_tokens"):
                self._pooling_mode = "mean"
        # fallback: 根据模型架构判断
        if not self._pooling_mode:
            self._pooling_mode = self._detect_pooling_mode(model_path)

        print(f"  [EMBED-LOCAL] Pooling mode: {self._pooling_mode}")
        print(f"  [EMBED-LOCAL] Loading model (transformers.AutoModel)...")
        _sys.stdout.flush()

        t0 = time.time()

        import torch
        from transformers import AutoModel, AutoTokenizer

        # 设备
        device_str = self.config.device
        self._device = torch.device(
            device_str if torch.cuda.is_available() or device_str == "cpu"
            else "cpu"
        )

        # dtype
        dtype = torch.float16 if (
            self._device.type == "cuda" and self.config.normalize
        ) else None

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        dt_tok = (time.time() - t0) * 1000
        print(f"  [EMBED-LOCAL] Tokenizer loaded ({dt_tok:.0f}ms)")
        _sys.stdout.flush()

        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        ).to(self._device)
        self._model.eval()  # ponytail: no grad overhead at inference

        dt_load = (time.time() - t0) * 1000
        print(f"  [EMBED-LOCAL] Model loaded ({dt_load:.0f}ms)")
        _sys.stdout.flush()

        # 确定维度
        self._dim = self._model.config.hidden_size

        # 预热
        print(f"  [EMBED-LOCAL] Warmup encode...")
        _sys.stdout.flush()
        t_warm = time.time()
        _ = self._encode_texts(["warmup"])
        dt_warm = (time.time() - t_warm) * 1000
        print(f"  [EMBED-LOCAL] Warmup done ({dt_warm:.0f}ms)")
        _sys.stdout.flush()

        print(f"  [EMBED-LOCAL] Ready: {self._dim}d, {self._pooling_mode} pooling, "
              f"device={self._device}, dtype={dtype}")

    # ----------------------------------------------------------------
    # Pooling mode detection
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_pooling_mode(model_path: str) -> str:
        """根据模型 config.json 判断 pooling 策略。"""
        import json as _json
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return "mean"  # safe default for encoder models

        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        archs = cfg.get("architectures", [])
        model_type = cfg.get("model_type", "")

        # Decoder-only → last token pooling (Qwen3-Embedding)
        if any("CausalLM" in a or "ForCausalLM" in a for a in archs):
            return "lasttoken"
        if "qwen" in model_type.lower():
            return "lasttoken"

        return "mean"

    # ----------------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------------

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []

        self._ensure_model()

        import sys as _sys
        import torch

        n = len(texts)
        print(f"  [EMBED-LOCAL] Encoding {n} texts (batch_size={self.config.batch_size})...")
        _sys.stdout.flush()
        t0 = time.time()

        try:
            all_vectors = []
            bs = self.config.batch_size

            for start in range(0, n, bs):
                batch = texts[start:start + bs]
                with torch.no_grad():
                    vecs = self._encode_texts(batch)
                all_vectors.extend(vecs.tolist())

            dt = (time.time() - t0) * 1000
            print(f"  [EMBED-LOCAL] Encoded {n} texts → {self._dim}d vectors "
                  f"in {dt:.0f}ms ({dt/n:.1f}ms/text)")
            return all_vectors

        except Exception as exc:
            dt = (time.time() - t0) * 1000
            print(f"  [EMBED-LOCAL] Batch encoding FAILED after {dt:.0f}ms: {exc}")
            # 降级：逐条重试
            results = []
            for i, text in enumerate(texts):
                try:
                    with torch.no_grad():
                        v = self._encode_texts([text])
                    results.append(v[0].tolist())
                except Exception as e2:
                    print(f"  [EMBED-LOCAL] Single [{i}] FAILED: {e2}")
                    results.append(None)
            return results

    def embed_single(self, text: str) -> list[float] | None:
        result = self.embed_batch([text])
        return result[0] if result else None

    # ----------------------------------------------------------------
    # Core: tokenize → forward → pool → normalize
    # ----------------------------------------------------------------

    def _encode_texts(self, texts: list[str]) -> "torch.Tensor":
        """ponytail: tokenize + forward + pool + normalize, no sentence_transformers."""
        import torch
        import torch.nn.functional as F

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model(**encoded)

        if self._pooling_mode == "lasttoken":
            # 取每个序列最后一个非 padding token 的 hidden state
            hidden = outputs.last_hidden_state  # [B, L, D]
            attn = encoded["attention_mask"]     # [B, L]
            # 每行的最后一个有效 token 位置
            seq_lens = attn.sum(dim=1) - 1      # [B]
            batch_indices = torch.arange(hidden.size(0), device=self._device)
            pooled = hidden[batch_indices, seq_lens]  # [B, D]
        else:
            # Mean pooling（仅对有效 token）
            hidden = outputs.last_hidden_state  # [B, L, D]
            attn = encoded["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
            pooled = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)

        # L2 normalize
        if self.config.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled.cpu()


# ================================================================
# Factory
# ================================================================

def create_embedding_adapter(config: EmbeddingConfig) -> EmbeddingAdapter:
    """根据配置创建 Embedding 适配器实例。

    用法:
        adapter = create_embedding_adapter(config.embedding)
    """
    if config.backend == "local":
        return SentenceTransformerAdapter(config.local)
    else:
        return APIEmbeddingAdapter(config.api)
