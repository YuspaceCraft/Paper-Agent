"""
embedder.py — 多粒度向量化器
===========================

三路向量化:
  1. Dense: OpenAI-compatible Embedding API → 语义向量
  2. Sparse/BM25: [KEYWORDS] 解析 + TF 关键词提取 → 倒排字段
  3. HyDE: 可选，生成假设性问题并单独向量化

含 PII 检测 — 在向量化前对文本进行敏感信息扫描。
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from openai import OpenAI

from .config import EmbeddingConfig, HyDEConfig, PIIConfig
from .embedding_adapters import EmbeddingAdapter, create_embedding_adapter

# ================================================================
# PII Detection
# ================================================================

# ponytail: 正则 PII 检测，覆盖常见敏感信息模式。
# presidio 可用时自动升级，默认不需要额外依赖。
_PII_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("EMAIL",    "Email address",      re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')),
    ("PHONE",    "Phone number",       re.compile(r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b')),
    ("IP",       "IP address",         re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
    ("SSN",      "SSN-like number",    re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("API_KEY",  "API key pattern",    re.compile(r'\b[A-Za-z0-9_]{20,}={0,2}\b')),
    ("CREDIT",   "Credit card number", re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')),
    ("ID_CARD",  "Chinese ID number",  re.compile(r'\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b')),
]


def detect_pii(text: str, config: PIIConfig) -> tuple[bool, list[dict]]:
    """检测文本中的 PII。返回 (has_pii, findings)。

    若 presidio 可用且配置为 presidio 模式，使用 presidio；
    否则使用正则匹配。
    """
    if not config.enabled:
        return False, []

    if config.detection_method == "presidio":
        return _detect_pii_presidio(text)

    return _detect_pii_regex(text)


def _detect_pii_regex(text: str) -> tuple[bool, list[dict]]:
    findings = []
    for pii_type, description, pattern in _PII_PATTERNS:
        for m in pattern.finditer(text):
            findings.append({
                "type": pii_type,
                "description": description,
                "match": m.group()[:30] + ("..." if len(m.group()) > 30 else ""),
                "position": m.start(),
            })
    return (len(findings) > 0, findings)


def _detect_pii_presidio(text: str) -> tuple[bool, list[dict]]:
    """使用 Microsoft Presidio 进行 PII 检测（可选依赖）。"""
    try:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        findings = [
            {"type": r.entity_type, "description": r.entity_type, "match": text[r.start:r.end]}
            for r in results
        ]
        return (len(findings) > 0, findings)
    except ImportError:
        # fallback to regex
        return _detect_pii_regex(text)
    except Exception:
        return False, []


# ================================================================
# Keyword Extraction (Sparse/BM25)
# ================================================================

_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'that', 'this', 'these', 'those', 'which',
    'and', 'but', 'or', 'if', 'because', 'while', 'although', 'we', 'our',
    'it', 'its', 'they', 'them', 'their', 'he', 'she', 'his', 'her',
    'using', 'used', 'use', 'based', 'also', 'et', 'al', 'via',
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
})


def parse_keywords_prefix(text: str) -> list[str]:
    """从 [KEYWORDS: ...] 前缀解析已有关键词。"""
    m = re.match(r'^\[KEYWORDS:\s*([^\]]+)\]', text)
    if not m:
        return []
    return [kw.strip() for kw in m.group(1).split(",") if kw.strip()]


def extract_sparse_keywords(text: str, top_n: int = 10) -> list[str]:
    """从文本提取 TF-based 关键词（用于 BM25 倒排）。

    优先使用 [KEYWORDS] 前缀中的关键词，不足时用 TF 补充。
    """
    keywords = parse_keywords_prefix(text)

    if len(keywords) >= top_n:
        return keywords[:top_n]

    # TF 补充
    clean = re.sub(r'\[(?:FORMULA_DESC|FIGURE_DESC|KEYWORDS):[^\]]+\]', ' ', text)
    clean = re.sub(r'[^a-zA-Z0-9\s一-鿿-]', ' ', clean.lower())

    words = [w.strip('-') for w in clean.split() if len(w.strip('-')) > 2]
    words = [w for w in words if w not in _STOP_WORDS]

    freq = Counter(words)
    existing = set(k.lower() for k in keywords)
    for word, _ in freq.most_common(top_n * 3):
        if len(keywords) >= top_n:
            break
        if word.isdigit() or word.lower() in existing:
            continue
        keywords.append(word)
        existing.add(word.lower())

    return keywords[:top_n]


# ================================================================
# Embedder
# ================================================================

class MultiGranularityEmbedder:
    """多粒度向量化器。

    - Dense: 调用兼容 OpenAI 的 Embedding API
    - Sparse: [KEYWORDS] 解析 + TF 关键词提取
    - HyDE: 可选，生成假设性问题（用 LLM 而非规则模板）
    """

    def __init__(
        self,
        emb_config: EmbeddingConfig | None = None,
        hyde_config: HyDEConfig | None = None,
        pii_config: PIIConfig | None = None,
        adapter: EmbeddingAdapter | None = None,
    ):
        self.emb_config = emb_config or EmbeddingConfig()
        self.hyde_config = hyde_config or HyDEConfig()
        self.pii_config = pii_config or PIIConfig()
        self._adapter = adapter or create_embedding_adapter(self.emb_config)
        self._hyde_client: OpenAI | None = None

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def embed(self, units: list[dict]) -> list[dict]:
        """对 assembled units 执行完整多粒度向量化。

        每个 unit (来自 ContextAssembler.assemble 的输出) 被扩展为:
          - dense_vector: float list
          - sparse_keywords: keyword list
          - hyde_questions: (可选) 假设性问题列表
          - pii_flagged / pii_findings: PII 检测结果
        """
        if not units:
            return []

        # Step 1: PII 检测
        t0 = time.time()
        for u in units:
            has_pii, findings = detect_pii(u["retrieval_text"], self.pii_config)
            u["pii_flagged"] = has_pii
            u["pii_findings"] = findings
        print(f"  [EMBED] PII scan: {sum(1 for u in units if u['pii_flagged'])} flagged "
              f"({(time.time()-t0)*1000:.0f}ms)")

        # Step 2: Dense embedding — 委托给适配器
        t0 = time.time()
        texts = [u["retrieval_text"] for u in units]
        vectors = self._adapter.embed_batch(texts)
        for u, vec in zip(units, vectors):
            u["dense_vector"] = vec
        n_success = sum(1 for v in vectors if v)
        print(f"  [EMBED] Dense ({self.emb_config.backend}): {n_success}/{len(units)} "
              f"embedded ({(time.time()-t0)*1000:.0f}ms)")

        # Step 3: Sparse keywords
        t0 = time.time()
        for u in units:
            u["sparse_keywords"] = extract_sparse_keywords(u["retrieval_text"])
        print(f"  [EMBED] Sparse: keywords extracted ({(time.time()-t0)*1000:.0f}ms)")

        # Step 4: HyDE (optional)
        if self.hyde_config.enabled:
            t0 = time.time()
            for u in units:
                u["hyde_questions"] = self._generate_hyde_questions(
                    u["retrieval_text"],
                )
            n_hyde = sum(1 for u in units if u["hyde_questions"])
            print(f"  [EMBED] HyDE: {n_hyde} chunks with questions "
                  f"({(time.time()-t0)*1000:.0f}ms)")

        return units

    # ----------------------------------------------------------------
    # HyDE (Hypothetical Document Embedding)
    # ----------------------------------------------------------------

    _HYDE_PROMPT = """You are a curious researcher. Given the following academic text excerpt,
generate {n} hypothetical search queries that a researcher might type to find this content.

Rules:
- Output ONLY the queries, one per line, no numbering, no preamble.
- Each query should be a natural language question or phrase (5-15 words).
- Vary the angle: one definitional, one methodological, one comparative.

Text excerpt:
{text}"""

    def _generate_hyde_questions(self, text: str) -> list[str]:
        """为文本生成假设性问题。"""
        if not text.strip():
            return []

        n = self.hyde_config.questions_per_chunk
        prompt = self._HYDE_PROMPT.format(n=n, text=text[:1500])

        try:
            client = self._get_hyde_client()
            resp = client.chat.completions.create(
                model=self.hyde_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=80 * n,
            )
            raw = resp.choices[0].message.content.strip()
            questions = [q.strip("-• ").strip() for q in raw.split("\n") if q.strip()]
            return questions[:n]
        except Exception as exc:
            print(f"  [EMBED] HyDE generation FAILED: {exc}")
            return []

    def _get_hyde_client(self) -> OpenAI:
        if self._hyde_client is None:
            # HyDE 使用 API 后端的 LLM 配置（与 embedding 后端无关）
            api_cfg = self.emb_config.api
            api_key = os.getenv(api_cfg.api_key_env)
            if not api_key:
                # 尝试加载 .env
                try:
                    from dotenv import load_dotenv
                    _env_path = Path(__file__).resolve().parent.parent / ".env"
                    if _env_path.exists():
                        load_dotenv(_env_path)
                        api_key = os.getenv(api_cfg.api_key_env)
                except ImportError:
                    pass
            if not api_key:
                raise RuntimeError(f"{api_cfg.api_key_env} 未设置")
            self._hyde_client = OpenAI(
                api_key=api_key,
                base_url=api_cfg.api_base,
            )
        return self._hyde_client


# ================================================================
# Pipeline 辅助: assembled unit → IndexUnit
# ================================================================

def assembled_to_index_unit(unit: dict, schema_version: str = "1.0") -> dict:
    """将 embedder 输出的 unit dict 转为可直接存储的 IndexUnit 数据。"""
    return {
        "chunk_id": unit["chunk_id"],
        "retrieval_text": unit["retrieval_text"],
        "generation_text": unit["generation_text"],
        "dense_vector": unit.get("dense_vector"),
        "sparse_keywords": unit.get("sparse_keywords", []),
        "hyde_questions": unit.get("hyde_questions", []),
        "source_chunk": unit.get("source_chunk", {}),
        "metadata": {
            **unit.get("metadata", {}),
            "content_type": unit.get("content_type", ""),
            "section_path": unit.get("section_path", ""),
            "token_count": unit.get("token_count", 0),
            "ref_ids": unit.get("ref_ids", []),
            "bound_elements": unit.get("bound_elements", []),
            "pii_flagged": unit.get("pii_flagged", False),
        },
        "pii_flagged": unit.get("pii_flagged", False),
        "pii_findings": unit.get("pii_findings", []),
        "schema_version": schema_version,
    }
