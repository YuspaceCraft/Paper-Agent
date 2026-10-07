"""Deterministic query rewriting for local paper retrieval.

The indexed papers are predominantly English, while users often ask in Chinese
or use short acronyms.  This module keeps the original query and adds
bilingual/expanded variants that can be retrieved independently and fused.

It is intentionally deterministic: no network call is required, failures cannot
block search, and the behaviour is straightforward to test.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class QueryRewrite:
    original: str
    primary: str
    variants: tuple[str, ...]
    translations: tuple[str, ...] = ()


# Longest phrases must come first so "遥感图像" is translated before "遥感".
_PHRASE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "遥感图像变化描述",
        ("remote sensing image change captioning", "remote sensing change captioning"),
    ),
    (
        "遥感影像变化描述",
        ("remote sensing image change captioning", "remote sensing change captioning"),
    ),
    (
        "遥感图像变化检测",
        ("remote sensing image change detection", "remote sensing change detection"),
    ),
    (
        "遥感影像变化检测",
        ("remote sensing image change detection", "remote sensing change detection"),
    ),
    ("遥感图像", ("remote sensing image", "remotely sensed image")),
    ("遥感影像", ("remote sensing image", "remotely sensed imagery")),
    ("遥感数据", ("remote sensing data", "remotely sensed data")),
    ("变化描述", ("change captioning",)),
    ("变化检测", ("change detection",)),
    ("图像描述", ("image captioning",)),
    ("图像字幕", ("image captioning",)),
    ("目标检测", ("object detection",)),
    ("语义分割", ("semantic segmentation",)),
    ("超分辨率", ("super-resolution", "super resolution")),
    ("数字高程模型", ("digital elevation model", "DEM")),
    ("高分辨率", ("high-resolution", "high resolution")),
    ("多模态", ("multimodal",)),
    ("注意力", ("attention",)),
    ("扩散模型", ("diffusion model",)),
    ("对比学习", ("contrastive learning",)),
    ("知识蒸馏", ("knowledge distillation",)),
    ("数据集", ("dataset",)),
    ("性能", ("performance",)),
    ("特征", ("feature", "features")),
    ("区别", ("difference", "comparison")),
    ("对比", ("compare", "comparison")),
    ("方法", ("method", "approach")),
    ("模型", ("model",)),
    ("遥感", ("remote sensing", "earth observation", "satellite imagery")),
)

_ACRONYM_ALIASES: dict[str, str] = {
    "rs": "remote sensing",
    "rsicc": "remote sensing image change captioning",
    "hrsi": "high-resolution remote sensing image",
    "dem": "digital elevation model",
    "sar": "synthetic aperture radar",
    "uav": "unmanned aerial vehicle",
    "mllm": "multimodal large language model",
    "vlm": "vision language model",
}

# Low-information request wrappers.  These are removed from the canonical
# search text but never from the original variant.
_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"只(?:给我|要|看|保留)",
        r"(?:请|麻烦)(?:帮我)?",
        r"(?:帮我|给我)",
        r"(?:搜索|检索|查找|查询|列出|看看|查看|找一下)",
        r"(?:本地|我的)?知识库",
        r"(?:相关|有关)的?",
        r"方面(?:的)?",
        r"(?:它们|它|他们|这些|那些|该论文|这篇|那篇|上述|刚才|其中|这个方法|该方法)",
        r"(?:论文|文献|资料|内容)",
        r"(?:这|那|两|几)篇",
        r"(?:有哪(?:些|几篇)|有哪些)",
        r"(?:有什么|是什么|什么是|如何|怎么样|怎么)",
        r"^(?:一下|一点|关于)",
    )
)

_REFERENCE_RE = re.compile(
    r"(?<![其])它(?:们)?"
    r"|他们|这些|那些|该论文|这篇|那篇|上述|刚才|其中|这个方法|该方法"
    r"|\b(?:it|they|these|those|this)\b",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _strip_fillers(text: str) -> str:
    cleaned = text
    for pattern in _FILLER_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"[的得地]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,，。.;；:：")
    return cleaned


def _expand_aliases(text: str) -> tuple[str, list[str]]:
    """Return (canonical English text, alternate translations)."""
    canonical = text
    alternates: list[str] = []

    for phrase, aliases in _PHRASE_ALIASES:
        if phrase not in canonical:
            continue
        canonical = canonical.replace(phrase, aliases[0])
        for alias in aliases[1:]:
            alternates.append(canonical.replace(aliases[0], alias))

    # Expand standalone acronyms without replacing the original query.
    for acronym, expansion in _ACRONYM_ALIASES.items():
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(acronym)}(?![A-Za-z0-9])", re.I)
        if pattern.search(canonical):
            alternates.append(pattern.sub(expansion, canonical))

    canonical = re.sub(
        r"\b(\w+)(?:\s+\1\b)+",
        r"\1",
        _clean_text(canonical),
        flags=re.IGNORECASE,
    )
    return canonical, alternates


def _dedupe(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = _clean_text(value)
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return tuple(out)


def rewrite_query(
    query: str,
    *,
    context: str = "",
    anchors: list[str] | tuple[str, ...] = (),
) -> QueryRewrite:
    """Build retrieval variants for one user query.

    ``context`` is only incorporated for anaphoric follow-ups.  ``anchors`` are
    already-resolved paper names and make vague references searchable.
    """
    original = _clean_text(query)
    if not original:
        return QueryRewrite("", "", ())

    contextual = bool(_REFERENCE_RE.search(original)) and bool(_clean_text(context))
    base = f"{_clean_text(context)[-800:]} {original}" if contextual else original
    if anchors:
        base = f"{base} {' '.join(str(a) for a in anchors if a)}"

    stripped = _strip_fillers(base)
    canonical, alternates = _expand_aliases(stripped or base)

    variants = _dedupe([
        original,
        canonical,
        *alternates,
        *_expand_aliases(original)[1],
    ])
    primary = canonical if canonical != original else original
    translations = _dedupe([canonical, *alternates])
    return QueryRewrite(
        original=original,
        primary=primary,
        variants=variants,
        translations=translations,
    )


def build_search_query(
    query: str,
    *,
    context: str = "",
    anchors: list[str] | tuple[str, ...] = (),
) -> str:
    """Return a single standalone query for the agent's tool call."""
    return rewrite_query(query, context=context, anchors=anchors).primary


def fuse_query_variants(
    result_lists: list[list[dict]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[dict]:
    """Fuse several rank lists while preserving the best raw score per chunk."""
    if not result_lists:
        return []
    if len(result_lists) == 1:
        return result_lists[0][:top_k]

    rank_scores: dict[str, float] = {}
    best_scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item.get("chunk_id", "")
            if not cid:
                continue
            rank_scores[cid] = rank_scores.get(cid, 0.0) + 1.0 / (
                rank_constant + rank + 1
            )
            try:
                raw_score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                raw_score = 0.0
            if cid not in best_scores or raw_score > best_scores[cid]:
                best_scores[cid] = raw_score
            docs.setdefault(cid, item)

    ranked = sorted(rank_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        {
            **docs.get(cid, {}),
            "chunk_id": cid,
            "score": rank_scores[cid],
            "raw_score": best_scores.get(cid, rank_scores[cid]),
        }
        for cid, _ in ranked
    ]
