"""
eval_dataset.py — 评估数据集生成器
===================================

从 rag_chunks.json 自动采样生成 (query, ground_truth_chunk_ids) 对。
三种生成模式: keyword / semantic / cross_chunk。

输出 eval_manifest.jsonl:
  {"id", "query", "ground_truth_ids", "metadata_filters", "difficulty_level"}
"""
from __future__ import annotations

import json
import re
import random
import hashlib
from pathlib import Path
from typing import Any

# ---- Constants ----

KEYWORD_RE = re.compile(r"\[KEYWORDS:\s*([^\]]+)\]")
RANDOM_SEED = 42


def _load_chunks(rag_path: str | Path) -> list[dict]:
    """Load chunks from rag_chunks.json."""
    data = json.loads(Path(rag_path).read_text(encoding="utf-8"))
    return data.get("chunks", [])


def _extract_keywords(content: str) -> list[str]:
    """Extract [KEYWORDS: ...] from chunk content. Returns list of keyword strings."""
    m = KEYWORD_RE.search(content)
    if not m:
        return []
    return [kw.strip() for kw in m.group(1).split(",") if kw.strip()]


def _chunk_difficulty(chunk: dict) -> str:
    """Assign difficulty based on content_type and token_count."""
    ct = chunk.get("content_type", "body")
    tc = chunk.get("token_count", 0)
    if ct in ("formula", "figure", "table"):
        return "hard"
    if tc < 300:
        return "easy"
    if tc > 800:
        return "hard"
    return "medium"


def _make_id(chunk_id: str, mode: str) -> str:
    """Deterministic ID for a QA pair."""
    h = hashlib.md5(f"{chunk_id}:{mode}:{RANDOM_SEED}".encode()).hexdigest()[:8]
    return f"qa_{h}"


# ================================================================
# Generation Modes
# ================================================================

def generate_keyword_queries(
    chunks: list[dict],
    sample_size: int | None = None,
) -> list[dict]:
    """关键词精确型: 提取 [KEYWORDS] 生成自然语言查询。

    每个 chunk 取前 3 个关键词拼接为查询，ground_truth 就是该 chunk。
    """
    random.seed(RANDOM_SEED)
    eligible = [
        c for c in chunks
        if c.get("content_type") in ("body", "formula")
        and _extract_keywords(c.get("content", ""))
    ]
    if sample_size and len(eligible) > sample_size:
        eligible = random.sample(eligible, sample_size)

    results = []
    for chunk in eligible:
        kws = _extract_keywords(chunk["content"])
        query = " ".join(kws[:3])
        results.append({
            "id": _make_id(chunk["chunk_id"], "kw"),
            "query": query,
            "ground_truth_ids": [chunk["chunk_id"]],
            "metadata_filters": {"content_type": chunk.get("content_type", "")},
            "difficulty_level": _chunk_difficulty(chunk),
            "generation_mode": "keyword",
        })
    return results


def generate_semantic_queries(
    chunks: list[dict],
    sample_size: int | None = None,
    llm_model: str = "qwen3.6-max-preview",
    llm_base_url: str = "",
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
) -> list[dict]:
    """语义摘要型: 用 LLM 基于 chunk 内容生成自然语言问题。

    LLM 被要求生成一个该 chunk 能够回答的问题。
    """
    random.seed(RANDOM_SEED)
    eligible = [
        c for c in chunks
        if c.get("content_type") in ("body",)
        and c.get("token_count", 0) >= 200
    ]
    if sample_size and len(eligible) > sample_size:
        eligible = random.sample(eligible, sample_size)

    client = _get_llm_client(llm_model, llm_base_url, llm_api_key_env)
    results = []
    failed = 0

    for i, chunk in enumerate(eligible):
        content = chunk["content"]
        # Truncate content to ~1500 chars for prompt budget
        snippet = content[:1500]
        prompt = (
            "You are generating test queries for a retrieval system. "
            "Read the following text chunk from an academic paper and generate ONE natural-language "
            "question that this specific chunk directly answers. "
            "The question should be specific enough that only this chunk (and very similar ones) "
            "would contain the answer. Output ONLY the question, no preamble.\n\n"
            f"Chunk section: {chunk.get('section_path', 'unknown')}\n"
            f"Chunk content:\n{snippet}\n\n"
            "Question:"
        )
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
                timeout=120,
            )
            question = resp.choices[0].message.content.strip()
            print(f"  [EVAL-DATASET]   {i+1}/{len(eligible)} {chunk['chunk_id']}: OK")
        except Exception as exc:
            print(f"  [EVAL-DATASET] LLM failed for {chunk['chunk_id']}: {exc}")
            failed += 1
            continue

        results.append({
            "id": _make_id(chunk["chunk_id"], "semantic"),
            "query": question,
            "ground_truth_ids": [chunk["chunk_id"]],
            "metadata_filters": {"content_type": chunk.get("content_type", "")},
            "difficulty_level": _chunk_difficulty(chunk),
            "generation_mode": "semantic",
        })

    total_eligible = len(eligible)
    generated = len(results)
    if failed > 0:
        print(f"  [EVAL-DATASET] WARNING: {failed}/{total_eligible} semantic chunks "
              f"failed QA generation ({failed/total_eligible:.0%})")
        if failed / total_eligible > 0.15:
            print(f"  [EVAL-DATASET] ERROR: failure rate exceeds 15% threshold. "
                  f"Eval dataset may be biased.")
    return results


def generate_cross_chunk_queries(
    chunks: list[dict],
    sample_size: int | None = None,
    llm_model: str = "qwen3.6-max-preview",
    llm_base_url: str = "",
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
) -> list[dict]:
    """跨块推理型: 利用 prev/next_chunk_id 生成多跳问题。

    选取有前后继的连续 chunk 对，用 LLM 生成需要同时查两个 chunk 才能回答的问题。
    """
    random.seed(RANDOM_SEED)
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Find chunk pairs linked via prev/next
    pairs = []
    for chunk in chunks:
        next_id = chunk.get("next_chunk_id", "")
        if next_id and next_id in chunk_map:
            pairs.append((chunk, chunk_map[next_id]))

    if not pairs:
        print("  [EVAL-DATASET] No linked chunk pairs found for cross-chunk generation")
        return []

    if sample_size and len(pairs) > sample_size:
        pairs = random.sample(pairs, sample_size)

    client = _get_llm_client(llm_model, llm_base_url, llm_api_key_env)
    results = []
    failed = 0

    for c1, c2 in pairs:
        s1 = c1["content"][:1000]
        s2 = c2["content"][:1000]
        prompt = (
            "You are generating test queries for a multi-hop retrieval system. "
            "Below are TWO consecutive chunks from an academic paper. "
            "Generate ONE question that requires information from BOTH chunks to answer fully. "
            "The question should require synthesizing information across the chunks, "
            "not just looking up facts from either one in isolation. "
            "Output ONLY the question, no preamble.\n\n"
            f"Chunk A (section: {c1.get('section_path', 'unknown')}):\n{s1}\n\n"
            f"Chunk B (section: {c2.get('section_path', 'unknown')}):\n{s2}\n\n"
            "Multi-hop question:"
        )
        try:
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
                timeout=120,
            )
            question = resp.choices[0].message.content.strip()
            print(f"  [EVAL-DATASET]   {len(results)+1} cross_chunk: OK")
        except Exception as exc:
            print(f"  [EVAL-DATASET] LLM failed for {c1['chunk_id']}+{c2['chunk_id']}: {exc}")
            failed += 1
            continue

        results.append({
            "id": _make_id(f"{c1['chunk_id']}_{c2['chunk_id']}", "cross"),
            "query": question,
            "ground_truth_ids": [c1["chunk_id"], c2["chunk_id"]],
            "metadata_filters": {},
            "difficulty_level": "hard",
            "generation_mode": "cross_chunk",
        })

    total_pairs = len(pairs)
    generated = len(results)
    if failed > 0:
        print(f"  [EVAL-DATASET] WARNING: {failed}/{total_pairs} cross-chunk pairs "
              f"failed QA generation ({failed/total_pairs:.0%})")
        if failed / total_pairs > 0.15:
            print(f"  [EVAL-DATASET] ERROR: failure rate exceeds 15% threshold. "
                  f"Eval dataset may be biased.")
    return results


# ================================================================
# LLM Client (ponytail: reused across all modes)
# ================================================================

_llm_client = None
_llm_client_key = ""


def _get_llm_client(model: str, base_url: str, api_key_env: str):
    """Lazy-init OpenAI-compatible client."""
    global _llm_client, _llm_client_key
    import os
    key = f"{model}:{base_url}:{api_key_env}"
    if _llm_client is not None and _llm_client_key == key:
        return _llm_client

    from openai import OpenAI
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv(api_key_env, "")
        except ImportError:
            pass

    _llm_client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=120.0)
    _llm_client_key = key
    return _llm_client


# ================================================================
# Main Entry Point
# ================================================================

def generate_manifest(
    rag_path: str | Path,
    modes: list[str] | None = None,
    samples_per_mode: int | None = None,
    llm_model: str = "qwen3.6-max-preview",
    llm_base_url: str = "",
    llm_api_key_env: str = "DASHSCOPE_API_KEY",
    output_path: str | Path | None = None,
) -> list[dict]:
    """Generate eval manifest from rag_chunks.json.

    Args:
        rag_path: Path to rag_chunks.json
        modes: List of modes to run. Default: ["keyword"] (no LLM needed)
        samples_per_mode: Max samples per mode. None = all eligible.
        output_path: If set, write JSONL file.

    Returns:
        List of QA dicts.
    """
    if modes is None:
        modes = ["keyword"]

    chunks = _load_chunks(rag_path)
    print(f"  [EVAL-DATASET] Loaded {len(chunks)} chunks from {rag_path}")

    all_qa: list[dict] = []
    for mode in modes:
        print(f"  [EVAL-DATASET] Generating {mode} queries...")
        if mode == "keyword":
            qa = generate_keyword_queries(chunks, samples_per_mode)
        elif mode == "semantic":
            qa = generate_semantic_queries(
                chunks, samples_per_mode, llm_model, llm_base_url, llm_api_key_env,
            )
        elif mode == "cross_chunk":
            qa = generate_cross_chunk_queries(
                chunks, samples_per_mode, llm_model, llm_base_url, llm_api_key_env,
            )
        else:
            print(f"  [EVAL-DATASET] Unknown mode: {mode}, skipping")
            continue
        all_qa.extend(qa)
        print(f"  [EVAL-DATASET]   → {len(qa)} queries generated")

    print(f"  [EVAL-DATASET] Total: {len(all_qa)} QA pairs ({len(modes)} modes)")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for qa in all_qa:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"  [EVAL-DATASET] Written to {output_path}")

    return all_qa


def review_manifest(
    manifest_path: str | Path,
    corrections_path: str | Path | None = None,
    output_path: str | Path | None = None,
    strict: bool = False,
) -> list[dict]:
    """CLI tool for human review: load manifest, apply corrections, write merged result.

    corrections_path should point to a JSONL file with same-id entries that replace
    or remove (by setting `removed: true`) specific QA pairs.

    If strict=True, refuses to write output if any QA pair (from LLM-generated modes)
    lacks a human review entry. Keyword-mode entries are exempt.
    """
    qa_list = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_list.append(json.loads(line))

    corrections: dict[str, dict] = {}
    if corrections_path and Path(corrections_path).exists():
        with open(corrections_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    corrections[c["id"]] = c

    merged = []
    unreviewed_llm: list[str] = []
    for qa in qa_list:
        cid = qa["id"]
        if cid in corrections:
            corr = corrections[cid]
            if corr.get("removed"):
                continue
            corr["reviewed"] = True
            merged.append(corr)
        else:
            merged.append(qa)
            # Track LLM-generated entries that were NOT reviewed
            if qa.get("generation_mode") in ("semantic", "cross_chunk"):
                unreviewed_llm.append(cid)

    if strict and unreviewed_llm:
        raise RuntimeError(
            f"Strict mode: {len(unreviewed_llm)} LLM-generated QA pairs have not been "
            f"reviewed. IDs: {unreviewed_llm[:5]}...\n"
            f"Run review with corrections first, or use --no-strict to skip."
        )

    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for qa in merged:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        if unreviewed_llm:
            print(f"  [EVAL-DATASET] WARNING: {len(unreviewed_llm)}/{len(merged)} "
                  f"LLM-generated QA pairs not reviewed — output is a DRAFT")
        print(f"  [EVAL-DATASET] Merged manifest written to {output_path}")

    return merged
