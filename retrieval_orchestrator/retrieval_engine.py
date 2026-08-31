"""
retrieval_engine.py — 多策略检索执行引擎
========================================

Pluggable retrieval pipeline supporting:
  - Dense Only / Sparse Only / Hybrid (RRF / Weighted)
  - With/Without HyDE query rewriting
  - With/Without metadata filtering
  - Configurable Top-K

Each experiment outputs retrieval_results.jsonl.

ponytail: 核心检索组件已提取到 retrieval/ 共享模块，
此文件仅保留评估特有的 HyDE 改写 + 策略网格构建 + 实验编排。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from retrieval import SparseRetriever, DenseRetriever, rrf_fuse, weighted_fuse


# ================================================================
# HyDE Query Rewriting
# ================================================================

def _validate_hypothetical(text: str, original_query: str) -> str | None:
    """Validate LLM-generated hypothetical text before using as search query.

    Returns validated text, or None if output should be discarded.
    """
    if not text or len(text) < 20:
        return None  # too short, likely failure
    if text.lower().startswith(("i cannot", "i'm sorry", "i am unable")):
        return None  # refusal pattern
    if text.lower() == original_query.lower():
        return None  # model just echoed the query
    # Dedup: if generated text is mostly just the query with minor changes
    overlap = len(set(text.lower().split()) & set(original_query.lower().split()))
    if overlap / max(len(text.split()), 1) > 0.9:
        return None  # too similar, no added value
    return text


def _hyde_rewrite(
    query: str,
    llm_model: str,
    llm_client,
) -> str:
    """Generate a hypothetical document from the query, then use that as search query.

    ponytail: concatenate original query + hypothetical answer for better recall.
    Validates LLM output before use; falls back to original query on invalid output.
    """
    prompt = (
        "You are helping improve a retrieval system. Given a search query, "
        "write a short paragraph (2-4 sentences) that a document answering this query "
        "would contain. Write in the style of an academic paper.\n\n"
        f"Query: {query}\n\n"
        "Hypothetical document snippet:"
    )
    try:
        resp = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        hypothetical = resp.choices[0].message.content.strip()
        validated = _validate_hypothetical(hypothetical, query)
        if validated is None:
            print(f"  [RETRIEVAL] HyDE output invalid (len={len(hypothetical)}), falling back to raw query")
            return query
        return f"{query}\n{validated}"
    except Exception as exc:
        print(f"  [RETRIEVAL] HyDE rewrite failed: {exc}")
        return query  # fallback: original query


# ================================================================
# Experiment Runner
# ================================================================

def run_experiments(
    manifest_path: str | Path,
    rag_path: str | Path,
    config: dict,
    indexer_config_path: str = "",
    output_dir: str | Path = "./eval_output",
) -> list[dict]:
    """Run all retrieval strategy experiments defined in config.

    Args:
        manifest_path: Path to eval_manifest.jsonl
        rag_path: Path to rag_chunks.json (for sparse index building)
        config: Parsed evaluation YAML config dict
        indexer_config_path: Path to indexer config (for vector store + embedding)
        output_dir: Directory for output files

    Returns:
        List of experiment summaries: [{experiment_id, config, results_path, ...}]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))
    print(f"  [RETRIEVAL] Loaded {len(manifest)} queries from manifest")

    # Load chunks for sparse index
    chunks = json.loads(Path(rag_path).read_text(encoding="utf-8")).get("chunks", [])

    # Init retrievers
    from indexer.config import load_config as load_index_cfg
    from indexer.vector_store import ChromaVectorStore, QdrantVectorStore
    from indexer.embedding_adapters import create_embedding_adapter

    idx_cfg = load_index_cfg(indexer_config_path)
    vs = idx_cfg.vector_store
    if vs.backend == "qdrant":
        store = QdrantVectorStore(url=vs.qdrant.url, collection_name=vs.qdrant.collection_name)
    else:
        store = ChromaVectorStore(persist_dir=vs.chroma.persist_dir, collection_name=vs.chroma.collection_name)

    embedder = create_embedding_adapter(idx_cfg.embedding)
    dense = DenseRetriever(store, embedder)

    sparse = SparseRetriever()
    sparse.index(chunks)
    print(f"  [RETRIEVAL] Sparse index built: {len(chunks)} documents")

    # LLM client for HyDE (lazy)
    hyde_client = None
    hyde_model = ""

    # Build strategy grid
    retrieval_cfg = config.get("retrieval", {})
    strategies = _build_strategy_grid(retrieval_cfg)

    experiment_results = []
    for strat in strategies:
        strat_id = _strategy_id(strat)
        print(f"  [RETRIEVAL] Running: {strat_id}")

        hyde_model = strat.get("hyde_model", "")
        if hyde_model and hyde_client is None:
            hyde_client = _init_llm(hyde_model, retrieval_cfg.get("llm", {}))

        results = []
        for qa in manifest:
            query = qa["query"]
            top_k = strat["top_k"]
            filters = qa.get("metadata_filters") or strat.get("metadata_filter") or None

            # Apply query rewriting
            if strat.get("query_rewriting") == "hyde" and hyde_client:
                query = _hyde_rewrite(query, hyde_model, hyde_client)

            # Retrieve
            start = time.perf_counter()
            if strat["method"] == "dense":
                hits = dense.search(query, top_k=top_k, filters=filters)
            elif strat["method"] == "sparse":
                hits = sparse.search(query, top_k=top_k)
            elif strat["method"] == "hybrid":
                d_hits = dense.search(query, top_k=max(top_k, 50), filters=filters)
                s_hits = sparse.search(query, top_k=max(top_k, 50))
                if strat.get("hybrid_mode", "rrf") == "weighted":
                    dw = strat.get("dense_weight", 0.7)
                    hits = weighted_fuse(d_hits, s_hits, dense_weight=dw, top_k=top_k)
                else:
                    hits = rrf_fuse(d_hits, s_hits, top_k=top_k)
            else:
                hits = []
            latency_ms = (time.perf_counter() - start) * 1000

            results.append({
                "query_id": qa["id"],
                "query": qa["query"],
                "ground_truth_ids": qa["ground_truth_ids"],
                "hits": [{"chunk_id": h.get("chunk_id", ""), "score": h.get("score", 0)} for h in hits],
                "latency_ms": round(latency_ms, 2),
            })

        # Write results
        results_path = output_dir / f"retrieval_{strat_id}.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        experiment_results.append({
            "experiment_id": strat_id,
            "config": strat,
            "results_path": str(results_path),
            "query_count": len(results),
        })
        print(f"  [RETRIEVAL]   → {len(results)} queries, saved to {results_path}")

    return experiment_results


def _build_strategy_grid(retrieval_cfg: dict) -> list[dict]:
    """Build the Cartesian product of retrieval strategy parameters."""
    methods = retrieval_cfg.get("methods", ["dense"])
    top_k_values = retrieval_cfg.get("top_k_values", [10])
    query_rewriting = retrieval_cfg.get("query_rewriting", ["none"])
    hybrid_modes = retrieval_cfg.get("hybrid_modes", ["rrf"])
    dense_weights = retrieval_cfg.get("dense_weights", [0.7])
    metadata_filters = retrieval_cfg.get("metadata_filters", [None])
    hyde_model = retrieval_cfg.get("hyde_model", "")

    strategies = []
    for method in methods:
        for top_k in top_k_values:
            for qr in query_rewriting:
                if method == "hybrid":
                    for hm in hybrid_modes:
                        for dw in dense_weights:
                            for mf in metadata_filters:
                                strategies.append({
                                    "method": method, "top_k": top_k,
                                    "query_rewriting": qr if qr != "none" else None,
                                    "hybrid_mode": hm, "dense_weight": dw,
                                    "metadata_filter": mf,
                                    "hyde_model": hyde_model if qr == "hyde" else "",
                                })
                else:
                    for mf in metadata_filters:
                        strategies.append({
                            "method": method, "top_k": top_k,
                            "query_rewriting": qr if qr != "none" else None,
                            "metadata_filter": mf,
                            "hyde_model": hyde_model if qr == "hyde" else "",
                        })
    return strategies


def _strategy_id(strat: dict) -> str:
    """Compact experiment ID for a strategy."""
    parts = [strat["method"], f"k{strat['top_k']}"]
    if strat.get("query_rewriting"):
        parts.append(strat["query_rewriting"])
    if strat.get("hybrid_mode") and strat["method"] == "hybrid":
        parts.append(strat["hybrid_mode"])
        parts.append(f"dw{strat.get('dense_weight', 0.7)}")
    if mf := strat.get("metadata_filter"):
        k = list(mf.keys())[0] if mf else ""
        parts.append(f"mf_{k}")
    return "_".join(parts)


def _init_llm(model: str, llm_cfg: dict):
    """Initialize OpenAI-compatible client for HyDE."""
    import os
    from openai import OpenAI

    api_key = os.getenv(llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"), "")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv(llm_cfg.get("api_key_env", "DASHSCOPE_API_KEY"), "")
        except ImportError:
            pass
    if not api_key:
        raise ValueError(f"API key not found. Set {llm_cfg.get('api_key_env', 'DASHSCOPE_API_KEY')} env var or .env file.")

    return OpenAI(
        api_key=api_key,
        base_url=llm_cfg.get("base_url", ""),
    )
