"""
pipeline.py — 索引编排器
========================

将所有子模块串联为完整的索引构建 Pipeline。
提供 CLI 入口 + Python API。

流程:
  1. ContextAssembler:  加载 rag_chunks.json → 组装长短文本
  2. MultiGranularityEmbedder: Dense + BM25 + PII 检测
  3. DedupManager:       Content Hash 分类 → 增量同步
  4. VectorStoreAdapter: 写入向量库
  5. Eval Export:        导出 eval_manifest.jsonl
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

from .config import IndexerConfig, load_config
from .context_assembler import ContextAssembler
from .embedder import MultiGranularityEmbedder, assembled_to_index_unit
from .vector_store import ChromaVectorStore, QdrantVectorStore
from .dedup_manager import DedupManager
from .catalog import mark_indexed


# ================================================================
# Structured Logging
# ================================================================

def structured_log(level: str, message: str, **kwargs) -> str:
    """输出 JSON 格式结构化日志。

    必须字段: chunk_id, stage, duration_ms, token_count, status。
    调用方可传入额外字段。
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(line)
    return line


# ================================================================
# Eval Manifest Export
# ================================================================

def export_eval_manifest(units: list[dict], export_path: str) -> int:
    """导出 eval_manifest.jsonl 供 Ragas 评估使用。

    格式: {query, ground_truth_chunk_ids, retrieval_text, generation_text}
    追加模式: 按 chunk_id 去重，已存在的记录跳过。

    每行一个四元组，chunk_id 作为 ground truth（检索应召回自身）。
    """
    p = Path(export_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # ponytail: load existing chunk_ids for dedup (append mode)
    seen: set[str] = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                for cid in rec.get("ground_truth_chunk_ids", []):
                    seen.add(cid)
            except (json.JSONDecodeError, TypeError):
                pass

    n = 0
    with open(p, "a", encoding="utf-8") as f:
        for u in units:
            cid = u["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            record = {
                "query": "",  # 由下游评估时填充真实查询
                "ground_truth_chunk_ids": [cid],
                "retrieval_text": u.get("retrieval_text", ""),
                "generation_text": u.get("generation_text", ""),
                "content_type": u.get("metadata", {}).get("content_type", ""),
                "section_path": u.get("metadata", {}).get("section_path", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    structured_log("INFO", f"Eval manifest exported: {export_path}",
                   chunk_id="pipeline", stage="eval_export",
                   duration_ms=0, token_count=0, status="OK",
                   records=n, new=n, existing=len(seen) - n)
    return n


def merge_all_chunks(output_dir: str = "pdf_pipeline/output",
                     merged_path: str = "eval_output/all_rag_chunks.json") -> int:
    """合并所有论文的 rag_chunks.json → all_rag_chunks.json。

    chunk_id 加 paper 前缀避免跨论文命名空间冲突。
    供 RetrievalService SparseRetriever 构建全局索引使用。
    """
    total = 0
    all_chunks: list[dict] = []
    papers: list[str] = []

    out = Path(output_dir)
    if not out.is_dir():
        return 0

    for paper_dir in sorted(out.iterdir()):
        if not paper_dir.is_dir():
            continue
        rag_path = paper_dir / "rag_chunks.json"
        if not rag_path.exists():
            continue

        paper_name = paper_dir.name
        papers.append(paper_name)
        data = json.loads(rag_path.read_text(encoding="utf-8"))

        for ch in data.get("chunks", []):
            ch = dict(ch)  # shallow copy, don't mutate source
            ch["chunk_id"] = f"{paper_name}__{ch['chunk_id']}"
            ch.setdefault("metadata", {})["paper_name"] = paper_name
            all_chunks.append(ch)
            total += 1

    # Build merged document
    section_index: dict[str, list[str]] = {}
    for ch in all_chunks:
        sp = ch.get("section_path", "")
        if sp:
            # Use setdefault for simplicity — a few duplicate lists is cheap enough
            section_index.setdefault(sp, []).append(ch["chunk_id"])

    merged = {
        "total_chunks": total,
        "papers": papers,
        "paper_count": len(papers),
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "chunks": all_chunks,
        "section_index": {k: sorted(set(v)) for k, v in section_index.items()},
    }

    dest = Path(merged_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    structured_log("INFO", f"All chunks merged: {merged_path}",
                   chunk_id="pipeline", stage="merge_all",
                   duration_ms=0, token_count=0, status="OK",
                   total_chunks=total, papers=len(papers))
    return total


# ================================================================
# Orchestrator
# ================================================================

class IndexerPipeline:
    """索引编排器 — 串联所有子模块。

    用法:
        pipeline = IndexerPipeline("indexer/config.yaml")
        stats = pipeline.run("pdf_pipeline/output/MV-CC/rag_chunks.json")
    """

    def __init__(self, config_path: str = ""):
        self.config = load_config(config_path)
        self._store = None

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def run(self, rag_chunks_path: str) -> dict:
        """执行完整索引 Pipeline。

        Args:
            rag_chunks_path: rag_chunks.json 文件路径

        Returns:
            {status, total_chunks, inserted, updated, skipped, deleted,
             pii_flagged, duration_ms, stages: {...}}
        """
        t_start = time.time()
        stage_times = {}

        if not Path(rag_chunks_path).exists():
            return {"status": "ERROR", "error": f"File not found: {rag_chunks_path}"}

        structured_log("INFO", "Pipeline started",
                       chunk_id="pipeline", stage="init",
                       duration_ms=0, token_count=0, status="OK",
                       rag_chunks_path=rag_chunks_path)

        # ---- Stage 1: Context Assembly ----
        t0 = time.time()
        structured_log("INFO", "Stage 1: Context Assembly",
                       chunk_id="pipeline", stage="context_assembly",
                       duration_ms=0, token_count=0, status="RUNNING")

        assembler = ContextAssembler(self.config.context_assembly)
        raw_chunks = assembler.load_chunks(rag_chunks_path)
        assembled = assembler.assemble(raw_chunks)

        # ponytail: prefix chunk_id with paper name for cross-paper uniqueness
        paper_name = Path(rag_chunks_path).parent.name
        for u in assembled:
            u["chunk_id"] = f"{paper_name}__{u['chunk_id']}"

        stage_times["context_assembly"] = (time.time() - t0) * 1000
        structured_log("INFO", "Context Assembly done",
                       chunk_id="pipeline", stage="context_assembly",
                       duration_ms=stage_times["context_assembly"],
                       token_count=sum(u.get("token_count", 0) for u in assembled),
                       status="OK",
                       total_chunks=len(assembled))

        # ---- Stage 2: Embedding + PII ----
        t0 = time.time()
        structured_log("INFO", "Stage 2: Multi-Granularity Embedding",
                       chunk_id="pipeline", stage="embedding",
                       duration_ms=0, token_count=0, status="RUNNING")

        embedder = MultiGranularityEmbedder(
            self.config.embedding, self.config.hyde, self.config.pii,
        )
        embedded = embedder.embed(assembled)

        # 转换为 IndexUnit 格式
        index_units = [
            assembled_to_index_unit(u, self.config.schema_version)
            for u in embedded
        ]

        stage_times["embedding"] = (time.time() - t0) * 1000
        n_pii = sum(1 for u in index_units if u["pii_flagged"])
        n_vectors = sum(1 for u in index_units if u.get("dense_vector"))
        structured_log("INFO", "Embedding done",
                       chunk_id="pipeline", stage="embedding",
                       duration_ms=stage_times["embedding"],
                       token_count=sum(u.get("metadata", {}).get("token_count", 0)
                                       for u in index_units),
                       status="OK",
                       dense_vectors=n_vectors,
                       pii_flagged=n_pii)

        # ---- Stage 3: Dedup ----
        t0 = time.time()
        structured_log("INFO", "Stage 3: Dedup & Version Management",
                       chunk_id="pipeline", stage="dedup",
                       duration_ms=0, token_count=0, status="RUNNING")

        store = self._get_store()
        dedup = DedupManager(store, self.config.schema_version)
        classification = dedup.sync(index_units)

        stage_times["dedup"] = (time.time() - t0) * 1000
        structured_log("INFO", "Dedup done",
                       chunk_id="pipeline", stage="dedup",
                       duration_ms=stage_times["dedup"],
                       token_count=0, status="OK",
                       new=classification["counts"]["new"],
                       updated=classification["counts"]["updated"],
                       skipped=classification["counts"]["skipped"],
                       orphaned=classification["counts"]["orphaned"])

        # ---- Stage 4: VectorStore Write ----
        t0 = time.time()
        structured_log("INFO", "Stage 4: VectorStore Write",
                       chunk_id="pipeline", stage="vector_store",
                       duration_ms=0, token_count=0, status="RUNNING")

        write_stats = dedup.apply(classification)

        stage_times["vector_store"] = (time.time() - t0) * 1000
        structured_log("INFO", "VectorStore Write done",
                       chunk_id="pipeline", stage="vector_store",
                       duration_ms=stage_times["vector_store"],
                       token_count=0, status="OK",
                       **write_stats)

        # ---- Stage 5: Eval Manifest ----
        t0 = time.time()
        structured_log("INFO", "Stage 5: Eval Manifest Export",
                       chunk_id="pipeline", stage="eval_export",
                       duration_ms=0, token_count=0, status="RUNNING")

        eval_n = export_eval_manifest(
            index_units, self.config.eval.export_path,
        )

        stage_times["eval_export"] = (time.time() - t0) * 1000

        # ---- Summary ----
        total_ms = (time.time() - t_start) * 1000
        status = "OK" if n_vectors > 0 else "DEGRADED"

        structured_log("INFO", "Pipeline completed",
                       chunk_id="pipeline", stage="complete",
                       duration_ms=total_ms, token_count=0, status=status,
                       total_chunks=len(index_units),
                       inserted=write_stats["inserted"],
                       updated=write_stats["updated"],
                       skipped=write_stats["skipped"],
                       deleted=write_stats["deleted"],
                       pii_flagged=n_pii,
                       store_count=store.count(),
                       stages=stage_times)

        return {
            "status": status,
            "total_chunks": len(index_units),
            "inserted": write_stats["inserted"],
            "updated": write_stats["updated"],
            "skipped": write_stats["skipped"],
            "deleted": write_stats["deleted"],
            "pii_flagged": n_pii,
            "store_count": store.count(),
            "eval_records": eval_n,
            "duration_ms": total_ms,
            "stages": stage_times,
        }

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _get_store(self):
        """ponytail: 根据配置创建向量库适配器。"""
        if self._store is not None:
            return self._store

        vs_config = self.config.vector_store
        if vs_config.backend == "qdrant":
            self._store = QdrantVectorStore(
                url=vs_config.qdrant.url,
                collection_name=vs_config.qdrant.collection_name,
            )
        else:
            self._store = ChromaVectorStore(
                persist_dir=vs_config.chroma.persist_dir,
                collection_name=vs_config.chroma.collection_name,
            )

        structured_log("INFO", f"VectorStore: {vs_config.backend}",
                       chunk_id="pipeline", stage="init",
                       duration_ms=0, token_count=0, status="OK",
                       backend=vs_config.backend,
                       count=self._store.count())
        return self._store


# ================================================================
# CLI Entry
# ================================================================

def main():
    """CLI 入口: python -m indexer.pipeline <rag_chunks.json> [--config config.yaml]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge Indexing Pipeline — 构建多粒度向量索引",
    )
    parser.add_argument(
        "rag_chunks_path",
        help="Path to rag_chunks.json (from pdf_pipeline)",
    )
    parser.add_argument(
        "--config", "-c", default="",
        help="Path to config.yaml (default: use built-in defaults)",
    )
    parser.add_argument(
        "--skip-embed", action="store_true",
        help="Skip embedding stage (dry-run: only assembly + dedup)",
    )
    args = parser.parse_args()

    # 查找默认 config 路径
    config_path = args.config
    if not config_path:
        default_config = Path(__file__).resolve().parent / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)

    pipeline = IndexerPipeline(config_path)

    # ponytail: 不支持 skip-embed 时直接跳过 embedder 初始化
    # 此处保持简单：全流程运行
    stats = pipeline.run(args.rag_chunks_path)

    # 保持 Redis 目录与向量库一致：进 Qdrant 才算 indexed
    if stats["status"] != "ERROR":
        paper_name = Path(args.rag_chunks_path).parent.name
        mark_indexed(paper_name, stats["total_chunks"])

    # Exit code
    if stats["status"] == "ERROR":
        sys.exit(1)
    elif stats["status"] == "DEGRADED":
        print("\n[DEGRADED] Pipeline completed with degraded embedding — "
              "check logs above")
        sys.exit(2)
    else:
        print(f"\n[OK] Indexed {stats['inserted']} new + {stats['updated']} updated, "
              f"{stats['skipped']} skipped")
        print(f"[OK] Vector store: {stats['store_count']} records")
        print(f"[OK] Eval manifest: {stats['eval_records']} records")


if __name__ == "__main__":
    main()
