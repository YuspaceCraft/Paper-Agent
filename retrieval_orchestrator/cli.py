"""
cli.py — retrieval_orchestrator CLI entry point.

Usage:
  python -m retrieval_orchestrator evaluate --config retrieval_orchestrator/evaluation.yaml
  python -m retrieval_orchestrator generate --rag-path <path> --output <path>
  python -m retrieval_orchestrator review --manifest <path> [--corrections <path>]
  python -m retrieval_orchestrator search --query "<text>" [--top-k 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_generate(args):
    """Generate eval manifest from rag_chunks.json."""
    from .eval_dataset import generate_manifest

    rag_path = Path(args.rag_path)
    if not rag_path.exists():
        print(f"Error: {rag_path} not found")
        sys.exit(1)

    modes = args.modes.split(",") if args.modes else ["keyword"]
    generate_manifest(
        rag_path=rag_path,
        modes=modes,
        samples_per_mode=args.samples,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        output_path=args.output,
    )


def cmd_review(args):
    """Review and merge corrections into eval manifest."""
    from .eval_dataset import review_manifest

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"Error: {args.manifest} not found")
        sys.exit(1)

    review_manifest(
        manifest_path=manifest,
        corrections_path=args.corrections,
        output_path=args.output or args.manifest,
    )


def cmd_evaluate(args):
    """Full evaluation pipeline: generate manifest → run experiments → evaluate → select."""
    import yaml
    from datetime import datetime
    from .eval_dataset import generate_manifest
    from .retrieval_engine import run_experiments
    from .evaluator import evaluate
    from .config_selector import select_optimal, QualityGateError

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    eval_cfg = cfg.get("evaluation", {})
    base_dir = Path(eval_cfg.get("output_dir", "./eval_output"))

    # Create timestamped run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [EVAL] Run directory: {output_dir}")

    # Save a copy of the config used for this run
    import shutil
    shutil.copy2(config_path, output_dir / "evaluation.yaml")

    # Step 1: Load manifest (priority: CLI arg > yaml manifest > auto-generate)
    manifest_path = args.manifest or eval_cfg.get("manifest", "")
    if manifest_path:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            print(f"Error: manifest not found: {manifest_path}")
            sys.exit(1)
        print(f"  [EVAL] Using existing manifest: {manifest_path}")
    else:
        rag_path = eval_cfg.get("rag_path", "")
        if not rag_path:
            print("Error: evaluation.rag_path or --manifest required")
            sys.exit(1)
        manifest_path = output_dir / "eval_manifest.jsonl"
        dataset_cfg = eval_cfg.get("dataset", {})
        generate_manifest(
            rag_path=rag_path,
            modes=dataset_cfg.get("modes", ["keyword"]),
            samples_per_mode=dataset_cfg.get("samples_per_mode"),
            llm_model=dataset_cfg.get("llm_model", ""),
            llm_base_url=dataset_cfg.get("llm_base_url", ""),
            llm_api_key_env=dataset_cfg.get("llm_api_key_env", "DASHSCOPE_API_KEY"),
            output_path=manifest_path,
        )

    # Step 2: Run retrieval experiments
    print("\n--- Step 2: Retrieval Experiments ---")
    indexer_cfg = eval_cfg.get("indexer_config", "")
    experiments = run_experiments(
        manifest_path=str(manifest_path),
        rag_path=eval_cfg.get("rag_path", ""),
        config=cfg,
        indexer_config_path=indexer_cfg,
        output_dir=output_dir,
    )

    # Write experiments manifest
    exp_manifest_path = output_dir / "experiments.json"
    exp_manifest_path.write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Step 3: Evaluate each experiment
    print("\n--- Step 3: Evaluation ---")
    k_values = eval_cfg.get("k_values", [5, 10, 20])
    llm_judge = eval_cfg.get("llm_judge", {})
    thresholds = eval_cfg.get("thresholds", {"recall@10": 0.85, "mrr": 0.7})

    for exp in experiments:
        report = evaluate(
            manifest_path=str(manifest_path),
            results_path=exp["results_path"],
            k_values=k_values,
            llm_judge=llm_judge.get("enabled", False),
            llm_model=llm_judge.get("model", ""),
            llm_base_url=llm_judge.get("base_url", ""),
            llm_api_key_env=llm_judge.get("api_key_env", "DASHSCOPE_API_KEY"),
            chunks_path=eval_cfg.get("rag_path", ""),
        )
        exp["evaluation_report"] = report

        # Write individual report
        report_path = Path(exp["results_path"]).with_suffix(".report.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{exp['experiment_id']}] Recall@10={report['overall_metrics'].get('recall@10', 0):.4f}  MRR={report['overall_metrics'].get('mrr', 0):.4f}")

    # Write full experiments with reports
    exp_manifest_path.write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Step 4: Select optimal config
    print("\n--- Step 4: Optimal Config Selection ---")
    try:
        best = select_optimal(
            experiment_reports=experiments,
            thresholds=thresholds,
            historical_best_path=eval_cfg.get("historical_best"),
            output_path=output_dir / "optimal_retrieval_config.yaml",
        )
        if best:
            print(f"\n✅ Optimal config: {best['experiment_id']}")
            r5 = best["evaluation_report"]["overall_metrics"].get("recall@5", 0)
            r10 = best["evaluation_report"]["overall_metrics"].get("recall@10", 0)
            mrr = best["evaluation_report"]["overall_metrics"].get("mrr", 0)
            print(f"   Recall@5={r5:.4f}  Recall@10={r10:.4f}  MRR={mrr:.4f}")
        else:
            print("\n❌ No configuration met all thresholds.")
            best = None
    except QualityGateError as e:
        print(f"\n🚫 QUALITY GATE FAILED: {e}")
        best = None

    # Write run summary (for cross-run comparison)
    top_metrics = sorted(
        [{"experiment_id": e["experiment_id"],
          "recall@5": e.get("evaluation_report", {}).get("overall_metrics", {}).get("recall@5", 0),
          "recall@10": e.get("evaluation_report", {}).get("overall_metrics", {}).get("recall@10", 0),
          "mrr": e.get("evaluation_report", {}).get("overall_metrics", {}).get("mrr", 0),
          "ndcg@10": e.get("evaluation_report", {}).get("overall_metrics", {}).get("ndcg@10", 0)}
         for e in experiments],
        key=lambda x: (-x["recall@10"], -x["mrr"]),
    )[:5]

    summary = {
        "run_ts": ts,
        "manifest": str(manifest_path),
        "config": str(config_path),
        "num_queries": sum(1 for _ in open(manifest_path, encoding="utf-8")) if manifest_path.exists() else 0,
        "thresholds": thresholds,
        "quality_gate": {"min_recall_at_5": 0.6, "passed": best is not None},
        "optimal": best["experiment_id"] if best else None,
        "top5_strategies": top_metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n  [EVAL] Run summary → {output_dir / 'run_summary.json'}")


def cmd_search(args):
    """Quick search test: embed query → search vector store."""
    from indexer.config import load_config
    from indexer.vector_store import ChromaVectorStore, QdrantVectorStore
    from indexer.embedding_adapters import create_embedding_adapter

    cfg = load_config(args.indexer_config or "")
    vs = cfg.vector_store
    if vs.backend == "qdrant":
        store = QdrantVectorStore(url=vs.qdrant.url, collection_name=vs.qdrant.collection_name)
    else:
        store = ChromaVectorStore(persist_dir=vs.chroma.persist_dir, collection_name=vs.chroma.collection_name)

    embedder = create_embedding_adapter(cfg.embedding)
    vec = embedder.embed_single(args.query)
    if vec is None:
        print("Error: embedding failed")
        sys.exit(1)

    results = store.search(vec, top_k=args.top_k)
    for i, r in enumerate(results):
        cid = r.get("chunk_id", "?")
        score = r.get("score", 0)
        doc = r.get("document", "")[:200]
        print(f"{i+1}. [{cid}] score={score:.4f}\n   {doc}\n")


def main():
    parser = argparse.ArgumentParser(
        description="retrieval_orchestrator — Offline retrieval evaluation framework",
    )
    sub = parser.add_subparsers(dest="command")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run full evaluation pipeline")
    p_eval.add_argument("--config", required=True, help="Path to evaluation.yaml")
    p_eval.add_argument("--manifest", default="", help="Existing manifest (skip generation)")

    # generate
    p_gen = sub.add_parser("generate", help="Generate eval manifest from rag_chunks.json")
    p_gen.add_argument("--rag-path", required=True, help="Path to rag_chunks.json")
    p_gen.add_argument("--output", default="./eval_output/eval_manifest.jsonl")
    p_gen.add_argument("--modes", default="keyword", help="Comma-separated: keyword,semantic,cross_chunk")
    p_gen.add_argument("--samples", type=int, default=None, help="Max samples per mode")
    p_gen.add_argument("--llm-model", default="kimi-k2.6")
    p_gen.add_argument("--llm-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p_gen.add_argument("--llm-api-key-env", default="DASHSCOPE_API_KEY")

    # review
    p_rev = sub.add_parser("review", help="Review and correct eval manifest")
    p_rev.add_argument("--manifest", required=True)
    p_rev.add_argument("--corrections", default="")
    p_rev.add_argument("--output", default="")

    # search
    p_search = sub.add_parser("search", help="Quick vector search test")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--indexer-config", default="./indexer/config.yaml")

    args = parser.parse_args()
    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
