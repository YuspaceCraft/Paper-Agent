"""
web CLI — dev helper commands
=============================

Usage:
  python -m web.cli status    # Show current state
  python -m web.cli reconcile # Reconcile catalog indexed flag vs vector store
  python -m web.cli reset     # Reset all state (Qdrant + Redis + local files)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import redis as _redis_lib

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "pdf_pipeline" / "output"
UPLOAD_DIR = ROOT / "data" / "uploads"
REGISTRY_PATH = ROOT / "eval_output" / "paper_registry.json"
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "rag_chunks"


def _redis():
    try:
        r = _redis_lib.Redis(host="127.0.0.1", port=6379, protocol=2, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


def _qdrant_info():
    try:
        from qdrant_client import QdrantClient
        c = QdrantClient("localhost", port=6333, timeout=5)
        cols = {col.name for col in c.get_collections().collections}
        info = c.get_collection(COLLECTION) if COLLECTION in cols else None
        return c, cols, info
    except Exception:
        return None, set(), None


def cmd_status():
    """Print current state of all stores."""
    print("=" * 50)
    print("  System Status")
    print("=" * 50)

    # Redis
    r = _redis()
    if r:
        n = r.dbsize()
        tasks = r.keys("task:*")
        print(f"\nRedis ({REDIS_URL}): {n} keys")
        if tasks:
            for t in sorted(tasks):
                t_str = t.decode() if isinstance(t, bytes) else t
                if t_str == "task:list":
                    continue
                status = r.hget(t_str, "status")
                print(f"  {t_str}: status={status}")
    else:
        print(f"\nRedis ({REDIS_URL}): unavailable")

    # Qdrant
    c, cols, info = _qdrant_info()
    if c:
        print(f"\nQdrant ({QDRANT_URL}): {len(cols)} collections")
        if info:
            vsize = info.config.params.vectors.size if info.config.params.vectors else "?"
            print(f"  {COLLECTION}: {info.points_count} points, vectors={vsize}d")
        else:
            print(f"  {COLLECTION}: not found")
    else:
        print(f"\nQdrant ({QDRANT_URL}): unavailable")

    # Local files
    print(f"\nOutputs ({OUTPUT_DIR}):")
    if OUTPUT_DIR.is_dir():
        entries = sorted(OUTPUT_DIR.iterdir())
        if entries:
            for d in entries:
                if d.is_dir():
                    files = list(d.iterdir())
                    print(f"  {d.name}/ ({len(files)} files)")
                else:
                    print(f"  {d.name}")
        else:
            print("  (empty)")
    else:
        print("  (empty)")

    print(f"\nUploads ({UPLOAD_DIR}):")
    if UPLOAD_DIR.is_dir():
        entries = sorted(UPLOAD_DIR.iterdir())
        if entries:
            for f in entries:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f}MB)")
        else:
            print("  (empty)")
    else:
        print("  (empty)")
    print()


def cmd_reset(force: bool = False):
    """Clear all state: Redis + Qdrant + local outputs + uploads."""
    if not force:
        resp = input("Reset ALL state? This clears Redis, Qdrant, and local files. [y/N]: ")
        if resp.strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return

    print("Clearing...")

    # 1. Redis: flush all keys
    r = _redis()
    if r:
        n = r.dbsize()
        r.flushdb()
        print(f"  Redis: cleared {n} keys")
    else:
        print("  Redis: unavailable, skip")

    # 2. Qdrant: delete + recreate collection
    c, cols, info = _qdrant_info()
    if c and COLLECTION in cols:
        points = info.points_count if info else 0
        c.delete_collection(COLLECTION)
        from qdrant_client.models import VectorParams, Distance
        c.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"  Qdrant: recreated {COLLECTION} (cleared {points} points)")
    elif c:
        print(f"  Qdrant: {COLLECTION} not found, skip")
    else:
        print("  Qdrant: unavailable, skip")

    # 3. Local outputs + uploads
    for d in [OUTPUT_DIR, UPLOAD_DIR]:
        if d.is_dir():
            count = 0
            for item in d.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                count += 1
            print(f"  {d}: deleted {count} items")
        else:
            d.mkdir(parents=True, exist_ok=True)

    # 4. Paper registry: clean JSON cold backup
    if REGISTRY_PATH.exists():
        import json as _json
        reg = _json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        n_before = len(reg.get("papers", {}))
        reg["papers"] = {}
        reg["by_hash"] = {}
        reg["by_doi"] = {}
        REGISTRY_PATH.write_text(_json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {REGISTRY_PATH}: cleared {n_before} paper entries")

    print("\n[DONE] Reset complete.")
    print("Also clear browser localStorage: F12 > Application > Local Storage > delete demo_* keys.")


def cmd_reconcile():
    """Reconcile catalog `indexed` flag against the vector store (backfill)."""
    from indexer.reconcile import reconcile

    print("Reconciling catalog ↔ vector store...")
    report = reconcile()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="web CLI — dev helpers")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Show current store state")
    sub.add_parser("reconcile", help="Reconcile catalog indexed flag vs vector store")
    p_reset = sub.add_parser("reset", help="Reset all state")
    p_reset.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.cmd == "reset":
        cmd_reset(force=args.force)
    elif args.cmd == "reconcile":
        cmd_reconcile()
    elif args.cmd == "status":
        cmd_status()
    else:
        cmd_status()


if __name__ == "__main__":
    main()
