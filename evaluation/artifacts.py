"""Content-addressed artifact spill for oversized evaluation tool results.

The trace table should stay queryable and small.  When a tool returns a large
payload during an evaluation run, keep the complete payload in a local
content-addressed file and put only a path/hash summary into the trace.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "eval_output" / "tool_results"


def persist_if_large(tool: str, content: str, *, threshold: int = 4000) -> dict:
    """Best-effort spill. Returns trace-safe metadata, never raises."""
    try:
        raw = str(content or "")
        size = len(raw.encode("utf-8"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        meta = {
            "result_bytes": size,
            "result_sha256": digest[:16],
            "result_stored": False,
        }
        if size <= threshold:
            return meta

        from .trace_store import eval_ctx

        if not eval_ctx.get():
            return meta

        root = Path(os.getenv("AGENT_TOOL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))
        target = root / digest[:2] / f"{digest}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(raw, encoding="utf-8")
        try:
            ref = str(target.relative_to(PROJECT_ROOT))
        except ValueError:
            ref = str(target)
        meta.update({
            "result_stored": True,
            "result_ref": ref,
        })
        return meta
    except Exception:  # noqa: BLE001 — artifact spill must never block a tool
        return {}
