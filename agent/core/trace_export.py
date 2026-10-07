"""Export a LangSmith run tree into the offline evaluation archive (ADR-0002).

Online diagnosis reads LangSmith; offline evaluation must not depend on the
retention window. After a batch (or for one interesting turn) this writes the
run tree to ``eval_output/runs/<run_id>/langsmith_runs.jsonl`` — one run per
line with join keys and metadata, so a report stays reproducible after the
trace ages out.

Privacy: only redacted metadata and identifiers are exported by default.
``AGENT_TRACE_EXPORT_FULL=1`` additionally keeps raw inputs/outputs and is
meant for an approved development environment.

Usage:
    python -m agent.core.trace_export <trace_id|run_id> [--project paper-agent]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "eval_output" / "runs"
_EXPORTED_FIELDS = (
    "id", "trace_id", "parent_run_id", "name", "run_type",
    "start_time", "end_time", "error", "session_id",
)


def export_full_content() -> bool:
    return os.getenv("AGENT_TRACE_EXPORT_FULL", "") not in {"", "0", "false"}


def _slug(value: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return str(value or "")


def run_dir_for(run_id: str, runs_dir: Path | str | None = None) -> Path:
    return Path(runs_dir or DEFAULT_RUNS_DIR) / str(run_id)


def _call_with_deadline(fn, deadline_seconds: float, what: str) -> Any:
    """Run a blocking SDK call in a daemon thread with a hard deadline.

    The LangSmith SDK has its own retry/timeout policy, which can outlive any
    caller-imposed budget when the endpoint is unreachable. A daemon thread can
    be abandoned safely, so a hanging HTTP call can never block the evaluation
    run or the process exit.
    """
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — reported to the caller
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True, name=f"trace-export-{what}")
    thread.start()
    thread.join(max(0.1, float(deadline_seconds)))
    if thread.is_alive():
        raise TimeoutError(f"timed out after {deadline_seconds:g}s calling {what}")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _build_client(deadline_seconds: float):
    """Create the SDK client under the same deadline.

    ``langsmith.Client()`` performs its own session handshake, so an
    unreachable endpoint must not be allowed to block the caller either.
    """
    import langsmith

    return _call_with_deadline(langsmith.Client, deadline_seconds, "client")


def read_root_run(client: Any, run_id: str):
    """Fetch one run, tolerating the SDK rename ``get_run`` → ``read_run``.

    langsmith >= 0.8 exposes ``read_run``; older releases expose ``get_run``.
    Supporting both keeps the exporter usable across environments.
    """
    for name in ("read_run", "get_run"):
        reader = getattr(client, name, None)
        if callable(reader):
            return reader(run_id)
    raise AttributeError("langsmith client has neither read_run nor get_run")


async def lookup_langsmith_run_id(trace_id: str) -> str:
    """Resolve the LangSmith root run id recorded for a local trace id."""
    try:
        from evaluation.trace_store import get_trace_store

        store = get_trace_store()
        rows = await store.query(trace_id=trace_id, limit=1)
    except Exception:  # noqa: BLE001 — export is best-effort by design
        return ""
    for row in rows or []:
        if row.get("run_id"):
            return str(row["run_id"])
    return ""


def _row(run: Any, *, include_io: bool) -> dict:
    row: dict[str, Any] = {}
    for field in _EXPORTED_FIELDS:
        value = getattr(run, field, None)
        row[field] = value.isoformat() if hasattr(value, "isoformat") else value
    metadata = {}
    extra = getattr(run, "extra", None)
    if isinstance(extra, dict):
        metadata = extra.get("metadata") or {}
    row["metadata"] = metadata
    if include_io:
        row["inputs"] = getattr(run, "inputs", None)
        row["outputs"] = getattr(run, "outputs", None)
    return row


def export_run_tree(
    trace_id: str,
    *,
    run_id: str | None = None,
    project_name: str | None = None,
    runs_dir: Path | str | None = None,
    client: Any = None,
    include_io: bool | None = None,
    wait_seconds: float = 20.0,
) -> Path:
    """Write the run tree for ``trace_id``; returns the JSONL path.

    Raises the underlying error when nothing could be exported — callers in the
    evaluation path wrap this so a missing archive never fails a run.
    """
    import time

    project = project_name or os.getenv("LANGSMITH_PROJECT", "paper-agent")
    budget = max(10.0, wait_seconds)
    client = client or _build_client(budget)
    include_io = export_full_content() if include_io is None else include_io

    resolved = _slug(run_id or "") or _slug(trace_id)
    root = None
    # At least one bounded attempt even when the caller asked for no waiting.
    deadline = time.monotonic() + budget
    while True:
        remaining = deadline - time.monotonic()
        try:
            root = _call_with_deadline(
                lambda: read_root_run(client, resolved), remaining, "read_run",
            )
            break
        except Exception:  # noqa: BLE001 — SDK indexing lags behind the call
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)

    spans = list(_call_with_deadline(
        lambda: list(client.list_runs(
            trace_id=_slug(str(root.trace_id or root.id)))),
        60.0, "list_runs",
    ))
    out_dir = run_dir_for(trace_id, runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "langsmith_runs.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for span in spans:
            handle.write(json.dumps(_row(span, include_io=include_io),
                                    ensure_ascii=False, default=str) + "\n")
    manifest = {
        "trace_id": trace_id,
        "langsmith_run_id": str(getattr(root, "id", "")),
        "project": project,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_count": len(spans),
        "full_content": include_io,
        "file": target.name,
    }
    (out_dir / "langsmith_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return target


async def export_for_trace(trace_id: str, **kwargs: Any) -> Path:
    """Async helper: resolve the run id from the local trace, then export."""
    run_id = kwargs.pop("run_id", "") or await lookup_langsmith_run_id(trace_id)
    return export_run_tree(trace_id, run_id=run_id or None, **kwargs)


def export_best_effort(trace_id: str, **kwargs: Any) -> Path | None:
    """Never raise: used from evaluation runs where export is a nice-to-have."""
    from ..observability import log_event

    try:
        return asyncio.run(export_for_trace(trace_id, **kwargs))
    except Exception as exc:  # noqa: BLE001
        log_event("langsmith_export_failed", node="trace_export", level="warning",
                  trace_id=trace_id, error=f"{type(exc).__name__}: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    """One-shot CLI. Exits via ``_halt`` to bypass asyncio teardown.

    The trace store's aiosqlite worker thread is not a daemon thread, so a
    one-shot CLI would hang at interpreter exit unless the process leaves
    immediately after flushing its result (see TROUBLESHOOTING).
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="导出 LangSmith run 树到评测归档")
    parser.add_argument("trace_id", help="本地 trace_id（线上路径等于 LangSmith run id）")
    parser.add_argument("--run-id", default="", help="显式指定 LangSmith 根 run id")
    parser.add_argument("--project", default="", help="LangSmith project（默认取环境变量）")
    parser.add_argument("--wait", type=float, default=20.0,
                        help="等待 SDK 索引根 run 的秒数")
    args = parser.parse_args(argv)
    try:
        path = asyncio.run(export_for_trace(
            args.trace_id, run_id=args.run_id or None,
            project_name=args.project or None, wait_seconds=args.wait,
        ))
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 导出失败：{type(exc).__name__}: {exc}")
        print("       检查 LANGSMITH_API_KEY / 网络可达性，或该 trace 是否已过保留期。")
        return _halt(1)
    print(f"[OK] 已导出：{path}")
    return _halt(0)


def _halt(code: int) -> int:
    """Flush output and leave the process without running async finalizers."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
