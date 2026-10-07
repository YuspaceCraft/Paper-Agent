"""Content-addressed storage for oversized execution results.

Artifacts are immutable and addressable by content hash.  Agent context can
carry a bounded preview plus an artifact reference, then read later pages only
when a task actually needs them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "artifacts"
DEFAULT_THRESHOLD_BYTES = int(os.getenv("AGENT_ARTIFACT_THRESHOLD", "4000"))
DEFAULT_PAGE_CHARS = int(os.getenv("AGENT_ARTIFACT_PAGE_CHARS", "6000"))
MAX_PAGE_CHARS = int(os.getenv("AGENT_ARTIFACT_MAX_PAGE_CHARS", "7000"))

_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{16,64}$")
_write_lock = Lock()


def artifact_root() -> Path:
    return Path(
        os.getenv("AGENT_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
    ).expanduser().resolve()


def _artifact_paths(artifact_id: str) -> tuple[Path, Path]:
    root = artifact_root()
    return (
        root / artifact_id[:2] / f"{artifact_id}.txt",
        root / artifact_id[:2] / f"{artifact_id}.json",
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def persist_text(
    content: str,
    *,
    tool_name: str = "",
    execution_id: str = "",
    thread_id: str = "",
    media_type: str = "application/json",
    threshold: int | None = None,
) -> dict[str, Any] | None:
    """Persist content above ``threshold`` and return its artifact reference."""
    raw = str(content or "")
    raw_bytes = raw.encode("utf-8")
    limit = DEFAULT_THRESHOLD_BYTES if threshold is None else max(0, int(threshold))
    if len(raw_bytes) <= limit:
        return None

    digest = hashlib.sha256(raw_bytes).hexdigest()
    artifact_id = digest[:32]
    content_path, metadata_path = _artifact_paths(artifact_id)
    reference = {
        "artifact_id": artifact_id,
        "sha256": digest,
        "bytes": len(raw_bytes),
        "chars": len(raw),
        "media_type": media_type,
        "tool_name": tool_name,
        "execution_id": execution_id,
        "thread_id": thread_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uri": str(content_path),
    }
    with _write_lock:
        if not content_path.exists():
            _atomic_write(content_path, raw)
        if not metadata_path.exists():
            _atomic_write(
                metadata_path,
                json.dumps(reference, ensure_ascii=False, indent=2),
            )
    return reference


def metadata(artifact_id: str) -> dict[str, Any] | None:
    """Read one artifact manifest, or ``None`` when it does not exist."""
    if not _ARTIFACT_ID_RE.fullmatch(str(artifact_id or "")):
        return None
    _content_path, metadata_path = _artifact_paths(str(artifact_id))
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_page(
    artifact_id: str,
    *,
    offset: int = 0,
    max_chars: int = DEFAULT_PAGE_CHARS,
) -> dict[str, Any] | None:
    """Read one deterministic character page from an artifact."""
    if not _ARTIFACT_ID_RE.fullmatch(str(artifact_id or "")):
        return None
    content_path, _metadata_path = _artifact_paths(str(artifact_id))
    try:
        full = content_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        start = max(0, int(offset))
    except (TypeError, ValueError):
        start = 0
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        limit = DEFAULT_PAGE_CHARS
    limit = max(1, min(limit, max(1, MAX_PAGE_CHARS)))
    if start > len(full):
        start = len(full)
    end = min(len(full), start + limit)
    return {
        "artifact_id": str(artifact_id),
        "offset": start,
        "end": end,
        "total_chars": len(full),
        "next_offset": None if end >= len(full) else end,
        "eof": end >= len(full),
        "text": full[start:end],
    }
