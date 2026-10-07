"""Privacy-safe historical cost observations for plan calibration."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import threading
from pathlib import Path
from typing import Any


_lock = threading.RLock()
_MAX_SAMPLES = 200


def cost_history_enabled() -> bool:
    return os.getenv("AGENT_COST_HISTORY", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def cost_history_path() -> Path:
    override = os.getenv("AGENT_COST_HISTORY_PATH", "")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parent.parent.parent
    return root / ".demo" / "cost_history.json"


def _default_payload() -> dict:
    return {"version": "1", "buckets": {}}


def _load() -> dict:
    path = cost_history_path()
    if not path.exists():
        return _default_payload()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return _default_payload()
    if not isinstance(payload, dict):
        return _default_payload()
    buckets = payload.get("buckets")
    payload["buckets"] = buckets if isinstance(buckets, dict) else {}
    return payload


def _save(payload: dict) -> None:
    path = cost_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def history_key(step: dict) -> str:
    return "|".join([
        str(step.get("target") or "auto"),
        str(step.get("required_scope") or "preview"),
        str(step.get("delivery") or "answer"),
    ])


def record_observation(
    key: str,
    *,
    tokens: float,
    seconds: float,
) -> None:
    if not cost_history_enabled():
        return
    token_value = max(0.0, float(tokens))
    second_value = max(0.0, float(seconds))
    if not key or token_value <= 0:
        return
    with _lock:
        payload = _load()
        bucket = payload["buckets"].setdefault(
            key, {"tokens": [], "seconds": []},
        )
        token_samples = list(bucket.get("tokens") or [])
        second_samples = list(bucket.get("seconds") or [])
        token_samples.append(token_value)
        second_samples.append(second_value)
        bucket["tokens"] = token_samples[-_MAX_SAMPLES:]
        bucket["seconds"] = second_samples[-_MAX_SAMPLES:]
        _save(payload)


def _scope_weight(step: dict) -> float:
    return {
        "preview": 1.0,
        "excerpt": 1.5,
        "section": 2.5,
        "full": 4.0,
    }.get(str(step.get("required_scope") or "preview"), 1.0)


def record_plan_cost(
    plan: list[dict],
    *,
    tokens_total: float | None,
    duration_ms: float | None,
) -> None:
    """Apportion one aggregate turn cost across plan steps without content."""
    steps = [step for step in (plan or []) if isinstance(step, dict)]
    if not steps or tokens_total is None or float(tokens_total) <= 0:
        return
    weights = [max(0.1, _scope_weight(step)) for step in steps]
    total_weight = sum(weights) or 1.0
    seconds_total = max(0.0, float(duration_ms or 0.0)) / 1000.0
    for step, weight in zip(steps, weights):
        share = weight / total_weight
        record_observation(
            history_key(step),
            tokens=float(tokens_total) * share,
            seconds=seconds_total * share,
        )


def cost_history_snapshot() -> dict[str, dict[str, Any]]:
    if not cost_history_enabled():
        return {}
    payload = _load()
    snapshot: dict[str, dict[str, Any]] = {}
    for key, bucket in payload["buckets"].items():
        tokens = [
            float(item) for item in (bucket.get("tokens") or [])
            if float(item) > 0
        ]
        seconds = [
            float(item) for item in (bucket.get("seconds") or [])
            if float(item) >= 0
        ]
        if not tokens:
            continue
        snapshot[str(key)] = {
            "samples": len(tokens),
            "tokens_p50": float(statistics.median(tokens)),
            "seconds_p50": (
                float(statistics.median(seconds)) if seconds else 0.0
            ),
        }
    return snapshot
