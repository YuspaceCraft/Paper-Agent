"""
tool_contract.py — 统一工具输出信封契约（INFO_FLOW_REVIEW P6 收敛）。

历史问题：库工具（builtin / creation / coding）成功与失败都返回 JSON envelope
（`{"ok": true/false, ...}`），而通用文件/数学工具（read_file、list_dir、
get_time、calculator、fetch_url）成功路径返回纯文本。于是 _salvage_tool_content、
_classify_tool_error、plan._ingest_guard 各自维护一份「双格式」试探解析 ——
模型上下文里同一批工具结果的形态不稳定，错误恢复字段名也各处硬编码。

本模块成为唯一权威：
- `success()` / `failure()` 生成统一 envelope（所有 provider 共用）；
- `parse_tool_result()` 唯一解析入口 —— 先试 envelope，非 envelope 一律按纯文本分流；
- `truncate_tool_result()` 截断时尽量保持 envelope 可解析（原理：字符级截断把
  JSON 切断后整个 envelope 作废并回退成纯文本，fetch_content 的章节内容会丢）。

格式契约（同步各 provider 模块 docstring）：
- 结构化 / 库 / 错误 → JSON envelope
    {"schema_version":"1.0","outcome":"succeeded","data":{...}}
    {"schema_version":"1.0","outcome":"failed",
     "error":"...","error_type":"...","code":"...","next":"..."}
- 纯文本工具（read_file / list_dir / get_time / calculator / fetch_url 成功路径）
  → 保证「不是合法 envelope」的 UTF-8 文本；parse_tool_result 据此确定性分流。
  非法 envelope（旧 `ok` 字段、矛盾 outcome、截断 JSON）按协议错误处理。

名称说明：`next` 只作 dict key，与内置函数名无冲突。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---- 生成端：统一信封 ----

def success(data: Any = None, *, meta: dict | None = None,
            warnings: list[dict] | None = None) -> str:
    """成功 envelope。data 为结构化负载（list/dict）；纯文本工具勿用本函数。"""
    payload = {
        "schema_version": "1.0",
        "outcome": "succeeded",
        "data": data,
    }
    if meta:
        payload["meta"] = meta
    if warnings:
        payload["warnings"] = warnings
    return json.dumps(payload, ensure_ascii=False, indent=2)


def failure(error_type: str = "unknown", detail: str = "",
            next_action: str = "", **ctx) -> str:
    """失败 envelope。ctx 可携带附加字段（available_papers / available_sections 等）。"""
    outcome = str(ctx.pop("outcome", "") or _outcome_for_error_type(error_type))
    payload: dict = {
        "schema_version": "1.0",
        "outcome": outcome,
        "error": detail,
        "next": next_action,
        "error_type": error_type,
    }
    payload.update(ctx)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _outcome_for_error_type(error_type: str) -> str:
    token = (error_type or "").strip().lower()
    if token in {"timeout", "tool_timeout"}:
        return "timed_out"
    if token in {"cancelled", "canceled", "tool_cancelled"}:
        return "cancelled"
    if token in {"interrupted", "approval_required"}:
        return "interrupted"
    return "failed"


# ---- 解析端：唯一入口 ----

@dataclass
class ToolResult:
    """解析后的统一视图。is_envelope 为 False ⇒ 纯文本成功结果（text 生效）。"""
    is_envelope: bool = False
    schema_version: str = ""
    outcome: str = "succeeded"
    data: Any = None               # 结构化负载（envelope 成功 / 失败时的 data 字段）
    text: str = ""                 # 纯文本结果（非 envelope）
    error: str = ""                # 失败详情
    error_type: str = ""           # param_error / transient / not_found / ...
    code: str = ""                 # 稳定机器码
    next_action: str = ""          # 建议的恢复动作
    retryable: bool | None = None
    retry_after_seconds: float | None = None
    meta: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    continuation: dict = field(default_factory=dict)
    protocol_error: str = ""       # envelope 格式非法时为非空
    extra: dict = field(default_factory=dict)  # envelope 的其余字段

    def to_operation_result(
        self,
        *,
        kind,
        operation_id: str,
        parent_operation_id: str | None = None,
        meta: dict | None = None,
    ):
        """Convert one parsed tool result into the shared operation contract."""
        from .core.contracts import (
            ErrorType,
            OperationKind,
            OperationOutcome,
            OperationResult,
            ResultMeta,
            WarningInfo,
        )
        from .core.errors import tool_envelope_error

        if isinstance(kind, str):
            kind = OperationKind(kind)
        merged_meta = {**self.meta, **(meta or {})}
        status = _operation_outcome(self)
        error = None
        if status in {
            OperationOutcome.FAILED,
            OperationOutcome.TIMED_OUT,
            OperationOutcome.CANCELLED,
        }:
            error = tool_envelope_error(
                str(merged_meta.get("tool_name") or ""),
                error_type=self.error_type,
                message=self.error or "工具执行失败。",
                recovery_action=self.next_action,
                retryable=bool(self.retryable),
            )
            error = error.model_copy(update={
                "code": self.code or error.code,
                "message": (
                    self.protocol_error or self.error
                    or "tool execution failed"
                ),
                "retry_after_seconds": self.retry_after_seconds,
            })
            if self.protocol_error:
                error = error.model_copy(update={
                    "error_type": ErrorType.PROTOCOL,
                    "code": "TOOL_PROTOCOL_INVALID",
                })
        data = self.data if self.is_envelope else (self.text or None)
        warnings = [
            warning if isinstance(warning, WarningInfo)
            else WarningInfo(**warning)
            for warning in self.warnings
            if isinstance(warning, (dict, WarningInfo))
        ]
        if status is OperationOutcome.PARTIAL and not warnings:
            warnings = [WarningInfo(
                code="PARTIAL_RESULT",
                message="Tool returned a partial result without warnings.",
            )]
        return OperationResult(
            kind=kind,
            operation_id=operation_id,
            parent_operation_id=parent_operation_id,
            outcome=status,
            data=data,
            error=error,
            warnings=warnings,
            meta=ResultMeta(**merged_meta),
        )


def _operation_outcome(result: ToolResult):
    from .core.contracts import OperationOutcome

    if result.protocol_error:
        return OperationOutcome.FAILED
    if result.outcome:
        try:
            return OperationOutcome(result.outcome)
        except ValueError:
            return OperationOutcome.FAILED
    return OperationOutcome.FAILED


def parse_tool_result(content) -> ToolResult:
    """唯一解析入口：先试 JSON envelope，失败按纯文本成功结果处理。"""
    raw = str(content)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        data = None

    if isinstance(data, dict) and "outcome" in data:
        if "ok" in data:
            return _protocol_error(
                raw, "legacy field 'ok' is not accepted; use outcome"
            )
        outcome = str(data.get("outcome") or "").strip().lower()
        valid_outcomes = {
            "succeeded", "partial", "failed", "timed_out",
            "cancelled", "interrupted", "skipped",
        }
        if outcome not in valid_outcomes:
            return _protocol_error(
                raw, f"unknown envelope outcome: {outcome!r}"
            )
        raw_retryable = data.get("retryable")
        if raw_retryable is not None and type(raw_retryable) is not bool:
            return _protocol_error(
                raw, "envelope field 'retryable' must be a JSON boolean"
            )
        r = ToolResult(
            is_envelope=True,
            schema_version=str(data.get("schema_version") or ""),
            outcome=outcome,
        )
        r.data = data.get("data")
        r.code = str(data.get("code", ""))
        r.retryable = raw_retryable
        try:
            r.retry_after_seconds = (
                float(data["retry_after_seconds"])
                if data.get("retry_after_seconds") is not None else None
            )
        except (TypeError, ValueError):
            return _protocol_error(
                raw, "envelope field 'retry_after_seconds' must be numeric"
            )
        if outcome in {"failed", "timed_out", "cancelled"}:
            r.error = str(data.get("error", ""))
            r.error_type = str(data.get("error_type", ""))
            r.next_action = str(data.get("next", ""))
        meta = data.get("meta")
        warnings = data.get("warnings")
        artifacts = data.get("artifacts")
        continuation = data.get("continuation")
        r.meta = meta if isinstance(meta, dict) else {}
        r.warnings = warnings if isinstance(warnings, list) else []
        r.artifacts = artifacts if isinstance(artifacts, list) else []
        r.continuation = continuation if isinstance(continuation, dict) else {}
        r.extra = {k: v for k, v in data.items()
                   if k not in (
                       "schema_version", "ok", "outcome", "data", "error",
                       "error_type", "next", "code", "retryable",
                       "retry_after_seconds", "meta", "warnings", "artifacts",
                       "continuation",
                   )}
        return r
    if isinstance(data, dict) and "ok" in data:
        return _protocol_error(
            raw, "legacy tool envelope with 'ok' is no longer supported"
        )
    if data is None and _looks_like_envelope(raw):
        return _protocol_error(raw, "malformed JSON tool envelope")
    # 非 envelope（合法 JSON 但没有顶层 outcome）→ 纯文本
    return ToolResult(is_envelope=False, outcome="succeeded", text=raw)


def _protocol_error(raw: str, detail: str) -> ToolResult:
    return ToolResult(
        is_envelope=True,
        outcome="failed",
        text=raw,
        error=detail,
        error_type="protocol_error",
        code="TOOL_PROTOCOL_INVALID",
        retryable=False,
        protocol_error=detail,
    )


def _looks_like_envelope(raw: str) -> bool:
    head = raw.lstrip()
    return head.startswith("{") and bool(
        re.search(
            r'["\']?(?:ok|outcome)["\']?\s*:',
            head[:512], re.IGNORECASE,
        )
    )


def operation_result_to_envelope(result) -> str:
    """Serialize the shared operation result through the tool wire contract."""
    outcome = result.outcome.value
    payload: dict[str, Any] = {
        "schema_version": result.schema_version,
        "outcome": outcome,
        "operation_id": result.operation_id,
        "kind": result.kind.value,
        "meta": result.meta.model_dump(mode="json"),
    }
    if result.parent_operation_id:
        payload["parent_operation_id"] = result.parent_operation_id
    if result.data is not None:
        payload["data"] = result.data
    if result.warnings:
        payload["warnings"] = [
            warning.model_dump(mode="json") for warning in result.warnings
        ]
    if result.error is not None:
        payload.update({
            "error": result.error.user_message,
            "error_type": result.error.error_type.value,
            "code": result.error.code,
            "next": result.error.recovery_action or "",
            "retryable": result.error.retryable,
            "retry_after_seconds": result.error.retry_after_seconds,
            "cause_ref": result.error.cause_ref,
            "tool_name": result.error.tool_name,
            "effect_applied": result.error.effect_applied,
        })
    return json.dumps(payload, ensure_ascii=False, indent=2)


def attach_artifact(content: str, artifact: dict) -> str:
    """Attach an artifact reference before ``data`` for truncation survival."""
    try:
        payload = json.loads(str(content))
    except (TypeError, ValueError):
        return str(content)
    if not isinstance(payload, dict):
        return str(content)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    artifact_id = str(artifact.get("artifact_id") or "")
    if artifact_id and not any(
        isinstance(item, dict) and item.get("artifact_id") == artifact_id
        for item in artifacts
    ):
        artifacts.append(artifact)
    payload["artifacts"] = artifacts

    preferred = (
        "schema_version", "outcome", "operation_id", "kind",
        "parent_operation_id", "error", "error_type", "code", "next",
        "retryable", "retry_after_seconds", "meta", "artifacts",
        "continuation", "data", "warnings",
    )
    ordered = {key: payload[key] for key in preferred if key in payload}
    ordered.update({key: value for key, value in payload.items()
                    if key not in ordered})
    return json.dumps(ordered, ensure_ascii=False, indent=2)


def paginate_tool_result(
    content: str, *, offset: int = 0, max_chars: int = 6000,
) -> str:
    """Project a successful string payload into one requested character page."""
    try:
        payload = json.loads(str(content))
    except (TypeError, ValueError):
        return str(content)
    if not isinstance(payload, dict) or payload.get("outcome") not in {
        "succeeded", "partial",
    }:
        return str(content)
    full = payload.get("data")
    if not isinstance(full, str):
        return str(content)
    try:
        start = max(0, int(offset))
    except (TypeError, ValueError):
        start = 0
    try:
        limit = max(1, min(int(max_chars), 7000))
    except (TypeError, ValueError):
        limit = 6000
    if start == 0 and len(full) <= limit:
        return str(content)
    if start > len(full):
        start = len(full)
    end = min(len(full), start + limit)
    payload["data"] = full[start:end]
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["truncated"] = end < len(full) or start > 0
    meta["original_chars"] = meta.get("original_chars") or len(full)
    payload["meta"] = meta
    artifacts = payload.get("artifacts")
    artifact_id = ""
    if isinstance(artifacts, list) and artifacts and isinstance(artifacts[0], dict):
        artifact_id = str(artifacts[0].get("artifact_id") or "")
    continuation = {
        "reason": "requested_page",
        "payload_chars": len(full),
        "shown_chars": end - start,
        "next_offset": None if end >= len(full) else end,
        "eof": end >= len(full),
    }
    if artifact_id:
        continuation["artifact_id"] = artifact_id
    payload["continuation"] = continuation
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---- 截断（对 envelope 保解析性） ----

def truncate_tool_result(text: str, limit: int) -> str:
    """Bound a tool result without destroying the useful payload.

    Structured envelopes keep their correlation fields and a bounded preview of
    ``data``.  A continuation block records the original size and the next
    payload offset so callers can fetch the remainder when the tool supports
    pagination.  Plain text keeps its prefix plus a visible truncation marker.
    """
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict) and ("ok" in data or "outcome" in data):
        return _truncate_envelope(data, len(text), limit)
    marker = f"\n…[truncated: result > {limit} chars]"
    if limit <= len(marker):
        return text[:limit]
    return text[:limit - len(marker)] + marker


def _truncate_envelope(data: dict, total: int, limit: int) -> str:
    """Return a valid envelope with a content preview and continuation cursor."""
    payload = data.get("data")
    base = {key: value for key, value in data.items() if key != "data"}
    meta = dict(base.get("meta") or {}) if isinstance(base.get("meta"), dict) else {}
    meta.update({"truncated": True, "original_chars": total})
    base["meta"] = meta

    if isinstance(payload, str):
        payload_len = len(payload)

        def render(shown: int, minimal: bool = False) -> str:
            if minimal:
                envelope = {
                    "schema_version": data.get("schema_version", "1.0"),
                    "outcome": data.get("outcome", "succeeded"),
                }
            else:
                envelope = dict(base)
            envelope["data"] = payload[:shown]
            continuation = {
                "reason": "output_too_large",
                "total_chars": total,
                "payload_chars": payload_len,
                "shown_chars": shown,
                "next_offset": shown,
                "next_max_chars": max(1, limit - 512),
            }
            artifacts = envelope.get("artifacts")
            if (
                isinstance(artifacts, list)
                and artifacts
                and isinstance(artifacts[0], dict)
            ):
                continuation["artifact_id"] = str(
                    artifacts[0].get("artifact_id") or ""
                )
            envelope["continuation"] = continuation
            return json.dumps(
                envelope, ensure_ascii=False, separators=(",", ":"),
            )

        return _fit_envelope_preview(render, payload_len, limit)

    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_len = len(rendered)
    original_type = (
        "array" if isinstance(payload, list)
        else "object" if isinstance(payload, dict)
        else type(payload).__name__
    )

    def render(shown: int, minimal: bool = False) -> str:
        if minimal:
            envelope = {
                "schema_version": data.get("schema_version", "1.0"),
                "outcome": data.get("outcome", "succeeded"),
            }
        else:
            envelope = dict(base)
        envelope["data"] = {
            "_truncated": True,
            "original_type": original_type,
            "preview": rendered[:shown],
        }
        continuation = {
            "reason": "output_too_large",
            "total_chars": total,
            "payload_chars": payload_len,
            "shown_chars": shown,
        }
        artifacts = envelope.get("artifacts")
        if (
            isinstance(artifacts, list)
            and artifacts
            and isinstance(artifacts[0], dict)
        ):
            continuation["artifact_id"] = str(
                artifacts[0].get("artifact_id") or ""
            )
        envelope["continuation"] = continuation
        return json.dumps(
            envelope, ensure_ascii=False, separators=(",", ":"),
        )

    return _fit_envelope_preview(render, payload_len, limit)


def _fit_envelope_preview(render, payload_len: int, limit: int) -> str:
    """Maximize a payload prefix while keeping serialized JSON within limit."""
    if limit <= 0:
        return ""

    def best_for(minimal: bool) -> tuple[int, str]:
        low, high = 0, payload_len
        best_n, best_text = 0, render(0, minimal)
        while low <= high:
            mid = (low + high) // 2
            candidate = render(mid, minimal)
            if len(candidate) <= limit:
                best_n, best_text = mid, candidate
                low = mid + 1
            else:
                high = mid - 1
        return best_n, best_text

    shown, rendered = best_for(minimal=False)
    if shown > 0 or len(rendered) <= limit:
        return rendered
    # Huge source metadata can leave no room for content. Keep the minimal
    # valid envelope; the payload preview is maximized again against it.
    _shown, rendered = best_for(minimal=True)
    if len(rendered) <= limit:
        return rendered
    minimal = json.dumps({
        "outcome": "succeeded",
        "data": "",
        "continuation": {"reason": "output_too_large", "total_chars": payload_len},
    }, ensure_ascii=False, separators=(",", ":"))
    if len(minimal) <= limit:
        return minimal
    marker = f"[result truncated: {payload_len} chars]"
    return marker[:limit]
