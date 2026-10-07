"""The single entry point every tool call must pass through (ADR-0003).

``ToolGateway.invoke`` runs the fixed sequence from the platform design:

1. resolve the tool version and validate its input against the spec,
2. ask :mod:`agent.core.policy` for allow / deny / approval,
3. derive the idempotency key for the call,
4. apply concurrency, timeout, circuit-breaker and retry limits,
5. call the adapter and normalise the result into the existing envelope,
6. record a redacted audit event,
7. return the result, its error and the retry decision to the caller.

Adapters keep their current shape (``call_fn(name, args) -> Any``), so this is
an additive layer rather than a rewrite of every provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..observability import log_event
from .approval import (
    NO_APPROVAL_CONTEXT,
    approval_granted,
    consume_preapproved,
    request_tool_approval,
)
from .contracts import (
    AgentError,
    ErrorType,
    OperationKind,
    OperationOutcome,
    OperationResult,
    ResultMeta,
    ToolSpec,
    WarningInfo,
)
from .errors import tool_envelope_error, tool_exception, tool_timeout
from .idempotency import (
    IdempotencyStore,
    InMemoryIdempotencyStore,
    idempotency_payload,
)
from .policy import ToolCallDecision, ToolPolicy, idempotency_key
from .tool_registry import ToolRegistry

# Fallback timeout: intentionally larger than the tools' own HTTP timeouts so
# the dispatcher only catches genuinely wedged calls (e.g. a dead MCP server).
DEFAULT_TIMEOUT_SECONDS = 130.0
# Per-tool parallelism cap for a single process.
DEFAULT_MAX_CONCURRENCY = 8
# Terminal transport failures tolerated before a tool is short-circuited.
DEFAULT_BREAKER_THRESHOLD = int(os.getenv("AGENT_TOOL_BREAKER_THRESHOLD", "3"))
DEFAULT_BREAKER_COOLDOWN = float(os.getenv("AGENT_TOOL_BREAKER_COOLDOWN", "30"))
DEFAULT_DECISION_HISTORY = int(
    os.getenv("AGENT_TOOL_DECISION_HISTORY", "512")
)
READ_CACHE_TTL = float(os.getenv("AGENT_TOOL_READ_CACHE_TTL", "30"))
READ_CACHE_MAX = int(os.getenv("AGENT_TOOL_READ_CACHE_MAX", "512"))
DEFAULT_READ_CACHE_TOOLS = {
    "search_papers",
    "fetch_content",
    "arxiv__search_papers",
    "arxiv__get_paper_data",
}

# Only transport-shaped failures open the breaker. Validation or authorization
# failures are the caller's fault and must not disable a healthy tool.
_BREAKER_ERRORS = {
    ErrorType.TOOL_TIMEOUT, ErrorType.TOOL_UNAVAILABLE, ErrorType.TRANSIENT,
    ErrorType.TOOL_RATE_LIMITED,
}

_SENSITIVE_ARG_PARTS = {
    "authorization", "credential", "key", "password", "secret", "token",
}


def _outcome_for_error(error: AgentError) -> OperationOutcome:
    if error.code == "TOOL_TIMEOUT":
        return OperationOutcome.TIMED_OUT
    if error.code == "TOOL_CANCELLED":
        return OperationOutcome.CANCELLED
    if error.code == "APPROVAL_REQUIRED":
        return OperationOutcome.INTERRUPTED
    return OperationOutcome.FAILED


def _outcome_from_parsed(parsed) -> OperationOutcome:
    if parsed.protocol_error:
        return OperationOutcome.FAILED
    try:
        return OperationOutcome(parsed.outcome)
    except ValueError:
        return OperationOutcome.FAILED


def _approval_preview(key: str, value: Any, depth: int = 0) -> Any:
    lowered = str(key or "").lower()
    if any(part in lowered for part in _SENSITIVE_ARG_PARTS):
        return "[redacted]"
    if depth >= 3:
        return "[depth limit]"
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        preview = [_approval_preview(key, item, depth + 1) for item in value[:10]]
        if len(value) > 10:
            preview.append(f"...[{len(value) - 10} more]")
        return preview
    if isinstance(value, dict):
        items = list(value.items())
        preview = {
            str(child_key): _approval_preview(str(child_key), child_value, depth + 1)
            for child_key, child_value in items[:20]
        }
        if len(items) > 20:
            preview["..."] = f"[{len(items) - 20} more]"
        return preview
    return str(value)[:200]


def redact_args(args: Any) -> dict:
    """Auditable approval view: digest plus a bounded, secret-redacted preview."""
    try:
        payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        payload = repr(args)
    keys: list[str] = []
    if isinstance(args, dict):
        keys = sorted(str(k) for k in args)
    return {
        "arg_keys": keys,
        "args_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "args_bytes": len(payload),
        "arg_preview": (
            {
                str(key): _approval_preview(str(key), value)
                for key, value in args.items()
            }
            if isinstance(args, dict) else _approval_preview("", args)
        ),
    }


def tool_approval_payload(
    *, name: str, spec: ToolSpec, args: dict, key: str, reason: str,
) -> dict:
    """Canonical payload shared by batch preflight and ToolGateway."""
    return {
        "kind": "tool_approval",
        "tool": name,
        "tool_version": spec.version,
        "side_effect": True,
        "permissions": sorted(p.value for p in spec.permissions),
        "args": redact_args(args),
        "idempotency_key": key,
        "reason": reason,
    }


@dataclass
class ToolCallOutcome:
    """Everything the caller (and the trace) needs about one tool call."""

    name: str
    envelope: str
    outcome: OperationOutcome
    decision: ToolCallDecision
    idempotency_key: str = ""
    attempts: int = 0
    duration_ms: float = 0.0
    error: AgentError | None = None
    deduplicated: bool = False
    cache_hit: bool = False
    audit: dict = field(default_factory=dict)

    def to_operation_result(
        self,
        *,
        spec: ToolSpec | None = None,
        ctx: Any = None,
    ) -> OperationResult:
        """Project the gateway result into the shared operation contract."""
        from ..tool_contract import parse_tool_result

        parsed = parse_tool_result(self.envelope)
        outcome = self.outcome
        data = parsed.data if parsed.is_envelope else (parsed.text or None)
        error = self.error
        if outcome is OperationOutcome.INTERRUPTED and error is not None:
            data = {
                "interrupt": error.model_dump(mode="json"),
                "request": parsed.data,
            }
            error = None
        if outcome in {
            OperationOutcome.FAILED,
            OperationOutcome.TIMED_OUT,
            OperationOutcome.CANCELLED,
        } and error is None:
            error = AgentError(
                error_type=ErrorType.PROTOCOL,
                code="TOOL_PROTOCOL_INVALID",
                message="Tool outcome has no AgentError",
                user_message="工具执行失败。",
                retryable=False,
            )
        return OperationResult(
            kind=OperationKind.TOOL,
            operation_id=self.idempotency_key or f"{self.name}:{uuid.uuid4().hex[:12]}",
            outcome=outcome,
            data=data if outcome not in {
                OperationOutcome.FAILED,
                OperationOutcome.TIMED_OUT,
                OperationOutcome.CANCELLED,
            } else None,
            error=error,
            warnings=[
                WarningInfo(
                    code="PARTIAL_RESULT",
                    message="Operation completed with incomplete output.",
                )
            ] if outcome is OperationOutcome.PARTIAL else [],
            meta=ResultMeta(
                trace_id=str(getattr(ctx, "trace_id", "") or ""),
                execution_id=str(
                    getattr(ctx, "execution_id", "")
                    or getattr(ctx, "request_id", "")
                    or ""
                ),
                thread_id=str(getattr(ctx, "thread_id", "") or ""),
                tool_name=self.name,
                tool_version=(spec.version if spec else ""),
                attempt=self.attempts,
                max_attempts=(
                    spec.effective_retry_policy().max_attempts if spec else 0
                ),
                duration_ms=self.duration_ms,
                cached=self.cache_hit,
                deduplicated=self.deduplicated,
                effect_applied=(
                    "yes" if outcome in {
                        OperationOutcome.SUCCEEDED, OperationOutcome.PARTIAL,
                    } and spec is not None and spec.side_effect
                    else (error.effect_applied if error is not None else None)
                ),
            ),
        )


class ToolGateway:
    """Policy-enforcing wrapper around a provider ``call_fn``."""

    def __init__(
        self,
        call_fn: Callable[[str, dict], Any],
        registry: ToolRegistry | None = None,
        *,
        policy: ToolPolicy | None = None,
        timeout_resolver: Callable[[str], float] | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
        breaker_cooldown: float = DEFAULT_BREAKER_COOLDOWN,
        idempotency_store: IdempotencyStore | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._call_fn = call_fn
        self._registry = registry or ToolRegistry()
        self._policy = policy or ToolPolicy()
        self._timeout_resolver = timeout_resolver
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._breaker_threshold = max(1, int(breaker_threshold))
        self._breaker_cooldown = max(0.0, float(breaker_cooldown))
        self._breakers: dict[str, dict] = {}
        self._idempotency_store = (
            idempotency_store if idempotency_store is not None
            else InMemoryIdempotencyStore()
        )
        self._clock = clock
        # Keep diagnostics bounded: long-lived desktop processes can issue
        # thousands of tool calls, and an unbounded list is a slow memory leak.
        self.decisions: deque[ToolCallDecision] = deque(
            maxlen=max(1, DEFAULT_DECISION_HISTORY)
        )
        self.decisions_total = 0
        raw_cache_tools = os.getenv("AGENT_TOOL_READ_CACHE_TOOLS", "").strip()
        self._read_cache_tools = (
            {item.strip() for item in raw_cache_tools.split(",") if item.strip()}
            if raw_cache_tools else set(DEFAULT_READ_CACHE_TOOLS)
        )
        self._read_cache: OrderedDict[
            str, tuple[float, ToolCallOutcome]
        ] = OrderedDict()

    # ---- introspection ----

    def spec(self, name: str) -> ToolSpec | None:
        return self._registry.get(name)

    def timeout_for(self, name: str) -> float:
        if self._timeout_resolver is not None:
            try:
                return float(self._timeout_resolver(name))
            except Exception:  # noqa: BLE001
                pass
        spec = self.spec(name)
        if spec is not None and spec.timeout_seconds:
            return float(spec.timeout_seconds)
        return DEFAULT_TIMEOUT_SECONDS

    # ---- main entry ----

    async def invoke(self, name: str, args: dict, *, ctx: Any = None,
                     intent: str = "") -> ToolCallOutcome:
        from ..tool_contract import parse_tool_result

        start = self._clock()
        spec = self.spec(name)

        decision = self._policy.decide(spec, ctx=ctx, tool_name=name)
        self.decisions.append(decision)
        self.decisions_total += 1
        if decision.action == "deny":
            error = self._policy.denial(decision, name)
            self._audit(name, spec, decision, error=error, attempts=0,
                        duration_ms=0.0, key="")
            return ToolCallOutcome(
                name=name, envelope=error.to_tool_envelope(),
                outcome=_outcome_for_error(error),
                decision=decision, error=error,
            )

        invalid = self._policy.validate_arguments(spec, args) if spec else None
        if invalid is not None:
            self._audit(name, spec, decision, error=invalid, attempts=0,
                        duration_ms=0.0, key="")
            return ToolCallOutcome(
                name=name, envelope=invalid.to_tool_envelope(),
                outcome=_outcome_for_error(invalid),
                decision=decision, error=invalid,
            )

        cache_key = self._read_cache_key(name, spec, args, ctx)
        if cache_key:
            cached = self._read_cache_get(cache_key)
            if cached is not None:
                log_event(
                    "tool_cache_hit", node="tool_gateway", tool=name,
                    tool_version=(spec.version if spec else ""),
                    cache_key=cache_key,
                )
                return ToolCallOutcome(
                    name=name,
                    envelope=cached.envelope,
                    outcome=cached.outcome,
                    decision=decision,
                    idempotency_key=cached.idempotency_key,
                    attempts=0,
                    duration_ms=0.0,
                    error=cached.error,
                    cache_hit=True,
                    audit={"cache_hit": True},
                )

        key = idempotency_key(
            thread_id=getattr(ctx, "thread_id", "") or "",
            execution_id=(
                getattr(ctx, "execution_id", "")
                or getattr(ctx, "request_id", "")
                or ""
            ),
            spec=spec, args=args, intent=intent,
        ) if spec else ""

        # Approval is a graph interrupt, not a policy denial.  If no graph
        # RunnableConfig is bound (direct provider/test call), fail closed with
        # the legacy APPROVAL_REQUIRED envelope.
        if decision.action == "approval":
            if consume_preapproved(key):
                log_event(
                    "tool_preapproved", node="tool_gateway", tool=name,
                    tool_version=spec.version, idempotency_key=key,
                )
                approval_value = {"approved": True}
            else:
                approval_value = request_tool_approval(
                    tool_approval_payload(
                        name=name, spec=spec, args=args, key=key,
                        reason=decision.reason,
                    )
                )
            if approval_value is NO_APPROVAL_CONTEXT:
                error = self._policy.denial(decision, name)
                self._audit(name, spec, decision, error=error, attempts=0,
                            duration_ms=0.0, key=key)
                return ToolCallOutcome(
                    name=name, envelope=error.to_tool_envelope(),
                    outcome=_outcome_for_error(error),
                    decision=decision, idempotency_key=key, error=error,
                )
            if not approval_granted(approval_value):
                error = self._policy.approval_denied(name)
                self._audit(name, spec, decision, error=error, attempts=0,
                            duration_ms=0.0, key=key)
                return ToolCallOutcome(
                    name=name, envelope=error.to_tool_envelope(),
                    outcome=_outcome_for_error(error),
                    decision=decision, idempotency_key=key, error=error,
                )
            log_event("tool_approval_granted", node="tool_gateway", tool=name,
                      tool_version=spec.version, idempotency_key=key)
            decision = ToolCallDecision("allow", "approved by user")

        if spec is not None and spec.side_effect:
            # A write can invalidate cached reads even when the adapter returns
            # an ambiguous error after touching the resource.
            self._read_cache.clear()

        blocked = self._breaker_error(name)
        if blocked is not None:
            self._audit(name, spec, decision, error=blocked, attempts=0,
                        duration_ms=0.0, key=key)
            return ToolCallOutcome(
                name=name, envelope=blocked.to_tool_envelope(),
                outcome=_outcome_for_error(blocked),
                decision=decision, idempotency_key=key, error=blocked,
            )

        # Persist a reservation before entering the adapter.  A successful
        # result is replayable after restart; an indeterminate result must not
        # be retried automatically because the side effect may already exist.
        # Every side effect gets an at-most-once journal for the current turn.
        # LangGraph replays an interrupted node from its start, so even tools
        # that are not naturally idempotent must not execute twice while the
        # same turn's approval chain is being completed.
        durable_key = key if (spec is not None and spec.side_effect) else ""
        if durable_key:
            try:
                reservation = await self._idempotency_store.reserve(
                    durable_key, tool_name=spec.name,
                    tool_version=spec.version,
                )
            except Exception as exc:  # noqa: BLE001 — fail closed
                error = AgentError(
                    error_type=ErrorType.CHECKPOINT,
                    code="IDEMPOTENCY_STORE_UNAVAILABLE",
                    message=f"Idempotency store failed: {type(exc).__name__}",
                    user_message="无法确认该副作用操作是否可安全重试，已停止执行。",
                    retryable=False,
                    recovery_action="Fix the idempotency store and retry explicitly.",
                    tool_name=name,
                )
                self._audit(name, spec, decision, error=error, attempts=0,
                            duration_ms=0.0, key=durable_key)
                return ToolCallOutcome(
                    name=name, envelope=error.to_tool_envelope(),
                    outcome=_outcome_for_error(error),
                    decision=decision, idempotency_key=durable_key,
                    error=error,
                )

            if not reservation.acquired and reservation.record is not None:
                record = reservation.record
                if record.status == "succeeded":
                    self._audit(
                        name, spec, decision, error=None, attempts=0,
                        duration_ms=0.0, key=durable_key,
                    )
                    return ToolCallOutcome(
                        name=name, envelope=record.envelope,
                        outcome=_outcome_from_parsed(
                            parse_tool_result(record.envelope)
                        ),
                        decision=decision, idempotency_key=durable_key,
                        attempts=0, deduplicated=True, error=None,
                        audit={"replayed": True, **idempotency_payload(record)},
                    )

                if record.status == "failed":
                    previous = record.error_code or "SIDE_EFFECT_PREVIOUSLY_FAILED"
                    error = AgentError(
                        error_type=ErrorType.VALIDATION,
                        code="SIDE_EFFECT_PREVIOUSLY_FAILED",
                        message=(
                            "The same side-effecting call previously failed "
                            f"without applying the effect: {previous}"
                        ),
                        user_message=(
                            "同一副作用操作此前已明确失败且未执行，"
                            "请修改参数或显式创建新的操作后再试。"
                        ),
                        retryable=False,
                        recovery_action=(
                            "Correct the request and create a new idempotency intent."
                        ),
                        tool_name=name,
                        effect_applied="no",
                    )
                elif record.status == "indeterminate":
                    error = AgentError(
                        error_type=ErrorType.CHECKPOINT,
                        code="SIDE_EFFECT_OUTCOME_UNKNOWN",
                        message="A previous side effect may have completed.",
                        user_message=(
                            "上一次副作用操作的结果无法确认，已禁止自动重试，"
                            "请先核对实际状态。"
                        ),
                        retryable=False,
                        recovery_action=(
                            "Inspect the external state, then retry with a new "
                            "intent or explicitly reconcile the operation."
                        ),
                        tool_name=name,
                        effect_applied="unknown",
                    )
                else:
                    error = AgentError(
                        error_type=ErrorType.TRANSIENT,
                        code="IDEMPOTENCY_IN_PROGRESS",
                        message="The same side-effecting call is already running.",
                        user_message="同一副作用操作正在执行，请勿重复提交。",
                        retryable=True,
                        recovery_action="Wait for the in-flight call to finish.",
                        tool_name=name,
                    )
                self._audit(name, spec, decision, error=error, attempts=0,
                            duration_ms=0.0, key=durable_key)
                return ToolCallOutcome(
                    name=name, envelope=error.to_tool_envelope(),
                    outcome=_outcome_for_error(error),
                    decision=decision, idempotency_key=durable_key,
                    error=error, audit=idempotency_payload(record),
                )

        timeout = self.timeout_for(name)
        retry = spec.effective_retry_policy() if spec else None
        attempts_allowed = retry.max_attempts if retry else 1
        error: AgentError | None = None
        attempts = 0

        for attempt in range(1, attempts_allowed + 1):
            attempts = attempt
            t0 = self._clock()
            try:
                async with self._semaphore:
                    result = await asyncio.wait_for(
                        self._call_fn(name, args), timeout=timeout,
                    )
            except asyncio.TimeoutError:
                error = tool_timeout(name, timeout)
            except asyncio.CancelledError:
                duration_ms = (self._clock() - t0) * 1000
                error = AgentError(
                    error_type=ErrorType.AGENT_RUNTIME,
                    code="TOOL_CANCELLED",
                    message="Tool execution was cancelled.",
                    user_message="工具执行已取消，副作用结果需要核对。",
                    retryable=False,
                    recovery_action=(
                        "Inspect external state before retrying the operation."
                    ),
                    tool_name=name,
                    effect_applied="unknown",
                )
                if durable_key:
                    try:
                        await self._idempotency_store.fail(
                            durable_key, error.code,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log_event(
                            "idempotency_mark_cancelled_error",
                            node="tool_gateway", tool=name,
                            idempotency_key=durable_key,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                self._audit(
                    name, spec, decision, error=error, attempts=attempt,
                    duration_ms=duration_ms, key=key,
                )
                raise
            except Exception as exc:  # noqa: BLE001 — adapters raise freely
                error = tool_exception(name, exc)
            else:
                duration_ms = (self._clock() - t0) * 1000
                envelope = str(result)
                parsed = parse_tool_result(envelope)
                if parsed.protocol_error:
                    error = AgentError(
                        error_type=ErrorType.PROTOCOL,
                        code="TOOL_PROTOCOL_INVALID",
                        message=parsed.protocol_error,
                        user_message="工具返回了不符合协议的数据。",
                        retryable=False,
                        recovery_action=(
                            "Fix the tool adapter to emit a valid result envelope."
                        ),
                        tool_name=name,
                        effect_applied=(
                            "unknown" if durable_key else None
                        ),
                    )
                    if durable_key:
                        await self._mark_uncertain(durable_key, error.code)
                    self._audit(
                        name, spec, decision, error=error, attempts=attempt,
                        duration_ms=duration_ms, key=key,
                    )
                    return ToolCallOutcome(
                        name=name, envelope=error.to_tool_envelope(),
                        outcome=_outcome_for_error(error),
                        decision=decision, idempotency_key=key,
                        attempts=attempt, duration_ms=duration_ms, error=error,
                    )
                if parsed.is_envelope and parsed.outcome in {
                    "failed", "timed_out", "cancelled",
                }:
                    raw_retry_after = parsed.extra.get("retry_after_seconds")
                    try:
                        retry_after = (
                            float(raw_retry_after)
                            if raw_retry_after is not None else None
                        )
                    except (TypeError, ValueError):
                        retry_after = None
                    raw_retryable = parsed.extra.get("retryable")
                    error = tool_envelope_error(
                        name,
                        error_type=parsed.error_type,
                        message=parsed.error,
                        recovery_action=parsed.next_action,
                        retry_after_seconds=retry_after,
                        retryable=(
                            bool(raw_retryable)
                            if isinstance(raw_retryable, bool) else None
                        ),
                    )
                    if durable_key:
                        try:
                            if error.effect_applied == "no":
                                await self._idempotency_store.reject(
                                    durable_key, error.code,
                                )
                            else:
                                await self._idempotency_store.fail(
                                    durable_key, error.code,
                                )
                        except Exception as exc:  # noqa: BLE001
                            log_event(
                                "idempotency_mark_semantic_error_failed",
                                node="tool_gateway", tool=name,
                                idempotency_key=durable_key,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                    retryable = error.retryable and attempt < attempts_allowed
                    if not retryable:
                        break
                    delay = retry.delay_for(attempt) if retry else 0.0
                    log_event(
                        "tool_retry", node="tool_gateway", tool=name,
                        attempt=attempt, delay_seconds=delay, code=error.code,
                    )
                    if delay:
                        await asyncio.sleep(delay)
                    continue
                output_error = self._validate_output(spec, parsed)
                if output_error is not None:
                    error = output_error
                    if durable_key:
                        await self._mark_uncertain(durable_key, error.code)
                    self._audit(
                        name, spec, decision, error=error, attempts=attempts,
                        duration_ms=duration_ms, key=durable_key or key,
                    )
                    return ToolCallOutcome(
                        name=name, envelope=error.to_tool_envelope(),
                        outcome=_outcome_for_error(error),
                        decision=decision, idempotency_key=key,
                        attempts=attempts, duration_ms=duration_ms, error=error,
                    )
                if durable_key:
                    try:
                        await self._idempotency_store.complete(
                            durable_key, envelope,
                        )
                    except Exception as exc:  # noqa: BLE001
                        error = AgentError(
                            error_type=ErrorType.CHECKPOINT,
                            code="IDEMPOTENCY_RECORD_FAILED",
                            message=(
                                "Side effect completed but its idempotency "
                                f"record failed: {type(exc).__name__}"
                            ),
                            user_message=(
                                "操作可能已执行，但结果未能持久记录；"
                                "请勿自动重试，先核对实际状态。"
                            ),
                            retryable=False,
                            recovery_action=(
                                "Inspect the external state and repair the "
                                "idempotency store before retrying."
                            ),
                            tool_name=name,
                            effect_applied="yes",
                        )
                        self._audit(
                            name, spec, decision, error=error, attempts=attempts,
                            duration_ms=duration_ms, key=durable_key,
                        )
                        return ToolCallOutcome(
                            name=name, envelope=error.to_tool_envelope(),
                            outcome=_outcome_for_error(error),
                            decision=decision, idempotency_key=durable_key,
                            attempts=attempts, duration_ms=duration_ms,
                            error=error,
                        )
                self._record_success(name)
                if not (spec is not None and spec.side_effect) and cache_key:
                    self._read_cache_put(
                        cache_key,
                        ToolCallOutcome(
                            name=name, envelope=envelope,
                            outcome=_outcome_from_parsed(parsed),
                            decision=decision, idempotency_key=key,
                            attempts=attempts, duration_ms=duration_ms,
                        ),
                    )
                self._audit(name, spec, decision, error=None, attempts=attempts,
                            duration_ms=duration_ms, key=key)
                return ToolCallOutcome(
                    name=name, envelope=envelope,
                    outcome=_outcome_from_parsed(parsed),
                    decision=decision,
                    idempotency_key=key, attempts=attempts,
                    duration_ms=duration_ms,
                )

            retryable = error.retryable and attempt < attempts_allowed
            if not retryable:
                break
            delay = retry.delay_for(attempt) if retry else 0.0
            log_event("tool_retry", node="tool_gateway", tool=name, attempt=attempt,
                      delay_seconds=delay, code=error.code)
            if delay:
                await asyncio.sleep(delay)

        duration_ms = (self._clock() - start) * 1000
        self._record_failure(name, error)
        if durable_key and error is not None:
            try:
                await self._idempotency_store.fail(
                    durable_key, error.code,
                )
            except Exception as exc:  # noqa: BLE001
                log_event(
                    "idempotency_mark_failed_error", node="tool_gateway",
                    tool=name, idempotency_key=durable_key,
                    error=f"{type(exc).__name__}: {exc}",
                )
        self._audit(name, spec, decision, error=error, attempts=attempts,
                    duration_ms=duration_ms, key=key)
        return ToolCallOutcome(
            name=name, envelope=error.to_tool_envelope() if error else "",
            outcome=(
                _outcome_for_error(error)
                if error is not None else OperationOutcome.FAILED
            ),
            decision=decision, idempotency_key=key, attempts=attempts,
            duration_ms=duration_ms, error=error,
        )

    # ---- internals ----

    def _validate_output(
        self, spec: ToolSpec | None, parsed,
    ) -> AgentError | None:
        """Validate the adapter's structured payload before it can be cached."""
        if spec is None or not spec.output_schema:
            return None
        instance = parsed.data if parsed.is_envelope else parsed.text
        try:
            from jsonschema import validate

            validate(instance=instance, schema=spec.output_schema)
            return None
        except Exception as exc:  # noqa: BLE001 — schema errors are data errors
            return AgentError(
                error_type=ErrorType.PROTOCOL,
                code="TOOL_OUTPUT_INVALID",
                message=f"Tool output failed schema validation: {type(exc).__name__}",
                user_message="工具返回结果不符合声明的数据结构。",
                retryable=False,
                recovery_action=(
                    "Fix the tool output or its output_schema; do not cache this result."
                ),
                tool_name=spec.name,
                effect_applied="unknown" if spec.side_effect else None,
            )

    async def _mark_uncertain(self, key: str, error_code: str) -> None:
        try:
            await self._idempotency_store.fail(key, error_code)
        except Exception as exc:  # noqa: BLE001
            log_event(
                "idempotency_mark_failed_error",
                node="tool_gateway",
                idempotency_key=key,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _read_cache_key(
        self, name: str, spec: ToolSpec | None, args: dict, ctx: Any,
    ) -> str:
        if spec is None or spec.side_effect or READ_CACHE_TTL <= 0:
            return ""
        if "*" not in self._read_cache_tools and name not in self._read_cache_tools:
            return ""
        from .policy import canonical_args

        thread_id = str(getattr(ctx, "thread_id", "") or "__global__")
        return "|".join([
            thread_id,
            f"{name}@{spec.version}",
            canonical_args(args),
        ])

    def _read_cache_get(self, key: str) -> ToolCallOutcome | None:
        item = self._read_cache.get(key)
        if item is None:
            return None
        expires_at, outcome = item
        if expires_at <= self._clock():
            self._read_cache.pop(key, None)
            return None
        self._read_cache.move_to_end(key)
        return outcome

    def _read_cache_put(self, key: str, outcome: ToolCallOutcome) -> None:
        self._read_cache[key] = (
            self._clock() + max(0.0, READ_CACHE_TTL),
            outcome,
        )
        self._read_cache.move_to_end(key)
        while len(self._read_cache) > max(1, READ_CACHE_MAX):
            self._read_cache.popitem(last=False)

    def _breaker_error(self, name: str) -> AgentError | None:
        state = self._breakers.get(name)
        # ``opened_at is None`` means "counting failures, not yet open".
        if not state or state.get("opened_at") is None:
            return None
        if self._clock() - state["opened_at"] < self._breaker_cooldown:
            return AgentError(
                error_type=ErrorType.TOOL_UNAVAILABLE,
                code="TOOL_CIRCUIT_OPEN",
                message=f"Tool '{name}' is temporarily short-circuited.",
                user_message=f"工具“{name}”连续失败，已暂时熔断，稍后再试。",
                retryable=True,
                retry_after_seconds=round(
                    self._breaker_cooldown - (self._clock() - state["opened_at"]), 1
                ),
                recovery_action="Wait for the cooldown or use a different tool.",
                tool_name=name,
            )
        # Cooldown elapsed → allow one probe call.
        self._breakers.pop(name, None)
        return None

    def _record_failure(self, name: str, error: AgentError | None) -> None:
        if error is None or error.error_type not in _BREAKER_ERRORS:
            return
        state = self._breakers.setdefault(name, {"failures": 0, "opened_at": None})
        state["failures"] += 1
        if state["failures"] >= self._breaker_threshold:
            state["opened_at"] = self._clock()

    def _record_success(self, name: str) -> None:
        self._breakers.pop(name, None)

    def _audit(self, name: str, spec: ToolSpec | None, decision: ToolCallDecision,
               *, error: AgentError | None, attempts: int, duration_ms: float,
               key: str) -> None:
        """Governance view of the call. Metrics keep their own ``tool_call`` row."""
        log_event(
            "tool_audit", node="tool_gateway", tool=name,
            tool_version=(spec.version if spec else ""),
            side_effect=(spec.side_effect if spec else None),
            decision=decision.action, decision_reason=decision.reason,
            attempts=attempts, duration_ms=round(duration_ms, 1),
            error_type=(error.error_type.value if error else ""),
            error_code=(error.code if error else ""),
            idempotency_key=key,
        )
