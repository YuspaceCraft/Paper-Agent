"""Factories for turning runtime failures into stable AgentError values."""

from __future__ import annotations

import asyncio

from .contracts import AgentError, ErrorType


class ModelGatewayError(RuntimeError):
    """All configured model candidates failed or are circuit-open."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.cause = cause


def tool_timeout(name: str, timeout_seconds: float) -> AgentError:
    return AgentError(
        error_type=ErrorType.TOOL_TIMEOUT,
        code="TOOL_TIMEOUT",
        message=f"Tool '{name}' exceeded its {timeout_seconds:g}s timeout.",
        user_message=f"工具“{name}”响应超时。",
        retryable=True,
        recovery_action="Retry once or use a different tool.",
        tool_name=name,
        effect_applied="unknown",
    )


def tool_exception(name: str, exc: BaseException) -> AgentError:
    """Map transport/runtime exceptions without exposing an exception message."""
    if isinstance(exc, asyncio.TimeoutError):
        return tool_timeout(name, 0)
    # Transport failures are transient by nature: a retry is legitimate (for
    # idempotent work only, enforced by ToolSpec.effective_retry_policy) and the
    # circuit breaker counts them so a wedged backend stops being hammered.
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return AgentError(
            error_type=ErrorType.TOOL_UNAVAILABLE,
            code="TOOL_UNAVAILABLE",
            message=f"{type(exc).__name__} while invoking tool '{name}'.",
            user_message=f"工具“{name}”暂时不可用。",
            retryable=True,
            retry_after_seconds=1.0,
            recovery_action="Retry once; if it keeps failing, degrade gracefully.",
            tool_name=name,
            effect_applied="unknown",
        )
    return AgentError(
        error_type=ErrorType.TOOL,
        code="TOOL_EXECUTION_FAILED",
        message=f"{type(exc).__name__} while invoking tool '{name}'.",
        user_message=f"工具“{name}”执行失败。",
        retryable=False,
        recovery_action="Check parameters or use a different tool.",
        tool_name=name,
        effect_applied="unknown",
    )


def tool_envelope_error(
    name: str,
    *,
    error_type: str,
    message: str,
    recovery_action: str = "",
    retry_after_seconds: float | None = None,
    retryable: bool | None = None,
) -> AgentError:
    """Map a structured tool failure into the shared runtime taxonomy."""
    token = (error_type or "").strip().lower()
    mapping = {
        "validation": ErrorType.VALIDATION,
        "param_error": ErrorType.VALIDATION,
        "authorization": ErrorType.AUTHORIZATION,
        "permission_denied": ErrorType.PERMISSION_DENIED,
        "policy": ErrorType.POLICY,
        "approval_required": ErrorType.POLICY,
        "tool_timeout": ErrorType.TOOL_TIMEOUT,
        "timeout": ErrorType.TOOL_TIMEOUT,
        "tool_rate_limited": ErrorType.TOOL_RATE_LIMITED,
        "rate_limited": ErrorType.TOOL_RATE_LIMITED,
        "tool_unavailable": ErrorType.TOOL_UNAVAILABLE,
        "backend_down": ErrorType.TOOL_UNAVAILABLE,
        "transient": ErrorType.TRANSIENT,
        "dependency": ErrorType.DEPENDENCY,
        "checkpoint": ErrorType.CHECKPOINT,
        "not_found": ErrorType.TOOL,
        "protocol_error": ErrorType.PROTOCOL,
        "tool_protocol_invalid": ErrorType.PROTOCOL,
        "tool_crash": ErrorType.AGENT_RUNTIME,
    }
    code_mapping = {
        "validation": "TOOL_ARGS_INVALID",
        "param_error": "TOOL_ARGS_INVALID",
        "authorization": "TOOL_NOT_PERMITTED",
        "permission_denied": "TOOL_NOT_PERMITTED",
        "policy": "POLICY_DENIED",
        "approval_required": "APPROVAL_REQUIRED",
        "tool_timeout": "TOOL_TIMEOUT",
        "timeout": "TOOL_TIMEOUT",
        "tool_rate_limited": "TOOL_RATE_LIMITED",
        "rate_limited": "TOOL_RATE_LIMITED",
        "tool_unavailable": "TOOL_UNAVAILABLE",
        "backend_down": "TOOL_UNAVAILABLE",
        "transient": "TOOL_TRANSIENT",
        "dependency": "TOOL_DEPENDENCY_FAILED",
        "checkpoint": "TOOL_CHECKPOINT_FAILED",
        "not_found": "TOOL_NOT_FOUND",
        "protocol_error": "TOOL_PROTOCOL_INVALID",
        "tool_protocol_invalid": "TOOL_PROTOCOL_INVALID",
        "tool_crash": "TOOL_EXECUTION_FAILED",
    }
    classified = mapping.get(token, ErrorType.TOOL)
    default_retryable = classified in {
        ErrorType.TOOL_TIMEOUT,
        ErrorType.TOOL_RATE_LIMITED,
        ErrorType.TOOL_UNAVAILABLE,
        ErrorType.TRANSIENT,
        ErrorType.DEPENDENCY,
    }
    effect_applied = (
        "no"
        if classified in {
            ErrorType.VALIDATION,
            ErrorType.AUTHORIZATION,
            ErrorType.PERMISSION_DENIED,
            ErrorType.POLICY,
        }
        else "unknown"
    )
    return AgentError(
        error_type=classified,
        code=code_mapping.get(token, "TOOL_EXECUTION_FAILED"),
        message=f"Tool '{name}' returned a structured error: {token or 'unknown'}.",
        user_message=message or f"工具“{name}”执行失败。",
        retryable=default_retryable if retryable is None else bool(retryable),
        retry_after_seconds=retry_after_seconds,
        recovery_action=recovery_action or "Inspect the tool result and retry safely.",
        tool_name=name,
        effect_applied=effect_applied,
    )
