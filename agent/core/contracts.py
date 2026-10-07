"""Stable runtime contracts for the LangGraph agent platform.

The project already exposes tool results through ``agent.tool_contract``.
These models complement that wire format: they make the error taxonomy and
tool governance metadata reusable without forcing providers to change their
public result shape in one migration.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .configuration import ConfigurationSnapshot


class ErrorType(str, Enum):
    """Classifications shared by retry, UX, audit and evaluation.

    ``TRANSIENT`` and ``UNKNOWN`` remain for backwards-compatible tool
    envelopes. New call sites should prefer the more specific values.
    """

    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    PERMISSION_DENIED = "permission_denied"
    POLICY = "policy"
    TOOL = "tool"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_RATE_LIMITED = "tool_rate_limited"
    MODEL = "model"
    SUBAGENT = "subagent"
    TASK = "task"
    AGENT_RUNTIME = "agent_runtime"
    GRAPH_TIMEOUT = "graph_timeout"
    PROTOCOL = "protocol"
    CONTEXT_BUDGET = "context_budget"
    CHECKPOINT = "checkpoint"
    DEPENDENCY = "dependency"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    ADMIN = "admin"
    SECRETS = "secrets"


class PromptType(str, Enum):
    SYSTEM = "system"
    ROUTER = "router"
    PLANNER = "planner"
    EXECUTOR = "executor"
    SYNTHESIZER = "synthesizer"
    JUDGE = "judge"
    SUMMARY = "summary"
    SAFETY = "safety"


class AgentError(BaseModel):
    """A safe error view that can cross runtime boundaries.

    ``message`` must already be redacted; stack traces belong in a protected
    audit/trace backend and must never be copied into this object.
    """

    model_config = ConfigDict(extra="forbid")

    error_type: ErrorType
    code: str
    message: str
    user_message: str
    retryable: bool = False
    retry_after_seconds: float | None = None
    recovery_action: str | None = None
    cause_ref: str | None = None
    tool_name: str | None = None
    effect_applied: Literal["no", "yes", "unknown"] | None = None

    def to_tool_envelope(self) -> str:
        """Render through the existing, backwards-compatible tool contract."""
        from ..tool_contract import failure

        return failure(
            self.error_type.value,
            self.user_message,
            self.recovery_action or "",
            code=self.code,
            retryable=self.retryable,
            retry_after_seconds=self.retry_after_seconds,
            cause_ref=self.cause_ref,
            tool_name=self.tool_name,
            effect_applied=self.effect_applied,
        )


class OperationKind(str, Enum):
    """The runtime layer that produced one result."""

    TOOL = "tool"
    AGENT = "agent"
    SUBAGENT = "subagent"
    TASK = "task"


class OperationOutcome(str, Enum):
    """The single authoritative lifecycle outcome for one operation."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"


class WarningInfo(BaseModel):
    """A non-fatal condition that callers and UIs must not report as success."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ResultMeta(BaseModel):
    """Correlation and execution metadata shared by every operation result.

    This object deliberately excludes exception objects, stack traces and raw
    secrets. Those belong in protected telemetry behind ``cause_ref``.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = ""
    execution_id: str = ""
    thread_id: str = ""
    tool_name: str = ""
    tool_version: str = ""
    attempt: int = 0
    max_attempts: int = 0
    timeout_ms: int | None = None
    duration_ms: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    truncated: bool = False
    original_chars: int | None = None
    cached: bool = False
    deduplicated: bool = False
    effect_applied: Literal["no", "yes", "unknown"] | None = None


class OperationResult(BaseModel):
    """One versioned result contract for tools, agents, subagents and tasks.

    The validator rejects ambiguous combinations such as success-with-error or
    failure-without-error. Serialize across a process/UI boundary with
    ``model_dump(mode="json")``; use :meth:`to_envelope` for the legacy string
    tool contract.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    kind: OperationKind
    operation_id: str
    parent_operation_id: str | None = None
    outcome: OperationOutcome
    data: Any = None
    error: AgentError | None = None
    warnings: list[WarningInfo] = Field(default_factory=list)
    meta: ResultMeta = Field(default_factory=ResultMeta)

    @model_validator(mode="after")
    def _validate_outcome(self) -> "OperationResult":
        error_required = {
            OperationOutcome.FAILED,
            OperationOutcome.TIMED_OUT,
            OperationOutcome.CANCELLED,
        }
        error_forbidden = {
            OperationOutcome.SUCCEEDED,
            OperationOutcome.PARTIAL,
            OperationOutcome.INTERRUPTED,
            OperationOutcome.SKIPPED,
        }
        if self.outcome in error_required and self.error is None:
            raise ValueError(f"{self.outcome.value} requires error")
        if self.outcome in error_forbidden and self.error is not None:
            raise ValueError(f"{self.outcome.value} must not carry error")
        if self.outcome is OperationOutcome.PARTIAL and not self.warnings:
            raise ValueError("partial result requires at least one warning")
        return self

    def to_envelope(self) -> str:
        """Render through the backwards-compatible string tool envelope."""
        from ..tool_contract import operation_result_to_envelope

        return operation_result_to_envelope(self)


class RetryPolicy(BaseModel):
    """Bounded exponential backoff for transient tool failures.

    Domain errors never use this: only timeouts, rate limits and transport
    failures are retryable, and only for tools whose work is idempotent.
    """

    max_attempts: int = 1
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait before ``attempt`` (1-based) is retried."""
        raw = self.backoff_seconds * (self.backoff_multiplier ** max(0, attempt - 1))
        return min(raw, self.max_backoff_seconds)


class ToolSpec(BaseModel):
    """Declarative metadata required before a tool can enter the registry."""

    name: str
    version: str = "1"
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    permissions: set[Permission] = Field(default_factory=lambda: {Permission.READ})
    side_effect: bool = False
    idempotency_scope: str = "request"
    timeout_seconds: float | None = None
    max_attempts: int = 1
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    owner: str = ""
    tags: set[str] = Field(default_factory=set)

    def requires_approval(self) -> bool:
        return self.side_effect or bool(
            {Permission.WRITE, Permission.EXECUTE, Permission.ADMIN} & self.permissions
        )

    def idempotent(self) -> bool:
        return self.idempotency_scope != "none"

    def effective_retry_policy(self) -> RetryPolicy:
        """Retry only reproducible work, and never more than the legacy cap.

        Non-idempotent tools keep a single attempt no matter what the caller
        declares, so a retry can never duplicate a side effect.
        """
        if self.side_effect or not self.idempotent():
            return self.retry_policy.model_copy(update={"max_attempts": 1})
        attempts = self.retry_policy.max_attempts
        if attempts <= 1:
            attempts = max(self.max_attempts, 3)
        return self.retry_policy.model_copy(update={"max_attempts": attempts})

class Budget(BaseModel):
    """Per-turn execution envelope frozen at bootstrap.

    Values come from ``agent/config.yaml`` and env overrides; they are copied
    into the trace so a run can be explained after the fact even if the file
    changed meanwhile.
    """

    max_steps: int = 30
    max_turns: int = 50
    plan_step_max_steps: int = 10
    turn_timeout_seconds: float = 900.0
    token_budget: int = 60000
    context_window_tokens: int = 32768
    max_output_tokens: int = 4096
    token_safety_ratio: float = 0.90

    def usable_context_tokens(self) -> int:
        """Context tokens available for prompt assembly.

        The reserve for tool results and model output is subtracted first so
        every caller uses the same safety boundary.
        """
        return max(0, self.context_window_tokens - self.max_output_tokens)


class PromptSpec(BaseModel):
    """One published prompt version with its input contract.

    ``template`` is the only prompt payload; every other field is metadata used
    for validation, trace binding and evaluation.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    version: str
    type: PromptType = PromptType.SYSTEM
    template: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    variables: list[str] = Field(default_factory=list)
    locale: str = "en"
    status: str = "active"
    evaluation_suite: str = ""

    @property
    def binding(self) -> str:
        """Stable ``prompt_id@version#checksum`` identifier for trace metadata."""
        return f"{self.id}@{self.version}#{self.checksum}"

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:12]

    def is_active(self) -> bool:
        return self.status == "active" and bool(self.template.strip())

    def render(self, **variables: Any) -> str:
        """Format the template, rejecting variables the spec did not declare."""
        declared = set(self.variables) or None
        unknown = set(variables) - declared if declared else set()
        if unknown:
            raise ValueError(f"undeclared prompt variables: {sorted(unknown)}")
        return self.template.format(**variables)


class ExecutionContext(BaseModel):
    """Everything a node, model call or tool needs that is not business state.

    The LangGraph ``RunnableConfig`` is deliberately *not* a model field: it
    holds live callbacks and must never be checkpointed. It travels through
    :func:`set_current_execution_context` instead, which is restored per turn.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    execution_id: str = ""
    trace_id: str
    thread_id: str
    actor_id: str = "local"
    roles: set[str] = Field(default_factory=lambda: {"user"})
    permissions: set[Permission] = Field(default_factory=lambda: {Permission.READ})
    config: ConfigurationSnapshot
    prompt_bindings: dict[str, str] = Field(default_factory=dict)
    prompt_evaluation_suites: dict[str, str] = Field(default_factory=dict)
    # Frozen prompt text, keyed by prompt id/type. Never copied into trace
    # metadata, but nodes resolve from this snapshot so a mid-turn publish
    # cannot make the recorded binding disagree with the text actually used.
    prompt_templates: dict[str, str] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)
    domain: str = ""
    graph_version: str = ""
    model_route: str = ""
    tool_versions: dict[str, str] = Field(default_factory=dict)
    eval_dataset_id: str = ""

    _runnable_config: Any = PrivateAttr(default=None)

    @property
    def runnable_config(self) -> Any:
        return self._runnable_config

    def bind_runnable_config(self, config: Any) -> "ExecutionContext":
        self._runnable_config = config
        return self

    def child_config(self, **overrides: Any) -> dict[str, Any]:
        """Derive a child ``RunnableConfig`` so model/tool runs nest correctly.

        Node code must call this instead of building a bare config, otherwise
        LangSmith records a detached root run (see ADR-0002).
        """
        base = dict(self._runnable_config or {})
        child = dict(base)
        child.update(overrides)
        metadata = dict(base.get("metadata") or {})
        metadata.setdefault("trace_id", self.trace_id)
        metadata.setdefault("thread_id", self.thread_id)
        child["metadata"] = metadata
        return child

    def allows(self, *permissions: Permission) -> bool:
        return set(permissions).issubset(self.permissions)

    def runtime_state(self) -> dict[str, Any]:
        """State values that must follow the frozen turn budget."""
        return {
            "execution_id": self.execution_id or self.request_id,
            "max_steps": self.budget.max_steps,
            "max_turns": self.budget.max_turns,
            "token_budget": self.budget.token_budget,
        }

    def model_for(self, task: str) -> str:
        """Resolve the frozen model route for one task class."""
        return (
            self.config.model_routes.get(str(task))
            or self.model_route
            or self.config.model
        )

    def trace_metadata(self) -> dict[str, Any]:
        """The metadata contract LangSmith traces and offline reports rely on."""
        return {
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "thread_id": self.thread_id,
            "request_id": self.request_id,
            "actor_id": self.actor_id,
            "actor_role": ",".join(sorted(self.roles)),
            "domain": self.domain,
            "graph_version": self.graph_version,
            "model_route": self.model_route,
            "eval_dataset_id": self.eval_dataset_id,
            "prompt_bindings": self.prompt_bindings,
            "prompt_evaluation_suites": self.prompt_evaluation_suites,
            "tool_versions": self.tool_versions,
            **self.config.trace_metadata(),
        }
