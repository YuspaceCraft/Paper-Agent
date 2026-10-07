"""Agent runtime core contracts shared by every domain and provider.

Only stable, domain-neutral types live here.  Domain nodes must depend on
these contracts rather than duplicating error, permission or tool metadata.
"""

from .contracts import (
    AgentError,
    Budget,
    ErrorType,
    ExecutionContext,
    OperationKind,
    OperationOutcome,
    OperationResult,
    Permission,
    PromptSpec,
    PromptType,
    ResultMeta,
    RetryPolicy,
    ToolSpec,
    WarningInfo,
)
from .approval import (
    NO_APPROVAL_CONTEXT,
    approval_granted,
    interrupt_values,
    pending_tool_approval,
    request_tool_approval,
    tool_approval_scope,
)
from .configuration import ConfigurationSnapshot, build_configuration_snapshot
from .capability_policy import CapabilityDecision, assess_capability
from .content_safety import (
    ContentSafetyDecision,
    assess_content_safety,
    classify_content_safety,
)
from .cost_history import (
    cost_history_snapshot,
    history_key,
    record_observation,
    record_plan_cost,
)
from .completion_report import build_completion_report
from .confidence_policy import FinalConfidence, calibrate_final_confidence
from .context_pack import ContextManager, ContextPack, ContextZone, get_context_manager
from .execution_context import (
    build_execution_context,
    get_current_execution_context,
    reset_current_execution_context,
    set_current_execution_context,
)
from .idempotency import (
    IdempotencyRecord,
    InMemoryIdempotencyStore,
    Reservation,
    SQLiteIdempotencyStore,
)
from .goal_policy import (
    GoalContract,
    GoalDrift,
    build_goal_contract,
    evaluate_goal_drift,
    goal_drift_feedback,
)
from .input_policy import (
    InputPolicyDecision,
    guard_user_input,
    scan_untrusted_content,
)
from .optimization_policy import (
    OptimizationDecision,
    OptimizationProfile,
    infer_optimization_profile,
    optimization_prompt,
)
from .memory_policy import MemoryPolicy, MemoryRecord, MemoryType, records_from_profile
from .memory_store import (
    approve_record,
    capture_explicit_memory,
    delete_record,
    get_policy,
    list_records,
    memory_path,
    update_policy,
    upsert_record,
)
from .policy import ToolCallDecision, ToolPolicy, permissions_for_roles
from .output_policy import OutputValidation, validate_final_output
from .plan_policy import (
    PlanCostEstimate,
    PlanIssue,
    PlanValidation,
    estimate_and_trim_plan,
    validate_and_repair_plan,
)
from .postmortem import record_turn_postmortem, should_record_postmortem
from .provider_canary import (
    run_model_canary,
    run_moderation_canary,
    run_provider_canary,
)
from .tool_gateway import ToolCallOutcome, ToolGateway
from .tool_registry import ToolRegistry

__all__ = [
    "AgentError", "ErrorType", "Permission", "PromptType", "ToolSpec",
    "OperationKind", "OperationOutcome", "OperationResult", "ResultMeta",
    "WarningInfo",
    "ConfigurationSnapshot", "build_configuration_snapshot",
    "CapabilityDecision", "assess_capability",
    "ContentSafetyDecision", "assess_content_safety",
    "classify_content_safety",
    "cost_history_snapshot", "history_key", "record_observation",
    "record_plan_cost",
    "build_completion_report", "FinalConfidence",
    "calibrate_final_confidence",
    "Budget", "ExecutionContext", "PromptSpec", "RetryPolicy",
    "ContextManager", "ContextPack", "ContextZone", "get_context_manager",
    "MemoryPolicy", "MemoryRecord", "MemoryType", "records_from_profile",
    "capture_explicit_memory", "approve_record", "delete_record",
    "get_policy", "list_records",
    "memory_path", "update_policy", "upsert_record",
    "NO_APPROVAL_CONTEXT", "approval_granted", "request_tool_approval",
    "tool_approval_scope", "interrupt_values", "pending_tool_approval",
    "ToolCallDecision", "ToolPolicy", "ToolGateway", "ToolCallOutcome",
    "ToolRegistry", "permissions_for_roles",
    "IdempotencyRecord", "Reservation", "InMemoryIdempotencyStore",
    "SQLiteIdempotencyStore",
    "GoalContract", "GoalDrift", "build_goal_contract",
    "evaluate_goal_drift", "goal_drift_feedback",
    "InputPolicyDecision", "guard_user_input", "scan_untrusted_content",
    "OptimizationDecision", "OptimizationProfile",
    "infer_optimization_profile", "optimization_prompt",
    "OutputValidation", "validate_final_output",
    "PlanCostEstimate", "PlanIssue", "PlanValidation",
    "estimate_and_trim_plan", "validate_and_repair_plan",
    "record_turn_postmortem", "should_record_postmortem",
    "run_model_canary", "run_moderation_canary", "run_provider_canary",
    "build_execution_context", "get_current_execution_context",
    "set_current_execution_context", "reset_current_execution_context",
]
