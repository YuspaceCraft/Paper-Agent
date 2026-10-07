"""Deterministic tool permission, approval and idempotency policy (ADR-0003).

Policy answers three questions before any adapter runs:

1. may this caller use this tool at all (permission matrix),
2. does the call need a human approval first,
3. what key makes the call safe to repeat after a crash.

The decision is data, not prompt text: it is recorded in the trace so a denied
or approved call can be explained afterwards. Nothing here calls an LLM.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .contracts import AgentError, ErrorType, Permission, ToolSpec

# Roles are deliberately coarse. The desktop app is single-user, so the local
# actor keeps every non-admin capability; restricted roles exist so a subagent
# or an evaluation harness can be scoped down without code changes.
ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    "user": frozenset({
        Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.NETWORK,
    }),
    "admin": frozenset({
        Permission.READ, Permission.WRITE, Permission.EXECUTE,
        Permission.NETWORK, Permission.ADMIN, Permission.SECRETS,
    }),
    "worker": frozenset({Permission.READ, Permission.NETWORK}),
    "observer": frozenset({Permission.READ}),
}

# Side effects fail closed by default. Operators may explicitly disable the
# gate for isolated tests or trusted automation with ``AGENT_TOOL_APPROVAL=0``.
APPROVAL_ENV = "AGENT_TOOL_APPROVAL"


def approval_enforced() -> bool:
    raw = (os.getenv(APPROVAL_ENV, "") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def permissions_for_roles(roles: Iterable[str]) -> set[Permission]:
    granted: set[Permission] = set()
    for role in roles or ():
        granted |= set(ROLE_PERMISSIONS.get(str(role), frozenset()))
    return granted


def _strict_schema(schema: dict) -> dict:
    """Copy a schema and reject undeclared object properties by default."""
    out = deepcopy(schema)

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        node_type = node.get("type")
        is_object = (
            node_type == "object"
            or (isinstance(node_type, list) and "object" in node_type)
        )
        if (
            is_object
            and isinstance(properties, dict)
            and "additionalProperties" not in node
        ):
            node["additionalProperties"] = False
        if isinstance(properties, dict):
            for child in properties.values():
                _walk(child)
        items = node.get("items")
        if isinstance(items, dict):
            _walk(items)
        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            _walk(additional)

    _walk(out)
    return out


def _fallback_schema_issues(schema: dict, args: dict) -> list[str]:
    """Minimal validation when ``jsonschema`` is not installed."""
    issues: list[str] = []
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required") or []
    for key in required:
        if key not in args:
            issues.append(f"$.{key}: required field is missing")
    extra = sorted(set(args) - set(properties))
    if extra and schema.get("additionalProperties") is not True:
        issues.append(f"$: unexpected field(s): {', '.join(extra)}")
    type_map = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": list, "object": dict,
    }
    for key, value in args.items():
        child = properties.get(key)
        if not isinstance(child, dict):
            continue
        expected = type_map.get(child.get("type"))
        if expected is not None and not isinstance(value, expected):
            issues.append(f"$.{key}: type validation failed")
        enum = child.get("enum")
        if isinstance(enum, list) and value not in enum:
            issues.append(f"$.{key}: enum validation failed")
    return issues


@dataclass(frozen=True)
class ToolCallDecision:
    """The outcome of a policy evaluation, safe to log verbatim."""

    action: Literal["allow", "deny", "approval"]
    reason: str
    missing_permissions: tuple[str, ...] = ()

    def allowed(self) -> bool:
        return self.action == "allow"


@dataclass
class ToolPolicy:
    """Permission matrix + approval gate + argument/idempotency helpers."""

    enforce_approval: bool | None = None
    # Set of tool names pre-approved by an operator (config centre / tests).
    preapproved: set[str] = field(default_factory=set)

    def _enforced(self) -> bool:
        return approval_enforced() if self.enforce_approval is None else self.enforce_approval

    def decide(self, spec: ToolSpec | None, *, ctx: Any = None,
               tool_name: str = "") -> ToolCallDecision:
        """Classify one call as allow / deny / approval."""
        if spec is None:
            # The registry is the authority on what exists; an unknown name at
            # this layer is a wiring bug, so fail closed as a policy decision.
            return ToolCallDecision("deny", "tool is not in the registry")

        if ctx is not None:
            granted = set(getattr(ctx, "permissions", ()) or ())
            missing = tuple(sorted(p.value for p in spec.permissions - granted))
            if missing:
                return ToolCallDecision(
                    "deny", f"missing permissions: {','.join(missing)}", missing,
                )

        if not spec.requires_approval():
            return ToolCallDecision("allow", "read-only tool")
        if spec.name in self.preapproved:
            return ToolCallDecision("allow", "pre-approved by operator")
        if self._enforced():
            return ToolCallDecision("approval", "side effect requires approval")
        return ToolCallDecision("allow", "approval gate disabled")

    def denial(self, decision: ToolCallDecision, tool_name: str) -> AgentError:
        """Turn a deny/approval decision into a typed, user-safe error."""
        if decision.action == "approval":
            return AgentError(
                error_type=ErrorType.POLICY,
                code="APPROVAL_REQUIRED",
                message=f"Tool '{tool_name}' has side effects and needs approval.",
                user_message=f"工具“{tool_name}”会修改数据，需要先确认。",
                retryable=False,
                recovery_action="Approve the call, then resume the task.",
                tool_name=tool_name,
            )
        return AgentError(
            error_type=ErrorType.AUTHORIZATION,
            code="TOOL_NOT_PERMITTED",
            message=f"Tool '{tool_name}' denied: {decision.reason}.",
            user_message=f"当前身份没有调用“{tool_name}”的权限。",
            retryable=False,
            recovery_action="Use a tool allowed for this role, or request access.",
            tool_name=tool_name,
        )

    def approval_denied(self, tool_name: str) -> AgentError:
        return AgentError(
            error_type=ErrorType.POLICY,
            code="APPROVAL_DENIED",
            message=f"Tool '{tool_name}' approval was denied.",
            user_message=f"已拒绝工具“{tool_name}”的本次操作。",
            retryable=False,
            recovery_action="No side effect was performed.",
            tool_name=tool_name,
        )

    def validate_arguments(self, spec: ToolSpec, args: dict) -> AgentError | None:
        """Validate against the tool's declared JSON schema before dispatch.

        Unknown top-level fields are rejected unless the schema explicitly sets
        ``additionalProperties: true``.  This prevents model hallucinations from
        being silently dropped or interpreted as defaults by Pydantic adapters.
        """
        schema = spec.input_schema if isinstance(spec.input_schema, dict) else {}
        if not schema:
            return None
        if not isinstance(args, dict):
            return AgentError(
                error_type=ErrorType.VALIDATION,
                code="TOOL_ARGS_INVALID",
                message=f"Tool '{spec.name}' arguments must be an object.",
                user_message=f"工具“{spec.name}”的参数必须是对象。",
                retryable=False,
                recovery_action="Provide arguments as a JSON object.",
                tool_name=spec.name,
            )

        issues: list[str] = []
        try:
            import jsonschema

            strict = _strict_schema(schema)
            validator_cls = jsonschema.validators.validator_for(strict)
            validator = validator_cls(strict)
            for err in sorted(
                validator.iter_errors(args),
                key=lambda e: (list(e.absolute_path), e.message),
            ):
                path = "$"
                for part in err.absolute_path:
                    path += f"[{part}]" if isinstance(part, int) else f".{part}"
                if err.validator == "additionalProperties":
                    local_props = err.schema.get("properties") or {}
                    extra = sorted(set(err.instance) - set(local_props))
                    issues.append(
                        f"{path}: unexpected field(s): {', '.join(extra)}"
                    )
                else:
                    issues.append(f"{path}: {err.validator} validation failed")
                if len(issues) >= 5:
                    break
        except ImportError:
            issues.extend(_fallback_schema_issues(schema, args))
        except Exception as exc:  # invalid schema must fail closed
            issues.append(f"invalid tool schema: {type(exc).__name__}")

        if issues:
            safe = "; ".join(issues[:5])
            return AgentError(
                error_type=ErrorType.VALIDATION,
                code="TOOL_ARGS_INVALID",
                message=f"Tool '{spec.name}' arguments invalid: {safe}",
                user_message=f"工具“{spec.name}”的参数不符合要求：{safe}",
                retryable=True,
                recovery_action="Correct the arguments using the tool schema.",
                tool_name=spec.name,
            )
        return None


def canonical_args(args: Any) -> str:
    """Deterministic, JSON-safe rendering of tool arguments."""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return repr(args)


def idempotency_key(*, thread_id: str, spec: ToolSpec, args: Any,
                    intent: str = "", execution_id: str = "") -> str:
    """``hash(thread, turn, tool@version, canonical_args, intent)``.

    The key is a digest, never the raw arguments, so it can be stored and
    logged without leaking user content or secrets.

    ``execution_id`` scopes replay to one user turn/task. Falling back to
    ``thread_id`` preserves compatibility for direct callers that do not yet
    provide a turn identity.
    """
    payload = "|".join([
        str(thread_id or ""),
        str(execution_id or thread_id or ""),
        f"{spec.name}@{spec.version}",
        canonical_args(args),
        str(intent or ""),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
