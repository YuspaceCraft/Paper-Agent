"""Shared operation contract regression tests."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent.core.contracts import (
    AgentError,
    ErrorType,
    OperationKind,
    OperationOutcome,
    OperationResult,
    Permission,
    ResultMeta,
    ToolSpec,
)
from agent.core.tool_gateway import ToolGateway
from agent.core.tool_registry import ToolRegistry
from agent.core.idempotency import SQLiteIdempotencyStore


def test_operation_result_rejects_ambiguous_outcomes():
    error = AgentError(
        error_type=ErrorType.TOOL,
        code="TOOL_EXECUTION_FAILED",
        message="failed",
        user_message="failed",
    )
    with pytest.raises(ValidationError):
        OperationResult(
            kind=OperationKind.TOOL,
            operation_id="op",
            outcome=OperationOutcome.SUCCEEDED,
            error=error,
            meta=ResultMeta(),
        )


def test_operation_envelope_uses_outcome_not_ok():
    result = OperationResult(
        kind=OperationKind.TOOL,
        operation_id="op",
        outcome=OperationOutcome.SUCCEEDED,
        data={"value": 1},
    )
    payload = json.loads(result.to_envelope())
    assert payload["outcome"] == "succeeded"
    assert "ok" not in payload
    with pytest.raises(ValidationError):
        OperationResult(
            kind=OperationKind.TOOL,
            operation_id="op",
            outcome=OperationOutcome.FAILED,
            meta=ResultMeta(),
        )


def test_tool_gateway_rejects_invalid_output_schema():
    async def call(name, args):
        return '{"outcome": "succeeded", "data": {"answer": 1}}'

    spec = ToolSpec(
        name="typed_tool",
        output_schema={
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
    )
    gateway = ToolGateway(call, ToolRegistry([spec]))
    ctx = SimpleNamespace(
        thread_id="test",
        execution_id="turn",
        permissions={Permission.READ},
    )
    outcome = asyncio.run(gateway.invoke("typed_tool", {}, ctx=ctx))
    assert outcome.outcome.value == "failed"
    assert outcome.error.code == "TOOL_OUTPUT_INVALID"
    assert outcome.to_operation_result(spec=spec, ctx=ctx).outcome.value == "failed"


def test_known_no_effect_failure_is_not_reported_as_indeterminate():
    async def call(name, args):
        return (
            '{"outcome": "failed", "error_type": "param_error", '
            '"error": "bad input"}'
        )

    spec = ToolSpec(
        name="write",
        side_effect=True,
        idempotency_scope="task",
        permissions={Permission.WRITE},
    )
    gateway = ToolGateway(call, ToolRegistry([spec]))
    ctx = SimpleNamespace(
        thread_id="test",
        execution_id="turn",
        permissions={Permission.WRITE},
    )
    first = asyncio.run(gateway.invoke("write", {"x": 1}, ctx=ctx))
    second = asyncio.run(gateway.invoke("write", {"x": 1}, ctx=ctx))
    assert first.error.code == "TOOL_ARGS_INVALID"
    assert second.error.code == "SIDE_EFFECT_PREVIOUSLY_FAILED"
    assert second.error.effect_applied == "no"


def test_sqlite_idempotency_records_known_no_effect_failure(tmp_path):
    async def run():
        store = SQLiteIdempotencyStore(tmp_path / "idempotency.db")
        first = await store.reserve("key", tool_name="write", tool_version="1")
        assert first.acquired is True
        await store.reject("key", "TOOL_ARGS_INVALID")
        second = await store.reserve("key", tool_name="write", tool_version="1")
        assert second.acquired is False
        assert second.record is not None
        assert second.record.status == "failed"
        assert second.record.error_code == "TOOL_ARGS_INVALID"

    asyncio.run(run())
