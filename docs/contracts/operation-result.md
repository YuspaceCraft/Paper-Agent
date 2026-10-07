# Operation Result Contract

`agent.core.contracts.OperationResult` is the shared business result model for
tools, agents, subagents and background tasks. LangGraph state, tool messages
and SSE events may carry this model, but they are not separate business
contracts.

## 1. Required fields

| field | meaning |
|---|---|
| `schema_version` | contract version, currently `1.0` |
| `kind` | `tool`, `agent`, `subagent` or `task` |
| `operation_id` | stable id for this operation |
| `parent_operation_id` | caller operation, when this is a child operation |
| `outcome` | one authoritative lifecycle outcome |
| `data` | successful or partial payload |
| `error` | typed error for failed, timeout and cancellation outcomes |
| `warnings` | non-fatal gaps; required for `partial` |
| `meta` | trace, attempt, timeout and side-effect metadata |

## 2. Outcomes

`succeeded | partial | failed | timed_out | cancelled | interrupted | skipped`

- `succeeded`: the operation completed and any declared output schema passed.
- `partial`: useful data exists, but the result is incomplete.
- `failed`: the operation did not produce a valid success result.
- `timed_out`: a declared deadline expired.
- `cancelled`: execution was cancelled by the caller or process.
- `interrupted`: execution intentionally paused for approval or input.
- `skipped`: the operation was deliberately not executed.

`interrupted` and `skipped` are not errors. `partial` must carry a warning.
Success and error are mutually exclusive.

## 3. Serialization

In-process code uses `OperationResult.model_dump(mode="json")`.

LangGraph stores the serialized projection in
`AgentState.operation_results`, keyed by operation id. `subagent_results` and
SSE event fields are compatibility views of the same facts.

The legacy string tool envelope remains supported while providers migrate:

```json
{
  "schema_version": "1.0",
  "outcome": "succeeded",
  "operation_id": "tool-call-id",
  "kind": "tool",
  "data": {},
  "meta": {}
}
```

The live parser rejects envelopes containing `ok` with
`TOOL_PROTOCOL_INVALID`. Database migration may read a historical `ok` column
once to backfill `outcome`, but that compatibility exists only in storage
migration and never enters the runtime contract.

## 4. Error and retry rules

Errors cross boundaries as `AgentError`, not as exception text. A retry decision
must consider:

- stable error code;
- idempotency;
- attempt and maximum attempts;
- `retry_after_seconds`;
- `effect_applied` (`no`, `yes` or `unknown`).

Known no-effect failures are recorded as `failed` in the idempotency store.
Ambiguous side-effect failures are recorded as `indeterminate`. This prevents
both duplicate side effects and unnecessarily blocking a request that was
definitively rejected before execution.
