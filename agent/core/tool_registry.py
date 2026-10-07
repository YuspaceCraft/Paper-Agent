"""Governance registry layered over legacy provider ``ToolDef`` objects."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from .contracts import Permission, ToolSpec


def tool_spec_from_def(tooldef) -> ToolSpec:
    """Adapt the existing provider description without changing providers."""
    annotations = getattr(tooldef, "annotations", {}) or {}
    source = str(getattr(tooldef, "source", "") or "")
    if source == "mcp":
        # MCP annotations are optional. Missing/None must fail closed because
        # an unclassified remote tool could write or execute.
        readonly = annotations.get("readOnlyHint") is True
    else:
        readonly = annotations.get("readOnlyHint", True) is not False
    open_world = annotations.get("openWorldHint", False) is True
    permissions = {Permission.READ} if readonly else {Permission.WRITE}
    if open_world:
        permissions.add(Permission.NETWORK)
    return ToolSpec(
        name=str(tooldef.name),
        version=str(annotations.get("version") or "1"),
        description=str(getattr(tooldef, "description", "")),
        input_schema=dict(getattr(tooldef, "parameters", {}) or {}),
        permissions=permissions,
        side_effect=not readonly or annotations.get("destructiveHint") is True,
        idempotency_scope="request" if annotations.get("idempotentHint") else "none",
        owner=source,
        tags={str(getattr(tooldef, "source", ""))} - {""},
    )


class ToolRegistry:
    """Immutable-per-build directory used by dispatch, config snapshots and UI."""

    def __init__(self, specs: Iterable[ToolSpec] = ()):
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"duplicate tool spec: {spec.name}")
            self._specs[spec.name] = spec

    @classmethod
    def from_tooldefs(cls, tooldefs: Iterable) -> "ToolRegistry":
        return cls(tool_spec_from_def(td) for td in tooldefs)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._specs.values())

    @property
    def registry_hash(self) -> str:
        payload = [spec.model_dump(mode="json") for spec in sorted(
            self._specs.values(), key=lambda item: item.name
        )]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
