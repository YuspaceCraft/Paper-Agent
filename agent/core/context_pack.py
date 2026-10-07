"""Budgeted Context Pack assembly (platform design §6.2, ADR-0005).

Every turn's prompt context is assembled from named zones with explicit token
budgets instead of one growing string:

    invariant     system policy / permissions / output contract (fixed cap)
    task          current goal, plan, approval state, workspace binding (15%)
    conversation  recent dialogue + compressed summary (25%)
    retrieved     retrieved chunks / citations (45%)
    memory        preferences, long-term facts, workspace state (10%)
    reserve       withheld for tool results and model output (5%)

Budgets are derived from the model's real context window, not from a character
estimate. Each zone reports tokens, provenance and whether it was curtailed;
that record (``ContextPack.decision``) goes to the trace so a truncated prompt
can be explained afterwards. Prompt text never enters the decision metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .contracts import ExecutionContext
from .memory_policy import MemoryPolicy, MemoryRecord, records_from_profile


class ContextZone(str, Enum):
    INVARIANT = "invariant"
    TASK = "task"
    CONVERSATION = "conversation"
    RETRIEVED = "retrieved"
    MEMORY = "memory"
    RESERVE = "reserve"


@dataclass(frozen=True)
class ZoneShare:
    """Share of the usable window plus how the zone may be curtailed."""

    share: float
    strategy: str
    keep_full: bool = False


# Shares of the usable window (context window minus the output reserve).
# ``invariant`` is a fixed cap and ``reserve`` is withheld, not rendered.
ZONE_SHARES: dict[ContextZone, ZoneShare] = {
    ContextZone.INVARIANT: ZoneShare(0.0, "never truncated", keep_full=True),
    ContextZone.TASK: ZoneShare(0.15, "keep whole"),
    ContextZone.CONVERSATION: ZoneShare(0.25, "pair-aware truncate"),
    ContextZone.RETRIEVED: ZoneShare(0.45, "dedupe + diversity, then truncate"),
    ContextZone.MEMORY: ZoneShare(0.10, "confidence / TTL filter"),
    ContextZone.RESERVE: ZoneShare(0.05, "withheld", keep_full=True),
}

INVARIANT_TOKEN_CAP = 1500


@dataclass
class ZoneContent:
    zone: ContextZone
    content: str
    tokens: int
    budget: int
    strategy: str
    truncated: bool = False
    entries: list[dict] = field(default_factory=list)

    def trace_view(self) -> dict:
        """Metadata only: counts, provenance and reasons — no prompt text."""
        return {
            "budget": self.budget,
            "tokens": self.tokens,
            "strategy": self.strategy,
            "truncated": self.truncated,
            "entries": self.entries,
        }


@dataclass
class ContextPack:
    zones: dict[str, ZoneContent]
    decision: dict

    def zone(self, zone: ContextZone | str) -> ZoneContent | None:
        key = zone.value if isinstance(zone, ContextZone) else str(zone)
        return self.zones.get(key)

    def render(self, zones: Iterable[ContextZone] | None = None) -> str:
        """Join the selected zones in canonical order into one prompt payload."""
        order = list(zones) if zones is not None else [
            ContextZone.INVARIANT, ContextZone.TASK, ContextZone.CONVERSATION,
            ContextZone.RETRIEVED, ContextZone.MEMORY,
        ]
        parts: list[str] = []
        for zone in order:
            item = self.zone(zone)
            if item is not None and item.content.strip():
                parts.append(item.content)
        return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    from ..memory import _estimate_tokens as estimate

    return estimate(text or "")


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    from ..memory import _truncate_to_tokens as truncate

    return truncate(text, max_tokens)


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _safe_score(value: Any) -> float:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _memory_source_rank(source_ref: str) -> int:
    source = str(source_ref or "").casefold()
    if source in {"user", "manual", "api:user"} or "user" in source:
        return 2
    if source in {"conversation", "chat"} or "thread" in source:
        return 1
    return 0


def _resolve_memory_conflicts(
    records: list[MemoryRecord],
) -> tuple[list[MemoryRecord], list[dict]]:
    """Resolve same-key values before any of them reach the prompt."""
    grouped: dict[str, list[MemoryRecord]] = {}
    passthrough: list[MemoryRecord] = []
    for record in records or ():
        key = str(record.semantic_key or "")
        if not key or record.status != "active":
            passthrough.append(record)
            continue
        grouped.setdefault(key, []).append(record)

    kept: list[MemoryRecord] = list(passthrough)
    audit: list[dict] = []
    for key, items in grouped.items():
        distinct = {
            " ".join(item.content.casefold().split()): item for item in items
        }
        values = list(distinct.values())
        if len(values) <= 1:
            kept.extend(values)
            continue
        ordered = sorted(
            values,
            key=lambda item: (
                _memory_source_rank(item.source_ref),
                item.confidence,
                item.created_at,
            ),
            reverse=True,
        )
        top = ordered[0]
        second = ordered[1]
        top_rank = _memory_source_rank(top.source_ref)
        second_rank = _memory_source_rank(second.source_ref)
        if top_rank > second_rank or top.confidence >= second.confidence + 0.1:
            kept.append(top)
            for item in ordered[1:]:
                audit.append({
                    **item.trace_view(),
                    "kept": False,
                    "reason": f"superseded by {top.memory_id} for {key}",
                })
            continue
        for item in ordered:
            audit.append({
                **item.trace_view(),
                "kept": False,
                "reason": f"conflicting value for {key} requires user resolution",
            })
    return kept, audit


def _retrieved_from_messages(messages: Iterable[Any]) -> list[dict]:
    """Extract retrieval evidence from recent tool messages.

    ``search_papers`` envelopes are expanded to individual chunk entries;
    ``fetch_content`` markdown is kept as one evidence item.  Other tools are
    ignored so file listings or task status do not crowd the retrieved budget.
    """
    evidence: list[dict] = []
    eligible = {"search_papers", "fetch_content"}
    seen_sources: set[str] = set()

    recent = list(messages or ())[-20:]
    for rank, message in enumerate(reversed(recent)):
        if getattr(message, "type", "") != "tool":
            continue
        name = str(getattr(message, "name", "") or "")
        if name not in eligible:
            continue
        content = str(getattr(message, "content", "") or "")
        if not content.strip():
            continue
        call_id = str(getattr(message, "tool_call_id", "") or f"tool-{rank}")

        if name == "search_papers":
            try:
                payload = json.loads(content)
                results = ((payload.get("data") or {}).get("results") or [])
            except (TypeError, ValueError):
                results = []
            expanded = False
            for item in results:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or item.get("generation_text") or "")
                source = str(item.get("chunk_id") or item.get("paper") or "")
                if not text.strip() or not source or source in seen_sources:
                    continue
                seen_sources.add(source)
                evidence.append({
                    "text": text,
                    "source_ref": source,
                    "score": float(item.get("score") or 0.0) + 1.0 / (rank + 2),
                })
                expanded = True
            if expanded:
                continue

        source = f"tool:{name}:{call_id}"
        if source in seen_sources:
            continue
        seen_sources.add(source)
        evidence.append({
            "text": content,
            "source_ref": source,
            "score": 1.0 / (rank + 1),
        })
    return evidence


class ContextManager:
    """Builds a :class:`ContextPack`; pure code, no LLM calls."""

    # Greedy diversity penalty for the retrieved zone (MMR-lite).
    DIVERSITY_PENALTY = 0.5

    def zone_budgets(
        self,
        ctx: ExecutionContext | None = None,
        *,
        usable_tokens: int | None = None,
        invariant_tokens: int | None = None,
    ) -> dict[ContextZone, int]:
        if usable_tokens is None:
            usable_tokens = (ctx.budget.usable_context_tokens() if ctx
                             else 32768 - 4096)
        usable_tokens = max(0, int(usable_tokens))
        # The invariant prompt has priority, but its actual size must still be
        # charged to the total.  A dynamic prompt that grows above the nominal
        # cap therefore shrinks flexible zones instead of silently overflowing
        # the model window.
        requested_invariant = (
            INVARIANT_TOKEN_CAP if invariant_tokens is None
            else max(0, int(invariant_tokens))
        )
        invariant_budget = min(usable_tokens, requested_invariant)
        flexible = max(0, usable_tokens - invariant_budget)
        budgets = {
            ContextZone.INVARIANT: invariant_budget,
            ContextZone.RESERVE: int(flexible * ZONE_SHARES[ContextZone.RESERVE].share),
        }
        spent = sum(budgets.values())
        for zone in (ContextZone.TASK, ContextZone.CONVERSATION,
                     ContextZone.RETRIEVED, ContextZone.MEMORY):
            budgets[zone] = int(flexible * ZONE_SHARES[zone].share)
            spent += budgets[zone]
        # Any rounding slack belongs to retrieval, the zone that benefits most
        # from extra room and is already deduplicated.
        budgets[ContextZone.RETRIEVED] += max(0, usable_tokens - spent)
        return budgets

    def build(
        self,
        ctx: ExecutionContext | None,
        state: dict,
        *,
        invariant: str = "",
        retrieved: list[dict] | None = None,
        memory_records: list[MemoryRecord] | None = None,
        memory_policy: MemoryPolicy | None = None,
    ) -> ContextPack:
        invariant_tokens = _estimate_tokens(invariant)
        budgets = self.zone_budgets(
            ctx, invariant_tokens=invariant_tokens,
        )
        zones: dict[str, ZoneContent] = {}

        zones[ContextZone.INVARIANT.value] = self._invariant_zone(
            invariant, budgets[ContextZone.INVARIANT]
        )
        zones[ContextZone.TASK.value] = self._task_zone(
            state, budgets[ContextZone.TASK]
        )
        zones[ContextZone.CONVERSATION.value] = self._conversation_zone(
            state, budgets[ContextZone.CONVERSATION]
        )
        zones[ContextZone.RETRIEVED.value] = self._retrieved_zone(
            _retrieved_from_messages(state.get("messages") or [])
            if retrieved is None else retrieved,
            budgets[ContextZone.RETRIEVED],
        )
        zones[ContextZone.MEMORY.value] = self._memory_zone(
            memory_records, memory_policy, budgets[ContextZone.MEMORY]
        )
        reserve = budgets[ContextZone.RESERVE]
        zones[ContextZone.RESERVE.value] = ZoneContent(
            zone=ContextZone.RESERVE, content="", tokens=0, budget=reserve,
            strategy=ZONE_SHARES[ContextZone.RESERVE].strategy,
        )

        used = sum(z.tokens for z in zones.values())
        usable = sum(budgets.values())
        overflow_tokens = max(0, used - usable)
        decision = {
            "max_tokens": usable,
            "estimated_tokens": used,
            "utilization": round(used / usable, 4) if usable else 0.0,
            "budget_exceeded": overflow_tokens > 0,
            "overflow_tokens": overflow_tokens,
            "strategy": "zone-budgeted context pack",
            "zones": {name: z.trace_view() for name, z in zones.items()},
            "truncated_zones": sorted(
                name for name, z in zones.items() if z.truncated
            ),
        }
        return ContextPack(zones=zones, decision=decision)

    # ---- per-zone assembly ----

    def _invariant_zone(self, text: str, budget: int) -> ZoneContent:
        content = text or ""
        tokens = _estimate_tokens(content)
        # Invariant policy is never silently cut: report the overrun instead so
        # the prompt version (not the truncation) gets fixed.
        return ZoneContent(
            zone=ContextZone.INVARIANT, content=content, tokens=tokens,
            budget=budget, strategy=ZONE_SHARES[ContextZone.INVARIANT].strategy,
            truncated=False,
            entries=[{"source_ref": "invariant:policy", "tokens": tokens,
                      "kept": True, "reason": "fixed cap, never truncated"}],
        )

    def _task_zone(self, state: dict, budget: int) -> ZoneContent:
        lines: list[str] = []
        context = state.get("context") or {}
        query = ""
        for m in reversed(state.get("messages") or []):
            if getattr(m, "type", "") == "human":
                query = str(getattr(m, "content", "") or "")
                break
        if query:
            lines.append(f"- Goal: {query[:400]}")
        if intent := state.get("intent"):
            lines.append(f"- Intent: {intent}")
        if mode := (state.get("mode") or state.get("requested_mode")):
            lines.append(f"- Mode: {mode}")
        if profile := state.get("optimization_profile"):
            lines.append(f"- Optimization profile: {profile}")
        for key, label in (("active_doc_id", "Active document"),
                           ("active_project", "Active experiment project"),
                           ("study_topic", "Study topic")):
            if value := context.get(key):
                lines.append(f"- {label}: {value}")
        if exps := context.get("recent_experiments"):
            lines.append(f"- Recent experiments: {', '.join(map(str, exps[:5]))}")
        plan = state.get("plan") or []
        if plan:
            done = sum(1 for s in plan if str(s.get("status")) == "done")
            lines.append(f"- Plan: {done}/{len(plan)} steps done")

        content = "\n".join(lines)
        tokens = _estimate_tokens(content)
        truncated = tokens > budget
        if truncated:
            content = _truncate_to_tokens(content, budget)
            tokens = _estimate_tokens(content)
        return ZoneContent(
            zone=ContextZone.TASK, content=content, tokens=tokens, budget=budget,
            strategy=ZONE_SHARES[ContextZone.TASK].strategy, truncated=truncated,
            entries=[{"source_ref": "task:goal+plan", "tokens": tokens,
                      "kept": True, "reason": "current objective"}],
        )

    def _conversation_zone(self, state: dict, budget: int) -> ZoneContent:
        from ..memory import get_memory_manager

        mm = get_memory_manager()
        content = mm.build_snapshot(
            state, max_tokens=budget, include_profile=False,
        ) if budget > 0 else ""
        tokens = _estimate_tokens(content)
        messages = state.get("messages") or []
        summary_used = bool(len(messages) > mm.BUFFER_SIZE and state.get("summary_cache"))
        return ZoneContent(
            zone=ContextZone.CONVERSATION, content=content, tokens=tokens,
            budget=budget,
            strategy=ZONE_SHARES[ContextZone.CONVERSATION].strategy,
            truncated=tokens >= max(1, int(budget * 0.95)),
            entries=[
                {"source_ref": "conversation:summary", "tokens": tokens,
                 "kept": summary_used, "reason": "compressed history"},
                {"source_ref": "conversation:buffer",
                 "tokens": tokens, "kept": bool(messages),
                 "reason": f"last {min(len(messages), mm.BUFFER_SIZE)} messages, "
                           "tool call/result pairs kept together"},
            ],
        )

    def _retrieved_zone(self, chunks: list[dict], budget: int) -> ZoneContent:
        """Dedupe, then pick a diverse subset that fits the budget."""
        entries: list[dict] = []
        seen: set[str] = set()
        candidates: list[tuple[float, set[str], str, str]] = []
        for item in chunks or []:
            text = str(item.get("text") or item.get("content") or "")
            if not text.strip():
                continue
            source = str(item.get("source_ref") or item.get("chunk_id") or "")
            norm = _normalize(text)
            if norm in seen:
                entries.append({"source_ref": source, "kept": False,
                                "reason": "duplicate of an already-included chunk"})
                continue
            seen.add(norm)
            score = _safe_score(item.get("score"))
            candidates.append((score, set(norm.split()), text, source))

        selected: list[tuple[float, set[str], str, str]] = []
        remaining = sorted(candidates, key=lambda c: c[0], reverse=True)
        while remaining:
            best_idx, best_value = 0, None
            for idx, cand in enumerate(remaining):
                penalty = max(
                    (_jaccard(cand[1], s[1]) for s in selected), default=0.0
                )
                value = cand[0] - self.DIVERSITY_PENALTY * penalty
                if best_value is None or value > best_value:
                    best_idx, best_value = idx, value
            chosen = remaining.pop(best_idx)
            if not selected or best_value > 0 or len(selected) < 3:
                selected.append(chosen)
            else:
                entries.append({"source_ref": chosen[3], "kept": False,
                                "reason": "redundant with selected chunks"})

        parts: list[str] = []
        used = 0
        truncated = False
        for _score, _tokens, text, source in selected:
            cost = _estimate_tokens(text)
            if used + cost > budget:
                truncated = True
                entries.append({"source_ref": source, "kept": False,
                                "reason": "did not fit the retrieved budget"})
                continue
            parts.append(text)
            used += cost
            entries.append({"source_ref": source, "tokens": cost, "kept": True,
                            "reason": "selected by score + diversity"})

        return ZoneContent(
            zone=ContextZone.RETRIEVED, content="\n\n".join(parts), tokens=used,
            budget=budget,
            strategy=ZONE_SHARES[ContextZone.RETRIEVED].strategy,
            truncated=truncated, entries=entries,
        )

    def _memory_zone(self, records: list[MemoryRecord] | None,
                     policy: MemoryPolicy | None, budget: int) -> ZoneContent:
        from .memory_store import get_policy, list_records

        if records is None:
            from ..memory import load_profile

            records = [
                *records_from_profile(load_profile()),
                *list_records(),
            ]
        records, conflict_entries = _resolve_memory_conflicts(records)
        policy = policy or get_policy()
        kept, dropped = policy.filter(records)
        entries = [*conflict_entries, *dropped]
        lines: list[str] = []
        used = 0
        truncated = False
        for record in kept:
            line = record.render()
            cost = _estimate_tokens(line)
            if used + cost > budget:
                truncated = True
                entries.append({"source_ref": record.memory_id, "kept": False,
                                "reason": "did not fit the memory budget"})
                continue
            lines.append(line)
            used += cost
            entries.append({"source_ref": record.memory_id, "tokens": cost,
                            "kept": True, "reason": f"{record.type.value}, "
                                                    f"confidence {record.confidence:.2f}"})
        content = "\n".join(lines)
        return ZoneContent(
            zone=ContextZone.MEMORY, content=content, tokens=used, budget=budget,
            strategy=ZONE_SHARES[ContextZone.MEMORY].strategy,
            truncated=truncated, entries=entries,
        )


def get_context_manager() -> ContextManager:
    return _manager


_manager = ContextManager()
