"""
memory.py — MemoryManager: structured context assembly for agent nodes.

v1: buffer + summary + profile → compact snapshot for downstream consumption.
Replaces ad-hoc context building scattered across nodes.

Architecture (Letta/LangGraph pattern):
  - Buffer zone: last K messages verbatim (~1200 tokens)
  - Summary zone: older messages compressed by LLM (~800 tokens)
  - Profile: user preferences persisted to disk (cross-session)

Pure-code snapshot assembly. Summary regeneration is lazy, triggered
when buffer overflows (~every 6 messages). Profile is JSON-file-backed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .prompts import MEMORY_SUMMARY
from .prompt_store import get_prompt


# ---- token estimation ----
# Try tiktoken for accurate counting; fall back to char/2 heuristic.
# cl100k_base is a reasonable cross-model proxy — within ±20% for
# qwen/dashscope tokenizers on mixed CJK/Latin text. The char/2
# heuristic underestimates CJK by ~40% (1 CJK char ≈ 1.5-2 tokens).

_tiktoken_enc = None

def _get_tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_enc = False  # sentinel: tried and failed
    return _tiktoken_enc if _tiktoken_enc is not False else None


def _estimate_tokens(text: str) -> int:
    enc = _get_tiktoken()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Conservative fallback when tiktoken is unavailable. CJK is close to one
    # token per character, while Latin text is roughly four characters/token.
    cjk = sum(
        1
        for ch in text
        if "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"
    )
    return max(1, cjk + (len(text) - cjk) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate by the same tokenizer used for budget accounting."""
    if max_tokens <= 0:
        return ""
    enc = _get_tiktoken()
    if enc is not None:
        try:
            token_ids = enc.encode(text)
            if len(token_ids) <= max_tokens:
                return text
            marker = "\n... (truncated)"
            marker_ids = enc.encode(marker)
            if len(marker_ids) >= max_tokens:
                return enc.decode(token_ids[:max_tokens])
            keep = max(0, max_tokens - len(marker_ids))
            return enc.decode(token_ids[:keep]) + marker
        except Exception:
            pass

    max_chars = max_tokens
    if len(text) <= max_chars:
        return text
    marker = "\n... (truncated)"
    if len(marker) >= max_chars:
        return text[:max_chars]
    return text[:max_chars - len(marker)] + marker


def _truncate_to_tokens_tail(text: str, max_tokens: int) -> str:
    """Token-budget truncation that preserves the newest tail of a transcript."""
    if max_tokens <= 0:
        return ""
    enc = _get_tiktoken()
    if enc is not None:
        try:
            token_ids = enc.encode(text)
            if len(token_ids) <= max_tokens:
                return text
            marker = "... (older content truncated)\n"
            marker_ids = enc.encode(marker)
            keep = max(0, max_tokens - len(marker_ids))
            return marker + enc.decode(token_ids[-keep:]) if keep else enc.decode(
                token_ids[-max_tokens:]
            )
        except Exception:
            pass
    if len(text) <= max_tokens:
        return text
    return text[-max_tokens:]


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace a small state file atomically so a crash cannot leave bad JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


# ---- profile persistence ----

def _profile_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    path = root / ".demo" / "memory" / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ponytail: cache profile in memory — read-once, invalidate on save.
# profile.json is <1KB and edited rarely, so per-turn disk I/O was
# pure waste. If multi-process writes become common, add mtime check.
_profile_cache: dict | None = None
_profile_mtime_ns: int | None = None


def load_profile() -> dict:
    """Load user profile from disk. Returns empty dict on first run.

    Cached in memory after first read. Call save_profile() to persist
    changes and invalidate the cache.
    """
    global _profile_cache, _profile_mtime_ns
    path = _profile_path()
    try:
        mtime_ns = path.stat().st_mtime_ns
        if _profile_cache is not None and mtime_ns == _profile_mtime_ns:
            return _profile_cache
        _profile_cache = json.loads(path.read_text(encoding="utf-8"))
        _profile_mtime_ns = mtime_ns
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        _profile_cache = {}
        _profile_mtime_ns = None
    return _profile_cache


def save_profile(profile: dict) -> None:
    """Persist user profile and invalidate in-memory cache."""
    global _profile_cache, _profile_mtime_ns
    path = _profile_path()
    _atomic_write_text(
        path, json.dumps(profile, ensure_ascii=False, indent=2),
    )
    _profile_cache = profile
    try:
        _profile_mtime_ns = path.stat().st_mtime_ns
    except OSError:
        _profile_mtime_ns = None


# ---- LLM helper (inline to avoid circular import from nodes.py) ----

async def _summarize_with_llm(prompt: str, config=None) -> str:
    """One-shot LLM call for conversation summarization.

    config：当前节点 runnable config（LangSmith 嵌套用，可空）。
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from .nodes import _get_model

    model = _get_model(config or {"configurable": {}}, task="summary")
    from evaluation.trace_wrap import traced_ainvoke
    response = await traced_ainvoke(model, [
        SystemMessage(content="You are a precise conversation summarizer."),
        HumanMessage(content=prompt),
    ], node="memory", config=config)
    return response.content if hasattr(response, "content") else str(response)


# ---- MemoryManager ----

class MemoryManager:
    """Assembles conversation context for downstream agent nodes.

    All mutable state lives in AgentState (persisted by LangGraph checkpointer).
    This class is stateless — a pure assembler.
    """

    BUFFER_SIZE = 6        # last N messages kept verbatim in snapshot
    # ponytail: sub-budgets scale with SNAPSHOT_MAX_TOKENS.
    # Summary: 30% (compressed history), Buffer: 50% (recent verbatim),
    # remaining 20% for profile + overhead.
    # Env vars override the computed defaults.
    SNAPSHOT_MAX_TOKENS = int(os.getenv("SNAPSHOT_MAX_TOKENS", "8000"))
    SUMMARY_MAX_TOKENS = int(os.getenv(
        "SUMMARY_MAX_TOKENS", str(int(SNAPSHOT_MAX_TOKENS * 0.30))))
    BUFFER_MAX_TOKENS = int(os.getenv(
        "BUFFER_MAX_TOKENS", str(int(SNAPSHOT_MAX_TOKENS * 0.50))))
    SUMMARY_REFRESH_MESSAGES = int(os.getenv(
        "MEMORY_SUMMARY_REFRESH_MESSAGES", "6"))

    # ---- public API ----

    def build_snapshot(
        self, state: dict, max_tokens: int | None = None, *,
        include_profile: bool = True,
    ) -> str:
        """Build a compact context snapshot from conversation history.

        Pure code — no LLM calls. Uses cached summary from state.
        Returns a string ready for injection into system prompts.
        """
        max_tokens = max_tokens or self.SNAPSHOT_MAX_TOKENS
        messages = state.get("messages", [])
        profile = load_profile() if include_profile else {}

        parts: list[str] = []
        tokens_used = 0

        # 1. Profile section
        profile_text = self._format_profile(profile)
        if profile_text:
            parts.append(profile_text)
            tokens_used += _estimate_tokens(profile_text)

        # 2. Summary section (older messages, cached)
        summary = state.get("summary_cache", "")
        if len(messages) > self.BUFFER_SIZE and summary:
            budget = min(self.SUMMARY_MAX_TOKENS, max_tokens - tokens_used - 200)
            if budget > 0:
                truncated = _truncate_to_tokens(summary, budget)
                parts.append(f"## Earlier Conversation (summary)\n{truncated}")
                tokens_used += _estimate_tokens(truncated) + 30

        # 3. Buffer section (recent messages verbatim)
        buffer_msgs = messages[-self.BUFFER_SIZE:]
        if buffer_msgs:
            budget = min(
                self.BUFFER_MAX_TOKENS, max_tokens - tokens_used - 100
            )
            # Format with a generous character allowance, then apply the real
            # tokenizer budget. This keeps Latin context from using only ~25%
            # of its token allowance while the tail truncation still bounds CJK.
            buffer_text = self._format_buffer(
                buffer_msgs, max(budget, budget * 8),
            )
            if buffer_text:
                buffer_text = _truncate_to_tokens_tail(buffer_text, budget)
                parts.append(f"## Recent Conversation\n{buffer_text}")

        return "\n\n".join(parts)

    def build_snapshot_with_decision(
        self, state: dict, max_tokens: int | None = None, *,
        include_profile: bool = True,
    ) -> tuple[str, dict]:
        """Build a snapshot plus the budget decision used to produce it.

        The snapshot string remains the only prompt payload; the accompanying
        metadata is stored in graph state/trace for debugging and evaluation.
        It deliberately contains no conversation text.
        """
        budget = max_tokens or self.SNAPSHOT_MAX_TOKENS
        snapshot = self.build_snapshot(
            state, max_tokens=budget, include_profile=include_profile,
        )
        estimated = _estimate_tokens(snapshot)
        messages = state.get("messages", [])
        decision = {
            "max_tokens": budget,
            "estimated_tokens": estimated,
            "utilization": round(estimated / budget, 4) if budget else 0.0,
            "message_count": len(messages),
            "buffer_message_count": min(len(messages), self.BUFFER_SIZE),
            "summary_used": bool(
                len(messages) > self.BUFFER_SIZE and state.get("summary_cache", "")
            ),
            "profile_used": bool(load_profile()) if include_profile else False,
            # The assembler reserves space before appending content. A near-full
            # result therefore means a later zone was curtailed, not an overflow.
            "truncated": estimated >= max(1, int(budget * 0.95)),
        }
        return snapshot, decision

    def needs_summary_update(self, state: dict) -> bool:
        """Check if older messages (beyond buffer) need re-summarization."""
        messages = state.get("messages", [])
        if len(messages) <= self.BUFFER_SIZE:
            return False
        older_count = len(messages) - self.BUFFER_SIZE
        try:
            through_seq = max(0, int(state.get("summary_through_seq", 0) or 0))
        except (TypeError, ValueError):
            through_seq = 0
        if not state.get("summary_cache"):
            return older_count > 0
        uncovered = older_count - min(older_count, through_seq)
        return uncovered >= max(1, self.SUMMARY_REFRESH_MESSAGES)

    async def regenerate_summary(self, state: dict, config=None) -> str:
        """Generate/update compressed summary of messages beyond buffer.

        Called inline on the turn where buffer overflows (~every 6 messages).
        Adds ~1-2s latency on that turn; subsequent turns use cached result.
        """
        messages = state.get("messages", [])
        older = messages[:-self.BUFFER_SIZE]
        existing = state.get("summary_cache", "")

        # Build summary input that prioritizes entity-carrying messages:
        # user questions + AI answers in full; tool results trimmed to header.
        older_text = self._format_for_summary(older)

        template = get_prompt("MEMORY_SUMMARY", MEMORY_SUMMARY)
        prompt = template.format(
            existing=existing if existing else "(none - first summary)",
            older_text=older_text,
        )
        return await _summarize_with_llm(prompt, config=config)

    @staticmethod
    def _format_for_summary(messages: list, max_chars: int = 4000) -> str:
        """Format messages for LLM summarization — prioritize entity-carrying
        messages (user questions, AI answers) over raw tool output.

        Tool results are trimmed to header only (paper name + first 200 chars)
        so they don't crowd out the actual Q&A flow the summary needs to capture.
        """
        import re as _re

        lines: list[str] = []
        chars = 0

        for m in messages:
            if not hasattr(m, "type"):
                continue
            content = m.content if hasattr(m, "content") else str(m)
            if not content:
                continue

            if m.type == "human":
                role = "User"
                text = content
            elif m.type == "ai":
                has_calls = hasattr(m, "tool_calls") and m.tool_calls
                if has_calls:
                    role = "Agent (tool call)"
                    text = content[:200]  # tool call text is short anyway
                else:
                    role = "Agent"
                    text = content
            elif m.type == "tool":
                role = "Tool result"
                # Keep header only: paper name + section + first 200 chars.
                # The summary LLM needs entity names, not raw chunk text.
                header = _re.match(
                    r'^(##\s+.+|#\s+.+|\{.+)', content
                )
                if header:
                    text = header.group(0)[:300]
                else:
                    text = content[:200]
            elif m.type == "system":
                continue
            else:
                role = "??"
                text = content[:200]

            remaining = max_chars - chars
            if remaining <= 0:
                lines.append("... (earlier messages omitted)")
                break
            if len(text) > remaining:
                text = text[:remaining] + "..."

            lines.append(f"[{role}]: {text}")
            chars += len(text)

        return "\n".join(lines)

    # ---- private helpers ----

    @staticmethod
    def _format_profile(profile: dict) -> str:
        if not profile:
            return ""
        lines = ["## User Profile"]
        if lang := profile.get("preferred_language"):
            lines.append(f"- Language: {lang}")
        if papers := profile.get("known_papers"):
            lines.append(f"- Known papers: {', '.join(papers[:10])}")
        if topics := profile.get("frequent_topics"):
            lines.append(f"- Frequent topics: {', '.join(topics[:5])}")
        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _format_buffer(messages: list, max_chars: int = 2400,
                       pair_aware: bool = True) -> str:
        """Format a message list as a readable transcript, respecting budget.

        Recent message groups win the budget. Groups are selected newest-first
        (Human/AI pair or tool-call/tool-result run), then rendered in
        chronological order. The previous oldest-first loop could spend the
        whole budget on stale turns and then roll back the newest pair, leaving
        only ``(earlier messages omitted)`` exactly when context was needed.
        """
        if not messages:
            return ""

        def _is_human(message: Any) -> bool:
            return getattr(message, "type", "") == "human"

        def _is_ai(message: Any) -> bool:
            return getattr(message, "type", "") == "ai"

        def _is_tool(message: Any) -> bool:
            return getattr(message, "type", "") == "tool"

        def _has_tool_calls(message: Any) -> bool:
            return bool(getattr(message, "tool_calls", None))

        def _unitize() -> list[list[Any]]:
            """Group messages that must not be separated by truncation."""
            units: list[list[Any]] = []
            index = 0
            while index < len(messages):
                current = messages[index]
                if (
                    pair_aware
                    and _is_human(current)
                    and index + 1 < len(messages)
                    and _is_ai(messages[index + 1])
                    and not _has_tool_calls(messages[index + 1])
                ):
                    units.append([current, messages[index + 1]])
                    index += 2
                    continue
                if pair_aware and _is_ai(current) and _has_tool_calls(current):
                    unit = [current]
                    index += 1
                    while index < len(messages) and _is_tool(messages[index]):
                        unit.append(messages[index])
                        index += 1
                    units.append(unit)
                    continue
                units.append([current])
                index += 1
            return units

        def _line(message: Any) -> str:
            role = "??"
            if hasattr(message, "type"):
                if message.type == "human":
                    role = "User"
                elif message.type == "ai":
                    role = (
                        "Agent (tool call)"
                        if _has_tool_calls(message) else "Agent"
                    )
                elif message.type == "tool":
                    role = "Tool result"
                elif message.type == "system":
                    return ""

            content = (
                message.content if hasattr(message, "content")
                else str(message)
            )
            if not content and _has_tool_calls(message):
                names = [
                    str(call.get("name", ""))
                    for call in message.tool_calls
                    if isinstance(call, dict)
                ]
                content = f"calls {', '.join(name for name in names if name)}"
            if not content:
                return ""
            return f"[{role}]: {content}"

        def _render_unit(unit: list[Any], limit: int) -> str:
            lines: list[str] = []
            used = 0
            for message in unit:
                line = _line(message)
                if not line:
                    continue
                separator = 1 if lines else 0
                remaining = limit - used - separator
                if remaining <= 0:
                    break
                if len(line) > remaining:
                    marker = "..."
                    keep = max(0, remaining - len(marker))
                    line = line[:keep] + (marker if remaining >= len(marker) else "")
                lines.append(line)
                used += len(line) + separator
                if line.endswith("..."):
                    break
            return "\n".join(lines)

        units = _unitize()
        selected: list[tuple[list[Any], str]] = []
        remaining = max(0, int(max_chars))
        omitted = False
        for unit in reversed(units):
            rendered = _render_unit(unit, remaining)
            if not rendered:
                omitted = True
                break
            cost = len(rendered) + (1 if selected else 0)
            if cost > remaining:
                # The newest unit may itself exceed the budget. Keep a
                # truncated prefix rather than dropping all recent context.
                if not selected:
                    rendered = _render_unit(unit, remaining)
                    if rendered:
                        selected.append((unit, rendered))
                omitted = True
                break
            selected.append((unit, rendered))
            remaining -= cost

        if not selected:
            return "... (earlier messages omitted)"
        selected.reverse()
        parts = [rendered for _unit, rendered in selected]
        if omitted:
            parts.insert(0, "... (earlier messages omitted)")
        output = "\n".join(parts)
        if len(output) <= max_chars:
            return output
        # Preserve the newest tail if the omission marker itself consumed the
        # tiny remaining budget.
        if max_chars <= len("... (earlier messages omitted)"):
            return output[-max_chars:] if max_chars else ""
        marker = "... (earlier messages omitted)"
        tail_size = max(0, max_chars - len(marker) - 1)
        return f"{marker}\n{output[-tail_size:]}" if tail_size else marker


# ---- singleton ----

_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
