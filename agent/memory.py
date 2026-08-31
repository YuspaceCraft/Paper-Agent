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
from pathlib import Path
from typing import Any


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
    return max(1, len(text) // 2)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 2
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


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


def load_profile() -> dict:
    """Load user profile from disk. Returns empty dict on first run.

    Cached in memory after first read. Call save_profile() to persist
    changes and invalidate the cache.
    """
    global _profile_cache
    if _profile_cache is not None:
        return _profile_cache
    try:
        _profile_cache = json.loads(_profile_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        _profile_cache = {}
    return _profile_cache


def save_profile(profile: dict) -> None:
    """Persist user profile and invalidate in-memory cache."""
    global _profile_cache
    _profile_path().write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _profile_cache = profile


# ---- LLM helper (inline to avoid circular import from nodes.py) ----

async def _summarize_with_llm(prompt: str) -> str:
    """One-shot LLM call for conversation summarization."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        temperature=0,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        request_timeout=60.0,
    )
    response = await model.ainvoke([
        SystemMessage(content="You are a precise conversation summarizer."),
        HumanMessage(content=prompt),
    ])
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

    # ---- public API ----

    def build_snapshot(
        self, state: dict, max_tokens: int | None = None
    ) -> str:
        """Build a compact context snapshot from conversation history.

        Pure code — no LLM calls. Uses cached summary from state.
        Returns a string ready for injection into system prompts.
        """
        max_tokens = max_tokens or self.SNAPSHOT_MAX_TOKENS
        messages = state.get("messages", [])
        profile = load_profile()

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
            buffer_text = self._format_buffer(buffer_msgs, budget)
            if buffer_text:
                parts.append(f"## Recent Conversation\n{buffer_text}")

        return "\n\n".join(parts)

    def needs_summary_update(self, state: dict) -> bool:
        """Check if older messages (beyond buffer) need re-summarization."""
        messages = state.get("messages", [])
        if len(messages) <= self.BUFFER_SIZE:
            return False
        older_count = len(messages) - self.BUFFER_SIZE
        through_seq = state.get("summary_through_seq", 0)
        return older_count > through_seq

    async def regenerate_summary(self, state: dict) -> str:
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

        prompt = f"""\
Summarize this conversation history for a research literature assistant. Focus on:
1. Papers discussed — names, key findings mentioned by user/agent
2. User's explicit questions and what was answered
3. Any unresolved or pending questions
4. User preferences observed — language, detail level, preferred sections

Existing summary (update/extend, don't repeat):
{existing if existing else "(none — first summary)"}

Conversation to summarize:
{older_text}

Output ONLY the updated summary text, no preamble. Keep under 300 words.
Write in the same language the user has been using."""
        return await _summarize_with_llm(prompt)

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

        When pair_aware=True (default), avoids splitting these pairs across
        the truncation boundary:
          - AIMessage(tool_calls) ↔ ToolMessage (tool call ↔ result)
          - HumanMessage ↔ AIMessage (Q&A)
        """
        lines: list[str] = []
        chars = 0

        # Pre-scan: map each message index to its pair partner index (if any).
        # ToolMessage(n) → AIMessage(n-1) if n-1 has tool_calls.
        # AIMessage(n, no tool_calls) → HumanMessage(n-1) for Q&A.
        pair_of: dict[int, int] = {}  # msg_idx → partner_idx
        for i, m in enumerate(messages):
            if not hasattr(m, "type"):
                continue
            if m.type == "tool" and i > 0:
                prev = messages[i - 1]
                if (hasattr(prev, "type") and prev.type == "ai"
                        and hasattr(prev, "tool_calls") and prev.tool_calls):
                    pair_of[i] = i - 1
                    pair_of[i - 1] = i
            elif m.type == "ai" and i > 0:
                has_calls = hasattr(m, "tool_calls") and m.tool_calls
                if not has_calls:
                    prev = messages[i - 1]
                    if hasattr(prev, "type") and prev.type == "human":
                        pair_of[i] = i - 1
                        pair_of[i - 1] = i

        last_included: int | None = None
        for i, m in enumerate(messages):
            role = "??"
            if hasattr(m, "type"):
                if m.type == "human":
                    role = "User"
                elif m.type == "ai":
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        role = "Agent (tool call)"
                    else:
                        role = "Agent"
                elif m.type == "tool":
                    role = "Tool result"
                    # If the paired AIMessage(tool_calls) was skipped (empty
                    # content), inject a synthetic tool-call line first.
                    # But only if there's enough budget for a meaningful
                    # result — don't start a pair we can't finish.
                    if pair_aware and i > 0:
                        partner = pair_of.get(i)
                        if partner is not None and partner == i - 1:
                            tc_msg = messages[partner]
                            tc_content = (
                                tc_msg.content if hasattr(tc_msg, "content")
                                else str(tc_msg)
                            )
                            if not tc_content and hasattr(tc_msg, "tool_calls"):
                                tc_names = [tc["name"] for tc in tc_msg.tool_calls]
                                synthetic = f"[Agent (tool call)]: calls {', '.join(tc_names)}"
                                syn_len = len(synthetic)
                                # Need at least 60 chars for tool result header
                                min_pair_budget = syn_len + 60
                                if max_chars - chars >= min_pair_budget:
                                    lines.append(synthetic)
                                    chars += syn_len
                                else:
                                    # Not enough budget for a complete pair —
                                    # skip both synthetic TC and the TR below.
                                    continue
                elif m.type == "system":
                    continue

            content = m.content if hasattr(m, "content") else str(m)
            if not content:
                continue

            remaining = max_chars - chars
            if remaining <= 0:
                # Pair-aware rollback: if the last included message is part
                # of an incomplete pair, remove it so the LLM doesn't see
                # orphaned tool results or answers without their question.
                if pair_aware and last_included is not None:
                    partner = pair_of.get(last_included)
                    if partner is not None:
                        is_orphan = partner > last_included
                        if is_orphan:
                            lines.pop()
                lines.append("... (earlier messages omitted)")
                break

            if len(content) > remaining:
                content = content[:remaining] + "..."

            lines.append(f"[{role}]: {content}")
            chars += len(content)
            last_included = i

        return "\n".join(lines)


# ---- singleton ----

_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
