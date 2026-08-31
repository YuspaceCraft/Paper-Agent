"""
nodes.py — LLM nodes for the agent graph.

v3 changes:
- understand_node: 3-way classification + confidence
- memory_node: context snapshot assembly (MemoryManager)
- chat_node: lightweight general conversation (no tools)
- clarify_node: targeted clarification questions
- agent_node: self-evaluation protocol + [FINAL_ANSWER] marker for fast path
- route_intent: confidence-gated 3-way router (replaces after_understand)
- after_agent: fast-path check for [FINAL_ANSWER] → END (skip synthesize)

Each node is a pure async function: (state, config) → partial state update.
Model name injected via config["configurable"]["model"], default from .env.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .state import AgentState, UnderstandResult
from .prompts import (
    UNDERSTAND_SYSTEM, AGENT_SYSTEM, SYNTHESIZE_SYSTEM,
    CHAT_SYSTEM, CLARIFY_SYSTEM,
)
from .tools import get_base_tools, get_cached_tools
from .resolution import normalize_name as _normalize_name
from .observability import log_event, timed, count


# ---- model factories ----

def _model_kwargs() -> dict:
    return {
        "temperature": 0,
        "base_url": os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "request_timeout": 120.0,  # 2 min per LLM call; avoids indefinite hang
        # 网络层重试：DashScope 连接抖动（APIConnectionError）会打穿节点级尽力
        # 重试（understand/plan 各 2 次），导致整轮 node_error。在 SDK 层统一
        # 加退避重试覆盖所有节点，而不是逐节点打补丁。
        "max_retries": 3,
    }


def _get_model(config: RunnableConfig) -> ChatOpenAI:
    """Non-streaming model for understand + agent + chat/clarify nodes."""
    model_name = config.get("configurable", {}).get(
        "model", os.getenv("LLM_MODEL", "qwen-plus")
    )
    return ChatOpenAI(model=model_name, **_model_kwargs())


def _get_bound_model(config: RunnableConfig, tool_names: list[str] | None = None):
    """Model with tools bound, for the agent node (tool-calling LLM).

    tool_names: subagent 的受限工具子集 → 从完整注册表（_BASE_TOOLS）按名挑选。
    None/empty: 父 agent → 绑定父工具面（_ALL_TOOLS）。
    """
    if tool_names:
        tools = [t for t in get_base_tools() if t.name in tool_names]
    else:
        tools = get_cached_tools()
    return _get_model(config).bind_tools(tools)


# [FINAL_ANSWER] is an agent→router protocol marker, not user-facing text.
# Streaming it token-by-token would leak the marker into the UI, so the first
# chunks are buffered until the leading marker can be stripped (if present).
_FINAL_MARKER_LITERAL = "[FINAL_ANSWER]"


async def _stream_llm(model, messages, *, emit_tokens: bool = True) -> AIMessage:
    """Stream model output token-by-token via the emit() event channel, returning
    the merged AIMessage (tool_calls preserved).

    get_stream_writer() is broken on async nodes under Python < 3.11 (see
    stream.py), so tokens flow through the same contextvar queue used for
    tool/plan events instead — the SSE endpoint already drains it.
    """
    from .stream import emit, current_scope

    full: AIMessageChunk | None = None
    # Buffer leading text so a leading [FINAL_ANSWER] marker can be stripped.
    prefix = ""
    prefix_resolved = False
    # Don't stream tokens while running inside a subagent — its answer is
    # returned to the parent as a tool result, not user-facing text.
    in_subagent = current_scope() is not None

    async for chunk in model.astream(messages):
        full = chunk if full is None else full + chunk
        if not emit_tokens or in_subagent:
            continue
        text = chunk.content
        if not isinstance(text, str) or not text:
            continue
        if not prefix_resolved:
            prefix += text
            if prefix.startswith(_FINAL_MARKER_LITERAL):
                if len(prefix) >= len(_FINAL_MARKER_LITERAL):
                    rest = prefix[len(_FINAL_MARKER_LITERAL):].lstrip("\n")
                    prefix_resolved = True
                    if rest:
                        emit({"type": "token", "content": rest})
            else:
                prefix_resolved = True
                emit({"type": "token", "content": prefix})
        else:
            emit({"type": "token", "content": text})

    if full is None:
        return AIMessage(content="")
    if isinstance(full, AIMessageChunk):
        return AIMessage(
            content=full.content,
            tool_calls=list(full.tool_calls) if full.tool_calls else [],
            additional_kwargs=dict(full.additional_kwargs or {}),
            response_metadata=dict(full.response_metadata or {}),
        )
    return full


# ---- memory assembly ----

@timed("memory")
async def memory_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Build context snapshot for downstream nodes. Runs after understand.

    Pure-code assembly + lazy summary regeneration. Summary updates happen
    inline (adds ~1s every ~6 turns when buffer overflows).
    """
    from .memory import get_memory_manager

    mm = get_memory_manager()
    summary = state.get("summary_cache", "")
    through_seq = state.get("summary_through_seq", 0)

    if mm.needs_summary_update(state):
        try:
            new_summary = await mm.regenerate_summary(state)
            if new_summary:
                summary = new_summary
                through_seq = len(state["messages"]) - mm.BUFFER_SIZE
        except Exception as exc:
            # ponytail: summary regeneration is best-effort — old summary
            # is still usable; don't block the turn on a summary LLM failure.
            log_event("memory_summary_failed", node="memory", level="warning",
                      error=f"{type(exc).__name__}: {exc}")

    # Build snapshot with current (possibly updated) summary
    merged = {**state, "summary_cache": summary}
    snapshot = mm.build_snapshot(merged)

    return {
        "context_snapshot": snapshot,
        "summary_cache": summary,
        "summary_through_seq": through_seq,
    }


async def _get_paper_names() -> list[str]:
    """Lazy fetch paper names via the shared TTL cache in resolution.py.

    Both resolve_node and agent_node pre-flight hit /api/reader/papers —
    the shared cache eliminates the duplicate HTTP call.
    """
    from .resolution import fetch_papers as _fetch_papers
    papers = await _fetch_papers()
    return [
        p.get("name", p.get("paper_name", ""))
        for p in papers
    ]


def _last_user_text(state: dict) -> str | None:
    """Extract the last HumanMessage text from state messages."""
    for m in reversed(state.get("messages", [])):
        if hasattr(m, "type") and m.type == "human":
            return m.content or ""
    return None


def _paper_matches(user_term: str, paper_name: str) -> bool:
    """Check if a user-provided term plausibly refers to a paper."""
    u = _normalize_name(user_term)
    p = _normalize_name(paper_name)
    if u == p:
        return True
    if len(u) >= 4 and (u in p or p in u):
        return True
    return False


def _has_prior_paper_access(state: dict) -> bool:
    """Check if conversation already contains successful paper access.

    Used to skip the pre-flight safety net on follow-up questions where the
    agent already knows which papers are available. The pre-flight was designed
    for first messages where the user asks about a paper that may not be in
    the library; on follow-ups, focus_papers may contain paper titles extracted
    from tool results, which won't match library directory names.
    """
    for m in state.get("messages", []):
        if not hasattr(m, "type") or m.type != "tool":
            continue
        content = str(m.content) if hasattr(m, "content") else ""
        cl = content.lower()
        # JSON response from search_papers
        if '"ok": true' in cl or '"ok":true' in cl:
            if any(kw in cl for kw in ('"paper"', '"chunk"', '"results"')):
                return True
    return False


# ---- resolved formatting ----

def _format_resolved(resolved: dict) -> str:
    """Format resolved references as Discovery Hints — clues, not facts.

    Every match must be verified through tool calls. The resolver accelerates
    discovery but does not replace it.
    """
    papers = resolved.get("papers", [])
    if not papers:
        return (
            "(no hints — discover papers via search_papers(query=''))"
        )

    lines: list[str] = []
    for p in papers:
        query = p.get("query", "")
        match = p.get("match", "")
        level = p.get("level", "NONE")
        match_type = p.get("match_type", "none")

        if level == "EXACT":
            lines.append(
                f'- User mentioned "{query}" → exact match: "{match}"\n'
                f'  Next: verify via search_papers("{match}")'
            )
        elif level == "HIGH":
            lines.append(
                f'- User mentioned "{query}" → likely: "{match}" '
                f'(HIGH confidence, {match_type})\n'
                f'  Next: verify via search_papers("{match}")'
            )
        elif level == "MEDIUM":
            lines.append(
                f'- User mentioned "{query}" → guess: "{match}" '
                f'(MEDIUM confidence — verify first)\n'
                f'  Next: confirm via search_papers()'
            )
        elif level == "LOW":
            lines.append(
                f'- User mentioned "{query}" → weak match: "{match}" '
                f'(LOW — likely wrong)\n'
                f'  Next: ignore hint, browse via search_papers()'
            )
        else:  # NONE
            lines.append(
                f'- User mentioned "{query}" → not in library.\n'
                f'  Next: try the arxiv subagent to find it externally, '
                f'or search_papers() to browse local papers'
            )

    section = resolved.get("section")
    if section:
        ordinal = section.get("ordinal", "?")
        text = section.get("text", "")
        # Pick the best paper match for the section hint
        best_paper = ""
        for p in papers:
            if p.get("level") in ("EXACT", "HIGH"):
                best_paper = p.get("match", "")
                break
        paper_ref = f'"{best_paper}"' if best_paper else "the confirmed paper"
        lines.append(
            f'\nSection reference: "{text}" → ordinal {ordinal}.\n'
            f'  Step 1: fetch_content({paper_ref}, section="") without a section filter\n'
            f'    to get the paper overview with ALL section headings.\n'
            f'  Step 2: find the {ordinal}th top-level section in the returned list\n'
            f'    and fetch_content({paper_ref}, section="<EXACT heading text>") with it.\n'
            f'  Do NOT guess the format (Roman numerals, Arabic, Chinese).'
            f'  Discover it from the overview.'
        )

    return "\n".join(lines)


# ---- error classification ----

def _classify_tool_error(content: str) -> dict | None:
    """Parse a tool error response, extract structured error info.

    Returns None if content is not a recognizable error.
    """
    import json as _json

    try:
        data = _json.loads(str(content))
    except (_json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict) or data.get("ok") is not False:
        return None

    error_type = data.get("error_type", "")
    available_papers = data.get("available_papers", [])
    available_sections = data.get("available_sections", [])
    error_msg = data.get("error", "")

    if not error_type:
        if "timeout" in error_msg.lower():
            error_type = "transient"
        elif available_papers or available_sections:
            error_type = "param_error"
        elif "not found" in error_msg.lower():
            error_type = "not_found"
        else:
            error_type = "unknown"

    return {
        "type": error_type,
        "error": error_msg,
        "next": data.get("next", ""),
        "available_papers": available_papers,
        "available_sections": available_sections,
    }


def _format_error_feedback(error_info: dict) -> str:
    """Format classified error as an actionable system note for the LLM."""
    etype = error_info["type"]
    papers = error_info.get("available_papers", []) or []
    sections = error_info.get("available_sections", []) or []

    lines = ["Tool error. Recovery:"]

    if etype == "transient":
        lines.append(
            "Temporary failure (timeout/server busy). Retry ONCE with same parameters."
        )
    elif etype == "param_error":
        if papers:
            names = ", ".join(papers[:5])
            lines.append(
                f"Available papers: [{names}]. Pick best match and retry."
            )
        if sections:
            names = ", ".join(sections[:8])
            lines.append(
                f"Available sections: [{names}]. Pick best match and retry."
            )
    elif etype == "not_found":
        lines.append(
            "Resource not in local library. Do NOT retry same parameters. "
            "Search the local library via search_papers(), try the arxiv "
            "subagent to find it externally, "
            "or tell user what IS available locally."
        )
    elif etype == "permission_denied":
        lines.append(
            "This action is not authorized for the current role. "
            "Tell the user it is not permitted and do NOT retry it."
        )
    else:
        lines.append(
            "Unexpected error. Fallback: browse the local library via search_papers(query='')."
        )

    return " ".join(lines)


# ---- nodes ----

# ---- follow-up detection (fast path, no LLM) ----

# 明显的 follow-up 信号 — 跳过 LLM 分类，复用上一轮 intent
_FOLLOW_UP_PATTERNS: list[re.Pattern] = [
    re.compile(r'^(继续|接着|然后呢|还有吗|还有呢|详细说|展开|具体点|往下说)$'),
    re.compile(r'^(go\s*on|continue|tell\s*me\s*more|more\s*details?|elaborate|expand)$',
              re.IGNORECASE),
    re.compile(r'^(然后|接下来|and\s*then|what\s*else|what\s*about)$', re.IGNORECASE),
    re.compile(r'^(能|可以|能否|can\s*you)\s*(再|更|多)?\s*(说|讲|解释|介绍|描述)',
              re.IGNORECASE),
]
_FOLLOW_UP_MAX_LEN = 30  # 短于此字符数的消息可能是 follow-up


def _detect_follow_up(state: dict) -> dict | None:
    """如果检测到明显的 follow-up 信号，返回应直接使用的 state 更新。

    纯启发式 — 无 LLM。只对高置信度模式生效，不确定时返回 None
    （走正常 LLM 分类路径）。
    """
    msgs = state.get("messages", [])
    if len(msgs) < 2:
        return None

    last_msg = msgs[-1]
    content = (last_msg.content if hasattr(last_msg, "content") else str(last_msg)).strip()
    if not content:
        return None

    # 必须有前序 intent 可复用
    prev_intent = state.get("intent", "")
    if not prev_intent:
        return None

    # 检查模式匹配
    matched = any(pat.match(content) for pat in _FOLLOW_UP_PATTERNS)
    if not matched:
        # 未命中模式但消息很短且有上下文 → 仍可能是 follow-up
        if len(content) >= _FOLLOW_UP_MAX_LEN:
            return None
        # 检查是否包含指代前文的词（"那个"/"这个"/"it"/"that"）
        if not re.search(r'(那个|这个|那|这|it|that|the\s+same|above)',
                         content, re.IGNORECASE):
            return None

    # 确认：复用上一轮 intent，高置信度
    return {
        "intent": prev_intent,
        "confidence": 0.92,  # 略低于显式匹配以保留 verify 行为
        "entities": state.get("entities", []),
        "focus_papers": state.get("focus_papers", []),
        "iteration": 0,
        "consecutive_failures": 0,
        # Reset per-turn execution state — a stale mode="plan" from a prior
        # turn otherwise leaks into synthesize_node and answers from the old
        # turn's subagent_results (root cause of the "wrong topic" reply).
        "mode": "react",
        "plan": [],
        "plan_progress": 0,
        "subagent_results": [],
    }


@timed("understand")
async def understand_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Router: classify intent + estimate confidence. 3-way classification.

    Injects recent conversation context for vague-reference resolution:
    "这段" / "that passage" / "它" can be resolved against prior exchange.

    Fast path: obvious follow-up signals ("继续" / "go on") skip the LLM call
    entirely — ~500ms saved per follow-up turn.
    """
    msgs = state["messages"]
    last_msg = msgs[-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # ---- Fast path: follow-up detection (no LLM) ----
    fast_result = _detect_follow_up(state)
    if fast_result is not None:
        return fast_result

    # ---- Normal path: LLM classification ----
    model = _get_model(config).with_structured_output(
        UnderstandResult, method="function_calling"
    )

    # Build recent context snippet for vague-reference resolution.
    # Exclude the last message itself (it's the query being classified).
    # Max ~600 chars — enough to see the prior Q&A pair.
    ctx_parts: list[str] = []
    for m in msgs[-5:-1]:
        if hasattr(m, "type"):
            if m.type == "human":
                role = "user"
            elif m.type == "tool":
                role = "tool_result"
            elif m.type == "ai":
                role = "assistant"
            else:
                role = "system"
        else:
            role = "assistant"
        c = (m.content if hasattr(m, "content") else str(m))[:300]
        if c.strip():
            ctx_parts.append(f"[{role}]: {c}")
    recent_context = "\n".join(ctx_parts) if ctx_parts else ""

    system_prompt = UNDERSTAND_SYSTEM
    if recent_context:
        system_prompt += (
            f"\n\n## Recent Conversation (for resolving vague references like "
            f"\"这段\"/\"this passage\"/\"它\")\n{recent_context}"
        )

    # Structured output can return None (model replied without a tool call) —
    # retry once, then default to the literature_search path (tools ground it).
    result: UnderstandResult | None = None
    for attempt in range(2):
        count("llm_calls")
        try:
            result = await model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ])
        except Exception as exc:
            log_event("understand_llm_failed", node="understand", level="warning",
                      attempt=attempt, error=f"{type(exc).__name__}: {exc}")
        if result is not None and getattr(result, "intent", None):
            break
        log_event("understand_empty_result", node="understand", level="warning",
                  attempt=attempt)

    if result is None:
        # degrade: literature_search + the react path is tool-grounded, so it
        # recovers even with empty entities/focus_papers.
        return {
            "intent": "literature_search",
            "confidence": 1.0,
            "entities": [],
            "focus_papers": [],
            "iteration": 0,
            "consecutive_failures": 0,
            "mode": "react",
            "plan": [],
            "plan_progress": 0,
            "subagent_results": [],
        }

    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "entities": result.entities,
        "focus_papers": result.focus_papers,
        "iteration": 0,
        "consecutive_failures": 0,
        # Reset per-turn execution state (see _detect_follow_up fast path).
        "mode": "react",
        "plan": [],
        "plan_progress": 0,
        "subagent_results": [],
    }


@timed("chat")
async def chat_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Lightweight general conversation — no tools, no retrieval overhead."""
    model = _get_model(config)
    msgs = [m for m in state["messages"] if hasattr(m, "type") and m.type in ("human", "ai")]
    recent = msgs[-4:] if len(msgs) > 4 else msgs

    system = CHAT_SYSTEM
    context = state.get("context_snapshot", "")
    if context:
        system += f"\n\n## Prior Conversation\n{context}"

    response = await _stream_llm(model, [
        SystemMessage(content=system),
        *recent,
    ])
    return {"messages": [response]}


@timed("clarify")
async def clarify_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Generate a targeted clarification question for ambiguous queries."""
    model = _get_model(config)
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    system = CLARIFY_SYSTEM
    context = state.get("context_snapshot", "")
    if context:
        system += f"\n\n## Prior Conversation\n{context}"

    response = await _stream_llm(model, [
        SystemMessage(content=system),
        HumanMessage(content=content),
    ])
    return {"messages": [response]}


@timed("agent")
async def agent_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM with bound tools. Self-evaluation in prompt drives tool decisions.

    Injects intent/entities/resolved as context. Pre-flight validates
    focus_papers on first iteration. Classifies tool errors for recovery.
    The [FINAL_ANSWER] marker in the prompt enables fast-path routing.
    """
    it = state.get("iteration", 0)
    focus = [p for p in state.get("focus_papers", []) if p]
    resolved = state.get("resolved", {})
    extra_msgs: list = []

    # ---- pre-flight: safety net for genuinely missing papers ----
    # Download/import intent: never bypass the local check — inject the mandated
    # tri-state ladder (check_paper → index if local, arXiv only if absent) so the
    # bypass becomes an explicit path. Non-download queries keep the not-found check;
    # follow-ups with existing paper context are skipped (library dir names won't
    # match tool-result titles → false positive "not found").
    _user_msg = _last_user_text(state)
    _is_download_intent = _user_msg and any(
        kw in _user_msg for kw in ("下载", "download", "导入", "import", "入库")
    )
    if focus and it == 0:
        if _is_download_intent:
            extra_msgs.append(SystemMessage(content=
                f"User asked to save/import: {focus}. "
                f"Before downloading or indexing, call check_paper(<the paper term>) "
                f"to detect local state (indexed / downloaded_not_indexed / absent). "
                f"Download or arXiv lookup is ONLY needed when state is 'absent'."
            ))
        elif not _has_prior_paper_access(state):
            available = await _get_paper_names()
            if available:
                missing = [
                    p for p in focus
                    if not any(_paper_matches(p, a) for a in available)
                ]
                if missing:
                    extra_msgs.append(SystemMessage(content=
                        f"Paper(s) not found in local library: {missing}. "
                        f"Look them up directly via search_papers() to re-check the local library, "
                        f"or use the arxiv subagent to find them externally. "
                        f"Use the ingest subagent only if the user asks to download/save a paper "
                        f"(a pure file download by default) — reading a paper's content is done "
                        f"via fetch_content(), not ingest."
                    ))

    # ---- failure tracking + error classification ----
    failures = state.get("consecutive_failures", 0)
    if state["messages"]:
        last = state["messages"][-1]
        error_info = (
            _classify_tool_error(str(last.content))
            if hasattr(last, "content") else None
        )
        if error_info is not None:
            failures += 1
            feedback = _format_error_feedback(error_info)
            extra_msgs.append(SystemMessage(content=feedback))
        elif hasattr(last, "type") and last.type == "tool":
            failures = 0  # tool returned ok, reset

    if failures >= 2:
        extra_msgs.append(SystemMessage(content=
            "Last 2 tool calls failed. "
            "STOP retrying the same parameters. Fall back: "
            "call search_papers(query='') to list available papers, "
            "then fetch_content() the one you want with its confirmed name. "
            "If nothing works, tell the user what's missing."
        ))

    # ---- LLM call ----
    # Subagent mode: a non-empty subagent_system overrides AGENT_SYSTEM, and
    # bound_tools restricts the tool set. Empty → parent behavior unchanged.
    subagent_system = state.get("subagent_system", "")
    model = _get_bound_model(config, state.get("bound_tools") or None)
    if subagent_system:
        system = subagent_system
        # Thread resolved refs across the subagent boundary. The subagent
        # state is fresh (only `task` + `resolved` from as_tool), so without
        # this it would re-search papers the parent already matched.
        _r = state.get("resolved", {}) or {}
        _rpapers = _r.get("papers", []) if isinstance(_r, dict) else []
        if _rpapers:
            _hints = "\n".join(
                f'- "{p.get("query", "")}" → "{p.get("match", "")}" '
                f'({p.get("level", "NONE")})'
                for p in _rpapers
            )
            system += (
                "\n\n## Resolved papers (trust these names — do NOT re-search)\n"
                f"{_hints}"
            )
    else:
        resolved_text = _format_resolved(resolved)
        system = AGENT_SYSTEM.format(
            intent=state.get("intent", "literature_search"),
            entities=", ".join(state.get("entities", [])) or "(none)",
            focus_papers=", ".join(focus) or "(none)",
            resolved=resolved_text,
        )

    # Inject context snapshot for multi-turn coherence
    context = state.get("context_snapshot", "")
    if context:
        system += f"\n\n## Conversation Context (for multi-turn reference)\n{context}"

    # ---- token budget guard: hard stop, force final answer ----
    budget = state.get("token_budget", 0)
    tokens_used = state.get("tokens_used", 0)
    if budget and tokens_used >= budget:
        plain = _get_model(config)
        count("llm_calls")
        response = await _stream_llm(plain, [
            SystemMessage(content=(
                "Token budget exhausted. Provide your best final answer now, "
                "citing any sources you already have. Do NOT call tools."
            )),
            *state["messages"],
        ])
        return {"messages": [response], "tokens_used": tokens_used}

    count("llm_calls")
    response = await _stream_llm(model, [
        SystemMessage(content=system),
        *state["messages"],
        *extra_msgs,
    ])
    if hasattr(response, "tool_calls") and response.tool_calls:
        count("tools_called", len(response.tool_calls))
    # Accumulate tokens consumed this call. Full input history is re-counted
    # each iteration — conservative over-count, so the budget fires early
    # rather than late. Bounded by max_iterations; a backstop, not a meter.
    from .memory import _estimate_tokens
    in_tokens = _estimate_tokens(system) + sum(
        _estimate_tokens(str(m.content))
        for m in [*state["messages"], *extra_msgs]
        if hasattr(m, "content") and m.content
    )
    out_tokens = _estimate_tokens(str(response.content)) if response.content else 0

    # ponytail: extra_msgs are this-turn-only context (pre-flight hints,
    # error feedback). Don't persist to checkpoint — they'd mislead the
    # LLM on subsequent turns with stale hints about prior errors.
    return {
        "messages": [response],
        "focus_papers": focus,
        "iteration": it + 1,
        "consecutive_failures": failures,
        "tokens_used": tokens_used + in_tokens + out_tokens,
    }


# ---- synthesize (safety net) ----

def _salvage_tool_content(messages: list) -> dict | None:
    """Salvage content from successful tool calls when the synthesize LLM fails.

    Handles both plain-text content tools (v2+) and legacy JSON-wrapped responses.
    """
    import json as _json

    best: dict | None = None

    for m in messages:
        if not hasattr(m, "content"):
            continue
        if hasattr(m, "type") and m.type != "tool":
            continue
        raw = str(m.content)

        # Skip error responses
        if '"ok": false' in raw:
            continue

        paper = ""
        section = ""
        text = ""

        # Try JSON format first (legacy + structured tools)
        try:
            data = _json.loads(raw)
        except (_json.JSONDecodeError, ValueError):
            data = None

        if isinstance(data, dict) and data.get("ok") is True:
            inner = data.get("data", {})
            chunks = inner.get("chunks", [])
            if chunks:
                paper = inner.get("paper_name", "")
                section = inner.get("section_query", "")
                text = "\n\n".join(
                    c.get("content", "") for c in chunks
                    if c.get("content", "").strip()
                )
        elif data is None and len(raw) > 100:
            # Plain-text content tool response (v2+): markdown
            text = raw.strip()
            h = re.match(r'^##\s+(.+?)\s+\((.+?)\)', text)
            if h:
                section = h.group(1).strip()
                paper = h.group(2).strip()
            else:
                h = re.match(r'^#\s+(.+)', text)
                if h:
                    paper = h.group(1).strip()

        if not text.strip():
            continue
        if best is None or len(text) > len(best.get("text", "")):
            best = {"paper": paper, "section": section, "text": text}

    return best


async def _synthesize_plan(state: AgentState, config: RunnableConfig) -> dict:
    """Plan-mode synthesis: merge subagent_results into a final answer."""
    desc = {s.get("id"): s.get("description", "") for s in state.get("plan", [])}
    parts: list[str] = []
    for r in state.get("subagent_results", []):
        label = desc.get(r.get("step_id")) or r.get("step_id", "")
        if r.get("ok"):
            out = (r.get("output") or "").strip()
            if out:
                parts.append(f"## {label}\n{out}")
        else:
            parts.append(f"## {label}\n(step failed: {r.get('error', '')})")
    context = "\n\n".join(parts)

    model = _get_model(config)
    user_q = _last_user_text(state) or ""

    answer = ""
    if context:
        try:
            response = await _stream_llm(model, [
                SystemMessage(content=SYNTHESIZE_SYSTEM.format(question=user_q)),
                HumanMessage(content=context),
            ])
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            log_event("synthesize_llm_failed", node="synthesize", level="warning",
                      error=f"{type(exc).__name__}: {exc}")

    if not answer or not answer.strip():
        # LLM failed or no subagent output → fall back to raw merged context
        answer = context or "抱歉，未能生成回答。"

    return {
        "messages": [AIMessage(content=answer)],
    }


@timed("synthesize")
async def synthesize_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Safety net: synthesize an answer if the agent didn't produce one.

    Plan mode: merge subagent_results (executor output) instead of raw tool
    messages. React mode keeps the two existing paths:
    - Fast path: agent already produced a text answer → reuse it, skip LLM call.
    - Slow path: agent exhausted without an answer → LLM synthesizes one.
    """
    # ── plan mode: merge subagent results ──
    if state.get("mode") == "plan":
        return await _synthesize_plan(state, config)

    # ── fast path: agent already answered ──
    # Scans for the last AI message with non-empty content (no tool_calls);
    # any substantive AI response counts as the final answer.
    # ([FINAL_ANSWER] marker, if present, is stripped in the streaming layer.)
    for m in reversed(state["messages"]):
        if not (hasattr(m, "type") and m.type == "ai" and hasattr(m, "content")):
            continue
        has_calls = hasattr(m, "tool_calls") and m.tool_calls
        if has_calls:
            break  # last AI msg was a tool call → no answer yet, fall to slow path
        content = str(m.content)
        if not content.strip():
            break  # empty content → no answer
        # Agent produced a text response — use it directly
        return {}
    # (if we reach here, fall through to slow path below)

    # ── slow path: agent didn't self-declare sufficiency ──
    model = _get_model(config)

    user_q = ""
    for m in state["messages"]:
        if hasattr(m, "type") and m.type == "human":
            user_q = m.content if hasattr(m, "content") else str(m)

    answer = ""
    try:
        response = await _stream_llm(model, [
            SystemMessage(content=SYNTHESIZE_SYSTEM.format(question=user_q)),
            *state["messages"],
        ])
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        log_event("synthesize_llm_failed", node="synthesize", level="warning",
                  error=f"{type(exc).__name__}: {exc}")

    # Guard: if model returns empty or call failed, salvage from tool results
    if not answer or not answer.strip():
        saved = _salvage_tool_content(state.get("messages", []))
        if saved:
            paper = saved.get("paper", "")
            section = saved.get("section", "")
            text = saved.get("text", "")
            answer = (
                f"根据检索到的内容，论文 **{paper}** 的 **{section}** 章节内容如下：\n\n"
                f"{text}\n\n"
                f"⚠️ 注意：以上内容可能不完整（模型生成中断）。"
                f"如需完整内容，请尝试重新提问。"
            )
        else:
            papers_hint = ""
            for m in state.get("messages", []):
                if hasattr(m, "content"):
                    import json as _json
                    try:
                        data = _json.loads(str(m.content))
                        if isinstance(data, dict) and data.get("available_papers"):
                            papers_hint = (
                                f"知识库中可用的论文: {data['available_papers']}。"
                                f"请使用完整论文名重试。"
                            )
                            break
                    except Exception:
                        pass
            if papers_hint:
                answer = f"抱歉，未能找到您指定的论文。{papers_hint}"
            else:
                answer = (
                    "抱歉，未能生成回答。请尝试使用 search_papers "
                    "查看可用论文，或换个问法。"
                )

    return {
        "messages": [AIMessage(content=answer)],
    }


# ---- routing (pure functions, no LLM) ----

def route_intent(state: AgentState) -> str:
    """Confidence-gated 3-way routing after understand_node.

    - Low confidence (< 0.5): route to clarify — avoid wasted tool calls
    - general_chat: route to chat — lightweight, no tools
    - needs_clarify: route to clarify
    - literature_search (default): route to resolve → full pipeline
    """
    intent = state.get("intent", "literature_search")
    confidence = state.get("confidence", 1.0)

    if confidence < 0.5:
        return "clarify"

    if intent == "general_chat":
        return "chat"
    elif intent == "needs_clarify":
        return "clarify"
    else:
        return "resolve"


def after_agent(state: AgentState) -> str:
    """Route after agent: tool loop continuation or terminal.

    - has tool calls + under max_iterations → tools (continue loop)
    - no tool calls → end (agent chose to answer — respect that)
    - max_iterations exhausted with tool calls → synthesize (safety net)

    The [FINAL_ANSWER] marker is encouraged by the prompt for explicitness
    but is no longer required: any non-empty AI message without tool_calls
    is treated as the agent's final answer.
    """
    msgs = state["messages"]
    if not msgs:
        return "synthesize"

    last = msgs[-1]
    has_tool_calls = hasattr(last, "tool_calls") and last.tool_calls

    if has_tool_calls:
        if state.get("iteration", 0) >= state.get("max_iterations", 5):
            return "synthesize"
        return "tools"

    # No tool calls → agent decided to answer. Treat as terminal.
    return "end"
