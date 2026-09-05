"""
nodes.py — LLM nodes for the agent graph.

v3 changes:
- understand_node: 3-way classification + confidence
- memory_node: context snapshot assembly (MemoryManager)
- chat_node: lightweight general conversation (no tools)
- clarify_node: targeted clarification questions
- agent_node: self-evaluation protocol driving tool decisions
- route_intent: confidence-gated 3-way router (replaces after_understand)
- after_agent: any non-empty AI message without tool_calls → END (fast path)

P1 (INFO_FLOW_REVIEW): [FINAL_ANSWER] marker 协议已从 prompt 删除——路由从未依赖
它。_stream_llm / router 保留统一正则兜底过滤，防御旧回合产出的存量 marker。

Each node is a pure async function: (state, config) → partial state update.
Model name injected via config["configurable"]["model"], default from .env.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, AIMessageChunk, ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .state import AgentState, UnderstandResult
from .prompts import (
    UNDERSTAND_SYSTEM, AGENT_SYSTEM, SYNTHESIZE_SYSTEM,
    CHAT_SYSTEM, CLARIFY_SYSTEM, TASK_SYSTEM,
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


# ---- tool executor（截断版，替代 langgraph.prebuilt.ToolNode） ----

_TOOL_RESULT_MAX = int(os.environ.get("AGENT_TOOL_RESULT_MAX", "8000"))


def build_tools_node(tools):
    """Factory for the tool-executor graph node.

    LangGraph tool loop 里的工具执行节点。替代 prebuilt.ToolNode，两道护栏：

    1. **结果截断**：工具返回在进 state 前截断到 _TOOL_RESULT_MAX 字符。多轮
       循环每次都把全量历史重发给模型，而工具正文是逐字存进 messages 的 ——
       trace 里一次 fetch_content 就 ≈9KB 原文，30 步后输入直接爆上下文窗口。
       截断后「逐篇验证」类循环才能跑满 max_steps 而不死于输入膨胀。
    2. **异常兜底**：单工具异常转为统一错误信封（{"ok": false, "error_type":
       "tool_crash"}），feed 给 _classify_tool_error 正常分类恢复，而不是像
       prebuilt.ToolNode 默认那样把异常抛穿整个 graph。

    工具查找用传入的精确工具列表（而非 get_cached_tools()）——subagent 执行
    的是受限子集（arxiv__* 等不在父工具面里），必须按传参执行。
    """
    tool_map = {t.name: t for t in tools}

    async def _node(state: dict) -> dict:
        msgs = state.get("messages", [])
        if not msgs:
            return {"messages": []}
        calls = getattr(msgs[-1], "tool_calls", None) or []
        if not calls:
            return {"messages": []}
        # 去重缓存 (v15): 单轮内相同(工具,参数)只真正执行一次后的结果复用。
        # subagent/父 react 循环共享本节点,所以两条路径都受益。
        cache = state.get("tool_result_cache", {}) or {}
        next_cache = dict(cache)

        def _tool_key(name: str, args: dict) -> str:
            try:
                args_sorted = json.dumps(args, sort_keys=True, ensure_ascii=False)
            except Exception:
                args_sorted = repr(args)
            return f"{name}|{args_sorted}"

        async def _run(tc: dict) -> ToolMessage:
            from .tool_contract import err as _err_contract
            from .tool_contract import parse_tool_result
            from .tool_contract import truncate_tool_result

            name = tc.get("name", "")
            cid = tc.get("id", "") or tc.get("resource_id", "") or ""
            args = tc.get("args") or {}
            tool = tool_map.get(name)
            key = _tool_key(name, args)

            hit = next_cache.get(key)
            if hit is not None:
                # 重复调用 → 复用上次结果,不重新执行副作用;前缀提示让 LLM
                # 知道命中缓存(避免它为一成不变的结果反复重试同一参数)。
                count = (hit.get("count", 1) or 1) + 1
                base = hit.get("content", "")
                next_cache[key] = {"content": base, "count": count}
                text = f"[重复调用 {count - 1} 次，结果与上次相同]\n{base}"
                return ToolMessage(
                    content=truncate_tool_result(text, _TOOL_RESULT_MAX),
                    tool_call_id=cid, name=name,
                )

            if tool is None:
                content = _err_contract("unknown", f"unknown tool: {name}")
            else:
                try:
                    content = await tool.ainvoke(args)
                except Exception as exc:
                    content = _err_contract(
                        "tool_crash", f"{type(exc).__name__}: {exc}"
                    )
            # P6: 截断走 tool_contract —— envelope 截在 data 内部保持可解析，
            # 纯文本保持字符级 (旧实现直接切字符会把 JSON 切成半截整体作废)。
            text = truncate_tool_result(str(content), _TOOL_RESULT_MAX)
            # 只缓存「成功」结果:错误信封让 LLM 据 error_type 恢复(重试/换工具),
            # 把错误也缓存会锁死恢复路径不让它重试。
            parsed = parse_tool_result(text)
            if not (parsed.is_envelope and not parsed.ok):
                next_cache[key] = {"content": text, "count": 1}
            return ToolMessage(content=text, tool_call_id=cid, name=name)

        results = await asyncio.gather(*[_run(tc) for tc in calls])
        return {"messages": list(results), "tool_result_cache": next_cache}

    return _node


# [FINAL_ANSWER] 是旧版 agent→router 协议 marker（P1 已从 prompt 删除——路由
# 从未依赖它：任何无 tool_calls 的文本即终局）。下面的兜底过滤专防旧回合产出
# 的存量 marker：正则覆盖全角括号 / 大小写 / final-answer 分隔符变体。

# 任意位置兜底（router 非流式 /_chat 与 _stream_llm 的后续 chunk 共用）
_FINAL_ANSWER_RE = re.compile(
    r"[\[【]\s*final\s*[-_ ]?\s*answer\s*[\]】]", re.IGNORECASE)
# 行级剥离：marker 独占一行（容忍行首空白、结尾冒号与换行）
_FINAL_ANSWER_LINE_RE = re.compile(
    r"^\s*[\[【]\s*final\s*[-_ ]?\s*answer\s*[\]】]\s*:?\s*\n?", re.IGNORECASE)


def _strip_lead_marker(text: str) -> str:
    """重复剥离开头的完整 marker 行（含相邻来自该行的换行）。"""
    s = text
    while True:
        stripped = _FINAL_ANSWER_LINE_RE.sub("", s, count=1)
        if stripped == s:
            return s
        s = stripped


def _may_be_marker_prefix(text: str) -> bool:
    """text 是否仍可能是一个 marker 行的前缀（行首空白后是 [ 或 【，后续
    与 final-answer 的规范串兼容）。是 → 继续缓冲，避免半截 marker 泄漏到 UI。"""
    t = text.lstrip(" \n\t")
    if not t:
        return True
    if t[0] not in "[【":
        return False
    seg = t[1:].casefold()
    for cand in ("final", "final answer]", "final_answer]", "final-answer]"):
        if cand.startswith(seg):
            return True
    return False


async def _stream_llm(model, messages, *, emit_tokens: bool = True) -> AIMessage:
    """Stream model output token-by-token via the emit() event channel, returning
    the merged AIMessage (tool_calls preserved).

    get_stream_writer() is broken on async nodes under Python < 3.11 (see
    stream.py), so tokens flow through the same contextvar queue used for
    tool/plan events instead — the SSE endpoint already drains it.

    P1: 前缀缓冲只拦「完整的 marker 行」且能容忍 ①首 chunk 是 "\n"（旧实现只
    startswith 精确字面量，换行后完全失效）②全角括号 ③final-answer 分隔符变体。
    缓冲区一旦不可能是 marker 前缀即整体发出；后续 chunk 逐段兜底过滤任意位置
    marker。token 事件只此一条路（见 P5：graph 级 messages 流没有 chunk 分支）。
    """
    from .stream import emit, current_scope

    full: AIMessageChunk | None = None
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
        if prefix_resolved:
            emit({"type": "token", "content": _FINAL_ANSWER_RE.sub("", text)})
            continue
        prefix = _strip_lead_marker(prefix + text)
        if _FINAL_ANSWER_RE.search(prefix):
            # 完整 marker 已出现（可能在行中）→ 不再等，过滤后发出
            prefix_resolved = True
            emit({"type": "token", "content": _FINAL_ANSWER_RE.sub("", prefix)})
            continue
        if _may_be_marker_prefix(prefix):
            continue  # 悬空的 marker 开头，继续缓冲
        prefix_resolved = True
        emit({"type": "token", "content": prefix})

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

def _format_work_context(ctx: dict) -> str:
    """Format the conversation's workspace binding (对话中心化重构) as a compact
    system-prompt section. Empty/unset keys are omitted — non-empty presence tells
    the parent agent which doc/project/study this conversation is currently on."""
    lines: list[str] = []
    doc = ctx.get("active_doc_id")
    if doc:
        lines.append(f"- Active writing document: {doc}")
    proj = ctx.get("active_project")
    if proj:
        lines.append(f"- Active experiment project: {proj}")
    topic = ctx.get("study_topic")
    if topic:
        lines.append(f"- Study topic: {topic}")
    exps = ctx.get("recent_experiments") or []
    if exps:
        lines.append(f"- This conversation's recent experiments: {', '.join(map(str, exps[:5]))}")
    if not lines:
        return ""
    return "\n".join(lines)


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

    Returns None if content is not a recognizable error. P6: 解析走
    tool_contract.parse_tool_result（唯一入口）——仅 envelope 失败判定为错误；
    纯文本结果不进入错误恢复。
    """
    from .tool_contract import parse_tool_result

    result = parse_tool_result(content)
    if not result.is_envelope or result.ok:
        return None

    error_type = result.error_type or ""
    available_papers = result.extra.get("available_papers", [])
    available_sections = result.extra.get("available_sections", [])
    error_msg = result.error or ""

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
        "next": result.next_action,
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
    elif etype == "backend_down":
        lines.append(
            "Local library backend is unreachable (backend_down). "
            "STOP calling library tools — they now fail fast on purpose. "
            "Report the outage to the user: the backend must be started "
            "(uvicorn web.api.main:app) or AGENT_API_BASE must point at the "
            "right port. Answer from knowledge you already have if enough, "
            "otherwise say plainly the local library is unavailable."
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
        # 每个新用户回合 +1（会话级 turn 粒度上限依据，见 state.max_turns）。
        "turn_count": state.get("turn_count", 0) + 1,
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
            "domain": "paper",
            "confidence": 1.0,
            "entities": [],
            "focus_papers": [],
            "iteration": 0,
            "consecutive_failures": 0,
            "mode": "react",
            "plan": [],
            "plan_progress": 0,
            "subagent_results": [],
            "turn_count": state.get("turn_count", 0) + 1,
        }

    return {
        "intent": result.intent,
        "domain": getattr(result, "domain", "paper"),
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
        "turn_count": state.get("turn_count", 0) + 1,
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


# ---- task supervision console (领导-部门制：监督台) ----

_TASK_ID_RE = re.compile(r'"task_id"\s*:\s*"([0-9a-f]{8,12})"')


def _collect_task_handles(state: dict, registry_entries: list[dict]) -> list[dict]:
    """会话内已知任务句柄：active_tasks 缓存 + 最近 messages 里的 task_id token。"""
    seen: dict[str, dict] = {}
    for t in state.get("active_tasks", []) or []:
        if isinstance(t, dict) and t.get("task_id"):
            seen[str(t["task_id"])] = {"task_id": str(t["task_id"])}
    for m in state.get("messages", []):
        c = str(getattr(m, "content", ""))
        for tid in _TASK_ID_RE.findall(c):
            seen.setdefault(tid, {"task_id": tid})
    # 与注册表并集回填 role/title（无则留空）
    by_id = {e.get("task_id"): e for e in registry_entries}
    out: list[dict] = []
    for tid, h in seen.items():
        e = by_id.get(tid) or {}
        out.append({
            "task_id": tid,
            "role": h.get("role") or e.get("kind", ""),
            "title": h.get("title") or e.get("title", ""),
        })
    return out


def _format_entries(entries: list[dict]) -> str:
    if not entries:
        return "(没有匹配的任务；可建议用户用 task_list 查看全部)"
    lines = []
    for e in entries[:15]:
        lines.append(
            f"- [{e.get('kind', '?')}] {e.get('task_id', '?')} "
            f"「{e.get('title', '')}」status={e.get('status', '?')} "
            f"progress={e.get('progress', '')}")
    return "\n".join(lines)


@timed("task")
async def task_node(state: AgentState, config) -> dict[str, Any]:
    """轻量任务监督台：不经 resolve/search/plan 漏斗，直达任务注册表回答进度问询。

    数据：本会话 active_tasks 句柄 + 最近 messages 里的 task_id + 当前问句术语，
    经 agent/task_registry.find_tasks 拿统一条目。LLM 依据 TASK_SYSTEM 简洁转述；
    LLM 失败 → 确定性条目表兜底。回写 active_tasks（去重 + cap 20）供后续引用。
    """
    from .task_registry import find_tasks
    from .stream import emit as _stream_emit

    query = _last_user_text(state) or ""
    try:
        entries = await find_tasks(query)
    except Exception:
        entries = []

    context = _format_entries(entries)
    handles = _collect_task_handles(state, entries)

    model = _get_model(config)
    answer = ""
    try:
        response = await _stream_llm(model, [
            SystemMessage(content=TASK_SYSTEM),
            HumanMessage(content=(
                f"## User question\n{query or '(查看全部任务)'}\n\n"
                f"## Task registry snapshot\n{context}")),
        ], emit_tokens=False)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        log_event("task_llm_failed", node="task", level="warning",
                  error=f"{type(exc).__name__}: {exc}")

    if not answer or not answer.strip():
        answer = f"本回合任务状态：\n{context}"

    _stream_emit({"type": "tasks", "entries": [
        {k: e.get(k) for k in ("task_id", "kind", "title", "status", "progress")}
        for e in entries[:20]
    ]})

    return {
        "messages": [AIMessage(content=answer)],
        "active_tasks": handles[:20],
    }


@timed("agent")
async def agent_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM with bound tools. Self-evaluation in prompt drives tool decisions.

    Injects intent/entities/resolved as context. Pre-flight validates
    focus_papers on first iteration. Classifies tool errors for recovery.
    Fast path = 无 tool_calls 的文本回答（after_agent 判定，与 marker 无关）。
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
    last_backend_down = False
    if state["messages"]:
        last = state["messages"][-1]
        error_info = (
            _classify_tool_error(str(last.content))
            if hasattr(last, "content") else None
        )
        if error_info is not None:
            failures += 1
            last_backend_down = error_info["type"] == "backend_down"
            feedback = _format_error_feedback(error_info)
            extra_msgs.append(SystemMessage(content=feedback))
        elif hasattr(last, "type") and last.type == "tool":
            failures = 0  # tool returned ok, reset

    if failures >= 2:
        if last_backend_down:
            # 后端 down 时不要再把 LLM 支去 search_papers()——那同样打死后端。
            extra_msgs.append(SystemMessage(content=
                "The local library backend is unreachable — library tools "
                "keep failing fast on purpose. STOP calling tools now. "
                "End the turn by telling the user the backend is down "
                "(start uvicorn web.api.main:app, or fix AGENT_API_BASE) "
                "and give what answer you can from existing knowledge."
            ))
        else:
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

    # Inject conversation workspace context (对话中心化重构): which writing doc /
    # experiment project / study topic this conversation is currently bound to.
    work_ctx = state.get("context", {}) or {}
    work_lines = _format_work_context(work_ctx)
    if work_lines:
        system += f"\n\n## Current Work Context\n{work_lines}"

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

    # ---- turn-limit guard (turn 粒度的会话上限，非 step 上限) ----
    # 超过 max_turns 个用户回合后禁止再调用工具：强制基于已有信息收尾，
    # 防止会话无限膨胀（memory 摘要是软压缩，这里是硬停）。
    if state.get("turn_count", 0) > state.get("max_turns", 50):
        plain = _get_model(config)
        count("llm_calls")
        max_turns = state.get("max_turns", 50)
        response = await _stream_llm(plain, [
            SystemMessage(content=(
                f"This session has reached its turn limit ({max_turns} turns). "
                "Answer from what's already in the conversation. Do NOT call tools. "
                "If further research is needed, suggest starting a new session."
            )),
            *state["messages"],
        ])
        return {"messages": [response], "tokens_used": state.get("tokens_used", 0)}

    count("llm_calls")
    response = await _stream_llm(model, [
        SystemMessage(content=system),
        *state["messages"],
        *extra_msgs,
    ])
    if hasattr(response, "tool_calls") and response.tool_calls:
        count("tools_called", len(response.tool_calls))
    # tokens_used = 本次调用的实际输入规模（不累加）。
    # 旧实现把每轮全量历史重复统计并累加（超线性增长），20k 预算下多轮工具
    # 任务在 3~4 轮即撞线，第 5 次等工具调用被提前截断成不完整 final answer。
    # 改为度量「当前喂给模型的上下文」后，预算语义 = 真实上下文上限兜底，
    # 仍保证终止，但合法多轮任务不再被掐断。
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
        "tokens_used": in_tokens + out_tokens,
    }


# ---- synthesize (safety net) ----

def _salvage_tool_content(messages: list) -> dict | None:
    """Salvage content from successful tool calls when the synthesize LLM fails.

    P6: 解析统一走 tool_contract.parse_tool_result——envelope（fetch_content
    的结构化 data.chunks / data.text）与纯文本工具（read_file 等）都归一为
    ToolResult，去掉旧版「先猜 JSON 再猜 markdown」的双格式试探。
    """
    from .tool_contract import parse_tool_result

    best: dict | None = None

    for m in messages:
        if not hasattr(m, "content"):
            continue
        if hasattr(m, "type") and m.type != "tool":
            continue

        result = parse_tool_result(m.content)
        paper = ""
        section = ""
        text = ""

        if result.is_envelope:
            if result.ok:
                inner = result.data
                if isinstance(inner, dict):
                    chunks = inner.get("chunks") or []
                    if chunks:
                        paper = inner.get("paper_name", "")
                        section = inner.get("section_query", "")
                        text = "\n\n".join(
                            c.get("content", "") for c in chunks
                            if c.get("content", "").strip()
                        )
                    elif inner.get("text"):
                        text = str(inner["text"]).strip()
        else:
            # 纯文本工具：markdown / 普通文本
            text = result.text.strip()

        if not text.strip():
            continue
        if not paper:
            h = re.match(r'^##\s+(.+?)\s+\((.+?)\)', text)
            if h:
                section = h.group(1).strip()
                paper = h.group(2).strip()
            else:
                h = re.match(r'^#\s+(.+)', text)
                if h:
                    paper = h.group(1).strip()

        if best is None or len(text) > len(best.get("text", "")):
            best = {"paper": paper, "section": section, "text": text}

    return best


async def _synthesize_plan(state: AgentState, config: RunnableConfig) -> dict:
    """Plan-mode synthesis: merge subagent_results into a final answer.

    creation 域例外: 终态是确定性写作进度报告,不合并 subagent 正文。creator 的
    raw output 可能是整章正文(模型未按 CREATOR 约定只回状态行),拼进 context 会
    把全文回给用户,而写作本体应落在 doc(写作工作区/导出 docx 消费)。doc 缺失或
    无大纲 → 退回通用合并兜底。
    """
    if state.get("domain") == "creation":
        doc_id = state.get("doc_id")
        if doc_id:
            from .domains.creation import doc_progress
            prog = await doc_progress(doc_id)
            if prog and prog.get("sections"):
                lines = [
                    f"- {i}. {s['title']} — {s['word_count']} 词 ✓"
                    if s["status"] == "done"
                    else f"- {i}. {s['title']} — 未写入"
                    for i, s in enumerate(prog["sections"], start=1)
                ]
                msg = (f"写作进度：文档《{prog['title']}》已完成 {prog['done']}/{prog['total']} 章\n"
                       + "\n".join(lines))
                if prog["done"] == prog["total"]:
                    msg += "\n全部章节已写完,可在「论文写作」工作区查看或导出 docx。"
                else:
                    msg += "\n可在「论文写作」工作区实时查看进度,或在对话里继续让我写剩余章节。"
                msg += f"\n(doc_id: {prog['doc_id']})"
                return {"messages": [AIMessage(content=msg)]}

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

    # 计划完成验证（报告式）：有未完成/失败步骤时把缺口明确带进 final answer，
    # 让模型用已有证据作答并如实标注缺口，而不是假装全部完成。
    verification = state.get("verification") or {}
    v_status = verification.get("status", "")
    if v_status and v_status != "satisfied":
        lines = [
            f"- {o.get('id')} | {o.get('description', '')} — {o.get('reason', '')}"
            for o in verification.get("outstanding", [])
        ]
        if lines:
            context = (
                "## 未完成步骤（以下步骤未成功执行或未满足条件，最终回答必须如实说明"
                "，再基于已有证据尽力回答）\n"
                + "\n".join(lines)
                + "\n\n" + context
            )

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
    # (P1: marker 协议已删除；残留 marker 由 _stream_llm/router 兜底过滤。)
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
    elif intent == "task_query":
        return "task"
    else:
        return "resolve"


# ---- domain routing (v10): paper / creation / coding ----
# Rule 只对「强行为动词」覆盖 LLM label——mid 性的内容词（指标/训练/实验结果
# 等）在 paper 问答里极常见（"RMNet 实验部分用了什么指标"），绝不能当 coding
# 信号；弱信号一律回退 understand 的 domain label（default "paper" 保零回归）。

_DOMAIN_CREATION_STRONG = (
    "写论文", "写综述", "写一篇", "撰写", "起草", "润色", "改写", "扩写",
    "生成大纲", "出大纲", "整理论文", "写文章", "写报告", "写摘要", "帮我写",
    "帮我整理", "write a paper", "write a review", "write a survey",
    "write a report", "write an article", "write a manuscript", "draft ",
)
_DOMAIN_CODING_STRONG = (
    "跑实验", "跑一下", "写代码", "改代码", "调参", "复现", "debug",
    "commit", " git ", "部署", "评测", "提升准确率", "提升性能",
    "优化代码", "跑通", "实验进度", "实验状态", "实验记录",
)


def _domain_label(state: AgentState) -> str:
    label = state.get("domain", "paper")
    return label if label in ("paper", "creation", "coding") else "paper"


def route_domain(state: AgentState) -> str:
    """Pick the working domain: "paper" | "creation" | "coding".

    Strong behavioral verbs override the LLM label (LLMs are unreliable at
    domain choice); mixed/no strong signals fall back to the understand label.
    """
    q = (_last_user_text(state) or "").lower()
    if not q:
        return _domain_label(state)
    coding_hits = [k for k in _DOMAIN_CODING_STRONG if k in q]
    creation_hits = [k for k in _DOMAIN_CREATION_STRONG if k in q]

    if coding_hits and not creation_hits:
        return "coding"
    if creation_hits and not coding_hits:
        return "creation"
    return _domain_label(state)


@timed("domain")
async def domain_node(state: AgentState, config) -> dict[str, Any]:
    """Pure-code domain routing node (resolve → domain → decide_mode)."""
    return {"domain": route_domain(state)}


def after_agent(state: AgentState) -> str:
    """Route after agent: tool loop continuation or terminal. Step 粒度上限。

    - has tool calls + 已执行轮数 < max_steps → tools（继续循环；**撞上限那一步
      也放行执行完**，绝不中途丢弃已发出待执行的 tool call——丢弃会让 SSE 只见
      tool_start 不见 tool_end，前端卡片永远悬挂）
    - has tool calls + 已执行轮数 >= max_steps → synthesize（安全网收尾）
    - no tool calls → end（agent 选择作答，尊重它）

    max_steps 语义 = 单 turn 内最多执行多少轮工具往返。iteration 是 agent LLM
    调用计数，「已完成轮数 = iteration - 1」（每轮 = 一次 agent 调用 + 一轮工具
    执行）；因此放行条件是 done < max_steps，即最多执行 max_steps 轮工具。

    P1: 终止判定与 marker 完全无关——任何无 tool_calls 且非空的 AI 文本
    即被视为最终答案（prompt 里的 marker 协议已删除）。
    """
    msgs = state["messages"]
    if not msgs:
        return "synthesize"

    last = msgs[-1]
    has_tool_calls = hasattr(last, "tool_calls") and last.tool_calls

    if has_tool_calls:
        # leader gate（领导-部门制）：调用 request_review → 挂起到 gate 节点
        # interrupt 等待领导输入。仅当 worker 绑定了 request_review 才可能触发
        # （父 agent / 普通 subagent 均未绑定 → 此分支不可达）。
        # 注意：tool_calls 项可能是 dict 或 AIMessageToolCall，取值需双兼容。
        if any(
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
            == "request_review"
            for tc in last.tool_calls
        ):
            return "gate"
        done = max(0, state.get("iteration", 0) - 1)
        if done >= state.get("max_steps", 30):
            return "synthesize"
        return "tools"

    # No tool calls → agent decided to answer. Treat as terminal.
    return "end"
