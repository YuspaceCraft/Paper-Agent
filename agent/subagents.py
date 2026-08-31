"""
subagents.py — Multi-Agent runtime (Phase 8).

build_subagent: compiles a subagent subgraph (agent_node + restricted ToolNode
    + loop) — reuses the same agent_node/after_agent as the parent search loop,
    differing only in restricted tools, dedicated system prompt, and isolated
    context (own message list, invisible to the parent).
as_tool: wraps a subgraph as a parent-callable tool ("task in → summary out").
build_subagents: config-driven factory — one SubagentSpec per subagent
    (arxiv / ingest — Claude Code 模式：仅写/外网操作隔离，库只读工具归父 agent),
    each injecting its own permitted tool subset.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .state import AgentState
from .nodes import agent_node, after_agent
from .tools import get_cached_tools
from .stream import emit, set_scope, reset_scope


# ---- runtime factory ----

async def _subagent_synthesize(state: AgentState, config) -> dict:
    """Safety net: subagent exhausted max_iterations without a final answer.

    Mirrors the parent's synthesize slow path — one final LLM call over the
    accumulated tool results, falling back to raw salvaged content. Without
    this node, after_agent's "synthesize" branch maps to END and as_tool()
    returns the opaque "subagent produced no final answer".
    """
    from langchain_core.messages import AIMessage, SystemMessage
    from .nodes import _get_model, _salvage_tool_content

    model = _get_model(config)
    try:
        response = await model.ainvoke([
            SystemMessage(content=(
                "Produce your final answer now, from the tool results above. "
                "Do NOT call any tools."
            )),
            *state["messages"],
        ])
        if getattr(response, "content", "").strip():
            return {"messages": [response]}
    except Exception:
        pass

    # LLM failed → salvage raw tool content (best-effort)
    saved = _salvage_tool_content(state.get("messages", []))
    text = (saved or {}).get("text", "").strip()
    return {"messages": [AIMessage(content=text or "No answer produced.")]}


def build_subagent(name, system_prompt, tools, *, max_iterations=5):
    """Compile a subagent subgraph + its initial-state overrides.

    Returns (subgraph, init_state). The subagent config (subagent_system +
    bound_tools + max_iterations) is returned separately instead of baked into
    a state subclass, because LangGraph does NOT pick up TypedDict class-attribute
    defaults — a subclass like `class S(AgentState): bound_tools = [...]` compiles
    but `state["bound_tools"]` is empty at runtime (KeyError). as_tool() merges
    init_state into the subgraph's input.
    """
    sg = StateGraph(AgentState)
    sg.add_node("agent", agent_node)
    sg.add_node("tools", ToolNode(tools))
    sg.add_node("synthesize", _subagent_synthesize)
    sg.set_entry_point("agent")
    sg.add_conditional_edges(
        "agent", after_agent,
        {"tools": "tools", "synthesize": "synthesize", "end": END},
    )
    sg.add_edge("tools", "agent")
    sg.add_edge("synthesize", END)
    init_state = {
        "subagent_system": system_prompt,
        "bound_tools": [t.name for t in tools],
        "max_iterations": max_iterations,
    }
    return sg.compile(), init_state


class SubagentArgs(BaseModel):
    task: str = Field(description="Self-contained task description for the subagent")


def as_tool(name, subgraph, description, args_model=SubagentArgs, init_state=None):
    """Wrap a subgraph as a parent-callable tool. Returns the subagent's final
    answer (last non-empty AI message) as a string — the parent never sees the
    subagent's internal tool messages (context isolation)."""
    from langchain_core.messages import HumanMessage

    async def _call(**kwargs) -> str:
        task = kwargs.get("task", "")
        # subagent 边界由这里权威发出（父层 router/executor 对 subagent 工具
        # 跳过 emit，避免重复卡片）。run_id 同时是叶子工具事件的 parent_id，
        # 让前端能把本次调用的内部工具精确挂到本节点下——同一 subagent 一轮
        # 内多次调用也不会串层。
        run_id = uuid.uuid4().hex[:8]
        token = set_scope(name, run_id)
        start = time.monotonic()
        emit({"type": "tool_start", "id": run_id, "name": name,
              "kind": "subagent", "args": {"task": task}})
        try:
            # Thread the parent's resolved references across the boundary.
            # The subgraph gets a fresh state (only `task`), so without this
            # it re-searches papers the parent already matched.
            from .resolution import get_resolved_ctx
            resolved = get_resolved_ctx()
            init: dict = {"messages": [HumanMessage(content=task)]}
            if init_state:
                init.update(init_state)
            if resolved and resolved.get("papers"):
                init["resolved"] = resolved
            result = await subgraph.ainvoke(init)
            answer = ""
            for m in reversed(result.get("messages", [])):
                if getattr(m, "type", "") == "ai" and getattr(m, "content", "").strip():
                    answer = str(m.content).strip()
                    break
            ok = bool(answer)
            if not ok:
                answer = json.dumps({
                    "ok": False,
                    "error": "subagent produced no final answer",
                    "error_type": "unknown",
                }, ensure_ascii=False)
            emit({"type": "tool_end", "id": run_id, "name": name,
                  "status": "success" if ok else "error",
                  "result": answer[:4000],
                  "execution_time": round(time.monotonic() - start, 2)})
            return answer
        except Exception as exc:
            emit({"type": "tool_end", "id": run_id, "name": name,
                  "status": "error", "result": f"{type(exc).__name__}: {exc}",
                  "execution_time": round(time.monotonic() - start, 2)})
            raise
        finally:
            reset_scope(token)

    return StructuredTool(
        name=name, description=description, args_schema=args_model, coroutine=_call,
    )


# ---- subagent config (single declarative table) ----
# Each SubagentSpec declares one subagent's name, parent-facing description,
# injected system prompt, and permitted tool subset. build_subagents() is a
# generic factory over this table — adding a subagent = adding one entry.

@dataclass
class SubagentSpec:
    name: str
    description: str      # shown to the parent agent when selecting the subagent
    system_prompt: str    # context injected into the subagent
    tools: list[str]      # permitted tool names (injected subset)
    max_iterations: int = 5


ARXIV_SYSTEM = """\
You are an arXiv search specialist. Your ONLY tools are the arxiv__* tools below
— the arXiv API is your entire scope; you cannot touch the local library
(that is the parent's job via its direct search_papers/fetch_content tools).

Toolset (all you may call):
- arxiv__search_papers(query, max_results, sort_by, sort_order, date_from, date_to)
- arxiv__get_paper_data(paper_id)
- arxiv__get_full_paper_text(paper_id)  # very large — prefer get_paper_data first
- arxiv__list_categories(primary_category)
- arxiv__update_categories()

Given a research topic or paper reference, find and read relevant arXiv papers.
Search tip: use field prefixes (ti:, cat:, au:) for precision; broad queries
return too many results.

Output rules — choose the format by the task's modality:
- FIND / IDENTIFY tasks (search, resolve which paper or arXiv ID matches a
  name/claim): Output ONLY a JSON object, no preamble, no prose, no code
  fences. The arxiv_id field is the caller's source of truth:
  {"papers": [{"arxiv_id": "...", "title": "...", "authors": "...", "abstract": "..."}]}
  ≤5 papers, ranked by relevance.
- READ tasks (abstract, full text, metadata review): Output ONLY the requested
  content as prose/markdown, no tool names.

Honesty — you CANNOT identify any paper from memory:
- If the arXiv API errors or returns no results, you have identified NOTHING.
  In FIND/IDENTIFY tasks output ONLY:
  {"error": "arxiv_api_unavailable", "detail": "<工具返回的具体错误原文>"}
  Never emit papers / arxiv_id / title / authors from memory or "domain consensus".
  In READ tasks, state plainly that the API is unavailable and stop.
- Every arxiv_id you output MUST come from an actual arxiv__* tool result in this
  conversation. Never invent or guess one.
- Use CONCRETE values in every tool argument — the actual id/title string from a
  result. Never emit placeholders like {{step1.arxiv_id}} or <search_result>;
  substitute the real value you received.

Keep the answer in the same language as the task."""

INGEST_SYSTEM = """\
You are the paper-ingestion executor. Your ONLY tools are download_paper and
ingest_paper. Execute EXACTLY the action field of the command block — nothing more.

The task MUST carry an explicit command block — one `key: value` line per field.
NEVER guess an action from prose:
  action: download | ingest | download_and_ingest
  arxiv_id: <arXiv ID, e.g. 2301.07093>
  paper_name: <name the paper should be known by locally, e.g. RMNet>
  pdf_path: <workspace-relative path of an existing local PDF, optional>
  destination: <download folder, default ./data/downloads>
  filename: <file stem without extension, optional>

Three request states — keep them distinct:
- download = fetch the PDF from arXiv to a local folder. PURE file, no parsing,
  no vectorization.
- ingest (入库) = ONE COMPLETE operation: parse the PDF INTO the vector library —
  the paper becomes searchable. Never split parsing and vectorization into two
  separate jobs; they are one task.
- download_and_ingest = download the PDF, then ingest it (one combined request).

Toolset (all you may call):
- download_paper(arxiv_id, destination="./data/downloads", filename=""): PURE
  download only. Saves to `destination` (workspace-relative — appears in the
  client's file explorer at exactly that folder), named `{filename}.pdf` (default:
  short name derived from the arXiv title, e.g. "RMNet"). Returns the ACTUAL
  saved path + filename — use them verbatim. NEVER parses/chunks/indexes.
- ingest_paper(paper_name, pdf_path=""): 入库 — the complete parse + vector-index
  job, ASYNC. Returns {task_id, status: "running"} IMMEDIATELY; it finishes in the
  background (1-2 min), progress is visible above the chat, completion announced
  to the user. Pass pdf_path (the path download_paper returned) when the file was
  saved to a custom folder. ALWAYS read the task_id from the result and include it
  in your summary so the parent can track it (check_task_status(task_id)).

Rules — execute ONLY the action field:
1. `action` is the single source of truth. If MISSING or not one of download /
   ingest / download_and_ingest: perform NO tool call and report back
   "缺少明确 action（download / ingest / download_and_ingest）".
2. download → call download_paper(arxiv_id, destination, filename) ONLY, then
   STOP. A plain download NEVER chains into 入库.
3. download_and_ingest → call download_paper FIRST, then call ingest_paper with
   the paper_name AND the actual relative path the download result returned.
4. ingest → call ingest_paper(paper_name, pdf_path) using only the given fields.
   The PDF already exists locally — do NOT download, do NOT search arXiv.
5. HONESTY: report the EXACT path/filename the tool results confirmed. Never
   claim a destination folder or filename the tool result does not confirm.

On download failure (ok:false), report the error as-is — do NOT retry the same
arxiv_id more than once. A download_paper result with error_type="unverified"
(422, 无法验证 arXiv ID) is FINAL: stop, do not retry, do not invent another
ID, do not fall back to a guessed PDF path — report it and stop.

In your summary ALWAYS include the download_paper-returned real `title` and
`relative_path` so the parent can verify the PDF matches the intended paper.
If `title` is missing or the download was unverified, state that clearly.

When the executed action was ingest / download_and_ingest, INCLUDE the returned
task_id and its "running" status in your summary — 入库 runs in the background
and the parent tracks it via check_task_status(task_id).

Output ONLY a short status summary. No preamble."""


SUBAGENTS: list[SubagentSpec] = [
    SubagentSpec(
        name="arxiv",
        description=(
            "Search and read papers on arXiv (external). Returns a ranked list "
            "of papers or requested content (metadata / abstract / full text)."
        ),
        system_prompt=ARXIV_SYSTEM,
        tools=[
            "arxiv__search_papers", "arxiv__get_paper_data",
            "arxiv__get_full_paper_text", "arxiv__list_categories",
            "arxiv__update_categories",
        ],
    ),
    SubagentSpec(
        name="ingest",
        description=(
            "Execute a paper ingestion command — the task MUST carry an explicit "
            "command block (action: download | ingest | download_and_ingest, plus "
            "arxiv_id/paper_name/pdf_path/destination as needed). Executes exactly "
            "that action: download = fetch the PDF from arXiv to a folder (pure file, "
            "no indexing); ingest = 入库 — the complete parse + vector-index job that "
            "makes the paper searchable (a single atomic ASYNC background task "
            "returning a task_id); download_and_ingest = download then ingest. "
            "download is synchronous (returns the saved path + status); ingest actions "
            "return a task_id the parent tracks via check_task_status(task_id)."
        ),
        system_prompt=INGEST_SYSTEM,
        tools=["download_paper", "ingest_paper"],
    ),
]


SUBAGENT_NAMES = {s.name for s in SUBAGENTS}


def build_subagents(tools: dict | None = None):
    """Build subagents from the SUBAGENTS config table.

    Args:
        tools: {tool_name: BaseTool} registry. Defaults to get_cached_tools()
            (kept for backward compatibility with tests that monkeypatch it).

    Tools not present in `tools` are silently skipped; a subagent whose whole
    toolset is missing is omitted.
    """
    if tools is None:
        tools = {t.name: t for t in get_cached_tools()}

    out = []
    for spec in SUBAGENTS:
        toolset = [tools[n] for n in spec.tools if n in tools]
        if not toolset:
            continue
        subgraph, init_state = build_subagent(
            spec.name, spec.system_prompt, toolset,
            max_iterations=spec.max_iterations,
        )
        out.append(as_tool(spec.name, subgraph, spec.description,
                           init_state=init_state))
    return out
