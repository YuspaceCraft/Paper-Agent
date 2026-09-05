"""
subagents.py — Multi-Agent runtime (Phase 8).

build_subagent: compiles a subagent subgraph (agent_node + restricted tool
    executor + loop) — reuses the same agent_node/after_agent/build_tools_node
    as the parent search loop, differing only in restricted tools, dedicated
    system prompt, and isolated context (own message list, invisible to the
    parent).
as_tool: wraps a subgraph as a parent-callable tool ("task in → summary out").
build_subagents: config-driven factory — one SubagentSpec per subagent
    (arxiv / ingest — Claude Code 模式：仅写/外网操作隔离，库只读工具归父 agent),
    each injecting its own permitted tool subset.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.tools import StructuredTool, tool
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from .state import AgentState
from .nodes import agent_node, after_agent, build_tools_node
from .tools import get_cached_tools
from .config import get_limits
from .stream import emit, set_scope, reset_scope


# ---- runtime factory ----

async def _subagent_synthesize(state: AgentState, config) -> dict:
    """Safety net: subagent exhausted max_steps without a final answer.

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


# ---- leader gate（领导-部门制：worker 需要领导输入时 interrupt 暂停） ----

@tool
async def request_review(question: str) -> str:
    """Pause this task and ask the leader (main agent) for input or a decision.

    Execution stops immediately at an interrupt checkpoint; after the leader
    replies via task_resume(), the run continues and this tool returns the
    leader's answer as its result. Use ONLY when the task truly needs leader
    input: approval, missing information, or a direction choice the task
    cannot make on its own. Never use for routine sub-steps."""
    return "PENDING_LEADER_REVIEW"  # gate 节点 interrupt 后合成的领导回复才是真实结果


async def _gate_node(state: AgentState, config):
    """Leader gate node: pause on a request_review tool call (interrupt()).

    LangGraph 原生干预机制：节点内 interrupt() 让运行停在已保存的 checkpoint 上
    （ainvoke 正常返回），worker 线程保持 interrupted 状态；领导经
    supervisor.resume(task_id, reply) 以 Command(resume=reply, thread_id=task_id)
    续跑。node 再执行时把领导回复伪造为该 request_review 调用的 ToolMessage，
    worker 下一次 agent 调用即把它当正常工具结果看到。
    """
    from langchain_core.messages import ToolMessage
    from langgraph.types import interrupt

    msgs = state["messages"]
    last = next((m for m in reversed(msgs) if getattr(m, "type", "") == "ai"), None)
    call = None
    if last is not None and getattr(last, "tool_calls", None):
        call = next((
            tc for tc in last.tool_calls
            if (tc.get("name") if isinstance(tc, dict)
                else getattr(tc, "name", "")) == "request_review"
        ), None)
    if isinstance(call, dict):
        args, call_id = call.get("args") or {}, call.get("id") or ""
    else:
        args, call_id = (getattr(call, "args", None) or {}), getattr(call, "id", "") or ""
    question = str(args.get("question", "")) or "需要领导确认"
    call_id = call_id or "gate"
    # Py3.10 + async 节点：LangGraph 不注入 var_child_runnable_config，interrupt()
    # 依赖的 get_config() 会抛 "outside of a runnable context"（与 get_stream_writer
    # 同类限制，见 stream.py）。手动把节点 config 桥进 contextvar，绕过硬限制。
    from langchain_core.runnables.config import var_child_runnable_config
    _conf = config if isinstance(config, dict) else {}
    _tok = var_child_runnable_config.set(_conf)
    try:
        reply = interrupt({"question": question})
    finally:
        var_child_runnable_config.reset(_tok)
    if not reply:
        reply = "(领导未提供输入，按最佳判断继续)"
    return {
        "messages": [ToolMessage(
            content=f"[领导回复]\n{reply}", tool_call_id=call_id,
            name="request_review")],
    }


def build_subagent(name, system_prompt, tools, *, max_steps=5, checkpointer=None,
                   leader_gate=False):
    """Compile a subagent subgraph + its initial-state overrides.

    Returns (subgraph, init_state). The subagent config (subagent_system +
    bound_tools + max_steps) is returned separately instead of baked into
    a state subclass, because LangGraph does NOT pick up TypedDict class-attribute
    defaults — a subclass like `class S(AgentState): bound_tools = [...]` compiles
    but `state["bound_tools"]` is empty at runtime (KeyError). as_tool() merges
    init_state into the subgraph's input.

    Step 上限按 subagent 工作负担显式设置（SubagentSpec.max_steps）：subagent
    都是单任务窄工具面（arxiv 检索 / ingest 入库），一般 1~5 步足够，默认保持
    5；父 agent 的复杂编排走 state.max_steps（默认 30）。

    checkpointer: 共享 checkpointer → worker 状态按 thread 持久化（supervisor 派
    发模式使用：thread_id = task_id = 子 agent 自己的状态栈，见领导-部门制）。
    None（默认）→ 现有同步一次性子图，行为不变。

    leader_gate: True → 挂 gate 节点 + request_review 工具，worker 需要在领导输入
    时 interrupt 暂停（supervisor.resume 续跑），见 request_review/_gate_node。
    默认 False，路径零变化。
    """
    # 节点名与父层 react 循环(search_loop 外的 "agent"/"tools")区分开:
    # "subagent_agent"/"subagent_tools" 让 SSE 端(web/api/routers/agent.py 的
    # _msg_pump)能按 langgraph_node 排除 subagent 内部消息——subagent 叶子工具的
    # 卡片唯一权威来源是 as_tool._call 边界 + ToolDispatcher(scope 嵌套),若沿用
    # "agent"/"tools" 会与父循环同名 → 同一调用被双发射、前端树重复挂卡。
    sg = StateGraph(AgentState)
    sg.add_node("subagent_agent", agent_node)
    sg.add_node("subagent_tools", build_tools_node(tools))
    sg.add_node("synthesize", _subagent_synthesize)
    sg.set_entry_point("subagent_agent")

    edge_map = {"tools": "subagent_tools", "synthesize": "synthesize", "end": END}
    if leader_gate:
        sg.add_node("gate", _gate_node)
        edge_map["gate"] = "gate"
        sg.add_edge("gate", "subagent_agent")
    else:
        edge_map["gate"] = "synthesize"  # request_review 未绑定 → 不可达，仅保映射完整

    sg.add_conditional_edges("subagent_agent", after_agent, edge_map)
    sg.add_edge("subagent_tools", "subagent_agent")
    sg.add_edge("synthesize", END)
    init_state = {
        "subagent_system": system_prompt,
        "bound_tools": [t.name for t in tools],
        "max_steps": max_steps,
    }
    return sg.compile(checkpointer=checkpointer), init_state


class SubagentArgs(BaseModel):
    task: str = Field(description="Self-contained task description for the subagent")


def as_tool(name, subgraph, description, args_model=SubagentArgs, init_state=None):
    """Wrap a subgraph as a parent-callable tool. Returns the subagent's final
    answer (last non-empty AI message) as a string — the parent never sees the
    subagent's internal tool messages (context isolation)."""
    from langchain_core.messages import HumanMessage

    # langchain-core 1.4 的 `_get_runnable_config_param` 用 `type_ is RunnableConfig`
    # 严格身份比较判定是否注入 config,而且 `get_type_hints` 会把「带 None 默认值
    # 的参数」推断成 Optional → 身份不符 → 不注入。所以这里必须: ①注解裸
    # RunnableConfig ②不能有 None 默认值(StructuredTool._arun 总会注入 config)。
    async def _call(config: RunnableConfig, **kwargs) -> str:
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
            # (P2) 零状态: 不再从 contextvar 注入父级的 `resolved`——subagent 的
            # system 反复声明自身的 "task 自包含"。已确认的论文名由父代理按
            # AGENT_SYSTEM「Delegation Priority」约定直接写进 task 字符串
            # （显式上下文，无隐藏通道；未来 contextvar 不跨线程传播也不受影响）。
            init: dict = {"messages": [HumanMessage(content=task)]}
            if init_state:
                init.update(init_state)
            # config 透传: 让 subgraph 作为当前 run 的子 run 挂到同一 LangSmith
            # trace。此前不带 config 地 ainvoke 会脱离父 trace(日志看不到 creation
            # 内部调用,SSE 却正常)。注意 arun 注入的 config 不带 callbacks,带
            # callbacks 的 child_config 在 var_child_runnable_config 里——优先用它。
            run_config = config
            try:
                from langchain_core.runnables.config import var_child_runnable_config
                ctx_cfg = var_child_runnable_config.get()
                if ctx_cfg is not None:
                    run_config = ctx_cfg
            except Exception:
                pass
            result = await subgraph.ainvoke(init, config=run_config)
            answer = ""
            # (P3) 只取「无 tool_calls 的 AI 消息」作为最终答案——带 tool_calls 的
            # AI 消息仍是中途状态（正计划下一步/被 max_steps 截断），其 content
            # 再长也不是答案。
            for m in reversed(result.get("messages", [])):
                if getattr(m, "type", "") != "ai":
                    continue
                if getattr(m, "tool_calls", None):
                    continue
                if getattr(m, "content", "").strip():
                    answer = str(m.content).strip()
                    break
            ok = bool(answer)
            if not ok:
                from .tool_contract import err as _err_contract
                answer = _err_contract("unknown", "subagent produced no final answer")
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
    max_steps: int = 5    # step 粒度的子代理轮次上限（单任务窄工具面够用）


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


CREATOR_SYSTEM = """\
You are the scientific writing specialist. Write the EXACT section requested in the
task — nothing more. The task is self-contained (zero-state): it carries the document
id, the section id/title, the reference papers, and the writing style. Never invent
information absent from the task or from tool results.

Toolset (all you may call):
- doc_write_section(doc_id, section_id, content): atomic write of one section's
  Markdown. The section must exist in the doc outline. ALWAYS finish by calling this
  with the full written section — a section without doc_write_section is not produced.
- doc_get_state(doc_id): read the doc outline / prior sections (avoid repetition and
  keep cross-section consistency).
- search_papers(query, top_k) / fetch_content(paper_name, section): gather comparison
  material from the LOCAL library when the task references papers. Cite with [N]
  markers inline next to the supported claim (keep them verbatim — the parent resolves
  them to [Paper, page N] later).
- read_file / list_dir: inspect workspace files only if the task explicitly needs them.
- experiment_list(project) / read_metrics(exp_id) / study_context(topic): when the
  task cites experiment results or compares against a baseline, pull the REAL
  numbers with these tools — never invent metrics or reuse figures from the task
  text verbatim; copy values from tool results.

Writing rules (scientific, structured):
1. Structure the section with short academic paragraphs; use bullets/tables/figure
   captions only when the section type calls for them. Concise, no fluff.
2. Every factual claim about a paper must trace to a tool result in this conversation
   (fetch_content / search_papers). Papers you could NOT read → either skip that claim
   or mark it "untreated in this section".
3. Inline citations [N]: place the marker immediately after the sentence it supports.
4. Do NOT write the section heading inside the content — the tool stores the title
   separately.
5. Zero-state: the task contains everything you need. Do not reference previous
   conversations or other agents.

MANDATORY workflow — reading is never the endpoint:
1. If reference papers exist in the library, read enough to ground this section
   (search_papers / fetch_content). Do NOT stop after reading.
2. Compose the full section as Markdown in your head.
3. Call doc_write_section(doc_id, section_id, content) with the ENTIRE content in
   ONE call. This tool call is the REQUIRED final action.
4. Never finish your turn with a plain-text answer instead of the tool call — a
   section that was not written via doc_write_section does not exist.

Output: after doc_write_section succeeds, output ONLY a short status line, no
preamble, no restated content:
<section_id> | <word count> words | wrote via doc_write_section\
"""


CODER_SYSTEM = """\
You are the experiment/code specialist. Execute the task inside an experiment project
under the configured experiments root / <project>. The task is self-contained (zero-state).

Toolset (all you may call):
- run_experiment(project, command, name): run a command in that project as a
  BACKGROUND experiment — logs stream to runs/<exp_id>/run.log, metrics.json/
  metrics.csv written by the command are parsed on completion. Returns exp_id.
- experiment_status(exp_id) / experiment_list(project) / read_metrics(exp_id):
  inspect results (status / exit_code / git_sha / metrics / log tail).
- set_experiment_project(project, paper, entry_run, description): bind this work
  to a project and establish/update its manifest (project.json) — the delegation
  contract. CALL IT when you establish which project you are working on.
- experiment_project_state(project): read the project manifest (entry.run /
  key_files / baseline / changelog) + recent experiments. Read it to learn the
  project's entry points / baseline before delegating or optimizing.
- git_status(project) / git_diff(project) / git_log(project) / git_commit(project, message):
  version-control the project (commit = destructive, gated).
- delegate_code_task(project, prompt): hand a coding job (implement/tune/fix/
  refactor) to the EXTERNAL coding agent — it edits files directly in the project
  and returns a summary + changed_files. Use this for CODE CHANGES; do not hand
  write files yourself.
- study_context(topic) / study_add_hypothesis(topic, hypothesis): read/write the
  study knowledge base (prior experiments = comparison baseline for optimization).
- read_file / list_dir / search_papers / fetch_content: inspect code and reference papers.

Workflow (recommended):
1. Explore: list_dir + git_status (+ read_file of the key script) to learn the project.
2. Read study_context(topic) for the baseline/SOTA and prior experiment metrics.
3. Run experiments: run_experiment → experiment_status (poll until done) →
   read_metrics. Compare numbers vs baseline.
4. To improve: delegate_code_task to change code, then re-run and re-compare
   (loop at most 2 improvement iterations unless the user wants more).
5. git_commit a logical checkpoint when a change is verified (message describes it).

Honesty:
- Never report metrics/files the tools did not return — copy values verbatim.
- If run_experiment fails, show the real error snippet, do not invent a fix as done.
- Keep the summary short: what was run (exp_id), the metrics delta vs baseline,
  what was changed (files), and the recommended next step.

Output ONLY a short status summary with a machine-parseable footer (the leader
derives the conversation context from it — NEVER omit):
PROJECT: <project name>
EXP: <exp_id for every experiment you ran, one per line>
"""


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
    SubagentSpec(
        name="creator",
        description=(
            "Write one section/chapter of a research document given a "
            "self-contained writing task (doc_id, section_id/title, reference "
            "papers, style). Gathers comparison material from the library and "
            "writes the section atomically via doc_write_section. Returns only "
            "a status line. Use for paper/review/report section drafting."
        ),
        system_prompt=CREATOR_SYSTEM,
        tools=[
            "doc_write_section", "doc_get_state", "doc_create",
            "doc_set_outline", "doc_list", "doc_export_docx",
            "search_papers", "fetch_content",
            # 实验引用（对话中心化：写完实验章要引用真实指标/基线）
            "experiment_list", "read_metrics", "study_context",
            "read_file", "list_dir",
        ],
        # 综述章节要先读多篇论文(fetch_content 多轮)再落盘;5 轮会中途打满 →
        # 路由到 _subagent_synthesize 用泛化 prompt 拼出正文、章节永不写盘。
        # 12 覆盖「读 3-5 篇 + doc_get_state + doc_write_section」的典型负担。
        # 实际生效值由 agent/config.yaml `subagents.creator.max_steps` 决定,
        # 此处为无配置文件时的代码兜底。
        max_steps=12,
    ),
    SubagentSpec(
        name="coder",
        description=(
            "Run experiments and improve code inside an experiment project "
            "(configured experiments root / <project>): run commands in the background, "
            "poll metrics, delegate code changes to the external coding agent, "
            "follow git versioning, read the study knowledge base for baselines. "
            "Use for 复现/跑实验/调参/优化代码/读指标 requests."
        ),
        system_prompt=CODER_SYSTEM,
        tools=[
            "run_experiment", "experiment_status", "experiment_list", "read_metrics",
            "git_status", "git_diff", "git_log", "git_commit",
            "delegate_code_task",
            "set_experiment_project", "experiment_project_state",
            "study_context", "study_add_hypothesis",
            "read_file", "list_dir",
            "search_papers", "fetch_content",
        ],
    ),
]


SUBAGENT_NAMES = {s.name for s in SUBAGENTS}


def build_subagents(tools: dict | None = None, checkpointer=None):
    """Build subagents from the SUBAGENTS config table.

    Args:
        tools: {tool_name: BaseTool} registry. Defaults to get_cached_tools()
            (kept for backward compatibility with tests that monkeypatch it).
        checkpointer: 透传给 build_subagent（supervisor 派发模式挂载；现有同步
            用途不传 → 一次性子图，行为不变）。

    Tools not present in `tools` are silently skipped; a subagent whose whole
    toolset is missing is omitted.

    Step 上限(tool 轮次)统一走 agent/config.yaml: `subagents.<name>.max_steps`
    覆盖 SUBAGENTS 表里的代码默认(表默认值作为无配置文件时的兜底)。
    """
    if tools is None:
        tools = {t.name: t for t in get_cached_tools()}

    limits = get_limits()

    out = []
    for spec in SUBAGENTS:
        toolset = [tools[n] for n in spec.tools if n in tools]
        if not toolset:
            continue
        cfg = limits.subagents.get(spec.name) if limits else None
        max_steps = cfg.max_steps if cfg else spec.max_steps
        subgraph, init_state = build_subagent(
            spec.name, spec.system_prompt, toolset,
            max_steps=max_steps, checkpointer=checkpointer,
        )
        out.append(as_tool(spec.name, subgraph, spec.description,
                           init_state=init_state))
    return out
