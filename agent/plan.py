"""
plan.py — Plan-and-Execute (Phase 7 / v14 LLM 逐步执行).

decide_mode: pure heuristic — "react" (simple query, zero regression) vs
    "plan" (multi-paper / comparison / multi-sub-question). 客户端可经
    state.requested_mode 显式覆盖。
plan_node: LLM structured output → ordered outcome steps (paper 域无 target).
executor_node: topological executor — 步骤顺序执行（无 asyncio.gather）：
    auto 步骤 → _run_step_agent（LLM 逐步执行，动态多次调工具）；
    tool / subagent 步骤 → 确定性 _run_step。

target semantics:
  - "auto" (paper 域默认) → LLM 逐步执行，模型动态选工具多次调用
  - "tool" → a DIRECT parent tool (search_papers / fetch_content / list_dir /
    read_file / write_file / check_paper / check_task_status). All LOCAL work.
  - a subagent name (arxiv / ingest / creator / coder) → subagent 工具 (Phase 8 / v10).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from .prompts import PLAN_SYSTEM
from .resolution import canonicalize
from .state import AgentState
from .observability import timed, count, log_event
from .stream import emit


# ---- mode heuristic (pure, no LLM) ----

_COMPARE_KEYWORDS = (
    "对比", "比较", "compare", "comparison", "vs", "versus", "区别", "差异",
    "哪个更好", "better", "survey", "综述",
)

def _last_user_text(state: dict) -> str:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human":
            return (m.content or "") if hasattr(m, "content") else str(m)
    return ""


# 显式异步写作信号：用户要求「后台写/异步」→ 不强制走同步 plan，让 react 循环
# 用 task_dispatch(role="creator", ...) 逐章派发（领导-部门制异步写作）。
_ASYNC_CREATION_HINTS = (
    "后台", "异步", "写好了叫我", "写完通知我", "先跑着",
    "background", "asynchronously", "in the background",
)


def _is_async_creation(q: str) -> bool:
    ql = (q or "").lower()
    return any(h in ql for h in _ASYNC_CREATION_HINTS)


def _heuristic_mode(state: dict) -> str:
    """The auto-detection heuristic (no override). Returns "react" or "plan"."""
    domain = state.get("domain")
    if domain in ("creation", "coding"):
        if domain == "creation" and _is_async_creation(_last_user_text(state)):
            return "react"
        return "plan"
    q = _last_user_text(state)
    if q:
        ql = q.lower()
        if any(kw in ql for kw in _COMPARE_KEYWORDS):
            return "plan"
        # multiple explicit sub-questions
        if q.count("？") + q.count("?") >= 2:
            return "plan"

    resolved = state.get("resolved", {}) or {}
    papers = resolved.get("papers", []) if isinstance(resolved, dict) else []
    confirmed = [p for p in papers if p.get("level") in ("EXACT", "HIGH", "MEDIUM")]
    if len({p.get("match") for p in confirmed}) >= 2:
        return "plan"
    return "react"


def decide_mode(state: dict) -> str:
    """Pick execution mode. "react" keeps the existing single-ReAct path.

    优先级最高是客户端显式覆盖（state.requested_mode: "react"/"plan"），
    之后才走 _heuristic_mode 自动判断；auto 时保持原行为。
    无论哪种都 emit {"type":"mode", source:"user"|"auto"} 供前端标注实际模式。
    """
    forced = str(state.get("requested_mode") or "auto")
    if forced in ("react", "plan"):
        emit({"type": "mode", "mode": forced, "source": "user"})
        return forced
    mode = _heuristic_mode(state)
    emit({"type": "mode", "mode": mode, "source": "auto"})
    return mode


# ---- structured plan output ----

class PlanStep(BaseModel):
    id: str
    description: str = Field(description="What this step must achieve/answer (outcome)")
    target: Literal["tool", "arxiv", "ingest", "creator", "coder", "auto"] = Field(
        default="auto",
        description='"auto" (paper 域默认): LLM 逐步执行，模型动态多次调工具 | '
        '"tool": 确定性单工具 | arxiv/ingest/creator/coder: subagent 边界 (v8/v10)',
    )
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class PlanResult(BaseModel):
    steps: list[PlanStep]


def _emit_plan(plan: list[dict]) -> None:
    """Push the plan event with per-step TODO status (all pending up front)."""
    emit({
        "type": "plan",
        "steps": [
            {
                k: s.get(k)
                for k in ("id", "description", "target", "depends_on")
            }
            | {"status": "pending"}
            for s in plan
        ],
    })


# ---- plan_node ----

async def _ask_for_plan(model, prompt: str, attempts: int = 2,
                        system: str | None = None) -> list[dict]:
    """Structured plan extraction with graceful degradation. Returns [] instead
    of raising/crashing when the model produces no plan.

    Extraction contract: PLAN_SYSTEM asks the model to output a JSON object
    directly ("Output ONLY a JSON object with a steps array"). We parse that
    from the reply content (strip code fences if present), and ALSO accept a
    function-call result if the provider emitted one. Previously this node
    used `with_structured_output(method="function_calling")`, which returns
    None whenever the model replies with JSON text instead of an OpenAI tool
    call — observed consistently with qwen-plus — so a valid plan was silently
    discarded and the turn degenerated into the fallback. Parsing the model's
    actual output is the root fix, not another retry.

    `system` overrides the system prompt (creation domain uses
    CREATION_PLAN_SYSTEM); default PLAN_SYSTEM keeps paper behavior unchanged.
    """
    msgs = [SystemMessage(content=system or PLAN_SYSTEM), HumanMessage(content=prompt)]
    for attempt in range(attempts):
        count("llm_calls")
        try:
            response = await model.ainvoke(msgs)
        except Exception as exc:
            log_event("plan_llm_failed", node="plan", level="warning",
                      attempt=attempt, error=f"{type(exc).__name__}: {exc}")
            continue

        # 1) provider emitted a function call (belt — rare but possible)
        for tc in getattr(response, "tool_calls", None) or []:
            args = tc.get("args") or (tc.get("function") or {}).get("arguments", "")
            steps = _parse_steps(args)
            if steps:
                return steps

        # 2) JSON object in the reply text (the contract PLAN_SYSTEM asks for)
        text = getattr(response, "content", "")
        if isinstance(text, str):
            raw = _extract_json_text(text)
            if raw:
                steps = _parse_steps(raw)
                if steps:
                    return steps

        log_event("plan_empty_result", node="plan", level="warning", attempt=attempt)
    return []


def _extract_json_text(text: str) -> str | None:
    """Pull the JSON object out of an LLM reply (no fences / preamble / prose)."""
    t = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fenced:
        t = fenced.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return t[start:end + 1]


def _parse_steps(raw_json: str) -> list[dict] | None:
    """Validate raw JSON against PlanResult; drop invalid steps. None = no plan."""
    try:
        data = json.loads(raw_json)
        result = PlanResult.model_validate(data)
    except Exception:
        return None
    steps = [s.model_dump() for s in result.steps]
    # a plan with only dangling/invalid steps is no plan — degrade cleanly.
    # v14: target 可缺省（pydantic 已填 "auto"），只要求有 id + 描述。
    valid = [s for s in steps if s.get("id") and s.get("description")]
    return valid or None


def _fallback_plan(state: dict) -> list[dict]:
    """Deterministic single-read plan when the LLM yields no steps.

    One direct-library step (target="tool" → fetch_content) per resolved paper —
    read-only, never schedules downloads/indexing, covers the common plan-mode
    intents (find/compare papers already in the library). Empty list when nothing
    was resolved → executor no-ops → synthesize replies gracefully.
    """
    resolved = state.get("resolved", {}) or {}
    papers = resolved.get("papers", []) if isinstance(resolved, dict) else []
    steps: list[dict] = []
    for i, p in enumerate(papers, start=1):
        name = p.get("match") or p.get("query") or "(referenced paper)"
        steps.append({
            "id": f"fb-{i}",
            "description": f"Read the local library for {name}",
            "target": "tool",
            "args": {"tool": "fetch_content", "paper_name": name},
            "depends_on": [],
        })
    return steps


@timed("plan")
async def plan_node(state: AgentState, config) -> dict:
    """LLM → structured ordered steps (or creation outline). Never raises:
    `_ask_for_plan` degrades to a deterministic fallback if the LLM yields none.

    Domain-aware:
      - paper (default) → PLAN_SYSTEM (zero regression)
      - creation → CREATION_PLAN_SYSTEM (章节大纲) + 建 doc/注入 doc_id
      - coding  → 默认 PLAN_SYSTEM（Phase C 配 CODING 专用 target 后细分）
    """
    from .nodes import _get_model  # lazy: avoid import cycle

    # Plain model — no with_structured_output. PLAN_SYSTEM's contract is direct
    # JSON text ("Output ONLY a JSON object"), which qwen-plus reliably follows;
    # with_structured_output(method="function_calling") waits for an OpenAI tool
    # call the model never emits and drops the reply as None (see _ask_for_plan).
    model = _get_model(config)

    query = _last_user_text(state) or "(none)"
    entities = ", ".join(e for e in state.get("entities", []) if e) or "(none)"
    resolved = state.get("resolved", {}) or {}
    papers = resolved.get("papers", []) if isinstance(resolved, dict) else []
    hints = "\n".join(
        f"- {p.get('query', '')} → {p.get('match', '')} ({p.get('level', 'NONE')})"
        for p in papers
    ) or "(no resolved hints)"

    if state.get("domain") == "creation":
        return await _creation_plan(model, state, query, entities, hints)
    if state.get("domain") == "coding":
        return await _coding_plan(model, query, entities, hints)

    prompt = (
        f"## User Question\n{query}\n\n"
        f"## Key Entities\n{entities}\n\n"
        f"## Resolved Paper References\n{hints}"
    )

    plan = await _ask_for_plan(model, prompt)
    if not plan:
        plan = _fallback_plan(state)
        log_event("plan_fallback", node="plan", level="warning", n_steps=len(plan))

    _emit_plan(plan)

    return {
        "mode": "plan",
        "plan": plan,
        "plan_progress": 0,
    }


async def _coding_plan(model, query: str, entities: str, hints: str) -> dict:
    """Coding-domain planning: 实验/代码请求 → coder/study 步骤表。

    MVP 不做确定性 fallback 步骤（无已知实验参数时空 plan → executor no-op →
    synthesize 兜底回答）；不建 state 额外字段。
    """
    from .prompts import CODING_PLAN_SYSTEM

    prompt = (
        f"## User Question\n{query}\n\n"
        f"## Key Entities\n{entities}\n\n"
        f"## Resolved Paper References\n{hints}"
    )
    plan: list[dict] = await _ask_for_plan(model, prompt, system=CODING_PLAN_SYSTEM)
    if not plan:
        log_event("coding_plan_fallback", node="plan", level="warning")

    _emit_plan(plan)
    return {"mode": "plan", "plan": plan, "plan_progress": 0}


async def _creation_plan(model, state: AgentState, query: str,
                        entities: str, hints: str) -> dict:
    """Creation-domain planning: 章节大纲 → 建 doc（确定性代码）→ 步骤注入 doc_id。

    `_ensure_writing_doc` 在 agent/domains/creation.py（业务模块）里建文档并写
    大纲（outline 来自本步骤产出的步骤表），doc_id 注入每个 creator 步骤的 args，
    使 executor 逐章调用 creator subagent 时能定位文档。任何失败都不 raise：
    空 plan → 不建 doc，executor no-op，synthesize 兜底回答。
    """
    from .prompts import CREATION_PLAN_SYSTEM

    prompt = (
        f"## User Writing Request\n{query}\n\n"
        f"## Key Entities\n{entities}\n\n"
        f"## Resolved Paper References\n{hints}"
    )
    plan: list[dict] = await _ask_for_plan(model, prompt, system=CREATION_PLAN_SYSTEM)
    if not plan:
        log_event("creation_plan_fallback", node="plan", level="warning")
        return {"mode": "plan", "plan": [], "plan_progress": 0, "doc_id": None}

    # 章节强制串行(覆盖 LLM 的空 depends_on): 并行 creator 同写一份 doc.json 是
    # read-modify-write 竞争,会丢章节状态;串行让后章 doc_get_state 能引用前章
    # 已写内容,交叉一致性才有意义。
    prev: str | None = None
    for _step in plan:
        if prev:
            _step["depends_on"] = [prev]
        prev = _step.get("id")

    outline = [
        (s.get("args") or {}).get("section_id", s.get("id", ""))
        for s in plan
    ]
    try:
        from .domains.creation import _ensure_writing_doc
        doc_id = await _ensure_writing_doc(query, outline, plan)
    except Exception as exc:
        log_event("creation_doc_failed", node="plan", level="warning",
                  error=f"{type(exc).__name__}: {exc}")
        return {"mode": "plan", "plan": [], "plan_progress": 0, "doc_id": None}

    for step in plan:
        args = dict(step.get("args") or {})
        args["doc_id"] = doc_id
        step["args"] = args

    _emit_plan(plan)

    return {
        "mode": "plan",
        "plan": plan,
        "plan_progress": 0,
        "doc_id": doc_id,
    }


# ---- executor ----

def _subagent_task(description: str, args: dict) -> str:
    """Fold a plan step into the single "task" string subagents accept.

    Subagent tools expose exactly one field (SubagentArgs.task), but plan_node
    emits natural arg names (query/paper_id/...). Deterministically re-fold
    description + args into one self-contained task so the call never fails
    schema validation.

    Non-empty args render as a `key: value` command block — the SAME contract the
    parent react loop is prompted to emit (esp. for ingest: action / arxiv_id /
    paper_name / pdf_path must survive the folding verbatim).
    """
    task = description or ""
    if not args:
        return task
    lines = []
    for k, v in args.items():
        lines.append(f"{k}: {v}" if not isinstance(v, (dict, list)) else f"{k}: {json.dumps(v, ensure_ascii=False)}")
    block = "\n".join(lines)
    return f"{task}\n{block}" if task else block


async def _verify_creator_step(step: dict, out: str) -> tuple[bool, str, str]:
    """Creator 步骤的权威校验: 该 section 必须已在 doc 落盘(status=done)。

    subagent 无论返回多完整的正文,只要没经过 doc_write_section 写进 doc 就
    等于未产出——返回 ok=False 且不转发正文,progress 由 synthesize 按 doc 状态
    生成(避免「聊天出全文、doc 没章节」的脱节)。
    """
    from .domains.creation import verify_section_written

    args = dict(step.get("args") or {})
    doc_id = str(args.get("doc_id") or "")
    section_id = str(args.get("section_id") or "")
    if not doc_id or not section_id:
        return False, "", f"creator 步骤缺少 doc_id/section_id: {args}"
    written, wc = await verify_section_written(doc_id, section_id)
    if written:
        return (True, f"{section_id} | {wc} words | wrote via doc_write_section (verified)", "")
    return (
        False, "",
        f"creator 未调用 doc_write_section，章节「{section_id}」未落盘。"
        f"subagent 仅返回文本: {str(out)[:160]}",
    )


async def _run_step(step: dict, state: dict, config) -> dict:
    """Execute one step. Returns {step_id, ok, output, error}.

    Looks up the target (subagent name or "tool") in get_cached_tools().
    Structured degradation: unknown target / missing tool → ok=False, never raises.
    Emits tool_start/tool_end (reusing the react-mode SSE shape) so the client
    renders each plan step as a collapsible card, plus plan_step lifecycle
    events (running → done/failed) that drive the plan TODO checklist.
    """
    step_id = step.get("id", "")
    target = step.get("target", "tool")
    args = dict(step.get("args") or {})
    start = time.monotonic()
    is_subagent = False

    def _plan_end(status: str) -> None:
        emit({"type": "plan_step", "id": step_id, "status": status})

    def _end(name: str, status: str, result: str) -> None:
        _plan_end("done" if status == "success" else "failed")
        emit({
            "type": "tool_end", "id": step_id, "name": name,
            "status": status, "result": str(result)[:4000],
            "execution_time": round(time.monotonic() - start, 2),
        })

    try:
        from .tools import get_cached_tools
        tools = {t.name: t for t in get_cached_tools()}

        if target == "tool":
            tool_name = args.pop("tool", None) or args.pop("name", None)
            name = tool_name or target
            tool = tools.get(tool_name) if tool_name else None
            call_args = args
        else:
            # subagent target → single "task" arg (see _subagent_task)
            name = target
            tool = tools.get(target)
            call_args = {"task": _subagent_task(step.get("description", ""), args)}
            is_subagent = True

        if tool is None:
            _end(name, "error", f"unknown target/tool: {target}")
            return {
                "step_id": step_id, "ok": False, "output": "",
                "error": f"unknown target/tool: {target}",
            }

        # TODO 列表驱动：真实执行前标 running（重试会重复 emit，前端幂等覆盖）
        emit({"type": "plan_step", "id": step_id, "status": "running",
              "name": name, "description": step.get("description", "")})

        if is_subagent:
            # subagent 的 as_tool._call 自己 emit 边界 + 叶子工具事件；
            # 这里只 await 拿结果，避免重复卡片。config 透传: subgraph 作为子
            # run 挂到父 trace(LangSmith 才能看到 creation 内部调用)。
            out = await tool.ainvoke(call_args, config=config)
            if target == "creator":
                ok, output, err = await _verify_creator_step(step, out)
                _plan_end("done" if ok else "failed")
                return {
                    "step_id": step_id, "ok": ok, "output": output, "error": err,
                }
            _plan_end("done")
            return {
                "step_id": step_id, "ok": True, "output": str(out),
            }

        emit({"type": "tool_start", "id": step_id, "name": name, "args": call_args})
        out_raw = await tool.ainvoke(call_args, config=config)
        out_str = str(out_raw)
        # (P4) 工具错误以错误信封形式正常返回（非异常）——统一解析后把
        # 「调用成功但语义失败」归一为 ok=False，好让 executor 走恢复/标注，
        # 而不是把 {"ok": false, ...} 当成功结果交给 synthesize。
        from .tool_contract import parse_tool_result
        parsed = parse_tool_result(out_str)
        if parsed.is_envelope and not parsed.ok:
            _end(name, "error", out_str)
            return {
                "step_id": step_id, "ok": False, "output": out_str,
                "error": parsed.error,
            }
        _end(name, "success", out_str)
        return {
            "step_id": step_id, "ok": True, "output": out_str,
        }
    except Exception as exc:
        # subagent 失败时 _call 已用 run_id emit tool_end(error)，这里不再重复。
        # 但 plan_step 终态仍要发出——否则 TODO 列表卡在 running。
        if not is_subagent:
            _end(target, "error", f"{type(exc).__name__}: {exc}")
        else:
            _plan_end("failed")
        return {
            "step_id": step_id, "ok": False, "output": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


@timed("executor")
async def executor_node(state: AgentState, config) -> dict:
    """Topological execution. Independent steps run in parallel; dependent steps
    wait for their depends_on. Never raises — bad steps degrade to ok=False.

    调用逻辑守卫（plan 模式唯一的分支点）：同一 plan 内若已执行 check_paper 且
    判定「论文已在本地/已入库」，同论文的下载/入库步骤在此被跳过，不再无条件执行。
    """
    plan = state.get("plan", [])
    by_id = {s["id"]: s for s in plan}
    results: dict[str, dict] = {
        r["step_id"]: r for r in state.get("subagent_results", [])
    }
    done: set[str] = set(results)
    # creator 失败重试一次: subagent 以纯文本作答(未调 doc_write_section)是高频
    # 模式,一次显式 "必须落盘" 提示通常足以修正,避免整章静默丢失。
    retried: set[str] = set()
    remaining = [s for s in plan if s.get("id") not in done]
    total = len(plan)

    def _emit_progress() -> None:
        emit({"type": "plan_progress", "done": len(done), "total": total})

    while remaining:
        ready = [
            s for s in remaining
            if all(d in done for d in s.get("depends_on", []))
        ]
        if not ready:
            # cycle or dangling dependency — degrade, don't spin
            break

        # 1) 同一批里的 check_paper 步骤先执行：它的确定性结论是后续入库/下载步骤
        #    的守卫依据（plan-and-execute 本身无分支，靠这里做分支）。
        for s in ready:
            if not _is_check_step(s, by_id):
                continue
            out = await _run_step(s, state, config)
            results[s["id"]] = out
            done.add(s["id"])
        _emit_progress()

        # 2) 其余步骤顺序执行（v14 去掉 asyncio.gather：LLM 逐步执行并发会交错
        #    工具事件、并发烧 LLM，顺序符合 Claude 逐步骤执行）。守卫命中的跳过。
        for s in ready:
            if s["id"] in done:
                continue
            note = _ingest_guard(s, results, by_id)
            if note is not None:
                emit({"type": "plan_step", "id": s["id"], "status": "skipped",
                      "name": "guard", "description": s.get("description", ""),
                      "output": note})
                results[s["id"]] = {
                    "step_id": s["id"], "ok": True,
                    "output": note, "error": "", "skipped": True,
                }
                done.add(s["id"])
                continue

            if _is_agent_step(s):
                # LLM 逐步执行：步骤=结果单元，模型动态多次调工具
                out = await _run_step_agent(s, state, config,
                                            _ok_outputs(results))
            else:
                out = await _run_step(s, state, config)

            # (P4) 直接工具步骤失败 → 确定性重试一次: transient 原参数; param_error
            # 且错误带 available_papers/sections → 修正参数。对比 react 模式的
            # LLM 错误恢复，这里不用 LLM,只做确定性的参数修正/重试(见
            # _retry_args_from_error)。auto 步骤内部已有 LLM 重试循环，不在此再重试。
            if (
                out.get("ok") is False
                and s.get("target") == "tool"
                and s.get("id") not in retried
            ):
                retried.add(s.get("id"))
                retry_args = _retry_args_from_error(s, out)
                if retry_args is not None:
                    retry = dict(s)
                    retry["args"] = retry_args
                    log_event("tool_step_retry", node="executor", level="warning",
                              step_id=s.get("id"))
                    retried_out = await _run_step(retry, state, config)
                    if retried_out.get("ok"):
                        out = retried_out
                    else:
                        retried_out["error"] = (
                            f"[重试一次仍失败] {retried_out.get('error', '')}"
                        )
                        out = retried_out
            # creator 落盘失败 → 重试一次(任务附明确落盘指令)
            if (
                out.get("ok") is False
                and s.get("target") == "creator"
                and s.get("id") not in retried
            ):
                retried.add(s.get("id"))
                retry = dict(s)
                retry_args = dict(s.get("args") or {})
                retry_args["_retry_hint"] = (
                    "上一轮没有调用 doc_write_section。现在必须调用 "
                    "doc_write_section(doc_id, section_id, content) 把整段内容写入 doc,"
                    "然后输出 ONLY 状态行: `<section_id> | <N> words | wrote via "
                    "doc_write_section`。不得以纯文本输出正文。"
                )
                retry["args"] = retry_args
                log_event("creator_step_retry", node="executor", level="warning",
                          step_id=s.get("id"))
                out = await _run_step(retry, state, config)
            results[s["id"]] = out
            done.add(s["id"])
        _emit_progress()
        remaining = [s for s in remaining if s.get("id") not in done]

    return {
        "subagent_results": list(results.values()),
        "plan_progress": len(done),
        # TODO 状态回填（也持久化进 checkpoint，synthesize/verify 消费）
        "plan": _statused_plan(plan, results),
        "plan_done": len(done),
        "plan_total": total,
    }


def _is_agent_step(step: dict) -> bool:
    """LLM 逐步执行步骤：target 缺省/auto = paper 域结果单元。"""
    return (step.get("target") or "auto") == "auto"


def _ok_outputs(results: dict) -> dict:
    """已完成（ok 且非 skipped）步骤产出，供后步复用（depends_on 语义）。"""
    return {
        sid: (r.get("output") or "")
        for sid, r in results.items()
        if r.get("ok") and not r.get("skipped")
    }


# ---- LLM 逐步执行（v14）：一个结果步骤 = agent 循环，动态多次调工具 ----


def _step_budget() -> int:
    """单步骤 agent 循环的工具调用轮次上限：env > agent/config.yaml > 默认 10。"""
    try:
        from .config import get_limits
        v = get_limits().plan_step_max_steps
        return v if v and v > 0 else 10
    except Exception:
        return 10


async def _run_step_agent(step: dict, state: dict, config,
                          prior_outputs: dict | None = None) -> dict:
    """Execute one outcome step via a bounded LLM agent loop.

    步骤 = 结果单元：per-step 纯净对话（中间产物不外泄到父线程），模型动态选择
    并多次调用工具。每次工具调用 emit tool_start/tool_end（父层工具卡片），
    文本产出 = 步骤答案，成为 synthesize 证据。never raise。
    """
    from .nodes import _get_bound_model, _stream_llm
    from .tool_contract import parse_tool_result, truncate_tool_result
    from .tools import get_cached_tools
    from .prompts import STEP_EXEC_SYSTEM

    step_id = step.get("id", "")
    description = (step.get("description") or "").strip() or "(步骤)"

    def _ps(status: str, output: str = "") -> None:
        emit({"type": "plan_step", "id": step_id, "status": status,
              **({"output": output} if output else {})})

    _ps("running")

    # 上下文：resolved 可信论文名（别重搜）+ 前序步骤产出（depends_on 引用）
    resolved = state.get("resolved", {}) or {}
    papers = resolved.get("papers", []) if isinstance(resolved, dict) else []
    hints = "\n".join(
        f'- "{p.get("query", "")}" → "{p.get("match", "")}" ({p.get("level", "NONE")})'
        for p in papers if p.get("match")
    )
    prior = ""
    if prior_outputs:
        lines = []
        for sid, txt in prior_outputs.items():
            t = str(txt).strip()
            if t:
                lines.append(f"- {sid}: {t[:400]}")
        if lines:
            prior = "\n" + "\n".join(lines[:8])

    system = STEP_EXEC_SYSTEM
    if hints:
        system += f"\n\n## Resolved paper references (trust these names)\n{hints}"
    if prior:
        system += f"\n\n## Previous steps completed\n{prior}"
    msgs: list = [SystemMessage(content=system), HumanMessage(content=description)]

    tools = {t.name: t for t in get_cached_tools()}
    # SUBAGENT_NAMES: subagent 工具(arxiv/ingest/…)的卡片由 as_tool._call 边界
    # 唯一发出,这里不再手动 emit,避免与边界卡重复(与 _run_step 的做法一致)。
    from .subagents import SUBAGENT_NAMES
    model = _get_bound_model(config)
    budget = _step_budget()
    last_text = ""
    consecutive_down = 0
    error = ""
    # 本步骤内相同(工具,参数)去重:命中直接复用上次结果,不再重复执行副作用。
    result_cache: dict[str, str] = {}

    def _tool_key(name: str, args: dict) -> str:
        try:
            args_sorted = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except Exception:
            args_sorted = repr(args)
        return f"{name}|{args_sorted}"

    try:
        for _ in range(budget):
            resp = await _stream_llm(model, msgs, emit_tokens=False)
            calls = getattr(resp, "tool_calls", None) or []
            text = str(getattr(resp, "content", "") or "").strip()
            if text:
                last_text = text
            if not calls:
                break  # 无工具调用 → 步骤完成

            for tc in calls:
                name = tc.get("name", "")
                targs = tc.get("args") or {}
                # 卡片 id 唯一:优先模型 tool_call id;缺失时生成随机 id,杜绝
                # 沿用 tc.name 兜底导致同名工具多次调用 id 碰撞(前端第二张卡永远
                # 转圈、React 同 key)。
                card_id = tc.get("id", "") or f"{name}-{uuid.uuid4().hex[:6]}"
                # ToolMessage.tool_call_id 维持原语义(仅作为字符串标签)。
                tc_id = tc.get("id", "") or name
                tool = tools.get(name)
                begin = time.monotonic()

                # 重复调用去重:命中缓存直接复用(不重新 invoke)。错误信封不
                # 缓存,给 LLM 留下 error_type 恢复(重试/换工具)的路径。
                ckey = _tool_key(name, targs)
                if ckey in result_cache:
                    content = result_cache[ckey]
                    status = "success"
                    if name not in SUBAGENT_NAMES:
                        emit({"type": "tool_start", "id": card_id, "name": name,
                              "args": targs})
                        emit({
                            "type": "tool_end", "id": card_id, "name": name,
                            "status": status,
                            "result": f"[重复调用,复用上次结果]\n{str(content)[:4000]}",
                            "execution_time": round(time.monotonic() - begin, 2),
                        })
                    msgs.append(ToolMessage(
                        content=truncate_tool_result(
                            f"[重复调用,复用上次结果]\n{content}", 8000),
                        tool_call_id=tc_id, name=name,
                    ))
                    continue

                if tool is None:
                    content = (f'{{"ok": false, "error_type": "unknown", '
                               f'"error": "unknown tool: {name}"}}')
                    status = "error"
                else:
                    if name not in SUBAGENT_NAMES:
                        emit({"type": "tool_start", "id": card_id, "name": name,
                              "args": targs})
                    try:
                        content = str(await tool.ainvoke(targs, config=config))
                    except Exception as exc:
                        content = (f'{{"ok": false, "error_type": "tool_crash", '
                                   f'"error": "{type(exc).__name__}: {exc}"}}')
                    parsed = parse_tool_result(content)
                    status = "error" if (parsed.is_envelope and not parsed.ok) else "success"
                    if parsed.is_envelope and not parsed.ok:
                        error = parsed.error or ""
                        if (parsed.error_type or "") == "backend_down":
                            consecutive_down += 1
                        else:
                            consecutive_down = 0
                    else:
                        consecutive_down = 0
                        result_cache[ckey] = content
                if name not in SUBAGENT_NAMES:
                    emit({
                        "type": "tool_end", "id": card_id, "name": name,
                        "status": status, "result": str(content)[:4000],
                        "execution_time": round(time.monotonic() - begin, 2),
                    })
                msgs.append(ToolMessage(
                    content=truncate_tool_result(content, 8000),
                    tool_call_id=tc_id, name=name,
                ))

            if consecutive_down >= 2:
                if not last_text:
                    last_text = ("本地知识库后端不可达，已停止工具调用，"
                                 "本步骤无法完成真实检索。")
                break
    except Exception as exc:
        log_event("step_agent_llm_failed", node="executor", level="warning",
                  step_id=step_id, error=f"{type(exc).__name__}: {exc}")
        error = f"{type(exc).__name__}: {exc}"

    ok = bool(last_text.strip())
    _ps("done" if ok else "failed")
    return {
        "step_id": step_id,
        "ok": ok,
        "output": last_text if ok else "",
        "error": error if not ok else "",
    }


def _statused_plan(plan: list[dict], results: dict) -> list[dict]:
    """回填每步 status。先全部 pending，再按 subagent_results 覆盖：
    ok+skipped → skipped / ok → done / !ok → failed；没出现在 results 的
    （cycle/悬空依赖）保持 pending。
    """
    out: list[dict] = []
    for s in plan:
        step = dict(s)
        r = results.get(s.get("id"))
        if r is None:
            step["status"] = "pending"
        elif r.get("ok"):
            step["status"] = "skipped" if r.get("skipped") else "done"
        else:
            step["status"] = "failed"
        out.append(step)
    return out


def _is_check_step(step: dict, by_id: dict) -> bool:
    """True if the step is a direct check_paper tool step (PLAN contract: tool=check_paper)."""
    return (
        step.get("target") == "tool"
        and (step.get("args") or {}).get("tool") == "check_paper"
    )


def _retry_args_from_error(step: dict, out: dict) -> dict | None:
    """(P4) 按 react 错误分类做确定性重试决策。返回修正后的 args 或 None(不重试)。

    - transient            → 原参数重试一次
    - param_error + 候选   → 用错误信封的 available_papers / available_sections
                             修正参数重试一次（同一步骤内，不级联后续依赖步骤）
    - not_found / backend_down / permission_denied / 未知 → 不重试，标注原因
      （synthesize 已有失败渲染）
    """
    from .nodes import _classify_tool_error

    payload = out.get("output") or out.get("error") or ""
    info = _classify_tool_error(payload)
    if not info:
        return None
    if info["type"] == "transient":
        return dict(step.get("args") or {})

    if info["type"] == "param_error":
        args = dict(step.get("args") or {})
        changed = False
        papers = info.get("available_papers") or []
        if papers and args.get("paper_name"):
            req = canonicalize(str(args["paper_name"]))
            exact = next((p for p in papers if canonicalize(str(p)) == req), None)
            if exact:
                args["paper_name"] = exact
                changed = True
            elif len(papers) == 1:
                args["paper_name"] = papers[0]
                changed = True
        sections = info.get("available_sections") or []
        if sections and args.get("section"):
            reql = str(args["section"]).lower()
            hit = next(
                (s for s in sections if reql in s.lower() or s.lower() in reql), None
            )
            if hit:
                args["section"] = hit
                changed = True
            elif sections:
                args["section"] = sections[0]
                changed = True
        return args if changed else None

    # not_found / backend_down / permission_denied / unknown → 不重试
    return None


def _ingest_guard(step: dict, results: dict, by_id: dict) -> str | None:
    """下载/入库步骤的守卫：同一 plan 中 check_paper 已判定本地状态时跳过。

    - indexed                → 已在库可检索，跳过下载/入库（action 任意）。
    - downloaded_not_indexed → PDF 已在本地：action=ingest 是正确路径（放行）；
                               download / download_and_ingest 跳过（不该再下载）。
    - absent                 → 放行（本来就该走 arXiv + download）。

    论文身份按 canonicalize 双向包含匹配（与 check_paper 的 match_local_state 一致）。
    返回 None 表示放行；返回字符串表示跳过理由。
    """
    if step.get("target") != "ingest":
        return None
    args = step.get("args") or {}
    action = str(args.get("action", ""))
    paper = str(args.get("paper_name", "") or "").strip()
    if not paper or action not in ("ingest", "download", "download_and_ingest"):
        return None
    pk = canonicalize(paper)
    if not pk:
        return None

    from .tool_contract import parse_tool_result

    for sid, r in results.items():
        st = by_id.get(sid)
        if not st or not _is_check_step(st, by_id):
            continue
        parsed = parse_tool_result(r.get("output") or "")
        if not parsed.is_envelope or not parsed.ok:
            continue
        inner = parsed.data or {}
        if not isinstance(inner, dict):
            continue
        term = str(inner.get("term", ""))
        local_state = str(inner.get("state", ""))
        tk = canonicalize(term)
        if not tk:
            continue
        hit = (
            tk == pk
            or (len(tk) >= 4 and tk in pk)
            or (len(pk) >= 4 and pk in tk)
        )
        if not hit:
            continue
        if local_state == "indexed":
            return (f"[guard] check_paper({term}) 返回 indexed —— 论文已在库中，"
                    f"跳过步骤「{step.get('description', action)}」。")
        if local_state == "downloaded_not_indexed" and action != "ingest":
            return (f"[guard] check_paper({term}) 返回 downloaded_not_indexed —— PDF 已在本地，"
                    f"跳过下载；如需入库直接用 action=ingest 处理本地 PDF。")
    return None


# ---- verify — 计划完成验证（报告式，不自动修复）----

_VERIFY_SYSTEM = """你是科研问答流程的验收员。判断「已执行的步骤产出」能否回答用户的原始问题。

Output ONLY a JSON object, no preamble, no markdown fences:
{"status":"satisfied|partial|failed|no_evidence","reason":"一句话","missing":["缺口1","缺口2"]}

- satisfied: 现有产出已充分回答用户问题
- partial: 能部分回答，仍有明确缺口
- failed: 关键产出缺失/根本性错误，不足以回答
- no_evidence: 没有任何可用执行产出
missing 最多 3 条，reason 不超过 40 字。"""

_VERIFY_EVIDENCE_MAX = int(os.environ.get("AGENT_VERIFY_EVIDENCE_MAX", "8000"))


def _verify_summary(plan: list[dict], results: list[dict]) -> dict:
    """确定性统计（零成本）：done/failed/pending 计数 + outstanding 列表。

    failed = !ok；pending = 未出现在 results（cycle/悬空依赖未执行）；
    skipped（ok + skipped）计入 done 但不进 outstanding（守卫生效不是失败）。
    """
    by_id = {r.get("step_id"): r for r in results}
    failed = [r for r in results if not r.get("ok")]
    pending = [s for s in plan if s.get("id") not in by_id]
    desc = {s.get("id"): s.get("description", "") for s in plan}
    outstanding: list[dict] = []
    for r in failed:
        outstanding.append({
            "id": r.get("step_id", ""),
            "description": desc.get(r.get("step_id", ""), ""),
            "reason": (r.get("error") or "执行失败")[:200],
        })
    for s in pending:
        outstanding.append({
            "id": s.get("id", ""),
            "description": s.get("description", ""),
            "reason": "步骤未执行（依赖不满足或计划空洞）",
        })
    return {
        "done": len([r for r in results if r.get("ok")]),
        "total": len(plan),
        "outstanding": outstanding,
    }


async def _verify_goal(model, query: str, plan: list[dict],
                       results: list[dict]) -> str | None:
    """LLM 目标满足度检查 → status；LLM 失败/不可解析 → None（调用方降级）。"""
    desc = {s.get("id"): s.get("description", "") for s in plan}
    parts: list[str] = []
    for r in results:
        label = desc.get(r.get("step_id"), "")
        if r.get("ok"):
            out = (r.get("output") or "").strip()
            if out and not r.get("skipped"):
                parts.append(f"## {label}\n{out[:1500]}")
        else:
            parts.append(f"## {label}\n(步骤失败: {(r.get('error') or '')[:200]})")
    evidence = "\n\n".join(parts) or "(无步骤产出)"
    from .tool_contract import truncate_tool_result
    evidence = truncate_tool_result(evidence, _VERIFY_EVIDENCE_MAX)

    steps = "\n".join(f"- {s.get('id')}: {s.get('description', '')}" for s in plan)
    prompt = (
        f"## User Question\n{query}\n\n"
        f"## Plan Steps\n{steps}\n\n"
        f"## Step Outputs\n{evidence}"
    )
    response = await model.ainvoke([
        SystemMessage(content=_VERIFY_SYSTEM),
        HumanMessage(content=prompt),
    ])
    text = getattr(response, "content", "")
    raw = _extract_json_text(text) if isinstance(text, str) else None
    if not raw:
        return None
    try:
        status = str(json.loads(raw).get("status", ""))
        if status in ("satisfied", "partial", "failed", "no_evidence"):
            return status
    except Exception:
        pass
    return None


@timed("verify")
async def verify_node(state: AgentState, config) -> dict:
    """计划完成验证（报告式）。确定性统计 + LLM 目标满足度检查。

    状态合成（不自动修复，只报告）：
      - 空 plan → no_evidence
      - 有 failed/pending 步骤 → partial（LLM 判 failed 才升格 failed；LLM 说
        satisfied 也钳制回 partial，绝不掩盖失败步骤）
      - 全部完成 → 以 LLM 结论为准（satisfied/partial）
    creation 域跳过 LLM（写作终态由 doc_progress 报告，verify 只做统计）。
    """
    plan = state.get("plan", [])
    results = state.get("subagent_results", [])
    summary = _verify_summary(plan, results)
    has_fail = bool(summary["outstanding"])

    status = "no_evidence"
    if len(plan) > 0:
        status = "partial" if has_fail else "satisfied"

    if state.get("domain") != "creation" and len(plan) > 0:
        from .nodes import _get_model  # lazy: avoid import cycle
        try:
            model = _get_model(config)
            llm_status = await _verify_goal(
                model, _last_user_text(state) or "(none)", plan, results,
            )
        except Exception as exc:
            log_event("verify_llm_failed", node="verify", level="warning",
                      error=f"{type(exc).__name__}: {exc}")
            llm_status = None
        if llm_status:
            if has_fail:
                status = "failed" if llm_status == "failed" else "partial"
            else:
                status = llm_status

    emit({
        "type": "plan_verify",
        "status": status,
        "done": summary["done"],
        "total": summary["total"],
        "outstanding": summary["outstanding"],
    })
    return {"verification": {**summary, "status": status}}
