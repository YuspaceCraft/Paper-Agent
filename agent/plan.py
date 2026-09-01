"""
plan.py — Plan-and-Execute (Phase 7).

decide_mode: pure heuristic — "react" (simple query, zero regression) vs
    "plan" (multi-paper / comparison / multi-sub-question).
plan_node: LLM structured output → ordered PlanStep list.
executor_node: no-LLM topological executor; runs independent steps in
    parallel via asyncio.gather, dependent steps in order.

target semantics (resolved against get_cached_tools()):
  - "tool" → a DIRECT parent tool (search_papers / fetch_content / list_dir /
    read_file / write_file / check_paper / check_task_status). All LOCAL work —
    filesystem & library reads live here, never in a subagent.
  - a subagent name (arxiv / ingest) → call the matching subagent tool (Phase 8).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
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


def decide_mode(state: dict) -> str:
    """Pick execution mode. "react" keeps the existing single-ReAct path.

    "plan" only on grounded signals:
      - domain is creation/coding (领域 agent 的写作/实验工作流必须走 plan 通道)
      - explicit comparison / multi-sub-question phrasing
      - ≥2 distinct papers CONFIRMED by the resolution layer (focus_papers and
        entity terms that actually matched a library paper — see resolve_node).
    No blacklist heuristics: an entity only counts when resolution matched it
    to a known paper, so 单论文单动作指令（入库/下载）never misroutes here.
    """
    if state.get("domain") in ("creation", "coding"):
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


# ---- structured plan output ----

class PlanStep(BaseModel):
    id: str
    description: str = Field(description="What this step answers (task-oriented)")
    target: Literal["tool", "arxiv", "ingest", "creator", "coder"] = Field(
        description='"tool" (a direct parent tool: search_papers / fetch_content / '
        'list_dir / read_file / write_file / check_paper / check_task_status) | '
        'arxiv (EXTERNAL arXiv only) | ingest (download/入库 write commands) | '
        'creator (创作域写作 subagent) | coder (编码域实验/委托 subagent，v10 Phase C)'
    )
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class PlanResult(BaseModel):
    steps: list[PlanStep]


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
    # a plan with only dangling/invalid steps is no plan — degrade cleanly
    valid = [s for s in steps if s.get("id") and s.get("target")]
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

    emit({
        "type": "plan",
        "steps": [
            {k: s.get(k) for k in ("id", "description", "target", "depends_on")}
            for s in plan
        ],
    })

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

    emit({
        "type": "plan",
        "steps": [
            {k: s.get(k) for k in ("id", "description", "target", "depends_on")}
            for s in plan
        ],
    })
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

    emit({
        "type": "plan",
        "steps": [
            {k: s.get(k) for k in ("id", "description", "target", "depends_on")}
            for s in plan
        ],
    })

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
    renders each plan step as a collapsible card.
    """
    step_id = step.get("id", "")
    target = step.get("target", "tool")
    args = dict(step.get("args") or {})
    start = time.monotonic()
    is_subagent = False

    def _end(name: str, status: str, result: str) -> None:
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

        if is_subagent:
            # subagent 的 as_tool._call 自己 emit 边界 + 叶子工具事件；
            # 这里只 await 拿结果，避免重复卡片。config 透传: subgraph 作为子
            # run 挂到父 trace(LangSmith 才能看到 creation 内部调用)。
            out = await tool.ainvoke(call_args, config=config)
            if target == "creator":
                ok, output, err = await _verify_creator_step(step, out)
                return {
                    "step_id": step_id, "ok": ok, "output": output, "error": err,
                }
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
        if not is_subagent:
            _end(target, "error", f"{type(exc).__name__}: {exc}")
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
        check_steps = [s for s in ready if _is_check_step(s, by_id)]
        if check_steps:
            outs = await asyncio.gather(*[_run_step(s, state, config) for s in check_steps])
            for s, out in zip(check_steps, outs):
                results[s["id"]] = out
                done.add(s["id"])

        # 2) 其余步骤：被 check_paper 结论守卫命中的下载/入库步骤 → 跳过（不调用）；
        #    其余照常执行。
        run: list[dict] = []
        for s in ready:
            if s["id"] in done:
                continue
            note = _ingest_guard(s, results, by_id)
            if note is not None:
                results[s["id"]] = {
                    "step_id": s["id"], "ok": True,
                    "output": note, "error": "", "skipped": True,
                }
                done.add(s["id"])
            else:
                run.append(s)
        outs = await asyncio.gather(*[_run_step(s, state, config) for s in run])
        for i, s in enumerate(run):
            out = outs[i]
            # (P4) 直接工具步骤失败 → 确定性重试一次: transient 原参数; param_error
            # 且错误带 available_papers/sections → 修正参数。对比 react 模式的
            # LLM 错误恢复，这里不用 LLM,只做确定性的参数修正/重试(见
            # _retry_args_from_error)。
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
        remaining = [s for s in remaining if s.get("id") not in done]

    return {
        "subagent_results": list(results.values()),
        "plan_progress": len(done),
    }


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
