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
      - explicit comparison / multi-sub-question phrasing
      - ≥2 distinct papers CONFIRMED by the resolution layer (focus_papers and
        entity terms that actually matched a library paper — see resolve_node).
    No blacklist heuristics: an entity only counts when resolution matched it
    to a known paper, so 单论文单动作指令（入库/下载）never misroutes here.
    """
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
    target: Literal["tool", "arxiv", "ingest"] = Field(
        description='"tool" (a direct parent tool: search_papers / fetch_content / '
        'list_dir / read_file / write_file / check_paper / check_task_status) | '
        'arxiv (EXTERNAL arXiv only) | ingest (download/入库 write commands)'
    )
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class PlanResult(BaseModel):
    steps: list[PlanStep]


# ---- plan_node ----

async def _ask_for_plan(model, prompt: str, attempts: int = 2) -> list[dict]:
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
    """
    msgs = [SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=prompt)]
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
    """LLM → structured ordered steps. Zero-state: full context injected.

    Never raises: `_ask_for_plan` degrades to `_fallback_plan` (deterministic
    library steps) if the LLM produces no steps.
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
            # 这里只 await 拿结果，避免重复卡片。
            out = await tool.ainvoke(call_args)
            return {
                "step_id": step_id, "ok": True, "output": str(out),
            }

        emit({"type": "tool_start", "id": step_id, "name": name, "args": call_args})
        out = await tool.ainvoke(call_args)
        _end(name, "success", str(out))
        return {
            "step_id": step_id, "ok": True, "output": str(out),
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
        for s, out in zip(run, outs):
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

    for sid, r in results.items():
        st = by_id.get(sid)
        if not st or not _is_check_step(st, by_id):
            continue
        try:
            payload = json.loads((r.get("output") or ""))
        except (ValueError, TypeError):
            continue
        inner = (payload or {}).get("data") if isinstance(payload, dict) else {}
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
