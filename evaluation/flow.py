"""
flow.py — 单条 turn 的分阶段透明视图（评估端第一层）。

把一条完整链路（提问 → 意图理解 → 上下文/记忆 → 计划 → 工具执行 → 回答）的
trace 事件按「节点归属」结构化聚合成 stages，每阶段给出耗时与 token 消耗；
阶段之上是 turn 级汇总指标（总时长/总 token/估算占比/成本/工具成败/任务成败）。

节点归属规则（只依赖事件顺序，不侵入运行时代码）：
- 计时：node_end 事件自带 node 名 + duration_ms → 按 node 名归并到阶段
- token：llm_call 事件若带 node 名直接用，否则用「node_start/node_end 栈」的
  当前栈顶（流式归纳到它运行的节点）
- 事件：按 node → 阶段映射分组

阶段名 → 节点：
  intent ← understand · memory ← memory · context ← resolve/context ·
  plan ← plan/verify/executor · tools ← 其余（agent 推理、tool 执行、空 node）·
  answer ← synthesize/chat/clarify/task/subagent_synthesize
"""

from __future__ import annotations

from typing import Any

from .metrics.task import task_success
from .metrics.tools import call_error

STAGE_ORDER = ["question", "intent", "context", "memory", "plan", "tools", "answer"]
STAGE_LABELS = {
    "question": "用户问题",
    "intent": "意图理解",
    "context": "上下文解析",
    "memory": "上下文/记忆",
    "plan": "计划",
    "tools": "工具执行",
    "answer": "问题回答",
}


def _stage_of(node: str) -> str:
    if node in ("understand",):
        return "intent"
    if node in ("memory",):
        return "memory"
    if node in ("resolve", "context"):
        return "context"
    if node in ("plan", "verify", "executor"):
        return "plan"
    if node in ("synthesize", "chat", "clarify", "task", "subagent_synthesize"):
        return "answer"
    return "tools"


def _ts_epoch(ts) -> float | None:
    """事件 ts（ISO 串）→ epoch 秒；解析不出来返回 None。"""
    if not ts:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _total_duration_ms(evs: list[dict], turn_end_its: list[float],
                       wall_ms: float | None) -> tuple[float | None, str]:
    """整轮完整用时（ms）+ 来源标注。

    只做「启停计时」或「时间戳相减」，不做步骤耗时累加——累加出来的数会随重试/
    并发/嵌套调用虚高，与用户看到的墙钟时间对不上。
    """
    if wall_ms is not None:
        return round(float(wall_ms), 1), "wall_clock"
    if turn_end_its:
        return round(float(max(turn_end_its)), 1), "turn_end"
    times = [t for t in (_ts_epoch(e.get("ts")) for e in evs) if t is not None]
    if len(times) >= 2:
        return round((max(times) - min(times)) * 1000, 1), "event_span"
    return None, "unknown"


def build_flow(events: list[dict], result: dict | None = None,
               query: str | None = None, wall_ms: float | None = None) -> dict:
    """把单条 turn 的事件链聚合成阶段视图。

    result: agent.graph.run 的返回态（可选，补充 context/context_snapshot）。
    query: 显式传入的用户问题（事件里 turn_end 兜底；优先用本参数）。
    wall_ms: 调用方单独计时得到的整轮墙钟耗时（ms）——「任务开始计时、任务结束
      停止计时」的权威值，优先于任何从事件里推导的时长。

    完整用时来源优先级（绝不把链路里每一步的 duration_ms 相加当总时长）：
      1. wall_ms（调用方启停计时）
      2. turn_end 的 duration_ms（graph 内部同样是启停计时）
      3. 首末事件的 ts 之差（墙钟跨度兜底）
    """
    evs = sorted(events, key=lambda e: (e.get("seq", 0)))
    seq_of = {e.get("seq"): e for e in evs}

    # 1) 目录节点：node_start/node_end/node_error → 用栈给中间事件定 node
    stack: list[str] = []
    eff_node: dict[int, str] = {}  # seq -> 归属 node
    for e in evs:
        t, node = e.get("event_type"), e.get("node") or ""
        if t == "node_start":
            stack.append(node)
        elif t in ("node_end", "node_error"):
            if stack and stack[-1] == node:
                stack.pop()
        else:
            eff_node[e.get("seq")] = node or (stack[-1] if stack else "")

    # 2) 各阶段聚合器
    stages: dict[str, dict] = {
        k: {"key": k, "label": STAGE_LABELS[k], "duration_ms": 0,
            "tokens": {"prompt": 0, "completion": 0,
                       "total": 0, "estimated": 0}, "items": []}
        for k in STAGE_ORDER
    }
    node_dur: dict[str, float] = {}

    def _stage_of_ev(e: dict) -> str:
        t = e.get("event_type")
        if t in ("node_start",):
            return _stage_of(e.get("node") or "")
        node = eff_node.get(e.get("seq"), "")
        if t in ("intent",):
            return "intent"
        if t in ("llm_call",):
            n = e.get("node") or node
            if n in ("synthesize", "chat", "clarify", "task", "subagent_synthesize"):
                return "answer"
            if n in ("", "agent"):
                # react 环里的推理调用；其后已无工具调用 → 视为「回答」成本。
                later_tool = any(
                    e2.get("event_type") == "tool_call" and e2.get("seq", 0) > e.get("seq", 0)
                    for e2 in evs)
                return "tools" if later_tool else "answer"
            return _stage_of(n)
        if t in ("tool_call", "retrieved_context"):
            return "tools"
        if t in ("plan", "plan_step", "plan_verify"):
            return "plan"
        if t in ("final_answer", "turn_end"):
            return "answer"
        return _stage_of(node)

    for e in evs:
        t, node = e.get("event_type"), e.get("node") or ""
        payload = e.get("payload") or {}

        # 节点计时（node_end 的 duration 归到该 node 名的阶段）
        if t in ("node_end", "node_error") and e.get("duration_ms") is not None:
            node_dur[node] = node_dur.get(node, 0.0) + (e.get("duration_ms") or 0)
            continue

        st = _stage_of_ev(e)
        stage = stages[st]

        if t == "turn_start":
            stage["items"].append({"type": "query", "query": payload.get("query", "")})
        elif t == "intent":
            stage["items"].append({
                "type": "intent",
                "intent": e.get("intent") or payload.get("intent") or "",
                "confidence": payload.get("confidence"),
                "entities": payload.get("entities") or [],
                "focus_papers": payload.get("focus_papers") or [],
                "needs_planning": payload.get("needs_planning"),
                "domain": payload.get("domain") or "",
            })
        elif t == "llm_call":
            tok = payload.get("tokens") or {}
            stage["tokens"]["prompt"] += int(tok.get("prompt_tokens") or 0)
            stage["tokens"]["completion"] += int(tok.get("completion_tokens") or 0)
            stage["tokens"]["total"] += int(tok.get("total_tokens") or 0)
            if tok.get("estimated"):
                stage["tokens"]["estimated"] += 1
            stage["items"].append({
                "type": "llm", "node": e.get("node") or eff_node.get(e.get("seq"), ""),
                "model": e.get("model") or "", "mode": payload.get("mode", ""),
                "tokens": tok, "duration_ms": e.get("duration_ms"),
                "error": payload.get("error") or e.get("error"),
            })
        elif t == "tool_call":
            parsed = payload.get("parsed") or {}
            operation = payload.get("operation") or {}
            outcome = (
                e.get("outcome")
                or operation.get("outcome")
                or parsed.get("outcome")
            )
            stage["items"].append({
                "type": "tool", "tool": e.get("tool") or "",
                "args": payload.get("args") or "",
                "result": operation or payload.get("result_summary") or "",
                "outcome": outcome,
                "duration_ms": e.get("duration_ms"),
                "error": e.get("error") or payload.get("error"),
                "parsed": parsed,
                "operation": operation,
            })
        elif t == "retrieved_context":
            stage["items"].append({
                "type": "retrieval", "tool": e.get("tool") or "",
                "query": payload.get("query") or "",
                "chunk_ids": payload.get("chunk_ids") or [],
                "snapshot": payload.get("snapshot") or "",
                "outcome": e.get("outcome") or "",
            })
        elif t == "plan":
            stage["items"].append({
                "type": "plan", "steps": payload.get("steps") or [],
                "mode": payload.get("mode", ""),
            })
        elif t in ("plan_step", "plan_verify"):
            stage["items"].append({
                "type": t, "status": payload.get("status") or "",
                "detail": str(payload.get("detail") or payload.get("verification") or "")[:2000],
                "step_id": payload.get("step_id"),
            })
        elif t == "final_answer":
            stage["items"].append({
                "type": "answer", "answer": payload.get("answer") or "",
                "mode": payload.get("mode", ""),
                "verification": payload.get("verification"),
            })
        elif t == "turn_end":
            stage["items"].append({
                "type": "turn_end", "status": payload.get("status", "ok"),
                "duration_ms": e.get("duration_ms"),
            })

    # 阶段耗时：node 名归并 + 直接带 stage 的事件
    for node, ms in node_dur.items():
        st = _stage_of(node)
        stages[st]["duration_ms"] += ms

    # 空阶段不展示（question 阶段并入 query 展示）
    ordered = [s for k in STAGE_ORDER if (s := stages[k]).get("items")]

    # 3) turn 级汇总
    llm_items = [it for s in ordered for it in s["items"] if it.get("type") == "llm"]
    tool_items = [it for s in ordered for it in s["items"] if it.get("type") == "tool"]
    raw_tool_events = [e for e in evs if e.get("event_type") == "tool_call"]
    total_tok = {"prompt": 0, "completion": 0, "total": 0}
    estimated = 0
    for it in llm_items:
        tk = it.get("tokens") or {}
        total_tok["prompt"] += int(tk.get("prompt_tokens") or 0)
        total_tok["completion"] += int(tk.get("completion_tokens") or 0)
        total_tok["total"] += int(tk.get("total_tokens") or 0)
        if tk.get("estimated"):
            estimated += 1

    intent_it = next((it for s in ordered for it in s["items"]
                      if it.get("type") == "intent"), None)
    answer_it = next((it for s in ordered for it in s["items"]
                      if it.get("type") == "answer"), None)
    turn_end_its = [it.get("duration_ms") for s in ordered
                    for it in s["items"] if it.get("type") == "turn_end"
                    and it.get("duration_ms") is not None]
    config_event = next(
        (e for e in evs if e.get("event_type") == "configuration_snapshot"),
        None,
    )
    config_payload = (config_event or {}).get("payload") or {}

    total_ms, duration_source = _total_duration_ms(evs, turn_end_its, wall_ms)
    task = task_success(evs)
    metrics = {
        "total_duration_ms": total_ms,
        "duration_source": duration_source,
        "llm_calls": len(llm_items),
        "tokens": total_tok,
        "estimated_calls": estimated,
        "tool_calls": len(tool_items),
        "tool_failures": sum(1 for e in raw_tool_events if call_error(e)),
        "task_success": bool(task.get("success")),
        "task_status": task.get("status", ""),
        "intent": (intent_it or {}).get("intent", ""),
        "mode": (answer_it or {}).get("mode", ""),
        "turn_status": next(
            (it.get("status", "") for s in ordered
             for it in s["items"] if it.get("type") == "turn_end"), ""),
        "config_revision": config_payload.get("config_revision", ""),
        "config_hash": config_payload.get("config_hash", ""),
        "prompt_versions": config_payload.get("prompt_versions", {}),
    }

    return {
        "query": (query or "").strip() or next(
            (it.get("query", "") for s in ordered
             for it in s["items"] if it.get("type") == "query"), ""),
        "stages": ordered,
        "metrics": metrics,
        "context": (result or {}).get("context") or {
            "active_doc_id": None, "active_project": None,
            "study_topic": None, "recent_experiments": []},
        "context_snapshot": (result or {}).get("context_snapshot", ""),
        # Full ordered trace makes the single-run UI able to render node-level
        # progress instead of only broad stage buckets.  Payloads are already
        # clipped by TraceStore and contain no environment credentials.
        "events": evs,
    }
