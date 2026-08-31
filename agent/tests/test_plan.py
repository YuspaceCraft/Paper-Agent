"""Phase 7 self-check — plan heuristic + executor topological order.

Run: python agent/tests/test_plan.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import agent.plan as plan
from agent.plan import decide_mode, executor_node, _subagent_task


class _Msg:
    type = "human"

    def __init__(self, content):
        self.content = content


def _state(query="", entities=None, focus=None, plan_steps=None, results=None,
           resolved=None):
    return {
        "messages": [_Msg(query)],
        "entities": entities or [],
        "focus_papers": focus or [],
        "plan": plan_steps or [],
        "subagent_results": results or [],
        "resolved": resolved or {"papers": [], "section": None},
    }


def _resolved(*pairs):
    """resolved.papers from (match, level) tuples."""
    return {"papers": [{"match": m, "level": lvl} for m, lvl in pairs],
            "section": None}


def test_decide_mode_comparison():
    # 对比关键词 → plan（即使只在 resolve 到一篇）
    assert decide_mode(_state(query="对比 RMNet 和 SRN 的 loss 设计")) == "plan"
    # 单论文无对比关键词 → react
    assert decide_mode(_state(query="RMNet 的 loss 是什么")) == "react"


def test_decide_mode_multi_target():
    # 多目标只认 resolve 确证过的论文（不是 entities 词袋计数）
    assert decide_mode(_state(entities=["RMNet"], focus=["SRN"],
                              resolved=_resolved(("RMNet", "EXACT"),
                                                 ("SRN", "EXACT")))) == "plan"
    # entities 里有名字但 resolve 没命中 → 不算目标 → react
    assert decide_mode(_state(entities=["RMNet"], focus=["SRN"])) == "react"
    assert decide_mode(_state(entities=["RMNet"])) == "react"
    # LOW 置信匹配不构成 multi-target
    assert decide_mode(_state(entities=["cv", "SRN"],
                              resolved=_resolved(("cv", "LOW"),
                                                 ("SRN", "HIGH")))) == "react"


def test_decide_mode_single_action_command():
    # 单论文单动作指令必须走 react：即使 entities/focus 各带一个候选，只要
    # resolve 没确证 ≥2 篇论文（或论文已在库、无需下载），就不进 plan。
    assert decide_mode(_state(entities=["该论文", "向量数据库"],
                              focus=["RMNet"])) == "react"
    assert decide_mode(_state(entities=["RMNet", "下载"],
                              focus=["RMNet"])) == "react"
    # 真正两篇确证论文（含 entities 来源）→ plan
    assert decide_mode(_state(entities=["RMNet", "SRN"], focus=["SRN"],
                              resolved=_resolved(("RMNet", "EXACT"),
                                                 ("SRN", "EXACT")))) == "plan"


def test_parse_plan_json():
    # 模型按 PLAN_SYSTEM 契约输出裸 JSON 文本（无 code fence / 有前后缀）
    # → 必须能被解析为 steps（回归：function_calling 空结果根因）
    raw = ('当然。以下是计划：\n```json\n'
           '{"steps": [{"id": "s1", "description": "Read RMNet", '
           '"target": "tool", "args": {"tool": "fetch_content", "paper_name": "RMNet"},'
           ' "depends_on": []}]}\n'
           '```\n希望有帮助。')
    assert plan._extract_json_text(raw) is not None
    steps = plan._parse_steps(plan._extract_json_text(raw))
    assert steps and steps[0]["target"] == "tool"
    assert steps[0]["args"] == {"tool": "fetch_content", "paper_name": "RMNet"}
    # 已下线的 target（"library" 不在 Literal 枚举）→ 计划整单拒绝 → None（降级，不抛）
    assert plan._parse_steps(
        '{"steps": [{"id": "s1", "description": "x", "target": "library",'
        ' "args": {}, "depends_on": []}]}'
    ) is None
    # 非法 JSON / 无 steps → None（降级，不抛）
    assert plan._parse_steps("no json here") is None
    assert plan._parse_steps("[]") is None
    assert plan._extract_json_text("no json") is None


def test_subagent_task_folding():
    # plan steps carry natural arg names (query/paper_id/...), but subagent
    # tools expose a single "task" field. _run_step must re-fold them as a
    # `key: value` command block (the same contract the react loop uses) so
    # ingest fields (action/arxiv_id/paper_name/pdf_path) survive verbatim.
    assert _subagent_task("find papers", {"query": "cv"}) == "find papers\nquery: cv"
    assert _subagent_task("", {"query": "cv"}) == "query: cv"
    assert _subagent_task("just read", {}) == "just read"
    assert _subagent_task("", {}) == ""
    assert _subagent_task("ingest RMNet", {"action": "download_and_ingest",
                                           "arxiv_id": "2301.07093",
                                           "paper_name": "RMNet"}) == (
        "ingest RMNet\naction: download_and_ingest\narxiv_id: 2301.07093\npaper_name: RMNet"
    )


def test_executor_topological():
    call_order = []

    async def fake_run(step, state, config):
        call_order.append(step["id"])
        return {"step_id": step["id"], "ok": True, "output": step["id"]}

    orig = plan._run_step
    plan._run_step = fake_run
    try:
        steps = [
            {"id": "a", "description": "", "target": "tool", "args": {}, "depends_on": []},
            {"id": "b", "description": "", "target": "tool", "args": {}, "depends_on": []},
            {"id": "c", "description": "", "target": "tool", "args": {}, "depends_on": ["a", "b"]},
        ]
        out = asyncio.run(executor_node(_state(plan_steps=steps), {}))
    finally:
        plan._run_step = orig

    ids = [r["step_id"] for r in out["subagent_results"]]
    assert set(ids) == {"a", "b", "c"}, "all steps must be recorded"
    # dependency: c runs only after a and b (both of which may be parallel)
    assert call_order.index("c") > call_order.index("a")
    assert call_order.index("c") > call_order.index("b")


if __name__ == "__main__":
    test_decide_mode_comparison()
    test_decide_mode_multi_target()
    test_decide_mode_single_action_command()
    test_parse_plan_json()
    test_subagent_task_folding()
    test_executor_topological()
    print("Phase 7 plan self-check OK")
