"""Phase 7 self-check — plan heuristic + executor topological order.

Run: python agent/tests/test_plan.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import agent.plan as plan
from agent.plan import decide_mode, executor_node, verify_node, _subagent_task


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


def test_parse_plan_targetless():
    # v14: 无 target 的结果步骤 → pydantic 默认 "auto"（LLM 逐步执行）
    steps = plan._parse_steps(
        '{"steps": [{"id": "s1", "description": "检索并精读相关论文", '
        '"depends_on": []}]}'
    )
    assert steps and steps[0]["target"] == "auto"
    assert steps[0]["description"] == "检索并精读相关论文"
    # 缺描述 → 整单拒绝（不降级为空洞计划）
    assert plan._parse_steps(
        '{"steps": [{"id": "s1", "depends_on": []}]}'
    ) is None


class _FakeStepTool:
    name = "fetch_content"

    async def ainvoke(self, args, config=None):
        return '{"ok": true, "data": {"content": "RMNet 的方法部分…"}}'


def test_step_agent_loop():
    """LLM 逐步执行：一步内多次工具调用，无工具调用的文本即步骤答案。"""
    import agent.plan as plan_mod
    import agent.tools as tools_mod
    import agent.nodes as nodes_mod
    from langchain_core.messages import AIMessage

    llm_rounds: list[int] = []

    async def fake_llm(model, msgs, *, emit_tokens=True):
        # 记录该轮对话里已有的 tool 消息数（验证 ToolMessage 确实回填）
        llm_rounds.append(
            sum(1 for m in msgs if getattr(m, "type", "") == "tool")
        )
        if len(llm_rounds) == 1:
            return AIMessage(content="", tool_calls=[{
                "id": "c1", "name": "fetch_content",
                "args": {"paper_name": "RMNet", "section": "method"},
            }])
        return AIMessage(content="RMNet 采用 xx 损失函数训练。", tool_calls=[])

    fake_tool = _FakeStepTool()
    orig_llm, orig_tools, orig_model = (
        nodes_mod._stream_llm, tools_mod.get_cached_tools, nodes_mod._get_bound_model,
    )
    try:
        nodes_mod._stream_llm = fake_llm
        tools_mod.get_cached_tools = lambda: [fake_tool]
        nodes_mod._get_bound_model = lambda *a, **k: object()  # fake_llm 不读它
        st = _state(query="RMNet 的 loss 是什么",
                    resolved=_resolved(("RMNet", "EXACT")))
        out = asyncio.run(plan_mod._run_step_agent(
            {"id": "s1", "description": "确定 RMNet 的损失函数", "depends_on": []},
            st, {"configurable": {}}, {"s9": "前序步骤结果"},
        ))
    finally:
        nodes_mod._stream_llm = orig_llm
        tools_mod.get_cached_tools = orig_tools
        nodes_mod._get_bound_model = orig_model

    assert out["step_id"] == "s1"
    assert out["ok"] is True
    assert "损失函数" in out["output"]
    assert len(llm_rounds) == 2          # 工具轮 + 收尾轮
    assert llm_rounds[1] == 1            # 第二轮对话里带回了该工具结果
    assert out["output"] == "RMNet 采用 xx 损失函数训练。"


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


def test_decide_mode_forced():
    # 客户端显式覆盖优先于启发式（requested_mode）
    s = _state(query="RMNet 的 loss 是什么")  # 启发式 → react
    s["requested_mode"] = "plan"
    assert decide_mode(s) == "plan"
    s["requested_mode"] = "react"
    assert decide_mode(s) == "react"

    # 对比类 query（启发式 → plan）也可被强制 react
    s = _state(query="对比 RMNet 和 SRN 的 loss 设计")
    s["requested_mode"] = "react"
    assert decide_mode(s) == "react"

    # 非法覆盖值回退启发式（auto 语义）
    s = _state(query="RMNet 的 loss 是什么")
    s["requested_mode"] = "hack"
    assert decide_mode(s) == "react"
    s = _state(query="对比 RMNet 和 SRN 的 loss 设计")
    s["requested_mode"] = "hack"
    assert decide_mode(s) == "plan"
    # 缺省 auto：完全走启发式，行为不变
    assert decide_mode(_state(query="对比 RMNet 和 SRN 的 loss 设计")) == "plan"


def test_executor_tracks_statuses():
    # executor 回填每步 status + done/total 计数
    async def fake_run(step, state, config):
        step_id = step["id"]
        if step_id == "bad":
            return {"step_id": step_id, "ok": False, "output": "", "error": "boom"}
        return {"step_id": step_id, "ok": True, "output": "ok"}

    orig = plan._run_step
    plan._run_step = fake_run
    try:
        steps = [
            {"id": "ok1", "description": "d1", "target": "tool", "args": {}, "depends_on": []},
            {"id": "ok2", "description": "d2", "target": "tool", "args": {}, "depends_on": []},
            {"id": "never", "description": "d4", "target": "tool", "args": {}, "depends_on": []},
            {"id": "bad", "description": "d3", "target": "tool", "args": {}, "depends_on": ["ok1"]},
        ]
        out = asyncio.run(executor_node(_state(plan_steps=steps), {}))
    finally:
        plan._run_step = orig

    statuses = {s["id"]: s["status"] for s in out["plan"]}
    assert statuses["ok1"] == "done"
    assert statuses["ok2"] == "done"
    assert statuses["never"] == "done"
    assert statuses["bad"] == "failed"
    # plan_done = 已处理（含失败的）步骤数；质量差异由 verify_node 单独报告
    assert out["plan_done"] == 4
    assert out["plan_total"] == 4
    # 全部步骤都进了 results
    assert len(out["subagent_results"]) == 4


def test_statused_plan_marks_skipped():
    # skipped（守卫生效）与 pending（未执行）在状态回填里的区分
    results = {
        "s1": {"step_id": "s1", "ok": True, "output": "x"},
        "s2": {"step_id": "s2", "ok": True, "output": "[guard] ...", "skipped": True},
    }
    statuses = {s["id"]: s["status"] for s in plan._statused_plan(
        [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}], results)}
    assert statuses == {"s1": "done", "s2": "skipped", "s3": "pending"}


def test_verify_deterministic():
    # 用 domain="creation" 跳过 LLM 目标检查，专测确定性统计/状态合成
    s = _state(query="对比 X 和 Y", plan_steps=[
        {"id": "s1", "description": "读 X", "target": "tool", "args": {}, "depends_on": []},
        {"id": "s2", "description": "读 Y", "target": "tool", "args": {}, "depends_on": []},
    ], results=[
        {"step_id": "s1", "ok": True, "output": "X 的内容"},
        {"step_id": "s2", "ok": True, "output": "Y 的内容"},
    ])
    s["domain"] = "creation"
    v = asyncio.run(verify_node(s, {}))["verification"]
    assert v["status"] == "satisfied"
    assert v["done"] == 2 and v["total"] == 2 and v["outstanding"] == []

    # 失败步骤 → partial（报告但不掩盖），reason 带进去
    s2 = _state(query="对比 X 和 Y", plan_steps=[
        {"id": "s1", "description": "读 X", "target": "tool", "args": {}, "depends_on": []},
        {"id": "s2", "description": "读 Y", "target": "tool", "args": {}, "depends_on": []},
    ], results=[
        {"step_id": "s1", "ok": True, "output": "X 的内容"},
        {"step_id": "s2", "ok": False, "output": "", "error": "backend_down"},
    ])
    s2["domain"] = "creation"
    v2 = asyncio.run(verify_node(s2, {}))["verification"]
    assert v2["status"] == "partial"
    assert len(v2["outstanding"]) == 1
    assert v2["outstanding"][0]["id"] == "s2"
    assert v2["outstanding"][0]["reason"].startswith("backend_down")

    # pending（依赖悬空未执行）→ 计入 outstanding，同样 partial
    s3 = _state(query="对比 X 和 Y", plan_steps=[
        {"id": "s1", "description": "读 X", "target": "tool", "args": {}, "depends_on": []},
        {"id": "s2", "description": "读 Y", "target": "tool", "args": {},
         "depends_on": ["s1", "s9"]},
    ], results=[{"step_id": "s1", "ok": True, "output": "X"}])
    s3["domain"] = "creation"
    v3 = asyncio.run(verify_node(s3, {}))["verification"]
    assert v3["status"] == "partial"
    assert any(o["id"] == "s2" for o in v3["outstanding"])

    # 空 plan → no_evidence
    s4 = _state(query="随便", results=[])
    s4["domain"] = "creation"
    assert asyncio.run(verify_node(s4, {}))["verification"]["status"] == "no_evidence"


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
    test_decide_mode_forced()
    test_parse_plan_targetless()
    test_step_agent_loop()
    test_parse_plan_json()
    test_subagent_task_folding()
    test_executor_topological()
    test_executor_tracks_statuses()
    test_statused_plan_marks_skipped()
    test_verify_deterministic()
    print("Phase 7 plan self-check OK")
