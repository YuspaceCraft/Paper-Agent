"""Phase 7 self-check — plan heuristic + executor topological order.

Run: python agent/tests/test_plan.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import agent.plan as plan
from agent.plan import decide_mode, executor_node, verify_node, _subagent_task
from agent.nodes import _synthesize_plan
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class _Msg:
    type = "human"

    def __init__(self, content):
        self.content = content


def _state(query="", entities=None, focus=None, plan_steps=None, results=None,
           resolved=None, needs_planning=None, domain=None):
    s = {
        "messages": [_Msg(query)],
        "entities": entities or [],
        "focus_papers": focus or [],
        "plan": plan_steps or [],
        "subagent_results": results or [],
        "resolved": resolved or {"papers": [], "section": None},
    }
    # needs_planning 缺省时**不写入**，专门测「旧状态无字段 → 兜底启发式」分支
    if needs_planning is not None:
        s["needs_planning"] = needs_planning
    if domain is not None:
        s["domain"] = domain
    return s


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
    assert steps[0]["required_scope"] == "preview"
    assert steps[0]["delivery"] == "answer"
    # 缺描述 → 整单拒绝（不降级为空洞计划）
    assert plan._parse_steps(
        '{"steps": [{"id": "s1", "depends_on": []}]}'
    ) is None


class _FakeStepTool:
    name = "fetch_content"

    async def ainvoke(self, args, config=None):
        return '{"outcome": "succeeded", "data": {"content": "RMNet 的方法部分…"}}'


def test_step_agent_loop():
    """LLM 逐步执行：一步内多次工具调用，无工具调用的文本即步骤答案。"""
    import agent.plan as plan_mod
    import agent.tools as tools_mod
    import agent.nodes as nodes_mod
    from langchain_core.messages import AIMessage

    llm_rounds: list[list[str]] = []

    async def fake_llm(model, msgs, *, emit_tokens=True, config=None):
        llm_rounds.append([getattr(m, "type", type(m).__name__) for m in msgs])
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
    assert out["outcome"] == "succeeded"
    assert "损失函数" in out["output"]
    assert len(llm_rounds) == 2          # 工具轮 + 收尾轮
    assert llm_rounds[1] == ["system", "human", "ai", "tool"]
    assert out["output"] == "RMNet 采用 xx 损失函数训练。"
    assert [m.type for m in out["messages"]] == ["ai", "tool"]
    assert out["messages"][0].tool_calls[0]["id"] == "c1"
    assert "RMNet 的方法部分" in out["messages"][1].content


def test_run_step_records_direct_tool_context():
    """Deterministic plan tools must also enter the standard message history."""
    import agent.plan as plan_mod
    import agent.tools as tools_mod
    from langchain_core.messages import ToolMessage

    fake_tool = _FakeStepTool()
    orig = tools_mod.get_cached_tools
    try:
        tools_mod.get_cached_tools = lambda: [fake_tool]
        out = asyncio.run(plan_mod._run_step(
            {
                "id": "s1",
                "description": "read",
                "target": "tool",
                "args": {"tool": "fetch_content", "paper_name": "RMNet"},
            },
            {"context": {}},
            {},
        ))
    finally:
        tools_mod.get_cached_tools = orig

    assert out["outcome"] == "succeeded"
    assert [m.type for m in out["messages"]] == ["ai", "tool"]
    ai, tool = out["messages"]
    assert ai.tool_calls[0]["name"] == "fetch_content"
    assert ai.tool_calls[0]["args"]["paper_name"] == "RMNet"
    assert isinstance(tool, ToolMessage)
    assert tool.tool_call_id == ai.tool_calls[0]["id"]
    assert '"outcome": "succeeded"' in tool.content


def test_step_retry_keeps_all_tool_call_result_pairs():
    """A retry is another call and must not overwrite the first attempt."""
    import agent.plan as plan_mod
    from langchain_core.messages import AIMessage, ToolMessage

    calls = {"n": 0}

    async def fake_run(step, state, config):
        calls["n"] += 1
        call_id = f"c{calls['n']}"
        messages = [
            AIMessage(content="", tool_calls=[{
                "name": "fetch_content",
                "args": {"paper_name": "RMNet"},
                "id": call_id,
            }]),
            ToolMessage(
                content=(
                    '{"outcome":"timed_out","error_type":"transient",'
                    '"error":"temporary"}'
                    if calls["n"] == 1
                    else '{"outcome":"succeeded","data":{"text":"ok"}}'
                ),
                tool_call_id=call_id,
                name="fetch_content",
            ),
        ]
        return {
            "step_id": "s1",
            "outcome": "timed_out" if calls["n"] == 1 else "succeeded",
            "output": messages[-1].content,
            "error": "temporary" if calls["n"] == 1 else "",
            "messages": messages,
        }

    orig = plan_mod._run_step
    plan_mod._run_step = fake_run
    try:
        out = asyncio.run(plan_mod._execute_step_with_retries(
            {
                "id": "s1",
                "description": "read",
                "target": "tool",
                "args": {"tool": "fetch_content", "paper_name": "RMNet"},
            },
            {"context": {}},
            {},
            set(),
        ))
    finally:
        plan_mod._run_step = orig

    assert calls["n"] == 2
    assert out["outcome"] == "succeeded"
    assert [
        m.tool_calls[0]["id"] if getattr(m, "tool_calls", None)
        else m.tool_call_id
        for m in out["messages"]
    ] == ["c1", "c1", "c2", "c2"]
    assert out["messages"][1].content.startswith('{"outcome":"timed_out"')
    assert out["messages"][3].content.startswith('{"outcome":"succeeded"')


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


def test_plan_step_budget_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_PLAN_STEP_MAX_STEPS", "3")
    assert plan._step_budget() == 3
    monkeypatch.setenv("AGENT_PLAN_STEP_MAX_STEPS", "0")
    assert plan._step_budget() == 10


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
    assert "required_scope: section" in _subagent_task(
        "read", {}, required_scope="section", delivery="answer",
    )


def test_fallback_plan_sets_scope_from_request():
    state = _state(query="给出这篇论文的完整原文",
                   resolved=_resolved(("RMNet", "EXACT")))
    steps = plan._fallback_plan(state)
    assert steps[0]["required_scope"] == "full"


def test_section_hints_preserve_ordinal_semantics():
    text = plan._format_section_hints({
        "sections": [
            {"ordinal": 3, "text": "第三章"},
            {"ordinal": 4, "text": "第四章"},
        ],
    })
    assert "section 3" in text
    assert "section 4" in text
    assert "do not require the literal ordinal phrase" in text


def test_agent_discovery_hints_include_sections_without_paper_match():
    import agent.nodes as nodes_mod

    text = nodes_mod._format_resolved({
        "papers": [],
        "sections": [
            {"ordinal": 3, "text": "第三章"},
            {"ordinal": 4, "text": "第四章"},
        ],
    })
    assert "ordinal 3" in text
    assert "ordinal 4" in text
    assert "Do not require" in text


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


def test_decide_mode_planning_signal():
    # 主信号 = 理解层按任务结构的 needs_planning，与 domain 无关 —— 通用覆盖一切
    # 单动作请求（下载/翻译/收藏/润色一句 → react），不枚举动词。
    assert decide_mode(_state(query="帮我从arXiv下载一篇NLP 2025年的论文，下载到本地",
                              needs_planning=False, domain="creation")) == "react"
    assert decide_mode(_state(query="翻译这篇论文的摘要",
                              needs_planning=False, domain="paper")) == "react"
    assert decide_mode(_state(query="润色一下这句话",
                              needs_planning=False, domain="creation")) == "react"
    # 多步任务 → plan（不看 domain）
    assert decide_mode(_state(query="写一篇 RMNet 综述", needs_planning=True)) == "plan"
    assert decide_mode(_state(query="对比 RMNet 和 SRN 的 loss 设计", needs_planning=True)) == "plan"
    # 标签漏判（False）但解析层观测到 ≥2 篇确证论文 → 结构约束强制 plan
    assert decide_mode(_state(query="RMNet 和 SRN", needs_planning=False,
                              resolved=_resolved(("RMNet", "EXACT"), ("SRN", "EXACT")))) == "plan"
    # 无 needs_planning 字段（旧 checkpoint / LLM degrade）→ 回落 v15 前启发式，零回归
    s = _state(query="对比 RMNet 和 SRN 的 loss 设计", domain="paper")
    assert decide_mode(s) == "plan"
    s = _state(query="RMNet 的 loss 是什么", domain="paper")
    assert decide_mode(s) == "react"
    s = _state(query="帮我从arXiv下载一篇NLP 2025年的论文，下载到本地", domain="creation")
    assert decide_mode(s) == "plan"  # 旧状态无信号 → 按域兜底（真实运行总有信号）


def test_executor_tracks_statuses():
    # executor 回填每步 status + done/total 计数
    async def fake_run(step, state, config):
        step_id = step["id"]
        if step_id == "bad":
            return {"step_id": step_id, "outcome": "failed", "output": "", "error": "boom"}
        return {"step_id": step_id, "outcome": "succeeded", "output": "ok"}

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


def test_executor_moves_step_messages_to_parent_history():
    """Step-local call pairs are flattened into the parent message stream."""
    import agent.plan as plan_mod
    from langchain_core.messages import AIMessage, ToolMessage

    async def fake_run(step, state, config):
        call_id = f"call-{step['id']}"
        return {
            "step_id": step["id"],
            "outcome": "succeeded",
            "output": step["id"],
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "name": "fetch_content",
                    "args": {"paper_name": step["id"]},
                    "id": call_id,
                }]),
                ToolMessage(
                    content='{"outcome":"succeeded","data":{"text":"evidence"}}',
                    tool_call_id=call_id,
                    name="fetch_content",
                ),
            ],
        }

    orig = plan_mod._run_step
    plan_mod._run_step = fake_run
    try:
        out = asyncio.run(plan_mod.executor_node(_state(plan_steps=[
            {"id": "a", "description": "", "target": "tool",
             "args": {}, "depends_on": []},
            {"id": "b", "description": "", "target": "tool",
             "args": {}, "depends_on": []},
        ]), {}))
    finally:
        plan_mod._run_step = orig

    assert [
        m.tool_calls[0]["id"] if getattr(m, "tool_calls", None)
        else m.tool_call_id
        for m in out["messages"]
    ] == ["call-a", "call-a", "call-b", "call-b"]
    assert all("messages" not in result for result in out["subagent_results"])


def test_plan_synthesis_receives_tool_history_and_legacy_operations(monkeypatch):
    """Synthesis must see call/result pairs and fall back for old checkpoints."""
    import agent.nodes as nodes_mod
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    captured = {}

    async def fake_llm(model, messages, *, emit_tokens=True, config=None):
        captured["messages"] = list(messages)
        return AIMessage(content="final")

    monkeypatch.setattr(nodes_mod, "_get_model", lambda *a, **k: object())
    monkeypatch.setattr(nodes_mod, "_stream_llm", fake_llm)

    state = {
        "messages": [
            HumanMessage(content="问题"),
            AIMessage(content="", tool_calls=[{
                "name": "fetch_content",
                "args": {"paper_name": "RMNet"},
                "id": "live-op",
            }]),
            ToolMessage(
                content='{"outcome":"succeeded","data":{"text":"live evidence"}}',
                tool_call_id="live-op",
                name="fetch_content",
            ),
        ],
        "plan": [{"id": "s1", "description": "read"}],
        "subagent_results": [{
            "step_id": "s1",
            "outcome": "succeeded",
            "output": "step summary",
        }],
        "operation_results": {
            "live-op": {
                "operation_id": "live-op",
                "outcome": "succeeded",
                "data": {"text": "live evidence"},
                "meta": {"tool_name": "fetch_content"},
            },
            "old-op": {
                "operation_id": "old-op",
                "outcome": "succeeded",
                "data": {"text": "legacy evidence"},
                "meta": {"tool_name": "fetch_content"},
            },
        },
        "verification": {"status": "satisfied"},
    }

    out = asyncio.run(nodes_mod._synthesize_plan(state, {}))
    assert out["messages"][-1].content == "final"
    sent = captured["messages"]
    assert any(
        getattr(message, "type", "") == "ai" and message.tool_calls
        for message in sent
    )
    assert any(
        getattr(message, "type", "") == "tool"
        and "live evidence" in message.content
        for message in sent
    )
    assert "legacy evidence" in sent[-1].content


def test_statused_plan_marks_skipped():
    # skipped（守卫生效）与 pending（未执行）在状态回填里的区分
    results = {
        "s1": {"step_id": "s1", "outcome": "succeeded", "output": "x"},
        "s2": {"step_id": "s2", "outcome": "skipped", "output": "[guard] ...", "skipped": True},
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
        {"step_id": "s1", "outcome": "succeeded", "output": "X 的内容"},
        {"step_id": "s2", "outcome": "succeeded", "output": "Y 的内容"},
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
        {"step_id": "s1", "outcome": "succeeded", "output": "X 的内容"},
        {"step_id": "s2", "outcome": "failed", "output": "", "error": "backend_down"},
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
    ], results=[{"step_id": "s1", "outcome": "succeeded", "output": "X"}])
    s3["domain"] = "creation"
    v3 = asyncio.run(verify_node(s3, {}))["verification"]
    assert v3["status"] == "partial"
    assert any(o["id"] == "s2" for o in v3["outstanding"])

    # 空 plan → no_evidence
    s4 = _state(query="随便", results=[])
    s4["domain"] = "creation"
    assert asyncio.run(verify_node(s4, {}))["verification"]["status"] == "no_evidence"


def test_verify_creation_uses_persisted_sections(monkeypatch):
    import agent.domains.creation as creation

    async def fake_progress(doc_id):
        assert doc_id == "doc-1"
        return {
            "doc_id": doc_id,
            "title": "review",
            "sections": [
                {"section_id": "intro", "title": "Intro", "status": "done"},
                {"section_id": "method", "title": "Method", "status": "pending"},
            ],
        }

    monkeypatch.setattr(creation, "doc_progress", fake_progress)
    state = _state(query="写综述", plan_steps=[
        {"id": "intro", "description": "写 Intro", "target": "creator",
         "args": {"doc_id": "doc-1", "section_id": "intro"}, "depends_on": []},
        {"id": "method", "description": "写 Method", "target": "creator",
         "args": {"doc_id": "doc-1", "section_id": "method"}, "depends_on": []},
    ], results=[
        {"step_id": "intro", "outcome": "succeeded", "output": "written"},
        {"step_id": "method", "outcome": "failed", "output": "", "error": "missing"},
    ], domain="creation")
    state["doc_id"] = "doc-1"

    verification = asyncio.run(verify_node(state, {}))["verification"]
    assert verification["status"] == "partial"
    assert verification["done"] == 1 and verification["total"] == 2
    assert verification["validator"] == "creation:doc_progress"


def test_verify_coding_checks_experiment_state(monkeypatch):
    import agent.domains.coding as coding

    monkeypatch.setattr(coding, "_load_exp", lambda exp_id: {
        "exp_id": exp_id,
        "name": "baseline",
        "status": "done",
        "exit_code": 0,
        "metrics": {"accuracy": 0.9},
    })
    state = _state(query="跑实验", plan_steps=[
        {"id": "run", "description": "run", "target": "coder",
         "args": {"project": "demo"}, "depends_on": []},
    ], results=[
        {"step_id": "run", "outcome": "succeeded", "output": "PROJECT: demo\nEXP: exp123"},
    ], domain="coding")

    verification = asyncio.run(verify_node(state, {}))["verification"]
    assert verification["status"] == "satisfied"
    assert verification["done"] == 1 and verification["total"] == 1
    assert verification["validator"] == "coding:experiment_store"


def test_verify_paper_requires_real_step_evidence(monkeypatch):
    async def fake_goal(*args, **kwargs):
        return "satisfied"

    monkeypatch.setattr(plan, "_verify_goal", fake_goal)
    state = _state(query="read paper", plan_steps=[
        {"id": "read", "description": "read", "target": "tool",
         "args": {}, "depends_on": []},
    ], results=[{
        "step_id": "read", "outcome": "failed", "output": "", "error": "empty",
    }], domain="paper")

    verification = asyncio.run(verify_node(state, {}))["verification"]
    assert verification["status"] == "no_evidence"
    assert verification["validator"] == "paper:plan_evidence"


def test_verify_does_not_report_no_evidence_when_operations_succeeded(monkeypatch):
    async def fake_goal(*args, **kwargs):
        raise AssertionError("complete deterministic plan must skip verifier LLM")

    import agent.nodes as nodes_mod

    monkeypatch.setattr(nodes_mod, "_get_model", lambda *a, **k: object())
    monkeypatch.setattr(plan, "_verify_goal", fake_goal)
    state = _state(query="read paper", plan_steps=[
        {"id": "read", "description": "read", "target": "auto",
         "args": {}, "depends_on": []},
    ], results=[{
        "step_id": "read",
        "outcome": "succeeded",
        "output": "section summaries are present",
    }], domain="paper")
    state["operation_results"] = {
        "op-1": {
            "outcome": "succeeded",
            "data": "raw section evidence",
            "meta": {"tool_name": "fetch_content"},
        },
    }

    verification = asyncio.run(verify_node(state, {}))["verification"]
    assert verification["status"] == "satisfied"
    assert verification["authoritative"] is True
    assert verification["evidence_steps"] == 1


def test_verify_complete_verbatim_plan_is_authoritative(monkeypatch):
    async def unexpected_goal_check(*_args, **_kwargs):
        raise AssertionError("complete verbatim output must not be downgraded by LLM")

    monkeypatch.setattr(plan, "_verify_goal", unexpected_goal_check)

    def exchange(call_id, section, text):
        assistant = AIMessage(content="", tool_calls=[{
            "name": "fetch_content",
            "args": {"paper_name": "paper", "section": section},
            "id": call_id,
            "type": "tool_call",
        }])
        result = ToolMessage(
            content=json.dumps({
                "schema_version": "1.0",
                "outcome": "succeeded",
                "data": text,
                "continuation": {"next_offset": None, "eof": True},
            }),
            tool_call_id=call_id,
            name="fetch_content",
        )
        return assistant, result

    messages = [HumanMessage(content="give Chapter 3 and Chapter 4 original text")]
    messages.extend(exchange("c3", "3", "CHAPTER THREE"))
    messages.extend(exchange("c4", "4", "CHAPTER FOUR"))
    state = {
        "messages": messages,
        "domain": "paper",
        "plan": [{
            "id": "s1",
            "description": "read chapters 3 and 4",
            "required_scope": "section",
            "delivery": "answer",
        }],
        "subagent_results": [{
            "step_id": "s1",
            "outcome": "succeeded",
            "output": "CHAPTER THREE\n\nCHAPTER FOUR",
        }],
    }

    verification = asyncio.run(verify_node(state, {}))["verification"]
    assert verification["status"] == "satisfied"
    assert verification["authoritative"] is True
    assert verification["evidence_steps"] == 1


def test_verify_operation_preview_contains_raw_tool_data():
    text = plan._operation_evidence_for_verify({
        "operation_results": {
            "op-1": {
                "outcome": "succeeded",
                "data": "complete section evidence",
                "meta": {"tool_name": "fetch_content"},
            },
        },
    }, 1000)
    assert "fetch_content" in text
    assert "complete section evidence" in text


def test_executor_topological():
    call_order = []

    async def fake_run(step, state, config):
        call_order.append(step["id"])
        return {"step_id": step["id"], "outcome": "succeeded", "output": step["id"]}

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


def test_executor_parallelizes_read_only_tool_steps():
    from agent.core.contracts import ToolSpec
    from agent.core.tool_registry import ToolRegistry
    import agent.tools as tools_mod

    active = {"n": 0, "max": 0}

    async def fake_run(step, state, config):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await asyncio.sleep(0.03)
        active["n"] -= 1
        return {"step_id": step["id"], "outcome": "succeeded", "output": step["id"]}

    old_registry = tools_mod._tool_registry
    old_run = plan._run_step
    try:
        tools_mod._tool_registry = ToolRegistry([
            ToolSpec(name="read_a"),
            ToolSpec(name="read_b"),
        ])
        plan._run_step = fake_run
        out = asyncio.run(executor_node(_state(plan_steps=[
            {"id": "a", "description": "", "target": "tool",
             "args": {"tool": "read_a"}, "depends_on": []},
            {"id": "b", "description": "", "target": "tool",
             "args": {"tool": "read_b"}, "depends_on": []},
        ]), {}))
    finally:
        tools_mod._tool_registry = old_registry
        plan._run_step = old_run

    assert active["max"] == 2
    assert [r["step_id"] for r in out["subagent_results"]] == ["a", "b"]


def test_executor_serializes_same_resource_writes():
    active = {"n": 0, "max": 0}

    async def fake_run(step, state, config):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await asyncio.sleep(0.03)
        active["n"] -= 1
        return {"step_id": step["id"], "outcome": "succeeded", "output": step["id"]}

    old_run = plan._run_step
    try:
        plan._run_step = fake_run
        out = asyncio.run(executor_node(_state(plan_steps=[
            {"id": "write-a", "description": "", "target": "creator",
             "args": {"doc_id": "doc-1"}, "depends_on": []},
            {"id": "write-b", "description": "", "target": "creator",
             "args": {"doc_id": "doc-1"}, "depends_on": []},
        ]), {}))
    finally:
        plan._run_step = old_run

    assert active["max"] == 1
    assert out["plan_progress"] == 2


def test_executor_allows_disjoint_write_resources_in_parallel():
    active = {"n": 0, "max": 0}

    async def fake_run(step, state, config):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await asyncio.sleep(0.03)
        active["n"] -= 1
        return {"step_id": step["id"], "outcome": "succeeded", "output": step["id"]}

    old_run = plan._run_step
    try:
        plan._run_step = fake_run
        asyncio.run(executor_node(_state(plan_steps=[
            {"id": "write-a", "description": "", "target": "creator",
             "args": {"doc_id": "doc-1"}, "depends_on": []},
            {"id": "write-b", "description": "", "target": "creator",
             "args": {"doc_id": "doc-2"}, "depends_on": []},
        ]), {}))
    finally:
        plan._run_step = old_run

    assert active["max"] == 2


def test_assemble_verbatim_sections_joins_complete_pages():
    def exchange(call_id, section, offset, text, eof):
        assistant = AIMessage(content="", tool_calls=[{
            "name": "fetch_content",
            "args": {
                "paper_name": "paper",
                "section": section,
                "offset": offset,
            },
            "id": call_id,
            "type": "tool_call",
        }])
        payload = {
            "schema_version": "1.0",
            "outcome": "succeeded",
            "data": text,
            "continuation": {
                "reason": "requested_page",
                "next_offset": None if eof else offset + len(text),
                "eof": eof,
            },
        }
        tool = ToolMessage(
            content=json.dumps(payload),
            tool_call_id=call_id,
            name="fetch_content",
        )
        return assistant, tool

    messages = []
    messages.extend(exchange("c1", "Chapter 3", 0, "abc", False))
    messages.extend(exchange("c2", "Chapter 3", 3, "def", True))
    result = plan._assemble_verbatim_sections(
        messages,
        {"description": "Return the original Chapter 3", "required_scope": "section"},
        {"messages": [HumanMessage(content="give Chapter 3 original text")]},
    )
    assert result == "abcdef"


def test_synthesize_verbatim_plan_skips_llm(monkeypatch):
    import agent.nodes as nodes
    from agent.stream import reset_event_queue, set_event_queue

    state = {
        "messages": [HumanMessage(content="give the original Chapter 3 and Chapter 4")],
        "mode": "plan",
        "domain": "paper",
        "plan": [
            {
                "id": "s3",
                "description": "Return Chapter 3",
                "required_scope": "section",
                "delivery": "answer",
            },
            {
                "id": "s4",
                "description": "Return Chapter 4",
                "required_scope": "section",
                "delivery": "answer",
            },
        ],
        "subagent_results": [
            {"step_id": "s3", "outcome": "succeeded", "output": "CHAPTER THREE"},
            {"step_id": "s4", "outcome": "succeeded", "output": "CHAPTER FOUR"},
        ],
        "verification": {"status": "satisfied"},
    }

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("synthesize LLM should not run for verbatim text")

    monkeypatch.setattr(nodes, "_get_model", fail_if_called)
    events = asyncio.Queue()
    token = set_event_queue(events)
    try:
        result = asyncio.run(_synthesize_plan(state, {}))
    finally:
        reset_event_queue(token)
    assert result["messages"][0].content == "CHAPTER THREE\n\nCHAPTER FOUR"
    assert events.get_nowait() == {
        "type": "token",
        "content": "CHAPTER THREE\n\nCHAPTER FOUR",
    }


def test_assemble_verbatim_sections_accepts_artifact_continuation():
    fetch_call = AIMessage(content="", tool_calls=[{
        "name": "fetch_content",
        "args": {"paper_name": "paper", "section": "Chapter 3"},
        "id": "fetch-1",
        "type": "tool_call",
    }])
    fetch_page = ToolMessage(
        content=json.dumps({
            "schema_version": "1.0",
            "outcome": "succeeded",
            "data": "abc",
            "artifacts": [{"artifact_id": "artifact-1"}],
            "continuation": {"next_offset": 3, "eof": False},
        }),
        tool_call_id="fetch-1",
        name="fetch_content",
    )
    artifact_call = AIMessage(content="", tool_calls=[{
        "name": "artifact_read",
        "args": {"artifact_id": "artifact-1", "offset": 3},
        "id": "artifact-1-call",
        "type": "tool_call",
    }])
    artifact_page = ToolMessage(
        content=json.dumps({
            "schema_version": "1.0",
            "outcome": "succeeded",
            "data": "def",
            "continuation": {"next_offset": None, "eof": True},
        }),
        tool_call_id="artifact-1-call",
        name="artifact_read",
    )

    result = plan._assemble_verbatim_sections(
        [fetch_call, fetch_page, artifact_call, artifact_page],
        {"description": "Return the original Chapter 3", "required_scope": "section"},
        {"messages": [HumanMessage(content="give Chapter 3 original text")]},
    )
    assert result == "abcdef"


def test_step_agent_stops_after_verbatim_pages_are_complete():
    import agent.plan as plan_mod
    import agent.tools as tools_mod
    import agent.nodes as nodes_mod

    rounds = {"count": 0}

    async def fake_llm(model, msgs, *, emit_tokens=True, config=None):
        rounds["count"] += 1
        offset = 0 if rounds["count"] == 1 else 3
        return AIMessage(content="", tool_calls=[{
            "id": f"c{rounds['count']}",
            "name": "fetch_content",
            "args": {
                "paper_name": "paper",
                "section": "Chapter 3",
                "offset": offset,
            },
        }])

    class FakeTool:
        name = "fetch_content"

        async def ainvoke(self, args, config=None):
            offset = int(args.get("offset") or 0)
            return json.dumps({
                "schema_version": "1.0",
                "outcome": "succeeded",
                "data": "abc" if offset == 0 else "def",
                "continuation": {
                    "next_offset": None if offset else 3,
                    "eof": bool(offset),
                },
            })

    orig_llm, orig_tools, orig_model = (
        nodes_mod._stream_llm,
        tools_mod.get_cached_tools,
        nodes_mod._get_bound_model,
    )
    try:
        nodes_mod._stream_llm = fake_llm
        tools_mod.get_cached_tools = lambda: [FakeTool()]
        nodes_mod._get_bound_model = lambda *_args, **_kwargs: object()
        state = _state(query="give Chapter 3 original text")
        result = asyncio.run(plan_mod._run_step_agent(
            {
                "id": "s1",
                "description": "Return Chapter 3 original text",
                "required_scope": "section",
                "delivery": "answer",
            },
            state,
            {"configurable": {}},
        ))
    finally:
        nodes_mod._stream_llm = orig_llm
        tools_mod.get_cached_tools = orig_tools
        nodes_mod._get_bound_model = orig_model

    assert rounds["count"] == 1
    assert result["output"] == "abcdef"


def test_step_agent_autofetches_missing_requested_sections():
    import agent.plan as plan_mod
    import agent.tools as tools_mod
    import agent.nodes as nodes_mod

    rounds = {"count": 0}

    async def fake_llm(model, msgs, *, emit_tokens=True, config=None):
        rounds["count"] += 1
        return AIMessage(content="", tool_calls=[{
            "id": "c1",
            "name": "fetch_content",
            "args": {"paper_name": "paper", "section": "3"},
        }])

    class FakeTool:
        name = "fetch_content"

        async def ainvoke(self, args, config=None):
            section = str(args.get("section") or "")
            return json.dumps({
                "schema_version": "1.0",
                "outcome": "succeeded",
                "data": f"SECTION {section}",
                "continuation": {"next_offset": None, "eof": True},
            })

    orig_llm, orig_tools, orig_model = (
        nodes_mod._stream_llm,
        tools_mod.get_cached_tools,
        nodes_mod._get_bound_model,
    )
    try:
        nodes_mod._stream_llm = fake_llm
        tools_mod.get_cached_tools = lambda: [FakeTool()]
        nodes_mod._get_bound_model = lambda *_args, **_kwargs: object()
        state = _state(query="给出第三章与第四章的原文内容")
        result = asyncio.run(plan_mod._run_step_agent(
            {
                "id": "s1",
                "description": "提取第三章与第四章原文内容",
                "required_scope": "section",
                "delivery": "answer",
            },
            state,
            {"configurable": {}},
        ))
    finally:
        nodes_mod._stream_llm = orig_llm
        tools_mod.get_cached_tools = orig_tools
        nodes_mod._get_bound_model = orig_model

    assert rounds["count"] == 1
    assert result["output"] == "SECTION 3\n\nSECTION 4"


def test_step_agent_autofetches_after_broad_search_without_more_llm():
    import agent.plan as plan_mod
    import agent.tools as tools_mod
    import agent.nodes as nodes_mod

    rounds = {"count": 0}

    async def fake_llm(model, msgs, *, emit_tokens=True, config=None):
        rounds["count"] += 1
        return AIMessage(content="", tool_calls=[{
            "id": "search-1",
            "name": "search_papers",
            "args": {"query": "remote sensing image"},
        }])

    class SearchTool:
        name = "search_papers"

        async def ainvoke(self, args, config=None):
            return json.dumps({
                "schema_version": "1.0",
                "outcome": "succeeded",
                "data": {
                    "results": [{"paper": "paper"}],
                    "papers": ["paper"],
                    "total": 1,
                },
            })

    class FetchTool:
        name = "fetch_content"

        async def ainvoke(self, args, config=None):
            section = str(args.get("section") or "")
            return json.dumps({
                "schema_version": "1.0",
                "outcome": "succeeded",
                "data": f"SECTION {section}",
                "continuation": {"next_offset": None, "eof": True},
            })

    orig_llm, orig_tools, orig_model = (
        nodes_mod._stream_llm,
        tools_mod.get_cached_tools,
        nodes_mod._get_bound_model,
    )
    try:
        nodes_mod._stream_llm = fake_llm
        tools_mod.get_cached_tools = lambda: [SearchTool(), FetchTool()]
        nodes_mod._get_bound_model = lambda *_args, **_kwargs: object()
        state = _state(
            query="在本地知识库找到一篇遥感图像论文，并给出第三章与第四章的原文内容"
        )
        result = asyncio.run(plan_mod._run_step_agent(
            {
                "id": "s1",
                "description": "找到一篇遥感论文并提取第三章与第四章原文",
                "required_scope": "section",
                "delivery": "answer",
            },
            state,
            {"configurable": {}},
        ))
    finally:
        nodes_mod._stream_llm = orig_llm
        tools_mod.get_cached_tools = orig_tools
        nodes_mod._get_bound_model = orig_model

    assert rounds["count"] == 1
    assert result["output"] == "SECTION 3\n\nSECTION 4"


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
